import os
import re

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors

from mobile_spec_model import load_model_bundle, predict_specifications

load_dotenv()
trained_bundle = load_model_bundle()
trained_model = trained_bundle["model"] if trained_bundle else None

SYSTEM_PROMPT = """
You are an AI Mobile Specification Advisor.

Help users choose mobile specifications based on their budget, preferred RAM
and storage, battery needs, and intended use (gaming, photography, studying,
or everyday use). Explain recommendations in simple language. Do not invent
real phone prices or claim to know current market prices without verified
data. Ask one follow-up question at a time when needed. Keep every answer
within 3–4 lines, with no headings or unnecessary explanations.

When an ML model prediction is included in the user's message, use those
predicted specifications as the recommendation baseline. Explain them simply
without substituting unprovided specs or claiming they are verified market data.
"""

SUGGESTIONS = [
    ("🎮", "Gaming", "I want a phone for gaming. What specifications should I prioritize?"),
    ("📸", "Photography", "What camera and other specs should I look for in a photography phone?"),
    ("🔋", "Long battery", "I need a phone that lasts all day. Which specifications matter most?"),
    ("✨", "Everyday use", "What are good balanced specifications for everyday use?"),
]

def parse_budget(value: str) -> float | None:
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*([kKmM]?)", value)
    if match is None:
        return None

    amount = float(match.group(1).replace(",", ""))
    suffix = match.group(2).lower()
    if suffix == "k":
        amount *= 1_000
    elif suffix == "m":
        amount *= 1_000_000
    return amount


def show_ml_specs(specs: dict[str, int]) -> None:
    st.markdown("**ML model suggestion**")
    columns = st.columns(4)
    labels = [
        ("RAM", f"{specs['ram_gb']} GB"),
        ("Storage", f"{specs['storage_gb']} GB"),
        ("Battery", f"{specs['battery_mah']} mAh"),
        ("Camera", f"{specs['camera_mp']} MP"),
    ]
    for column, (label, value) in zip(columns, labels):
        column.metric(label, value)


st.set_page_config(
    page_title="Mobile Specification Advisor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #f4f5fb;
        --muted: #a3a9bc;
        --accent: #a78bfa;
        --accent-bright: #c4b5fd;
        --paper: #0d0f16;
        --surface: #151823;
        --surface-raised: #1b1f2c;
        --line: #292d3b;
    }
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Segoe UI', sans-serif;
        color: var(--ink) !important;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, rgba(124, 90, 220, .11), transparent 34%),
            var(--paper);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1180px; padding-top: 2rem; }
    [data-testid="stSidebar"] {
        background: #11131b;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
    .brand {
        display: flex; align-items: center; gap: 10px;
        font-family: 'Segoe UI', sans-serif; font-size: 19px;
        font-weight: 800; letter-spacing: -0.6px; color: var(--ink);
        margin: 0 0 26px 2px;
    }
    .brand-mark {
        display: grid; place-items: center; width: 36px; height: 36px;
        border-radius: 12px; background: linear-gradient(135deg, #8b5cf6, #6d5dfc); color: white;
        font-size: 18px;
    }
    .eyebrow {
        color: var(--accent-bright); font-size: 11px; font-weight: 700;
        letter-spacing: 1.4px; text-transform: uppercase; margin-bottom: 10px;
    }
    .hero-title {
        font-family: 'Segoe UI', sans-serif; font-size: clamp(34px, 5vw, 54px);
        font-weight: 800; letter-spacing: -2.5px; line-height: 1.08;
        margin: 0; color: var(--ink);
    }
    .hero-title span { color: var(--accent-bright); }
    .hero-copy {
        max-width: 580px; margin: 14px 0 0; color: var(--muted);
        font-size: 15px; line-height: 1.65;
    }
    .hero-card {
        position: relative; overflow: hidden; padding: 34px 38px;
        border: 1px solid #343044; border-radius: 24px;
        background:
            radial-gradient(ellipse at 90% 10%, rgba(139, 92, 246, .2), transparent 36%),
            linear-gradient(115deg, #1b1d2a 0%, #191b27 72%, #201d31 100%);
        margin: 4px 0 28px;
    }
    .hero-card:after {
        content: '✳'; position: absolute; right: 7%; top: 5%;
        font-size: 148px; line-height: 1; color: rgba(167,139,250,.13);
        transform: rotate(-14deg);
    }
    .hero-content { position: relative; z-index: 1; }
    .section-title {
        font-family: 'Segoe UI', sans-serif; font-size: 18px;
        font-weight: 700; letter-spacing: -.4px; margin: 0 0 4px;
    }
    .section-copy { color: var(--muted); font-size: 13px; margin: 0 0 14px; }
    div.stButton > button {
        min-height: 58px; width: 100%; text-align: left;
        border: 1px solid #303444; border-radius: 14px;
        background: var(--surface); color: var(--ink); font-weight: 600;
        transition: border-color .15s ease, transform .15s ease, box-shadow .15s ease;
    }
    div.stButton > button:hover {
        border-color: #8970d9; color: var(--accent-bright);
        transform: translateY(-1px); box-shadow: 0 5px 20px rgba(124,90,220,.13);
    }
    div.stButton > button:focus { box-shadow: 0 0 0 2px rgba(167,139,250,.3); }
    [data-testid="stChatInput"] > div {
        border: 1px solid #343849;
        background: #171a25;
    }
    [data-testid="stChatInput"] textarea { color: var(--ink); }
    [data-testid="stChatInput"] textarea::placeholder { color: #858ba0; }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--line); border-radius: 16px;
        background: var(--surface); padding: 16px 19px; margin-bottom: 12px;
    }
    [data-testid="stChatMessage"] p { color: var(--ink); }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
        color: var(--ink);
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        color: var(--ink);
        background: var(--surface-raised);
        border-color: #343849;
    }
    .sidebar-note {
        border: 1px solid #373149; background: #1d1b29; border-radius: 13px;
        padding: 12px 14px; font-size: 12px; line-height: 1.5;
        color: #b9b1cf; margin-top: 18px;
    }
    .footer-note { color: #777d91; font-size: 11px; margin-top: 22px; }
    [data-testid="stAlert"] {
        background: #1b1b29;
        border: 1px solid #3a3454;
        color: var(--ink);
    }
    @media (max-width: 700px) {
        .hero-card { padding: 27px 23px; }
        .hero-title { letter-spacing: -1.5px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><span class="brand-mark">◈</span> Mobile Advisor</div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Make it personal")
    st.caption("A little context helps shape a better recommendation.")
    trained_use_cases = trained_bundle["use_cases"] if trained_bundle else []
    trained_priorities = trained_bundle["priorities"] if trained_bundle else []
    use_case_options = ["Not sure yet", *trained_use_cases] if trained_use_cases else [
        "Not sure yet", "Everyday use", "Gaming", "Photography", "Studying", "Work"
    ]
    priority_options = trained_priorities or [
        "Balanced overall", "Battery life", "Camera quality", "Performance", "Value for money"
    ]
    if st.session_state.get("advisor_use_case") not in use_case_options:
        st.session_state.pop("advisor_use_case", None)
    if st.session_state.get("advisor_priority") not in priority_options:
        st.session_state.pop("advisor_priority", None)
    budget = st.text_input(
        "Your budget (training data units)",
        placeholder="e.g. 25,000 or 25k",
        key="advisor_budget",
    )
    use_case = st.selectbox(
        "Main use",
        use_case_options,
        key="advisor_use_case",
    )
    priority = st.selectbox(
        "What matters most?",
        priority_options,
        key="advisor_priority",
    )
    st.markdown(
        '<div class="sidebar-note">Recommendations focus on specifications, not '
        'unverified prices or specific phone deals.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="footer-note">Thoughtful specs. Smarter choices.</div>', unsafe_allow_html=True)

main_left, main_right = st.columns([8, 2])
with main_left:
    st.markdown(
        """
        <div class="hero-card"><div class="hero-content">
          <div class="eyebrow">Your personal mobile guide</div>
          <h1 class="hero-title">Find the right fit.<br><span>Not the hype.</span></h1>
          <p class="hero-copy">Tell us what you need from your next phone. Get clear, practical
          advice on the specifications that matter to you.</p>
        </div></div>
        """,
        unsafe_allow_html=True,
    )
with main_right:
    st.write("")
    if st.button("＋  New chat", width="stretch"):
        st.session_state.messages = []
        st.session_state.pop("mobile_advisor_chat", None)
        st.rerun()

if trained_model is None:
    st.info(
        "No trained ML model found. Train one from the terminal with "
        "`python ml_trainer.py`; its predictions will then be included in this Gemini chat."
    )

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.warning(
        "Add your Gemini API key as `GEMINI_API_KEY` in a `.env` file to start chatting."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown('<p class="section-title">A good place to start</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Choose a topic, or ask your own question below.</p>',
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(SUGGESTIONS), 2):
        columns = st.columns(2)
        for column, (icon, label, prompt) in zip(columns, SUGGESTIONS[row_start : row_start + 2]):
            with column:
                if st.button(
                    f"{icon}  {label}",
                    key=f"suggestion_{label}",
                    width="stretch",
                    disabled=not api_key,
                ):
                    st.session_state.pending_prompt = prompt

for message in st.session_state.messages:
    with st.chat_message("assistant" if message["role"] == "error" else message["role"]):
        if message.get("ml_specs"):
            show_ml_specs(message["ml_specs"])
        if message["role"] == "error":
            st.error(message["content"])
        else:
            st.markdown(message["content"])

submitted_prompt = st.chat_input("Ask about specs, trade-offs, or what suits your needs…", disabled=not api_key)
prompt = st.session_state.pop("pending_prompt", None) or submitted_prompt

if prompt:
    prompt = prompt.strip()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        preference_lines = []
        if budget.strip():
            preference_lines.append(f"Budget: {budget.strip()}")
        if use_case != "Not sure yet":
            preference_lines.append(f"Main use: {use_case}")
        if priority != "Balanced overall":
            preference_lines.append(f"Top priority: {priority}")
        context = "\n".join(preference_lines)
        ml_specs = None
        ml_context = ""
        if trained_model is not None:
            parsed_budget = parse_budget(budget)
            budget_range = trained_bundle["budget_range"]
            budget_is_supported = (
                parsed_budget is not None
                and budget_range[0] <= parsed_budget <= budget_range[1]
            )
            if budget_is_supported and use_case != "Not sure yet":
                ml_specs = predict_specifications(
                    trained_model,
                    parsed_budget,
                    use_case,
                    priority,
                )
                ml_context = (
                    "\n\nML model prediction from the user's trained examples:\n"
                    f"- RAM: {ml_specs['ram_gb']} GB\n"
                    f"- Storage: {ml_specs['storage_gb']} GB\n"
                    f"- Battery: {ml_specs['battery_mah']} mAh\n"
                    f"- Camera: {ml_specs['camera_mp']} MP\n"
                    "Explain these predicted specs as the baseline recommendation. "
                    "Note that they reflect the training data and are not verified market facts."
                )
        advisor_message = (
            f"Use this context when relevant:\n{context}\n\nQuestion: {prompt}{ml_context}"
            if context
            else f"{prompt}{ml_context}"
        )

        error_message = None
        try:
            client = st.session_state.get("mobile_advisor_client")
            if client is None:
                client = genai.Client(api_key=api_key)
                st.session_state.mobile_advisor_client = client
                st.session_state.pop("mobile_advisor_chat", None)

            chat = st.session_state.get("mobile_advisor_chat")
            if chat is None:
                chat = client.chats.create(
                    model="gemini-flash-latest",
                    config={"system_instruction": SYSTEM_PROMPT},
                )
                st.session_state.mobile_advisor_chat = chat

            try:
                response = chat.send_message(advisor_message)
            except RuntimeError as error:
                if "client has been closed" not in str(error).lower():
                    raise

                client = genai.Client(api_key=api_key)
                chat = client.chats.create(
                    model="gemini-flash-latest",
                    config={"system_instruction": SYSTEM_PROMPT},
                )
                st.session_state.mobile_advisor_client = client
                st.session_state.mobile_advisor_chat = chat
                response = chat.send_message(advisor_message)

            answer = response.text
        except (errors.APIError, RuntimeError) as error:
            answer = None
            error_message = f"Couldn't get a recommendation: {error}"

        if answer:
            assistant_message = {"role": "assistant", "content": answer}
            if ml_specs:
                assistant_message["ml_specs"] = ml_specs
            st.session_state.messages.append(assistant_message)
            with st.chat_message("assistant"):
                if ml_specs:
                    show_ml_specs(ml_specs)
                st.markdown(answer)
        else:
            if not error_message:
                error_message = "The advisor returned an empty response. Please try again."
            error_entry = {"role": "error", "content": error_message}
            if ml_specs:
                error_entry["ml_specs"] = ml_specs
            st.session_state.messages.append(error_entry)
            with st.chat_message("assistant"):
                if ml_specs:
                    show_ml_specs(ml_specs)
                st.error(error_message)
