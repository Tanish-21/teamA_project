# Mobile Specification Advisor

A Streamlit app that combines a conversational Gemini advisor with a locally
trained machine-learning model. The ML model predicts phone specifications from
budget and preferences; Gemini explains those predictions in a conversational
way.

> **Data note:** `phone_specs_training.csv` contains illustrative starter
> examples, not verified phone-market data. Replace it with accurate,
> consistently labeled data before relying on the recommendations.

## Features

- Ask mobile-specification questions in a Gemini-powered chat.
- Train the recommendation model from the terminal using a labeled CSV.
- Predict RAM, storage, battery capacity, and camera resolution from budget,
  use case, and priority.
- Show predictions in the chat and ask Gemini to explain them.
- Report mean absolute error (MAE) for a held-out test split during training.

The chat and ML model are separate components: Gemini provides the
conversation, while the local scikit-learn model provides the structured
specification predictions.

## Requirements

- Python and pip installed.
- A Gemini API key for chat. ML model training and prediction do not require
  the Gemini API key.

Install the packages from the project directory:

```powershell
pip install -r requirements.txt
```

## Configure Gemini

Create a `.env` file in the project directory and add your own API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Keep `.env` private. Do not commit or share your API key.

## Prepare the training data

The default training file is `phone_specs_training.csv`. It must contain these
columns:

| Column | Meaning | Example |
| --- | --- | --- |
| `budget` | Numeric budget, using the same currency/unit for every row | `30000` |
| `use_case` | Intended use | `Gaming` |
| `priority` | Main preference | `Performance` |
| `ram_gb` | Example RAM recommendation in GB | `8` |
| `storage_gb` | Example storage recommendation in GB | `256` |
| `battery_mah` | Example battery recommendation in mAh | `5000` |
| `camera_mp` | Example camera resolution in MP | `50` |

Use positive numeric values for the budget and specification columns, and
non-empty values for `use_case` and `priority`. At least 8 complete rows are
required. Keep budget units consistent; the app uses the range in the training
data to decide whether to make an ML prediction.

## Train the model

From the project directory, run:

```powershell
python ml_trainer.py
```

The trainer reads `phone_specs_training.csv`, reserves 25% of the rows for
testing, trains a Random Forest regressor on the other rows, prints held-out
MAE metrics, and saves the model bundle to `mobile_specs_model.joblib`.

To train from a different CSV:

```powershell
python ml_trainer.py "C:\path\to\your_training_data.csv"
```

To choose a different output path:

```powershell
python ml_trainer.py "C:\path\to\your_training_data.csv" --output "C:\path\to\model.joblib"
```

## Run the app

Start Streamlit from the project directory:

```powershell
streamlit run mobile_advisor.py
```

The app loads `mobile_specs_model.joblib` when it starts. If you retrain the
model while Streamlit is running, stop and restart the app so it loads the new
model.

In the sidebar, enter a budget within the training-data range and select a
known use case and priority. When these inputs are supported, the app displays
the ML-predicted specs and includes them in the prompt sent to Gemini. If the
inputs are missing or outside the model's training range, chat continues
without an ML prediction.

## Project files

- `mobile_advisor.py` — Streamlit chat interface and Gemini integration.
- `ml_trainer.py` — command-line training program.
- `mobile_spec_model.py` — shared model creation, validation, loading, and
  prediction helpers.
- `phone_specs_training.csv` — starter training data; replace with your own
  trusted examples for real recommendations.
- `mobile_specs_model.joblib` — generated trained model; created by the trainer.
- `requirements.txt` — Python dependencies.

Only load a `.joblib` model file that you generated or otherwise trust.
