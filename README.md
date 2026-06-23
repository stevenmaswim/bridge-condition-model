# Bridge Condition Prediction Model

Predicts bridge condition ratings (deck, superstructure, substructure, culvert) for Texas bridges using XGBoost regression. Predictions are grouped by TxDOT district and Texas climate zone.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Place bridge data CSV in `data/raw/bridge_data.csv`, then run:

```powershell
python main.py
```

Outputs land in `data/outputs/` and trained models in `models/`.

## Project Structure

- `src/data_loader.py` — load and clean raw bridge data
- `src/features.py` — feature engineering and encoding
- `src/model.py` — train/evaluate XGBoost models per target
- `src/predict.py` — generate predictions and grouped summaries
- `config.yaml` — all configurable settings
