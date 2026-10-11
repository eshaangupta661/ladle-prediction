# Ladle Life Prediction

Cycle-aware time-series prediction of current ladle life. Date, shift, and heat number establish record order only; they are not model features.

## VS Code workflow

1. Open this folder in VS Code and select the project virtual environment as the Python interpreter.
2. Install packages: `python -m pip install -r requirements.txt`.
3. Open `notebooks/01_ladle_life_time_series.ipynb` and run every cell. This trains a time-forward model and creates the model artifacts.
4. Copy `.env.example` to `.env`, then start the API with `docker compose up --build` or `uvicorn api.main:app --app-dir src --reload`.

## Live prediction inputs

Send `ladle_no`, opening/despatch temperatures, arcing/purging/treatment times, optional `heat_no`, and `new_campaign=true` only after a confirmed reline. The API retrieves earlier heats for that ladle and calculates the rolling, lag, and cumulative features itself.

Raw source data belongs in `data/raw`; generated datasets go to `data/processed`; trained artifacts go to `src/models`.
