# Ladle Life Prediction

Cycle-aware time-series prediction of current ladle life (heats since the last reline), served through a FastAPI service. Date, shift, and heat number establish record order only; they are not model features.

## Results

Walk-forward validation (train before a cutoff date, test after it; cutoffs at 60/70/80% of dates, no shuffling). Holdout = the last 6 days, 324 heats.

| Model | Walk-forward MAE (heats) | Holdout MAE (heats) | Holdout R² |
|---|---|---|---|
| Direct ExtraTrees (original approach) | 23.82 | 16.61 | −0.645 |
| Heat counter only (reference) | 0.044 | 0.046 | 0.9996 |
| Counter + Ridge correction | 0.068 | 0.064 | 0.9996 |
| Counter + XGBoost correction | 0.081 | 0.082 | 0.9996 |
| **Counter + ExtraTrees correction (deployed)** | **0.059** | **0.061** | **0.9996** |

What changed from the original approach:

- **No heats are dropped during cleaning.** Missing or out-of-range readings are imputed per ladle. Dropping them had removed ~20% of heats and broken the per-campaign heat count.
- **The model predicts life as the heat counter plus a learned correction.** Tree models trained directly on life cannot predict above the largest life seen in training, so they stalled on long campaigns.
- **No target-derived features at inference.** The starting life of a ladle already in service is supplied once by the operator (`starting_life`).

On this dataset, ladle life is driven almost entirely by the heat count. The learned correction matches the counter within ~0.02 heats and is there to absorb missed or duplicated heats in real plant logs. Process times rise as the lining wears (see the notebook), which is the basis for a future remaining-life model.

## VS Code workflow

1. Open this folder in VS Code and select the project virtual environment as the Python interpreter.
2. Install packages: `python -m pip install -r requirements.txt`.
3. Open `notebooks/01_ladle_life_time_series.ipynb` and run every cell, or run `python -m pipeline.train` from `src`. This trains the model and writes the artifacts.
4. Copy `.env.example` to `.env`, then start the API with `docker compose up --build` or `uvicorn api.main:app --app-dir src --reload`.
5. Run the tests with `pytest`.

## Live prediction inputs

`POST /predict` with `ladle_no`, opening/despatch temperatures, arcing/purging/treatment times, and optionally `heat_no`. Set `new_campaign=true` only after a confirmed reline. For a ladle already in service when it is first tracked, also send `starting_life` (its life counter for that heat). The API retrieves earlier heats of the campaign and calculates the rolling, lag, and cumulative features itself.

```json
{"ladle_no": 8, "opening_temp": 1550, "despatch_temp": 1580, "arcing_time": 20, "purging_time": 12, "treatment_time": 40}
```

Other endpoints: `GET /health`, `GET /predict/health` (model name and holdout metrics), `GET /predictions`, `PATCH /predictions/{id}/actual` (record ground truth). Interactive docs are at `/docs`.

Raw source data belongs in `data/raw` (only the synthetic sample is committed); generated datasets go to `data/processed`; trained artifacts go to `src/models`.
