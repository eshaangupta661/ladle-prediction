"""Train time-forward models and save a single production inference pipeline."""
from pathlib import Path
import json
from datetime import datetime, timezone
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .clean import load_and_clean
from .features import FEATURE_COLUMNS, create_training_features
from .model import CounterBaseline, CounterResidualRegressor

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "src" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_DATA = ROOT / "data" / "raw" / "ladle_data_synthetic.xlsx"
CV_FRACTIONS = (0.6, 0.7, 0.8)   # walk-forward cutoffs; the last one is the reported holdout
BASELINE_NAME = "heat_counter_baseline"


def time_forward_split(frame, train_fraction=0.80):
    """Hold out the newest dates; never randomly shuffle time-series records."""
    dates = sorted(frame["date"].dropna().unique())
    if len(dates) < 5:
        raise ValueError("At least five distinct dates are needed for time-forward validation.")
    cutoff = dates[max(1, int(len(dates) * train_fraction))]
    return frame[frame["date"] < cutoff].copy(), frame[frame["date"] >= cutoff].copy(), str(cutoff)


def candidate_models():
    def residual(model, scale=False):
        steps = [("imputer", SimpleImputer(strategy="median"))]
        steps += [("scaler", StandardScaler())] if scale else []
        return CounterResidualRegressor(Pipeline(steps + [("model", model)]))

    return {
        BASELINE_NAME: CounterBaseline(),
        "direct_extra_trees": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                        ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                                                      random_state=42, n_jobs=-1))]),
        "residual_ridge": residual(Ridge(alpha=10.0), scale=True),
        "residual_xgboost": residual(XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                  subsample=0.8, colsample_bytree=0.8, random_state=42)),
        "residual_extra_trees": residual(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                                             random_state=42, n_jobs=-1)),
    }


def score(y_true, y_pred):
    return {"mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4)}


def train_and_save(data_path=None):
    featured = create_training_features(load_and_clean(Path(data_path or DEFAULT_DATA)))
    labelled = featured[featured["ladle_life"].notna()]
    candidates = candidate_models()

    folds = {name: [] for name in candidates}
    holdout_predictions = {}
    for fraction in CV_FRACTIONS:
        train, test, cutoff = time_forward_split(labelled, fraction)
        for name, estimator in candidates.items():
            prediction = estimator.fit(train[FEATURE_COLUMNS], train["ladle_life"]).predict(test[FEATURE_COLUMNS])
            folds[name].append(score(test["ladle_life"], prediction))
            holdout_predictions[name] = prediction
    results = {name: {"holdout": scores[-1],
                      "cv_mean": {key: round(float(np.mean([fold[key] for fold in scores])), 4)
                                  for key in ("mae", "rmse", "r2")},
                      "cv_folds": scores}
               for name, scores in folds.items()}

    learned = [name for name in candidates if name != BASELINE_NAME]
    chosen_name = min(learned, key=lambda name: results[name]["cv_mean"]["mae"])
    final_model = candidates[chosen_name].fit(labelled[FEATURE_COLUMNS], labelled["ladle_life"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "ladle_life_pipeline.pkl")
    featured.to_csv(PROCESSED_DIR / "ladle_features.csv", index=False)
    _, holdout, cutoff = time_forward_split(labelled, CV_FRACTIONS[-1])
    holdout = holdout[["ladle_no", "campaign_id", "date", "ladle_life"]].assign(
        **{f"pred_{name}": prediction for name, prediction in holdout_predictions.items()})
    holdout.to_csv(PROCESSED_DIR / "holdout_predictions.csv", index=False)

    metadata = {"model_name": chosen_name, "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "feature_columns": FEATURE_COLUMNS, "target": "ladle_life", "time_split_cutoff": cutoff,
                "selection": "lowest mean walk-forward MAE; final model refit on all labelled heats",
                "metrics": results, "training_rows": int(len(labelled) - len(holdout)),
                "test_rows": int(len(holdout)), "total_heats": int(len(featured))}
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
