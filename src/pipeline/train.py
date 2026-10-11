"""Train time-forward models and save a single production inference pipeline."""
from pathlib import Path
import json
from datetime import datetime, timezone
import joblib
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .clean import load_and_clean
from .features import FEATURE_COLUMNS, create_training_features

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "src" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"


def time_forward_split(frame, test_fraction=0.20):
    """Hold out the newest observations; never randomly shuffle time-series records."""
    dates = sorted(frame["date"].dropna().unique())
    if len(dates) < 5:
        raise ValueError("At least five distinct dates are needed for time-forward validation.")
    cutoff = dates[max(1, int(len(dates) * (1 - test_fraction)))]
    return frame[frame["date"] < cutoff].copy(), frame[frame["date"] >= cutoff].copy(), str(cutoff)


def train_and_save(data_path=None):
    data_path = Path(data_path or ROOT / "data" / "raw" / "ladle_data_synthetic.xlsx")
    cleaned = load_and_clean(data_path, require_target=True)
    featured = create_training_features(cleaned)
    train, test, cutoff = time_forward_split(featured)
    X_train, y_train = train[FEATURE_COLUMNS], train["ladle_life"]
    X_test, y_test = test[FEATURE_COLUMNS], test["ladle_life"]
    candidates = {
        "extra_trees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "random_forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        results[name] = {"mae": round(mean_absolute_error(y_test, prediction), 4),
                         "rmse": round(mean_squared_error(y_test, prediction) ** 0.5, 4),
                         "r2": round(r2_score(y_test, prediction), 4)}
        fitted[name] = pipeline
    chosen_name = min(results, key=lambda name: results[name]["mae"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[chosen_name], MODELS_DIR / "ladle_life_pipeline.pkl")
    featured.to_csv(PROCESSED_DIR / "ladle_features.csv", index=False)
    metadata = {"model_name": chosen_name, "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "feature_columns": FEATURE_COLUMNS, "target": "ladle_life", "time_split_cutoff": cutoff,
                "metrics": results, "training_rows": len(train), "test_rows": len(test)}
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2))
