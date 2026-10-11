"""
predict.py
----------
POST /predict   - run inference and log to DB
GET  /predict/health - model health check
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db import get_db, Prediction, utc_now
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from pipeline.features import PROCESS_COLUMNS, feature_row_from_history

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse, status_code=201)
def predict(req: PredictionRequest, db: Session = Depends(get_db)):
    """
    Build features from this campaign's earlier heats, run the trained model,
    store the result in the DB, and return the predicted ladle life.
    """
    from api.main import MODEL_ARTIFACTS   # imported here to avoid circular import

    pipeline = MODEL_ARTIFACTS.get("pipeline")
    feat_cols = MODEL_ARTIFACTS.get("feature_cols")
    version = MODEL_ARTIFACTS.get("model_version")
    if pipeline is None or not feat_cols or version is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")

    latest = (
        db.query(Prediction)
        .filter(Prediction.ladle_no == req.ladle_no)
        .order_by(Prediction.id.desc())
        .first()
    )
    if req.new_campaign or latest is None:
        campaign_id = 1 if latest is None else latest.campaign_id + 1
        history_rows = []
        prior_heats = req.starting_life - 1 if req.starting_life else 0
    else:
        campaign_id = latest.campaign_id
        history_rows = (
            db.query(Prediction)
            .filter(Prediction.ladle_no == req.ladle_no, Prediction.campaign_id == campaign_id)
            .order_by(Prediction.id.asc())
            .all()
        )
        prior_heats = history_rows[0].feature_snapshot.get("prior_unobserved_heats", 0)

    history_frame = pd.DataFrame([row.feature_snapshot for row in history_rows])
    current_frame = feature_row_from_history(req.model_dump(include=set(PROCESS_COLUMNS)), history_frame,
                                             prior_unobserved_heats=prior_heats)
    # JSON columns (PostgreSQL) reject NaN, so store missing lag features as null.
    current_features = {key: None if pd.isna(value) else float(value) for key, value in current_frame.iloc[0].items()}
    current_features["ladle_no"] = req.ladle_no

    pred_life = float(pipeline.predict(current_frame[feat_cols])[0])
    pred_life = round(max(1.0, pred_life), 2)   # life can't be < 1

    db_record = Prediction(
        requested_at   = utc_now(),
        ladle_no       = req.ladle_no,
        heat_no        = req.heat_no,
        campaign_id    = campaign_id,
        predicted_life = pred_life,
        actual_life    = None,
        model_version  = version,
        feature_snapshot = current_features,
        is_partial_campaign = bool(current_features.get("is_partial_campaign", 0)),
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return PredictionResponse(
        prediction_id  = db_record.id,
        predicted_life = pred_life,
        model_version  = version,
        requested_at   = db_record.requested_at,
        ladle_no       = req.ladle_no,
        heat_no        = req.heat_no,
        campaign_id    = campaign_id,
        history_heats_used = len(history_rows),
    )


@router.get("/health", response_model=HealthResponse)
def health():
    """Returns model metadata to confirm the service is up and which model is loaded."""
    from api.main import MODEL_ARTIFACTS
    meta = MODEL_ARTIFACTS.get("meta", {})
    model_name = meta.get("model_name", "unknown")
    holdout = meta.get("metrics", {}).get(model_name, {}).get("holdout", {})
    return HealthResponse(
        status        = "ok" if MODEL_ARTIFACTS.get("pipeline") is not None else "model_not_trained",
        model_version = MODEL_ARTIFACTS.get("model_version"),
        model_name    = model_name,
        test_r2       = holdout.get("r2", 0.0),
        test_mae      = holdout.get("mae"),
    )
