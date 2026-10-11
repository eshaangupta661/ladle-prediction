"""
predict.py
----------
POST /predict   - run inference and log to DB
GET  /predict/health - model health check
"""

from datetime import datetime
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db import get_db, Prediction
from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from pipeline.features import feature_row_from_history

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictionResponse, status_code=201)
def predict(req: PredictionRequest, db: Session = Depends(get_db)):
    """
    Accept feature values, run the XGBoost model, store result in DB,
    and return the predicted ladle life.
    """
    from api.main import MODEL_ARTIFACTS   # imported here to avoid circular import

    pipeline = MODEL_ARTIFACTS.get("pipeline")
    feat_cols = MODEL_ARTIFACTS.get("feature_cols")
    version = MODEL_ARTIFACTS.get("model_version")
    if pipeline is None or not feat_cols or version is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")

    feature_dict = req.model_dump(exclude={"ladle_no", "heat_no"})

    latest = (
        db.query(Prediction)
        .filter(Prediction.ladle_no == req.ladle_no)
        .order_by(Prediction.requested_at.desc())
        .first()
    )
    if req.new_campaign or latest is None:
        campaign_id = 1 if latest is None else latest.campaign_id + 1
        history_rows = []
    else:
        campaign_id = latest.campaign_id
        history_rows = (
            db.query(Prediction)
            .filter(Prediction.ladle_no == req.ladle_no, Prediction.campaign_id == campaign_id)
            .order_by(Prediction.requested_at.asc())
            .all()
        )

    history_frame = pd.DataFrame([row.feature_snapshot for row in history_rows]) if history_rows else pd.DataFrame()
    current_frame = feature_row_from_history(req.ladle_no, feature_dict, history_frame, campaign_id=campaign_id)
    current_features = current_frame.iloc[0].to_dict()
    current_features["ladle_no"] = req.ladle_no

    pred_life = float(pipeline.predict(current_frame[feat_cols])[0])
    pred_life = round(max(1.0, pred_life), 2)   # life can't be < 1

    # Store in DB
    db_record = Prediction(
        requested_at   = datetime.utcnow(),
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
        history_heats_used = int(current_features.get("campaign_heats_observed", 1)) - 1,
    )


@router.get("/health", response_model=HealthResponse)
def health():
    """Returns model metadata to confirm the service is up and which model is loaded."""
    from api.main import MODEL_ARTIFACTS
    meta = MODEL_ARTIFACTS.get("meta", {})
    return HealthResponse(
        status        = "ok",
        model_version = MODEL_ARTIFACTS.get("model_version"),
        model_name    = meta.get("model_name", "unknown"),
        test_r2       = meta.get("metrics", {}).get(meta.get("model_name", ""), {}).get("r2", 0.0),
    )
