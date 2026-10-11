"""Prediction audit and ground-truth feedback endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from api.db import Prediction, get_db
from api.schemas import PredictionRecord

router = APIRouter(prefix="/predictions", tags=["History"])


def as_record(row):
    return PredictionRecord(prediction_id=row.id, predicted_life=row.predicted_life, model_version=row.model_version,
        requested_at=row.requested_at, ladle_no=row.ladle_no, heat_no=row.heat_no, campaign_id=row.campaign_id,
        history_heats_used=int(row.feature_snapshot.get("campaign_heats_observed", 1)) - 1,
        actual_life=row.actual_life, feature_snapshot=row.feature_snapshot)


@router.get("", response_model=list[PredictionRecord])
def list_predictions(ladle_no: Optional[int] = None, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    query = db.query(Prediction).order_by(Prediction.requested_at.desc())
    if ladle_no is not None:
        query = query.filter(Prediction.ladle_no == ladle_no)
    return [as_record(row) for row in query.limit(limit).all()]


@router.patch("/{prediction_id}/actual", response_model=PredictionRecord)
def record_actual_life(prediction_id: int, actual_life: float = Query(..., ge=1), db: Session = Depends(get_db)):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(404, "Prediction not found")
    row.actual_life = actual_life
    db.commit(); db.refresh(row)
    return as_record(row)
