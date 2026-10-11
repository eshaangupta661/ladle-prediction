"""API schemas: operators submit raw current-heat readings only."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    ladle_no: int = Field(..., gt=0)
    heat_no: Optional[str] = None
    opening_temp: float = Field(..., ge=1400, le=1700)
    despatch_temp: float = Field(..., ge=1400, le=1700)
    arcing_time: float = Field(..., ge=0, le=240)
    purging_time: float = Field(..., ge=0, le=240)
    treatment_time: float = Field(..., ge=0, le=240)
    new_campaign: bool = Field(False, description="Set true only immediately after a confirmed reline.")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction_id: int
    predicted_life: float
    model_version: str
    requested_at: datetime
    ladle_no: int
    heat_no: Optional[str]
    campaign_id: int
    history_heats_used: int


class PredictionRecord(PredictionResponse):
    actual_life: Optional[float]
    feature_snapshot: dict


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: Optional[str] = None
    model_name: str
    test_r2: float
