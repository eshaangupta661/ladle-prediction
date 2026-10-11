import pytest
from fastapi.testclient import TestClient

from api.db import Base, engine
from api.main import MODEL_ARTIFACTS, app
from api.schemas import PredictionRequest
from pipeline.features import FEATURE_COLUMNS
from pipeline.model import CounterBaseline

HEAT = dict(ladle_no=20, opening_temp=1550, despatch_temp=1580, arcing_time=20, purging_time=12, treatment_time=40)


@pytest.fixture
def client():
    Base.metadata.drop_all(bind=engine)
    with TestClient(app) as test_client:
        # A deterministic stand-in model: predicted life equals the campaign heat counter.
        MODEL_ARTIFACTS.clear()
        MODEL_ARTIFACTS.update({"pipeline": CounterBaseline(), "model_version": "test",
                                "feature_cols": FEATURE_COLUMNS,
                                "meta": {"model_name": "stub", "metrics": {"stub": {"holdout": {"r2": 0.99, "mae": 0.1}}}}})
        yield test_client
    MODEL_ARTIFACTS.clear()


def post(client, **overrides):
    response = client.post("/predict", json={**HEAT, **overrides})
    assert response.status_code == 201, response.text
    return response.json()


def test_operator_payload_uses_raw_heat_inputs_only():
    request = PredictionRequest(**HEAT)
    assert request.new_campaign is False and request.starting_life is None


def test_out_of_range_reading_is_rejected(client):
    assert client.post("/predict", json={**HEAT, "opening_temp": 1300}).status_code == 422


def test_missing_model_returns_503(client):
    MODEL_ARTIFACTS.clear()
    assert client.post("/predict", json=HEAT).status_code == 503
    assert client.get("/predict/health").json()["status"] == "model_not_trained"


def test_life_counts_up_within_a_campaign(client):
    results = [post(client) for _ in range(3)]
    assert [r["predicted_life"] for r in results] == [1, 2, 3]
    assert [r["history_heats_used"] for r in results] == [0, 1, 2]
    assert {r["campaign_id"] for r in results} == {1}


def test_new_campaign_resets_life(client):
    post(client); post(client)
    reset = post(client, new_campaign=True)
    assert (reset["campaign_id"], reset["predicted_life"], reset["history_heats_used"]) == (2, 1, 0)
    assert post(client)["predicted_life"] == 2


def test_starting_life_onboards_ladle_already_in_service(client):
    assert post(client, starting_life=40)["predicted_life"] == 40
    assert post(client)["predicted_life"] == 41


def test_ladles_are_tracked_independently(client):
    post(client); post(client)
    assert post(client, ladle_no=7)["predicted_life"] == 1


def test_actual_life_feedback_and_history(client):
    first = post(client)
    updated = client.patch(f"/predictions/{first['prediction_id']}/actual", params={"actual_life": 1})
    assert updated.status_code == 200 and updated.json()["actual_life"] == 1
    assert client.patch("/predictions/999/actual", params={"actual_life": 1}).status_code == 404
    post(client, ladle_no=7)
    assert len(client.get("/predictions").json()) == 2
    assert len(client.get("/predictions", params={"ladle_no": 7}).json()) == 1


def test_health_reports_loaded_model(client):
    assert client.get("/health").json()["status"] == "ok"
    body = client.get("/predict/health").json()
    assert body["status"] == "ok" and body["test_mae"] == 0.1
