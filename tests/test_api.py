import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from api.schemas import PredictionRequest


def test_operator_payload_uses_raw_heat_inputs_only():
    request = PredictionRequest(ladle_no=20, opening_temp=1550, despatch_temp=1580,
        arcing_time=20, purging_time=12, treatment_time=40)
    assert request.ladle_no == 20
    assert request.new_campaign is False
