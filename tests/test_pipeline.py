import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from pipeline.features import FEATURE_COLUMNS, create_training_features


def sample_frame():
    return pd.DataFrame({"ladle_no": [1, 1, 1], "date": pd.date_range("2026-01-01", periods=3),
        "ladle_life": [1., 2., 3.], "opening_temp": [1550.] * 3, "despatch_temp": [1580.] * 3,
        "arcing_time": [20., 21., 22.], "purging_time": [10.] * 3, "treatment_time": [40.] * 3})


def test_past_only_features():
    result = create_training_features(sample_frame())
    assert result.loc[0, "cum_arcing_time"] == 0
    assert result.loc[2, "cum_arcing_time"] == 41


def test_metadata_excluded():
    assert not {"date", "shift", "heat_no"}.intersection(FEATURE_COLUMNS)
    assert "ladle_no" in FEATURE_COLUMNS
