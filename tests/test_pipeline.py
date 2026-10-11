import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor

from pipeline.clean import load_and_clean, parse_numeric
from pipeline.features import FEATURE_COLUMNS, create_training_features, feature_row_from_history
from pipeline.model import CounterResidualRegressor


def sample_frame(lives=(1., 2., 3.)):
    n = len(lives)
    return pd.DataFrame({"ladle_no": [1] * n, "date": pd.date_range("2026-01-01", periods=n),
        "ladle_life": list(lives), "opening_temp": [1550.] * n, "despatch_temp": [1580.] * n,
        "arcing_time": [20. + i for i in range(n)], "purging_time": [10.] * n, "treatment_time": [40.] * n})


def test_parse_plant_style_values():
    assert parse_numeric("28+2") == 28
    assert parse_numeric("1550+4M") == 1550
    assert np.isnan(parse_numeric("--")) and np.isnan(parse_numeric("`"))


def test_cleaning_keeps_heats_with_missing_values(tmp_path):
    raw = pd.DataFrame({"Date": ["2026-01-01"] * 3, "Shift": ["A", "b", "C"], "Heat No": [1, 2, 3],
        "Ldl No": [5, 5, 5], "Ladle Life": [1, None, 3], "Opening Temp.": ["1550", "--", "1300"],
        "Despatch Temp.": [1580, 1581, 1582], "LF Arcing Time (Mins)": ["20+2", 21, 22],
        "Purging Time (Mins)": [10, 11, 12], "Treatment Time (Mins)": [40, 41, 42]})
    path = tmp_path / "heats.csv"
    raw.to_csv(path, index=False)
    cleaned = load_and_clean(path)
    assert len(cleaned) == 3                          # nothing dropped
    assert cleaned["opening_temp"].tolist() == [1550, 1550, 1550]   # missing/out-of-range imputed
    assert cleaned["shift"].tolist() == ["A", "B", "C"]


def test_past_only_features():
    result = create_training_features(sample_frame())
    assert result.loc[0, "cum_arcing_time"] == 0
    assert result.loc[2, "cum_arcing_time"] == 41
    assert result["expected_life"].tolist() == [1, 2, 3]


def test_reline_starts_new_campaign_and_partial_start_is_detected():
    result = create_training_features(sample_frame((40., np.nan, 42., 1., 2.)))
    assert result["campaign_id"].tolist() == [1, 1, 1, 2, 2]
    assert result["expected_life"].tolist() == [40, 41, 42, 1, 2]
    assert result["is_partial_campaign"].tolist() == [1, 1, 1, 0, 0]


def test_inference_features_match_training_features():
    training = create_training_features(sample_frame((10., 11., 12., 13., 14., 15., 16.)))
    for position in range(len(training)):
        history = training.iloc[:position]
        row = feature_row_from_history(training.iloc[position], history, prior_unobserved_heats=9)
        expected = training.iloc[[position]][FEATURE_COLUMNS].reset_index(drop=True)
        pd.testing.assert_frame_equal(row, expected, check_dtype=False)


def test_residual_model_extrapolates_beyond_training_range():
    training = create_training_features(sample_frame(tuple(float(i) for i in range(1, 61))))
    model = CounterResidualRegressor(ExtraTreesRegressor(n_estimators=20, random_state=0))
    model.fit(training[FEATURE_COLUMNS].fillna(0), training["ladle_life"])
    later = feature_row_from_history(training.iloc[0], training, prior_unobserved_heats=39).fillna(0)
    assert model.predict(later)[0] == pytest.approx(100, abs=0.5)  # 39 prior + 61st heat


def test_metadata_excluded():
    assert not {"date", "shift", "heat_no", "ladle_no"}.intersection(FEATURE_COLUMNS)
