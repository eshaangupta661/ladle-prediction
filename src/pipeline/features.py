"""Leakage-safe, cycle-aware features. Dates/shifts/heat IDs are never model inputs."""
import numpy as np
import pandas as pd

from .clean import PROCESS_COLUMNS

FEATURE_COLUMNS = [
    *PROCESS_COLUMNS, "temp_gain", "total_process_time",
    "campaign_heats_observed", "prior_unobserved_heats", "is_partial_campaign", "expected_life",
    "cum_arcing_time", "cum_purging_time", "cum_treatment_time", "cum_temp_gain",
    "prev_arcing_time", "prev_purging_time", "prev_treatment_time", "prev_temp_gain",
    "roll3_arcing_time", "roll3_purging_time", "roll3_treatment_time", "roll3_temp_gain",
    "roll5_arcing_time", "roll5_purging_time", "roll5_treatment_time", "roll5_temp_gain",
]
HISTORY_SOURCES = ["arcing_time", "purging_time", "treatment_time", "temp_gain"]


def _first_known_life(series):
    """Life counter at the first heat of a campaign, inferred from its first recorded value."""
    known = series.reset_index(drop=True).dropna()
    if known.empty:
        return 1.0
    return max(1.0, float(known.iloc[0]) - known.index[0])


def assign_campaigns(frame, reset_threshold=10):
    """Assign campaigns from historical life resets (relines); training only.

    ``prior_unobserved_heats`` is the ladle's life counter before its first recorded heat,
    which is non-zero only for ladles already in service when data collection began. In
    production the operator supplies it once via ``starting_life`` when onboarding the ladle.
    """
    frame = frame.copy()
    known_life = frame.groupby("ladle_no")["ladle_life"].ffill()
    previous = known_life.groupby(frame["ladle_no"]).shift()
    resets = (frame["ladle_life"] < previous - reset_threshold).fillna(False)
    frame["campaign_id"] = resets.groupby(frame["ladle_no"]).cumsum().astype(int) + 1
    start_life = frame.groupby(["ladle_no", "campaign_id"])["ladle_life"].transform(_first_known_life)
    frame["prior_unobserved_heats"] = start_life - 1
    frame["is_partial_campaign"] = (start_life > 1).astype(int)
    return frame


def create_training_features(frame):
    """Build one feature row per heat using only past values in its campaign."""
    frame = assign_campaigns(frame)
    frame["temp_gain"] = frame["despatch_temp"] - frame["opening_temp"]
    frame["total_process_time"] = frame["arcing_time"] + frame["purging_time"] + frame["treatment_time"]
    groups = frame.groupby(["ladle_no", "campaign_id"], sort=False)
    frame["campaign_heats_observed"] = groups.cumcount() + 1
    frame["expected_life"] = frame["prior_unobserved_heats"] + frame["campaign_heats_observed"]
    for source in HISTORY_SOURCES:
        frame[f"cum_{source}"] = groups[source].transform(lambda series: series.shift().expanding().sum()).fillna(0.0)
        frame[f"prev_{source}"] = groups[source].shift(1)
        for window in (3, 5):
            frame[f"roll{window}_{source}"] = groups[source].transform(
                lambda series: series.shift().rolling(window, min_periods=1).mean())
    return frame


def feature_row_from_history(current, history, prior_unobserved_heats=0.0):
    """Create an inference row from stored prior heats of the campaign and current measurements."""
    current = {key: float(current[key]) for key in PROCESS_COLUMNS}
    history = history.copy()
    for column in PROCESS_COLUMNS:
        if column not in history:
            history[column] = pd.Series(dtype=float)
    history["temp_gain"] = history["despatch_temp"] - history["opening_temp"]
    current["temp_gain"] = current["despatch_temp"] - current["opening_temp"]
    current["total_process_time"] = current["arcing_time"] + current["purging_time"] + current["treatment_time"]
    current["campaign_heats_observed"] = len(history) + 1
    current["prior_unobserved_heats"] = float(prior_unobserved_heats)
    current["is_partial_campaign"] = int(prior_unobserved_heats > 0)
    current["expected_life"] = current["prior_unobserved_heats"] + current["campaign_heats_observed"]
    for source in HISTORY_SOURCES:
        values = history[source].dropna()
        current[f"cum_{source}"] = float(values.sum()) if not values.empty else 0.0
        current[f"prev_{source}"] = float(values.iloc[-1]) if not values.empty else np.nan
        for window in (3, 5):
            tail = values.tail(window)
            current[f"roll{window}_{source}"] = float(tail.mean()) if not tail.empty else np.nan
    return pd.DataFrame([{column: current.get(column, np.nan) for column in FEATURE_COLUMNS}])
