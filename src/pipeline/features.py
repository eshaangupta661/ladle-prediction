"""Leakage-safe, cycle-aware features. Dates/shifts/heat IDs are never model inputs."""
import numpy as np
import pandas as pd

PROCESS_COLUMNS = ["opening_temp", "despatch_temp", "arcing_time", "purging_time", "treatment_time"]
FEATURE_COLUMNS = [
    "ladle_no", *PROCESS_COLUMNS, "temp_gain", "total_process_time",
    "campaign_heats_observed", "prior_unobserved_heats", "is_partial_campaign",
    "cum_arcing_time", "cum_purging_time", "cum_treatment_time", "cum_temp_gain",
    "prev_arcing_time", "prev_purging_time", "prev_treatment_time", "prev_temp_gain",
    "roll3_arcing_time", "roll3_purging_time", "roll3_treatment_time", "roll3_temp_gain",
    "roll5_arcing_time", "roll5_purging_time", "roll5_treatment_time", "roll5_temp_gain",
]


def assign_campaigns(frame, reset_threshold=10):
    """Assign campaigns from known historical life resets; training only."""
    frame = frame.copy()
    previous = frame.groupby("ladle_no")["ladle_life"].shift()
    resets = (frame["ladle_life"] < previous - reset_threshold).fillna(False)
    frame["campaign_id"] = resets.groupby(frame["ladle_no"]).cumsum().astype(int) + 1
    first_life = frame.groupby(["ladle_no", "campaign_id"])["ladle_life"].transform("first")
    frame["prior_unobserved_heats"] = (first_life - 1).clip(lower=0)
    frame["is_partial_campaign"] = (first_life != 1).astype(int)
    return frame


def create_training_features(frame):
    """Build one feature row per heat using only past values in its campaign."""
    frame = assign_campaigns(frame)
    frame["temp_gain"] = frame["despatch_temp"] - frame["opening_temp"]
    frame["total_process_time"] = frame["arcing_time"] + frame["purging_time"] + frame["treatment_time"]
    groups = frame.groupby(["ladle_no", "campaign_id"], sort=False)
    frame["campaign_heats_observed"] = groups.cumcount() + 1
    for source, output in [("arcing_time", "cum_arcing_time"), ("purging_time", "cum_purging_time"),
                           ("treatment_time", "cum_treatment_time"), ("temp_gain", "cum_temp_gain")]:
        frame[output] = groups[source].transform(lambda series: series.shift().expanding().sum()).fillna(0.0)
    for source in ["arcing_time", "purging_time", "treatment_time", "temp_gain"]:
        frame[f"prev_{source}"] = groups[source].shift(1)
        for window in (3, 5):
            frame[f"roll{window}_{source}"] = groups[source].transform(
                lambda series: series.shift().rolling(window, min_periods=1).mean())
    return frame


def feature_row_from_history(ladle_no, current, history, campaign_id=1):
    """Create an inference row from stored prior heat records and current measurements."""
    current = {key: float(current[key]) for key in PROCESS_COLUMNS}
    current["ladle_no"] = int(ladle_no)
    history = history.copy()
    for column in PROCESS_COLUMNS:
        if column not in history:
            history[column] = pd.Series(dtype=float)
    history["temp_gain"] = history["despatch_temp"] - history["opening_temp"]
    current["temp_gain"] = current["despatch_temp"] - current["opening_temp"]
    current["total_process_time"] = sum(current[key] for key in ["arcing_time", "purging_time", "treatment_time"])
    current["campaign_heats_observed"] = len(history) + 1
    current["prior_unobserved_heats"] = 0.0
    current["is_partial_campaign"] = 0
    for source, output in [("arcing_time", "cum_arcing_time"), ("purging_time", "cum_purging_time"),
                           ("treatment_time", "cum_treatment_time"), ("temp_gain", "cum_temp_gain")]:
        values = history[source].dropna()
        current[output] = float(values.sum()) if not values.empty else 0.0
        current[f"prev_{source}"] = float(values.iloc[-1]) if not values.empty else np.nan
        for window in (3, 5):
            tail = values.tail(window)
            current[f"roll{window}_{source}"] = float(tail.mean()) if not tail.empty else np.nan
    return pd.DataFrame([{column: current.get(column, np.nan) for column in FEATURE_COLUMNS}])
