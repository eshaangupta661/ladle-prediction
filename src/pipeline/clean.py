"""Cleaning and chronological preparation for ladle heat data."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"date", "shift", "heat_no", "ladle_no", "ladle_life",
                    "opening_temp", "despatch_temp", "arcing_time",
                    "purging_time", "treatment_time"}
PROCESS_COLUMNS = ["opening_temp", "despatch_temp", "arcing_time", "purging_time", "treatment_time"]
NUMERIC_COLUMNS = ["ladle_no", "ladle_life", *PROCESS_COLUMNS]
VALID_RANGES = {"opening_temp": (1400, 1700), "despatch_temp": (1400, 1700),
                "arcing_time": (0, 240), "purging_time": (0, 240), "treatment_time": (0, 240)}
COLUMN_ALIASES = {
    "ldl_no": "ladle_no", "opening_temp_": "opening_temp",
    "despatch_temp_": "despatch_temp", "lf_arcing_time_mins": "arcing_time",
    "purging_time_mins": "purging_time", "treatment_time_mins": "treatment_time",
}


def parse_numeric(value):
    """Convert plant-style values such as '28+2' or '--' to a float."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in {"", "-", "--", "nan", "none", "`"}:
        return np.nan
    text = re.split(r"[+/]", text)[0].strip().rstrip("MmHh")
    return pd.to_numeric(text, errors="coerce")


def _normalise_columns(frame):
    frame = frame.copy()
    frame.columns = (frame.columns.astype(str).str.strip().str.lower()
                     .str.replace(r"[\s/().\\]+", "_", regex=True)
                     .str.replace(r"_+", "_", regex=True).str.strip("_"))
    return frame.rename(columns={key: value for key, value in COLUMN_ALIASES.items()
                                 if key in frame.columns})


def load_and_clean(path):
    """Load Excel/CSV, clean values, and sort records without using metadata as features.

    Every heat with a date and ladle number is kept, even when its life or a process
    reading is missing: dropping heats would break the per-campaign heat count.
    ``date``, ``shift``, and ``heat_no`` are retained only to establish chronology.
    """
    path = Path(path)
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    frame = _normalise_columns(frame)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        frame[column] = frame[column].map(parse_numeric)
    frame["heat_no"] = frame["heat_no"].astype(str).str.strip()
    frame["shift"] = frame["shift"].astype(str).str.strip().str.upper()
    frame.loc[frame["shift"].isin(["", "NAN", "NONE"]), "shift"] = np.nan

    frame = frame.dropna(subset=["date", "ladle_no"])
    frame = frame[frame["ladle_no"] > 0].copy()
    # Out-of-range readings are treated as missing and imputed, not dropped with their heat.
    for column, (low, high) in VALID_RANGES.items():
        frame.loc[~frame[column].between(low, high, inclusive="both"), column] = np.nan
    frame.loc[frame["ladle_life"] < 1, "ladle_life"] = np.nan

    frame = frame.drop_duplicates(subset=["ladle_no", "date", "heat_no"], keep="last")
    shift_order = {"A": 0, "B": 1, "C": 2}
    frame["_shift_order"] = frame["shift"].map(shift_order).fillna(99)
    frame = frame.sort_values(["ladle_no", "date", "_shift_order", "heat_no"]).reset_index(drop=True)

    for column in PROCESS_COLUMNS:
        by_ladle = frame.groupby("ladle_no")[column].transform("median")
        frame[column] = frame[column].fillna(by_ladle).fillna(frame[column].median())
    frame["ladle_no"] = frame["ladle_no"].astype(int)
    frame["ladle_life"] = frame["ladle_life"].astype(float)
    return frame.drop(columns="_shift_order")
