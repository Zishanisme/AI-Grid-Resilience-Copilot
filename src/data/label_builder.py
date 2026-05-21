"""
label_builder.py
================
Assigns physical-threshold-based risk labels to buses and lines.

FIX (C1 / W5):  Previous version used per-timestep quantile thresholds
(e.g. top-30% of loading = "congested").  That guaranteed a fixed positive
rate at every snapshot regardless of absolute grid stress, making the labels
relative-rank predictors rather than absolute-risk predictors.

All thresholds below are grounded in IEEE / ANSI standards:
  - Congestion  : loading_pct > 80 %   (ANSI / IEEE thermal limit threshold)
  - Overload    : loading_pct > 100 %  (thermal overload)
  - Volt. viol. : v_min_pu < 0.95      (ANSI C84.1 Range A lower limit)
  - Fault risk  : v_min_pu < 0.90      (severe undervoltage — high fault risk)

These thresholds produce sparse labels under light operating conditions,
which is physically correct.  Class-weighted loss in train_model.py and
AUPRC as the primary metric handle the resulting imbalance properly.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Physical threshold constants (IEEE / ANSI)
# ---------------------------------------------------------------------------
CONGESTION_THRESHOLD_PCT   = 80.0   # % of thermal rating
OVERLOAD_THRESHOLD_PCT     = 100.0  # % of thermal rating
VOLT_VIOLATION_THRESHOLD   = 0.95   # p.u. — ANSI C84.1 Range A lower limit
FAULT_RISK_THRESHOLD       = 0.90   # p.u. — severe undervoltage


def add_bus_labels(bus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bus-level risk labels using absolute voltage thresholds.

    Expects column 'v_min_pu' (minimum per-unit voltage at the bus).
    Falls back gracefully if the column is missing.

    Labels
    ------
    voltage_violation : 1 if v_min_pu < 0.95 p.u. (ANSI C84.1 Range A)
    fault_risk        : 1 if v_min_pu < 0.90 p.u. (severe undervoltage)
    """
    df = bus_df.copy()

    if "v_min_pu" not in df.columns:
        # Graceful fallback — will be overwritten by temporal_dataset.py anyway
        df["voltage_violation"] = 0
        df["fault_risk"]        = 0
        return df

    df["voltage_violation"] = (
        df["v_min_pu"] < VOLT_VIOLATION_THRESHOLD
    ).astype(int)

    df["fault_risk"] = (
        df["v_min_pu"] < FAULT_RISK_THRESHOLD
    ).astype(int)

    return df


def add_line_labels(line_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add line-level risk labels using absolute loading thresholds.

    Expects column 'loading_pct' (% of thermal rating).

    Labels
    ------
    congestion_risk : 1 if loading_pct > 80 % (ANSI thermal limit)
    overload        : 1 if loading_pct > 100 % (actual overload)
    """
    df = line_df.copy()

    if "loading_pct" not in df.columns:
        df["congestion_risk"] = 0
        df["overload"]        = 0
        return df

    df["congestion_risk"] = (
        df["loading_pct"] > CONGESTION_THRESHOLD_PCT
    ).astype(int)

    df["overload"] = (
        df["loading_pct"] > OVERLOAD_THRESHOLD_PCT
    ).astype(int)

    return df


def map_line_labels_to_buses(
    bus_df: pd.DataFrame,
    line_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Propagate line-level congestion risk to both endpoint buses.

    FIX (W2 from data-pipeline review):  Previous version used only
    'from_bus' via groupby, missing 'to_bus' endpoints entirely.
    This version maps congestion to both endpoints of each congested line.
    """
    bus_df = bus_df.copy()

    if line_df.empty or "congestion_risk" not in line_df.columns:
        bus_df["congestion_risk"] = 0
        return bus_df

    congestion_map: dict = {}

    for _, row in line_df.iterrows():
        if row.get("congestion_risk", 0) == 1:
            for col in ("from_bus", "to_bus"):
                bus = row.get(col)
                if bus is not None:
                    key = str(bus).split(".")[0].lower()
                    congestion_map[key] = 1

    if "bus_name" in bus_df.columns:
        name_col = bus_df["bus_name"].astype(str).str.lower()
    elif "bus" in bus_df.columns:
        name_col = bus_df["bus"].astype(str).str.lower()
    elif "name" in bus_df.columns:
        name_col = bus_df["name"].astype(str).str.lower()
    else:
        bus_df["congestion_risk"] = 0
        return bus_df

    bus_df["congestion_risk"] = name_col.apply(
        lambda b: congestion_map.get(b, 0)
    ).astype(int)

    return bus_df
