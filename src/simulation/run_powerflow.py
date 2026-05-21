"""
run_powerflow.py
================
OpenDSS power flow runner for IEEE 33-bus distribution feeder.

Fixes applied
-------------
R1 — Voltage p.u. conversion now uses the feeder rated voltage (IEEE 33-bus:
     12.66 kV line-to-line = 7310 V line-to-neutral) rather than the
     per-timestep maximum observed bus voltage.  The previous approach
     produced per-timestep relative values that were incompatible with the
     absolute ANSI C84.1 thresholds (0.95 p.u., 0.90 p.u.) used in
     label_builder.py.

R2 — NormAmps fallback changed from `1.0` (which produced loading_pct
     values in the thousands when line current > 1 A) to 0.0 with a
     critical warning.  Lines without rated capacity do not trigger
     congestion labels rather than corrupting the label distribution.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import opendssdirect as dss


# ---------------------------------------------------------------------------
# IEEE 33-bus feeder rated voltage (line-to-neutral, volts)
# 12.66 kV (line-to-line) / sqrt(3) = 7308.8 V ≈ 7310 V
# ---------------------------------------------------------------------------
IEEE33_RATED_V_LN = 7310.0   # line-to-neutral volts


def run_power_flow(
    dss_file,
    mode="snapshot",
    load_mult=1.0,
    save_dir=None,
):
    """
    Run OpenDSS power flow and extract bus/line/transformer data.

    Returns
    -------
    dict with keys: converged, bus_df, line_df, transformer_df, summary.
    """
    dss_file = Path(dss_file)
    if not dss_file.exists():
        raise FileNotFoundError(f"DSS file not found: {dss_file}")

    dss.Basic.ClearAll()
    dss.Text.Command(f'compile "{dss_file}"')
    dss.Solution.Mode(0)
    dss.Solution.LoadMult(float(load_mult))

    start = time.time()
    dss.Solution.Solve()
    elapsed = time.time() - start

    converged = bool(dss.Solution.Converged())

    bus_df         = _extract_buses()
    line_df        = _extract_lines()
    transformer_df = _extract_transformers()

    total_p_kw    = float(dss.Circuit.TotalPower()[0])
    total_q_kvar  = float(dss.Circuit.TotalPower()[1])
    losses        = dss.Circuit.Losses()
    total_loss_kw = float(losses[0]) / 1000.0

    summary = {
        "circuit_name":    dss.Circuit.Name(),
        "converged":       converged,
        "n_buses":         len(bus_df),
        "n_lines":         len(line_df),
        "n_transformers":  len(transformer_df),
        "total_p_kw":      abs(total_p_kw),
        "total_q_kvar":    abs(total_q_kvar),
        "total_loss_kw":   total_loss_kw,
        "loss_pct": (
            abs(total_loss_kw / total_p_kw * 100.0)
            if total_p_kw != 0 else 0.0
        ),
        "elapsed_s": elapsed,
    }

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        bus_df.to_csv(save_dir / "bus_df.csv", index=False)
        line_df.to_csv(save_dir / "line_df.csv", index=False)
        transformer_df.to_csv(save_dir / "transformer_df.csv", index=False)
        pd.DataFrame([summary]).to_json(save_dir / "summary.json", indent=2)

    return {
        "converged":       converged,
        "bus_df":          bus_df,
        "line_df":         line_df,
        "transformer_df":  transformer_df,
        "summary":         summary,
    }


# ============================================================
# BUS EXTRACTION
# ============================================================
def _get_feeder_base_v() -> float:
    """
    Determine the feeder base voltage in volts (line-to-neutral).

    Priority:
    1. Read Bus.kVBase() from OpenDSS (line-to-neutral kV).
    2. Fall back to IEEE 33-bus rated value (7310 V) if kVBase is
       unavailable or zero.

    R1 fix: using rated voltage base makes v_min_pu physically
    consistent with ANSI C84.1 thresholds in label_builder.py.
    """
    try:
        all_buses = dss.Circuit.AllBusNames()
        if all_buses:
            dss.Circuit.SetActiveBus(all_buses[0])
            kv_base = float(dss.Bus.kVBase())   # line-to-neutral kV
            if kv_base > 0:
                return kv_base * 1000.0
    except Exception:
        pass

    return IEEE33_RATED_V_LN


def _extract_buses() -> pd.DataFrame:
    """
    Extract bus voltages and convert to ANSI-standard per-unit.

    v_min_pu is the minimum phase voltage divided by the rated
    line-to-neutral voltage, consistent with ANSI C84.1 Range A.
    """
    feeder_base_v = _get_feeder_base_v()

    raw_rows = []

    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)

        try:
            v_mag_angle = dss.Bus.VMagAngle()
            v_mag = np.array(v_mag_angle[0::2], dtype=float)
        except Exception:
            v_mag = np.array([], dtype=float)

        v_mag = v_mag[np.isfinite(v_mag)]
        v_mag = v_mag[v_mag > 0]

        if len(v_mag) == 0:
            v_avg_v = 0.0
            v_min_v = 0.0
            v_max_v = 0.0
        else:
            v_avg_v = float(np.mean(v_mag))
            v_min_v = float(np.min(v_mag))
            v_max_v = float(np.max(v_mag))

        raw_rows.append({
            "bus_name":    str(bus),
            "v_avg_v":     v_avg_v,
            "v_min_v":     v_min_v,
            "v_max_v":     v_max_v,
            "p_load_kw":   0.0,
            "q_load_kvar": 0.0,
        })

    bus_df = pd.DataFrame(raw_rows)

    # R1 fix: divide by rated voltage, not per-timestep max
    bus_df["v_avg_pu"] = bus_df["v_avg_v"] / feeder_base_v
    bus_df["v_min_pu"] = bus_df["v_min_v"] / feeder_base_v
    bus_df["v_max_pu"] = bus_df["v_max_v"] / feeder_base_v

    bus_df = bus_df[[
        "bus_name", "v_avg_pu", "v_min_pu", "v_max_pu",
        "p_load_kw", "q_load_kvar",
    ]]

    # Add load data
    if dss.Loads.First() > 0:
        while True:
            bus  = dss.CktElement.BusNames()[0].split(".")[0].lower()
            p    = float(dss.Loads.kW())
            q    = float(dss.Loads.kvar())
            mask = bus_df["bus_name"].str.lower() == bus
            if mask.any():
                bus_df.loc[mask, "p_load_kw"]   += p
                bus_df.loc[mask, "q_load_kvar"]  += q
            if dss.Loads.Next() == 0:
                break

    return bus_df


# ============================================================
# LINE EXTRACTION
# ============================================================
def _extract_lines() -> pd.DataFrame:
    """
    Extract line parameters and thermal loading.

    R2 fix: if NormAmps is not set (returns 0), loading_pct is set to 0.0
    with a warning rather than dividing by 1.0 (which produced multi-thousand
    percent loading values, corrupting congestion labels for those lines).
    """
    rows = []
    lines_without_rating: list = []

    if dss.Lines.First() == 0:
        return pd.DataFrame(columns=[
            "line_name", "from_bus", "to_bus",
            "r_ohm", "x_ohm", "length", "loading_pct",
        ])

    while True:
        name = dss.Lines.Name()
        buses = dss.CktElement.BusNames()
        from_bus = buses[0].split(".")[0]
        to_bus   = buses[1].split(".")[0] if len(buses) > 1 else from_bus

        norm_amps = float(dss.Lines.NormAmps())

        currents = dss.CktElement.CurrentsMagAng()[0::2]
        max_current = float(max(currents)) if currents else 0.0

        # R2 fix: do not use 1.0 fallback — that produces absurd loading_pct
        if norm_amps > 0:
            loading_pct = 100.0 * max_current / norm_amps
        else:
            loading_pct = 0.0
            lines_without_rating.append(name)

        rows.append({
            "line_name":   str(name),
            "from_bus":    str(from_bus),
            "to_bus":      str(to_bus),
            "r_ohm":       float(dss.Lines.R1()),
            "x_ohm":       float(dss.Lines.X1()),
            "length":      float(dss.Lines.Length()),
            "loading_pct": float(loading_pct),
        })

        if dss.Lines.Next() == 0:
            break

    if lines_without_rating:
        print(
            f"[WARN] {len(lines_without_rating)} line(s) have no NormAmps rating "
            f"in the DSS file — loading_pct set to 0.0 for these lines. "
            f"They will not trigger congestion labels. "
            f"Lines: {lines_without_rating[:5]}{'...' if len(lines_without_rating) > 5 else ''}"
        )

    return pd.DataFrame(rows)


# ============================================================
# TRANSFORMER EXTRACTION
# ============================================================
def _extract_transformers() -> pd.DataFrame:
    rows = []
    if dss.Transformers.First() == 0:
        return pd.DataFrame(columns=["transformer_name"])
    while True:
        rows.append({"transformer_name": str(dss.Transformers.Name())})
        if dss.Transformers.Next() == 0:
            break
    return pd.DataFrame(rows)
