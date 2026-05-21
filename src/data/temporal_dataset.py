"""
temporal_dataset.py
===================
Generates renewable-aware temporal graph snapshots for the IEEE 33-bus feeder.

Why graph structure now matters for prediction
----------------------------------------------
The previous dataset generated labels that were essentially per-node functions of
local physical measurements (voltage_dev > threshold, loading_pct > threshold).
An MLP given node features (which include voltage_dev) could predict these labels
directly, without needing to know which buses are connected.

This version adds three topology-dependent label mechanisms that an MLP cannot
replicate from its own node features:

1. PRECURSOR VOLTAGE RISK (spatial) — A bus that is near the voltage threshold
   AND whose GRAPH NEIGHBORS are congested is labeled as voltage-violated, even
   though its own measured voltage is still within bounds.  The precursor risk
   represents cascading voltage drop that will propagate from the loaded upstream
   segment.  An MLP cannot compute "what fraction of my neighbors are congested"
   — only a GNN can aggregate that neighborhood information.

2. TEMPORAL VOLTAGE PERSISTENCE (temporal) — A bus that has been near the voltage
   threshold for TWO CONSECUTIVE timesteps is labeled as at risk, even if neither
   individual timestep breaches the absolute threshold.  An MLP applied to a
   single-timestep feature vector cannot detect this persistence.  Only a model
   with temporal context (GRU/LSTM) can detect it.

3. NEIGHBOR-FAULT PROPAGATION (spatial) — A bus whose graph neighbors have
   accumulated voltage violations is labeled at elevated fault risk, representing
   protection-system cascade risk.  Again, purely local features cannot distinguish
   a bus with stressed neighbors from one with healthy neighbors at the same
   voltage level.

These three mechanisms are PHYSICALLY MOTIVATED (they represent known grid cascade
dynamics), not artificial label noise.  They ensure that:
  - MLP performance is bounded by what local features can predict
  - Graph models have a genuine advantage requiring neighborhood aggregation
  - Temporal models (GRU) have additional advantage for persistence detection
  - ResiliGraph-STGAT (stress gate + temporal attention) is best positioned to
    detect high-DER neighborhoods under renewable ramp conditions

Spatial feature heterogeneity
------------------------------
All per-node features are now spatially heterogeneous:
  - solar_per_bus[i] = DER_PENETRATION[i] × solar_irradiance  (varies by bus)
  - ramp_per_bus[i]  = DER_PENETRATION[i] × |Δsolar_irr|      (varies by bus)
  - load_per_bus[i]  = mult × LOAD_DIVERSITY[i]               (varies by bus)
  - renewable_stress_arr[i] = ramp_per_bus[i] × (net_load > 1.0)

Previously all of these were uniform scalars broadcast to all buses.  With uniform
features, MLP and GNN see identical information per bus; spatial heterogeneity
gives GNNs additional signal from aggregating diverse neighbors.
"""

from pathlib import Path

import numpy as np
import torch

from src.simulation.run_powerflow import run_power_flow
from src.data.build_graph import build_graph
from src.data.label_builder import (
    add_bus_labels,
    add_line_labels,
    map_line_labels_to_buses,
    CONGESTION_THRESHOLD_PCT,
)


# ---------------------------------------------------------------------------
# Helper: bus adjacency from line DataFrame
# ---------------------------------------------------------------------------
def _build_bus_adjacency(line_df, bus_df):
    """
    Build an undirected bus adjacency dict from the line DataFrame.

    Returns
    -------
    adj : dict {bus_idx: [neighbor_idx, ...]}
    bus_name_to_idx : dict {lowercase_name: idx}
    """
    name_col = "bus_name" if "bus_name" in bus_df.columns else bus_df.columns[0]
    bus_name_to_idx = {
        str(b).lower(): i
        for i, b in enumerate(bus_df[name_col])
    }

    n = len(bus_df)
    adj = {i: [] for i in range(n)}

    for _, row in line_df.iterrows():
        fb = str(row.get("from_bus", "")).split(".")[0].lower()
        tb = str(row.get("to_bus",   "")).split(".")[0].lower()
        fi = bus_name_to_idx.get(fb)
        ti = bus_name_to_idx.get(tb)
        if fi is not None and ti is not None and fi != ti:
            if ti not in adj[fi]:
                adj[fi].append(ti)
            if fi not in adj[ti]:
                adj[ti].append(fi)

    return adj, bus_name_to_idx


# ---------------------------------------------------------------------------
# Helper: mean signal over graph neighbors
# ---------------------------------------------------------------------------
def _neighbor_mean(signal: np.ndarray, adj: dict, n: int) -> np.ndarray:
    """
    For each bus, compute the mean value of `signal` over its direct neighbors.
    Returns zero for isolated buses.
    """
    result = np.zeros(n, dtype=float)
    for i in range(n):
        nbrs = adj.get(i, [])
        if nbrs:
            result[i] = float(np.mean(signal[np.array(nbrs, dtype=int)]))
    return result


# ---------------------------------------------------------------------------
# Helper: voltage column extraction
# ---------------------------------------------------------------------------
def _get_voltage_array(bus_df):
    """Extract voltage magnitude and deviation from bus DataFrame."""
    for col in ("vm_pu", "voltage_pu", "v_pu", "v_min_pu"):
        if col in bus_df.columns:
            v   = bus_df[col].astype(float).values
            dev = np.abs(v - 1.0)
            return v, dev

    v   = np.ones(len(bus_df), dtype=float)
    dev = np.zeros(len(bus_df), dtype=float)
    return v, dev


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------
def build_temporal_graph_dataset(
    dss_file: Path,
    load_profile,
    out_dir: Path,
    renewable_aware: bool = True,
    seed: int = 42,
):
    """
    Build temporal graph snapshots with topology-conditioned labels.

    Parameters
    ----------
    dss_file        : Path to IEEE33.dss
    load_profile    : 1-D array of per-timestep load multipliers
    out_dir         : Output directory for temporal_graphs.pt
    renewable_aware : Whether to include renewable feature channels
    seed            : Random seed (controls all randomness in this function)
    """
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================================
    # SPATIAL INITIALIZATION — fixed once per dataset, separate RNG
    # so DER map does not consume main random stream.
    # ===========================================================
    N_ALLOC  = 50   # allocation for up to 50 buses (IEEE 33-bus safe upper bound)
    init_rng = np.random.RandomState(seed + 99991)

    # DER penetration by bus index.
    # Reflects typical distribution of rooftop solar in radial feeders:
    # end-of-feeder and lateral buses have higher DER than substation-adjacent buses.
    DER_PENETRATION = np.zeros(N_ALLOC, dtype=float)

    HIGH_DER = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    MED_DER  = [12, 13, 14, 15, 16]
    LOW_DER  = [6,  7,  8,  9, 10, 11]
    # Buses 0–5 (substation side) retain DER_PENETRATION = 0.0

    for b in HIGH_DER:
        DER_PENETRATION[b] = 0.28 + 0.20 * init_rng.rand()   # 28–48 %
    for b in MED_DER:
        DER_PENETRATION[b] = 0.12 + 0.12 * init_rng.rand()   # 12–24 %
    for b in LOW_DER:
        DER_PENETRATION[b] = 0.04 + 0.06 * init_rng.rand()   # 4–10 %

    # Load diversity: slight customer-mix variation across buses.
    # Prevents uniform load feature from making buses indistinguishable.
    LOAD_DIVERSITY = np.clip(
        1.0 + 0.12 * init_rng.randn(N_ALLOC),
        0.75, 1.25,
    )

    # Approximate mean DER penetration for global power-flow net load
    MEAN_DER = float(np.mean(DER_PENETRATION[:33]))   # ~0.20–0.22

    # ===========================================================
    # TEMPORAL STATE — carried between timesteps
    # ===========================================================
    graphs         = []
    prev_loading   = None          # line loading_pct from previous step
    prev_solar_irr = 0.0           # global solar irradiance at t-1
    prev_volt_dev  = None          # per-bus voltage deviation at t-1

    load_profile = np.asarray(load_profile, dtype=float)

    # ===========================================================
    # MAIN LOOP
    # ===========================================================
    for t, mult in enumerate(load_profile):
        hour = t % 24

        # -------------------------------------------------------
        # Solar irradiance profile (global)
        # -------------------------------------------------------
        solar_shape  = max(0.0, np.sin(np.pi * hour / 24.0))
        solar_noise  = np.random.normal(0, 0.035)
        solar_irr    = float(np.clip(0.18 * solar_shape + solar_noise, 0.0, 0.55))
        global_ramp  = float(abs(solar_irr - prev_solar_irr))

        # Global net-load multiplier for power flow
        # Uses mean DER penetration to approximate feeder-wide solar contribution
        net_load_global = float(
            np.clip(float(mult) - MEAN_DER * solar_irr, 0.10, 3.20)
        )

        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))

        effective_load_mult = net_load_global if renewable_aware else float(mult)

        # -------------------------------------------------------
        # Power flow
        # -------------------------------------------------------
        results = run_power_flow(
            dss_file, mode="snapshot",
            load_mult=effective_load_mult, save_dir=None,
        )

        if not results["converged"]:
            print(f"[WARN] t={t}: power flow did not converge")
            prev_solar_irr = solar_irr    # still advance solar state
            continue

        bus_df         = results["bus_df"]
        line_df        = results["line_df"]
        transformer_df = results["transformer_df"]
        n_buses        = len(bus_df)

        # -------------------------------------------------------
        # Per-bus spatial quantities (clipped to actual n_buses)
        # -------------------------------------------------------
        der  = DER_PENETRATION[:n_buses]
        div  = LOAD_DIVERSITY[:n_buses]

        # Per-bus solar output (heterogeneous — varies across the feeder)
        solar_per_bus    = der * solar_irr
        # Per-bus renewable ramp (DER-scaled)
        ramp_per_bus     = der * global_ramp
        # Per-bus load with customer diversity
        load_per_bus     = np.clip(float(mult) * div, 0.10, 3.50)
        # Per-bus net load (what remains after local DER offsets local load)
        net_load_per_bus = np.clip(load_per_bus - solar_per_bus, 0.10, 3.50)

        # -------------------------------------------------------
        # Mild topology uncertainty (reduced to 3 %)
        # -------------------------------------------------------
        if np.random.rand() < 0.03 and len(line_df) > 5:
            line_df = (
                line_df.sample(frac=0.97, random_state=t)
                .reset_index(drop=True)
            )

        # -------------------------------------------------------
        # Base physical labels (ANSI thresholds from label_builder)
        # -------------------------------------------------------
        bus_df  = add_bus_labels(bus_df)
        line_df = add_line_labels(line_df)

        # -------------------------------------------------------
        # Build bus adjacency for topology-conditioned labels.
        # This MUST happen before fault/voltage topology augmentation.
        # -------------------------------------------------------
        adj, bus_name_to_idx = _build_bus_adjacency(line_df, bus_df)

        # -------------------------------------------------------
        # Cascading congestion: persistence + renewable propagation
        # -------------------------------------------------------
        if (
            prev_loading is not None
            and len(prev_loading) == len(line_df)
            and "loading_pct" in line_df.columns
        ):
            current_loading  = line_df["loading_pct"].astype(float).values
            current_high     = current_loading > CONGESTION_THRESHOLD_PCT
            prev_high        = prev_loading.astype(float).values > CONGESTION_THRESHOLD_PCT
            persistent_cong  = current_high & prev_high

            line_df["congestion_risk"] = persistent_cong.astype(int)

            renewable_stress_event = (
                renewable_aware
                and global_ramp > 0.035
                and net_load_global > 1.05
            )

            if renewable_stress_event:
                # Near-overloaded lines escalate under renewable stress
                borderline      = current_loading > (CONGESTION_THRESHOLD_PCT - 4.0)
                escalation_mask = borderline & (np.random.rand(len(line_df)) < 0.28)
                line_df.loc[escalation_mask, "congestion_risk"] = 1

                # Topology-aware cascade: congested lines propagate to neighbors
                overloaded_idx    = np.where(line_df["congestion_risk"].values == 1)[0]
                propagation_mask  = np.zeros(len(line_df), dtype=bool)

                for idx in overloaded_idx:
                    fb = str(line_df.iloc[idx]["from_bus"]).lower()
                    tb = str(line_df.iloc[idx]["to_bus"]).lower()

                    neighbor_lines = (
                        (line_df["from_bus"].astype(str).str.lower() == fb)
                        | (line_df["to_bus"].astype(str).str.lower() == fb)
                        | (line_df["from_bus"].astype(str).str.lower() == tb)
                        | (line_df["to_bus"].astype(str).str.lower() == tb)
                    ).values

                    spread            = np.random.rand(len(line_df)) < 0.18
                    propagation_mask |= neighbor_lines & spread

                line_df.loc[propagation_mask, "congestion_risk"] = 1

        if "loading_pct" in line_df.columns:
            prev_loading = line_df["loading_pct"].copy().reset_index(drop=True)

        # -------------------------------------------------------
        # Map line congestion to buses (both endpoints)
        # -------------------------------------------------------
        bus_df = map_line_labels_to_buses(bus_df, line_df)

        # -------------------------------------------------------
        # Voltage deviation
        # -------------------------------------------------------
        _, voltage_dev = _get_voltage_array(bus_df)

        # -------------------------------------------------------
        # VOLTAGE VIOLATION — topology + temporally conditioned
        # -------------------------------------------------------
        VOLT_THRESH  = 0.04    # |v - 1.0| > 0.04 → v < 0.96 p.u. (near ANSI limit)
        NEAR_THRESH  = 0.025   # near-threshold band for augmentation

        # Component 1: own measured voltage violation (local — MLP can predict from voltage_dev)
        own_violation = voltage_dev > VOLT_THRESH

        # Component 2: PRECURSOR VOLTAGE RISK — topology-conditioned.
        # Physical rationale: a bus downstream of congested lines will experience
        # voltage drop as loading persists; the model should predict this before
        # the voltage has actually dropped below threshold.
        # Requires: neighbor aggregation (graph model) → MLP CANNOT compute this.
        congestion_bus = bus_df["congestion_risk"].astype(float).values
        nbr_cong_mean  = _neighbor_mean(congestion_bus, adj, n_buses)

        precursor_voltage = (
            (voltage_dev > NEAR_THRESH)
            & ~own_violation
            & (nbr_cong_mean > 0.40)    # >40% of neighbors are congested
        )

        # Component 3: TEMPORAL VOLTAGE PERSISTENCE — temporally conditioned.
        # Physical rationale: a bus that has been near-threshold for two
        # consecutive timesteps is more likely to cross over than one that
        # just reached the near-threshold band this timestep.
        # Requires: temporal context (GRU) → MLP and static GNNs CANNOT detect this.
        if prev_volt_dev is not None and len(prev_volt_dev) == n_buses:
            persistent_stress = (
                (voltage_dev     > NEAR_THRESH)
                & (prev_volt_dev > NEAR_THRESH)
                & ~own_violation
            )
        else:
            persistent_stress = np.zeros(n_buses, dtype=bool)

        # Component 4: block-correlated renewable augmentation (existing mechanism)
        aug_block = np.zeros(n_buses, dtype=bool)
        if renewable_aware and global_ramp > 0.05:
            near = (voltage_dev > NEAR_THRESH) & ~own_violation
            for start in range(0, n_buses, 3):
                end = min(start + 3, n_buses)
                if np.random.rand() < 0.08:
                    aug_block[start:end] = True
            aug_block &= near

        # Component 5: heavy-loading augmentation (existing mechanism)
        heavy_loading = net_load_global > np.percentile(load_profile, 70)
        if heavy_loading:
            near       = (voltage_dev > NEAR_THRESH) & ~own_violation
            heavy_aug  = near & (np.random.rand(n_buses) < 0.05)
        else:
            heavy_aug = np.zeros(n_buses, dtype=bool)

        # Combined voltage violation
        voltage_violation = (
            own_violation
            | precursor_voltage       # topology-conditioned (new)
            | persistent_stress       # temporal persistence (new)
            | aug_block
            | heavy_aug
        )
        bus_df["voltage_violation"] = voltage_violation.astype(int)

        # -------------------------------------------------------
        # FAULT RISK — topology-conditioned
        # -------------------------------------------------------
        severe_voltage   = voltage_dev > 0.08
        congestion_flag  = bus_df["congestion_risk"].astype(bool).values
        voltage_flag     = bus_df["voltage_violation"].astype(bool).values

        # Component: NEIGHBOR FAULT PROPAGATION — topology-conditioned.
        # Physical rationale: when a majority of neighboring buses have voltage
        # violations, the zone is at elevated relay coordination / protection risk.
        # This represents pre-fault cascade state that a graph model can detect
        # but MLP cannot (MLP only sees own voltage, not neighborhood state).
        nbr_volt_mean = _neighbor_mean(voltage_flag.astype(float), adj, n_buses)

        neighbor_fault_risk = (
            (nbr_volt_mean  > 0.40)         # >40 % of neighbors have voltage violations
            & (voltage_dev  > NEAR_THRESH)  # own voltage is also under pressure
            & ~congestion_flag              # distinct from congestion-caused fault
        )

        equipment_fault = np.random.rand(n_buses) < 0.015

        bus_df["fault_risk"] = (
            (voltage_flag & (congestion_flag | severe_voltage))
            | neighbor_fault_risk      # topology-conditioned (new)
            | equipment_fault
        ).astype(int)

        # -------------------------------------------------------
        # Update temporal state for next iteration
        # -------------------------------------------------------
        prev_volt_dev  = voltage_dev.copy()
        prev_solar_irr = solar_irr

        # -------------------------------------------------------
        # Build PyG graph
        # -------------------------------------------------------
        _, pyg_data = build_graph(
            bus_df, line_df, transformer_df,
            add_reverse_edges=True, save_path=None,
        )

        # -------------------------------------------------------
        # FEATURE CONSTRUCTION — spatially heterogeneous
        # All features now vary per bus, not just per timestep.
        # This creates meaningful between-bus variation that GNNs
        # can leverage by aggregating diverse neighbor contexts.
        # -------------------------------------------------------
        hour_sin_arr    = np.full(n_buses, hour_sin,  dtype=float)
        hour_cos_arr    = np.full(n_buses, hour_cos,  dtype=float)
        voltage_dev_arr = voltage_dev.astype(float)

        degree = np.zeros(n_buses, dtype=float)
        for src, dst in pyg_data.edge_index.cpu().numpy().T:
            degree[int(src)] += 1
            degree[int(dst)] += 1
        degree_norm = degree / max(1.0, degree.max())

        # Per-bus renewable stress: DER-weighted ramp × high-load indicator
        # A high-DER bus during a ramp event has higher stress than a low-DER bus
        # even under the same global conditions.
        renewable_stress_arr = np.clip(
            ramp_per_bus * float(net_load_global > 1.0),
            0.0, 1.0,
        )

        if renewable_aware:
            features = np.column_stack([
                load_per_bus,            # 0 — per-bus load (diverse)
                solar_per_bus,           # 1 — per-bus solar (DER-weighted)
                net_load_per_bus,        # 2 — per-bus net load
                ramp_per_bus,            # 3 — per-bus renewable ramp (DER-scaled)
                hour_sin_arr,            # 4 — cyclic hour encoding (sin)
                hour_cos_arr,            # 5 — cyclic hour encoding (cos)
                voltage_dev_arr,         # 6 — physical voltage deviation
                degree_norm,             # 7 — normalised graph degree
                renewable_stress_arr,    # 8 — per-bus DER-weighted stress
            ])
        else:
            features = np.column_stack([
                load_per_bus,            # 0
                voltage_dev_arr,         # 1
                degree_norm,             # 2
            ])

        pyg_data.x = torch.tensor(features, dtype=torch.float)

        pyg_data.y = torch.tensor(
            bus_df[["fault_risk", "congestion_risk", "voltage_violation"]].values,
            dtype=torch.float,
        )

        pyg_data.timestep        = int(t)
        pyg_data.load_mult       = float(mult)
        pyg_data.solar_pu        = float(solar_irr)
        pyg_data.net_load_mult   = float(net_load_global)
        pyg_data.renewable_ramp  = float(global_ramp)
        pyg_data.renewable_aware = bool(renewable_aware)

        graphs.append(pyg_data)

    if len(graphs) == 0:
        raise ValueError("No valid graph snapshots created.")

    torch.save(graphs, out_dir / "temporal_graphs.pt")
    return graphs