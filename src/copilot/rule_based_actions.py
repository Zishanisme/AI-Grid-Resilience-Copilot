"""
rule_based_actions.py
=====================
Converts predicted fault, congestion, and voltage probabilities into
a single operator-facing action recommendation.

Fix: Added explicit else guard at the end of recommend_action() to
prevent a silent None return if risk_type somehow falls outside the
known set.  Python's dict max() guarantees one of the three keys is
always returned, but the guard makes the contract explicit and
prevents KeyError downstream if the function signature ever changes.
"""


def recommend_action(
    fault_prob:      float,
    congestion_prob: float,
    voltage_prob:    float,
    threshold:       float = 0.5,
) -> dict:
    """
    Return a single operator-facing recommendation.

    Logic
    -----
    1. If all risk probabilities are below threshold → normal / monitor.
    2. Otherwise, the risk type with the highest probability determines
       the recommended action.
    3. Tie-breaking: fault > congestion > voltage (insertion-order of
       the risks dict below).  This ordering reflects operational severity.

    Returns
    -------
    dict with keys: risk_type, recommended_action, explanation.
    """
    risks = {
        "fault":      fault_prob,
        "congestion": congestion_prob,
        "voltage":    voltage_prob,
    }

    risk_type = max(risks, key=risks.get)
    max_risk  = risks[risk_type]

    if max_risk < threshold:
        return {
            "risk_type":          "normal",
            "recommended_action": "monitor",
            "explanation": (
                "All predicted risks are below the operator action threshold. "
                "Continue normal monitoring."
            ),
        }

    if risk_type == "fault":
        return {
            "risk_type":          "fault",
            "recommended_action": "isolate_fault_or_dispatch_crew",
            "explanation": (
                "Fault probability is the dominant risk. Isolate the affected "
                "feeder section and dispatch inspection crew immediately."
            ),
        }

    if risk_type == "congestion":
        return {
            "risk_type":          "congestion",
            "recommended_action": "network_reconfiguration_or_reroute",
            "explanation": (
                "Congestion probability is the dominant risk. Consider feeder "
                "switch reconfiguration or power-flow rerouting to relieve "
                "thermal loading."
            ),
        }

    if risk_type == "voltage":
        return {
            "risk_type":          "voltage",
            "recommended_action": "voltage_regulation_or_reactive_support",
            "explanation": (
                "Voltage risk is the dominant risk. Deploy voltage regulation "
                "equipment or reactive power support (capacitor banks, STATCOM) "
                "to restore voltage within ANSI C84.1 limits."
            ),
        }

    # Explicit fallback — should be unreachable given risks dict structure
    return {
        "risk_type":          "unknown",
        "recommended_action": "escalate_to_operator",
        "explanation": (
            f"Risk type could not be resolved (risk_type={risk_type!r}). "
            "Escalate to human operator for manual assessment."
        ),
    }
