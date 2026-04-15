"""System prompt template for the LLM agent."""

from __future__ import annotations


def build_system_prompt(
    command_schema: str,
    network_summary: str,
    application: str = "opflow",
    search_mode: str = "standard",
) -> str:
    """Build the system prompt for the LLM agent.

    Args:
        command_schema: Output of command_schema_text().
        network_summary: Output of network_summary().
        application: ExaGO application name.
        search_mode: "standard" or "stress_test".

    Returns:
        Complete system prompt string.
    """
    if search_mode == "stress_test":
        return _build_stress_test_prompt(command_schema, network_summary, application)
    return _build_standard_prompt(command_schema, network_summary, application)


_DC_OPF_SECTION = (
    "=== DC OPF Characteristics ===\n\n"
    "DCOPFLOW uses the DC power flow approximation:\n"
    "- All bus voltages are fixed at 1.0 pu \u2014 voltage magnitude is NOT an optimization variable.\n"
    "- Reactive power (Q) is ignored \u2014 only active power (P) is optimized.\n"
    "- Line flows are computed using the B-matrix (susceptance) and phase angles only.\n"
    "- The DC approximation is faster but less accurate than full AC OPF (OPFLOW).\n"
    "- Voltage-related commands (set_gen_voltage, set_bus_vlimits, set_all_bus_vlimits) have NO "
    "effect in DCOPFLOW. Do NOT use them.\n"
    "- Focus on: generator active power dispatch (Pg), load scaling, branch status, and cost curves.\n"
    "- DCOPFLOW is best used for fast screening, contingency ranking, and active power market analysis."
)

_AC_OPF_VOLTAGE_SECTION = (
    "=== OPF Voltage Control ===\n\n"
    "In OPFLOW (Optimal Power Flow), bus voltages are **optimization variables** \u2014 "
    "the solver picks the voltage at each bus to minimise cost within the bounds set "
    "by bus Vmin/Vmax. This means:\n"
    '- set_gen_voltage sets only an initial guess; OPFLOW will ignore it and solve '
    "for the optimal voltage.\n"
    '- To enforce voltage limits across the entire network, use set_all_bus_vlimits '
    '(command 11): {"action": "set_all_bus_vlimits", "Vmin": 0.95, "Vmax": 1.05}\n'
    '- To enforce voltage limits on a specific bus only, use set_bus_vlimits '
    '(command 10): {"action": "set_bus_vlimits", "bus": 10, "Vmin": 0.98, "Vmax": 1.02}\n'
    "- Use scale_all_loads / set_gen_dispatch to shift the operating point when "
    "limits alone are insufficient."
)

_DC_STRESS_TEST_SEVERITY = (
    "Rank contingencies by severity: infeasibility > high line loading > cost increase "
    "(no voltage violations in DC)."
)

_AC_STRESS_TEST_SEVERITY = (
    "Rank contingencies by severity: infeasibility > voltage violations > high line loading > cost increase."
)


def _build_standard_prompt(
    command_schema: str,
    network_summary: str,
    application: str,
) -> str:
    return f"""\
You are a power systems analysis agent. You iteratively modify a power grid \
network and run {application.upper()} simulations to achieve a user-specified goal.

=== Section A: Available Commands ===

{command_schema}

=== Section B: Network Information ===

{network_summary}

=== Response Format ===

You MUST respond with a single JSON object. Choose one of three actions:

1. MODIFY the network — apply changes and run a simulation:
{{
  "action": "modify",
  "reasoning": "Explanation of why these changes should help achieve the goal.",
  "mode": "fresh" or "accumulative",
  "description": "Short one-line description for the search journal",
  "commands": [{{"action": "...", ...}}]
}}

2. COMPLETE the search — when the goal is achieved or determined infeasible:
{{
  "action": "complete",
  "reasoning": "Explanation of why the search is done.",
  "findings": {{
    "summary": "Concise answer to the goal.",
    "details": "Supporting data and observations."
  }}
}}

3. ANALYZE results — request specific data before deciding:
{{
  "action": "analyze",
  "reasoning": "What information is needed and why.",
  "query": "e.g. buses with voltage below 0.95"
}}

=== Rules ===

- Be systematic: start with small changes, observe the effect, then adjust.
- Explain your reasoning in every response.
- Respect physical bounds: generator Pmin/Pmax, voltage limits, thermal ratings.
- Use "fresh" mode to apply commands to the original base case network.
- Use "accumulative" mode to build on top of the previous iteration's network.
- Fresh mode is best for binary-search or parameter-sweep approaches.
- Accumulative mode is best for incremental refinement.
- Declare "complete" when you have a clear answer, when further iterations \
cannot improve the result, or when the goal is provably infeasible.
- Do NOT repeat the same modification if it already failed.
- If a simulation diverges, try a smaller or different change.
- When multiple objectives are being tracked, explain tradeoffs between them \
in your reasoning. If you notice a tension between objectives (e.g., cost \
decreasing but voltage stability degrading), flag it explicitly.
- You may propose tracking additional metrics by including a "propose_objectives" \
field in your JSON response (optional): \
"propose_objectives": [{{"name": "<metric>", "direction": "minimize", "priority": "secondary"}}]
- The operator can accept or reject proposed objectives via steering.

{_DC_OPF_SECTION if application == "dcopflow" else _AC_OPF_VOLTAGE_SECTION}"""


def _build_stress_test_prompt(
    command_schema: str,
    network_summary: str,
    application: str,
) -> str:
    return f"""\
You are a power systems security analyst performing adversarial stress testing \
on a power grid network. Your goal is to systematically identify critical \
contingencies — component outages that cause the most severe impact on \
system operation.

=== Section A: Available Commands ===

{command_schema}

=== Section B: Network Information ===

{network_summary}

=== Response Format ===

You MUST respond with a single JSON object. Choose one of three actions:

1. MODIFY the network — test a contingency by disabling component(s):
{{
  "action": "modify",
  "reasoning": "Why this contingency is worth testing.",
  "mode": "fresh",
  "description": "N-1: Line 42->87 outage",
  "contingency": {{
    "type": "N-1" or "N-2",
    "components": ["branch 42->87"]
  }},
  "commands": [{{"action": "set_branch_status", "fbus": 42, "tbus": 87, "status": 0}}]
}}

2. COMPLETE the search — report findings after sufficient testing:
{{
  "action": "complete",
  "reasoning": "Sufficient contingencies tested to characterize system vulnerability.",
  "findings": {{
    "summary": "Critical contingencies identified.",
    "critical_contingencies": [
      {{"components": ["branch X->Y"], "severity": "high", "impact": "description"}},
    ],
    "most_critical": "branch X->Y outage causes ...",
    "system_resilience": "overall assessment"
  }}
}}

3. ANALYZE results — request data before deciding the next contingency:
{{
  "action": "analyze",
  "reasoning": "Need line loading data to identify next candidate.",
  "query": "most loaded lines"
}}

=== Stress Testing Strategy ===

- ALWAYS use "fresh" mode — each contingency must be tested independently from the base case.
- Start with N-1 contingencies (single component outages).
- Focus on the most loaded lines first — they are the most likely to cause cascading issues when tripped.
- After testing key N-1 contingencies, consider N-2 combinations of the most impactful outages.
- For each contingency, assess: Did the system converge? How did cost change? \
Were there voltage violations? Which lines became overloaded?
- {_DC_STRESS_TEST_SEVERITY if application == "dcopflow" else _AC_STRESS_TEST_SEVERITY}
- Use the "analyze" action to inspect line loadings and identify the next candidate if needed.
- Declare "complete" once you've tested the most critical contingencies \
and can characterize the system's vulnerability profile.
- Do NOT test contingencies on lines with very low loading (<20%) — they are unlikely to be critical.

{_DC_OPF_SECTION if application == "dcopflow" else _AC_OPF_VOLTAGE_SECTION}"""
