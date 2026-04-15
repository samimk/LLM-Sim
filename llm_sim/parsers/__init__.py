"""File parsers for MATPOWER and simulation results."""

from __future__ import annotations

import logging

from llm_sim.parsers.matpower_model import (
    Branch,
    Bus,
    GenCost,
    Generator,
    MATNetwork,
)
from llm_sim.parsers.matpower_parser import parse_matpower
from llm_sim.parsers.matpower_writer import write_matpower
from llm_sim.parsers.network_summary import network_summary
from llm_sim.parsers.opflow_parser import parse_opflow_output, parse_simulation_result
from llm_sim.parsers.opflow_results import (
    BranchResult,
    BusResult,
    GenResult,
    OPFLOWResult,
)
from llm_sim.parsers.results_summary import results_summary
from llm_sim.parsers.dcopflow_summary import dcopflow_results_summary

_logger = logging.getLogger("llm_sim.parsers")

# Applications that use the same output format as OPFLOW
_OPFLOW_COMPATIBLE_APPS = {"opflow", "dcopflow"}


def parse_simulation_result_for_app(sim_result, application: str, bus_limits=None):
    """Dispatch to the correct parser based on application name.

    For 'opflow' and 'dcopflow', both use parse_simulation_result (same output format).
    Returns parsed result object, or None if parsing fails.
    """
    if application in _OPFLOW_COMPATIBLE_APPS:
        return parse_simulation_result(sim_result, bus_limits=bus_limits)
    _logger.warning(
        "Unknown application '%s' for parse_simulation_result_for_app — "
        "falling back to OPFLOW parser.",
        application,
    )
    return parse_simulation_result(sim_result, bus_limits=bus_limits)


def results_summary_for_app(result: OPFLOWResult, application: str) -> str:
    """Dispatch to the correct results summary generator based on application.

    For 'opflow': use existing results_summary().
    For 'dcopflow': use dcopflow_results_summary() — a DC-aware summary
    that skips voltage magnitude analysis (meaningless in DC) and focuses
    on phase angles, active power dispatch, and line loading.
    For unknown applications: fall back to results_summary() with a warning.
    """
    if application == "dcopflow":
        return dcopflow_results_summary(result)
    if application == "opflow":
        return results_summary(result)
    _logger.warning(
        "Unknown application '%s' for results_summary_for_app — "
        "falling back to OPFLOW summary.",
        application,
    )
    return results_summary(result)


__all__ = [
    "MATNetwork",
    "Bus",
    "Generator",
    "Branch",
    "GenCost",
    "parse_matpower",
    "write_matpower",
    "network_summary",
    "OPFLOWResult",
    "BusResult",
    "BranchResult",
    "GenResult",
    "parse_opflow_output",
    "parse_simulation_result",
    "parse_simulation_result_for_app",
    "results_summary",
    "results_summary_for_app",
    "dcopflow_results_summary",
]
