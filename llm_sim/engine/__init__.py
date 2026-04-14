"""Simulation orchestration engine — modification commands and application."""

from __future__ import annotations

import logging

from llm_sim.engine.agent_loop import AgentLoopController, SearchSession
from llm_sim.engine.commands import ModCommand, parse_command
from llm_sim.engine.executor import SimulationExecutor, SimulationResult
from llm_sim.engine.journal import JournalEntry, ObjectiveEntry, ObjectiveRegistry, SearchJournal
from llm_sim.engine.metric_extractor import available_metrics, extract_all_metrics, extract_metric
from llm_sim.engine.modifier import ModificationReport, apply_modifications
from llm_sim.engine.objective_parser import build_objective_extraction_prompt, parse_objective_extraction
from llm_sim.engine.schema_description import command_schema_text
from llm_sim.engine.validation import ValidationResult, validate_command
from llm_sim.parsers.matpower_model import MATNetwork

logger = logging.getLogger("llm_sim.engine")

__all__ = [
    "AgentLoopController",
    "JournalEntry",
    "ModCommand",
    "ModificationReport",
    "ObjectiveEntry",
    "ObjectiveRegistry",
    "SearchJournal",
    "SearchSession",
    "SimulationExecutor",
    "SimulationResult",
    "ValidationResult",
    "apply_modifications",
    "available_metrics",
    "build_objective_extraction_prompt",
    "command_schema_text",
    "extract_all_metrics",
    "extract_metric",
    "parse_command",
    "parse_objective_extraction",
    "process_commands",
    "validate_command",
]


def process_commands(
    net: MATNetwork,
    raw_commands: list[dict],
) -> tuple[MATNetwork, ModificationReport]:
    """Parse, validate, and apply raw JSON commands to a network.

    This is the main entry point for the Modification Engine.

    Args:
        net: Base network (not mutated).
        raw_commands: List of JSON dicts from the LLM.

    Returns:
        Tuple of (modified_network, report).
    """
    commands: list[ModCommand] = []
    parse_errors: list[tuple[dict, str]] = []

    for raw in raw_commands:
        try:
            cmd = parse_command(raw)
            commands.append(cmd)
        except ValueError as exc:
            logger.warning("Failed to parse command %s: %s", raw, exc)
            parse_errors.append((raw, str(exc)))

    modified, report = apply_modifications(net, commands)

    # Add parse errors as skipped (using a sentinel command-like dict)
    for raw, err in parse_errors:
        # Store raw dict and error for reporting
        report.skipped.append((raw, [err]))  # type: ignore[arg-type]

    return modified, report
