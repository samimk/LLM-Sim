"""Concurrent PFLOW explore/select data structures and result formatting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from llm_sim.engine.commands import ModCommand
from llm_sim.engine.executor import SimulationResult
from llm_sim.engine.journal import ObjectiveEntry
from llm_sim.engine.metric_extractor import extract_all_metrics
from llm_sim.engine.pareto import ParetoCandidate, pareto_filter
from llm_sim.parsers.matpower_model import MATNetwork
from llm_sim.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("llm_sim.engine.explore")


@dataclass
class VariantResult:
    """Result of a single variant simulation within an explore action."""

    label: str
    description: str
    commands: list[ModCommand]
    raw_commands: list[dict]
    modified_net: MATNetwork
    sim_result: SimulationResult
    opflow_result: Optional[OPFLOWResult]
    is_pareto: bool = False


@dataclass
class ExploreCache:
    """Temporary storage for explore results awaiting a select action."""

    variants: dict[str, VariantResult] = field(default_factory=dict)
    description: str = ""
    reasoning: str = ""
    iteration: int = 0
    base_network_snapshot: Optional[MATNetwork] = None
    base_mode: str = "accumulative"


def variant_metrics(
    variant: VariantResult,
    objectives: list[ObjectiveEntry],
    gencost: list | None = None,
) -> dict[str, float]:
    """Extract metrics from a variant for Pareto comparison.

    For PFLOW, if no objectives are registered, uses computed generation
    cost as the default metric.
    """
    if variant.opflow_result is None:
        return {}

    metrics: dict[str, float] = {}

    if objectives:
        obj_names = [o.name for o in objectives]
        extracted = extract_all_metrics(variant.opflow_result, obj_names)
        metrics.update(extracted)

        # For PFLOW: override generation_cost with computed cost if gencost is available
        is_pflow = any(
            g.fuel.lower() == "" or g.fuel.lower() != ""
            for g in variant.opflow_result.generators[:1]
        )
        # Always compute generation cost for PFLOW since objective_value is 0.0
        if "generation_cost" in obj_names and gencost is not None:
            computed = variant.opflow_result.compute_generation_cost(gencost)
            if computed > 0:
                metrics["generation_cost"] = computed
    else:
        # No objectives registered: use computed generation cost as default
        if gencost is not None:
            computed = variant.opflow_result.compute_generation_cost(gencost)
            metrics["generation_cost"] = computed

    return metrics


def compute_pareto_labels(
    variants: dict[str, VariantResult],
    objectives: list[ObjectiveEntry],
    gencost: list | None = None,
) -> list[str]:
    """Compute Pareto-optimal labels and mark variants accordingly.

    Returns the list of Pareto-optimal labels. Variants in the dict
    are modified in-place: ``is_pareto`` is set on each.
    """
    candidates: list[ParetoCandidate] = []
    for label, v in variants.items():
        feasible = (
            v.opflow_result is not None
            and v.opflow_result.feasibility_detail == "feasible"
        )
        metrics = variant_metrics(v, objectives, gencost)
        candidates.append(ParetoCandidate(
            label=label, metrics=metrics, feasible=feasible,
        ))

    pareto_labels = pareto_filter(candidates, objectives)

    for label, v in variants.items():
        v.is_pareto = label in pareto_labels

    return pareto_labels


def format_variant_results(
    variants: dict[str, VariantResult],
    pareto_labels: list[str],
    gencost: list | None = None,
) -> str:
    """Format variant results as a comparative table for the LLM prompt.

    Shows a compact table with key metrics for each variant, with Pareto-optimal
    variants marked with a star. Includes guidance on how to select a variant.
    """
    if not variants:
        return "No variant results to display."

    lines: list[str] = []
    lines.append(f"=== Neighborhood Exploration Results ({len(variants)} variants) ===")
    lines.append("")

    # Header
    lines.append(
        f"{'Variant':<8} | {'Description':<28} | {'Cost($)':>12} | "
        f"{'Feas.':>5} | {'V_range':>13} | {'Max_load':>8} | "
        f"{'V viol':>5} | {'Pareto':>6}"
    )
    lines.append("-" * 100)

    for label, v in variants.items():
        desc = v.description[:28]
        if v.opflow_result is not None:
            opf = v.opflow_result
            cost_str = "N/A"
            if gencost is not None:
                try:
                    computed = opf.compute_generation_cost(gencost)
                    cost_str = f"${computed:,.2f}"
                except Exception:
                    pass

            feas = opf.feasibility_detail if opf.feasibility_detail else (
                "Yes" if opf.converged else "No"
            )

            if opf.voltage_min > 0 and opf.voltage_max > 0:
                vr = f"{opf.voltage_min:.3f}-{opf.voltage_max:.3f}"
            else:
                vr = "N/A"

            ml = f"{opf.max_line_loading_pct:.1f}%"
            vv = str(opf.num_violations)
        else:
            cost_str = "FAIL"
            feas = "No"
            vr = "N/A"
            ml = "N/A"
            vv = "-"

        pareto_mark = "  ★" if v.is_pareto else ""
        lines.append(
            f"{label:<8} | {desc:<28} | {cost_str:>12} | "
            f"{ feas:>5} | {vr:>13} | {ml:>8} | "
            f"{vv:>5} | {pareto_mark:>6}"
        )

    lines.append("")

    # Pareto summary
    if pareto_labels:
        lines.append(f"Pareto-optimal variants (★): {', '.join(pareto_labels)}")
    else:
        lines.append("No Pareto-optimal variants (all infeasible or marginal).")

    lines.append("")
    lines.append(
        'To adopt a variant as the new current point, respond with:\n'
        '{"action": "select", "choice": "<label>", '
        '"reasoning": "why this variant is best"}\n\n'
        'You can also:\n'
        '- "explore" to evaluate a different neighborhood\n'
        '- "analyze" to inspect detailed results of any variant\n'
        '- "modify" to make a single-point change (clears explore state)\n'
        '- "complete" to end the search'
    )

    return "\n".join(lines)