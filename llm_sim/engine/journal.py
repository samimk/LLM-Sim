"""Search journal — records the trajectory of the LLM-driven search."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_sim.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("llm_sim.engine.journal")


@dataclass
class JournalEntry:
    """One iteration's record in the search journal."""

    iteration: int
    description: str
    commands: list[dict]
    objective_value: Optional[float]
    feasible: bool
    convergence_status: str
    violations_count: int
    voltage_min: float
    voltage_max: float
    max_line_loading_pct: float
    total_gen_mw: float
    total_load_mw: float
    llm_reasoning: str
    mode: str
    elapsed_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SearchJournal:
    """Maintains the search trajectory across iterations."""

    def __init__(self) -> None:
        self._entries: list[JournalEntry] = []

    def add_entry(self, entry: JournalEntry) -> None:
        """Append an entry to the journal."""
        self._entries.append(entry)

    def add_from_results(
        self,
        iteration: int,
        description: str,
        commands: list[dict],
        opflow_result: Optional[OPFLOWResult],
        sim_elapsed: float,
        llm_reasoning: str,
        mode: str,
    ) -> JournalEntry:
        """Create and append a journal entry from OPFLOW results.

        If opflow_result is None (simulation failed), fills in defaults
        indicating failure. Returns the created entry.
        """
        if opflow_result is not None:
            entry = JournalEntry(
                iteration=iteration,
                description=description,
                commands=commands,
                objective_value=opflow_result.objective_value,
                feasible=opflow_result.converged,
                convergence_status=opflow_result.convergence_status,
                violations_count=opflow_result.num_violations,
                voltage_min=opflow_result.voltage_min,
                voltage_max=opflow_result.voltage_max,
                max_line_loading_pct=opflow_result.max_line_loading_pct,
                total_gen_mw=opflow_result.total_gen_mw,
                total_load_mw=opflow_result.total_load_mw,
                llm_reasoning=llm_reasoning,
                mode=mode,
                elapsed_seconds=sim_elapsed,
            )
        else:
            entry = JournalEntry(
                iteration=iteration,
                description=description,
                commands=commands,
                objective_value=None,
                feasible=False,
                convergence_status="FAILED",
                violations_count=0,
                voltage_min=0.0,
                voltage_max=0.0,
                max_line_loading_pct=0.0,
                total_gen_mw=0.0,
                total_load_mw=0.0,
                llm_reasoning=llm_reasoning,
                mode=mode,
                elapsed_seconds=sim_elapsed,
            )

        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[JournalEntry]:
        """All journal entries (read-only copy)."""
        return list(self._entries)

    @property
    def latest(self) -> Optional[JournalEntry]:
        """The most recent entry, or None if empty."""
        return self._entries[-1] if self._entries else None

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_row(e: JournalEntry) -> str:
        """Format a single entry as a table row."""
        desc = e.description[:35].ljust(35)
        if e.feasible and e.objective_value is not None:
            cost = f"{e.objective_value:>12,.2f}"
            feas = "Yes"
            vrange = f"{e.voltage_min:.3f} - {e.voltage_max:.3f}"
            load = f"{e.max_line_loading_pct:>6.1f}%"
        else:
            cost = "         N/A"
            feas = "No "
            vrange = "N/A          "
            load = "   N/A"
        return (
            f"{e.iteration:>4} | {desc} | {cost} | {feas}   "
            f"| {vrange:<13} | {load}"
        )

    def format_for_prompt(self, max_entries: Optional[int] = None) -> str:
        """Format the journal as a compact text table for LLM prompt injection."""
        if not self._entries:
            return "Search Journal: no iterations yet."

        header = (
            "Iter | Description                         "
            "| Cost($)      | Feas. | V_range(pu)   | Max_load(%)"
        )
        sep = (
            "-----|-------------------------------------"
            "|--------------|-------|---------------|------------"
        )

        lines = [f"Search Journal ({len(self._entries)} iterations):", "", header, sep]

        if max_entries is None or len(self._entries) <= max_entries:
            for e in self._entries:
                lines.append(self._format_row(e))
        else:
            # Always include entry 1, then last (max_entries - 1)
            lines.append(self._format_row(self._entries[0]))
            tail_start = len(self._entries) - (max_entries - 1)
            lines.append(
                f"  ... (entries 2-{tail_start} omitted)"
            )
            for e in self._entries[tail_start:]:
                lines.append(self._format_row(e))

        return "\n".join(lines)

    def format_detailed(self) -> str:
        """Format the full journal with all fields for post-session reporting."""
        if not self._entries:
            return "Search Journal: empty."

        parts: list[str] = [
            f"Search Journal — {len(self._entries)} iterations",
            "=" * 60,
        ]

        for e in self._entries:
            parts.append("")
            parts.append(f"--- Iteration {e.iteration} [{e.timestamp}] ---")
            parts.append(f"Mode: {e.mode}")
            parts.append(f"Description: {e.description}")
            parts.append(f"Convergence: {e.convergence_status}")
            if e.objective_value is not None:
                parts.append(f"Objective value: ${e.objective_value:,.2f}")
            else:
                parts.append("Objective value: N/A")
            parts.append(f"Feasible: {e.feasible}")
            parts.append(f"Violations: {e.violations_count}")
            parts.append(f"Voltage: {e.voltage_min:.3f} - {e.voltage_max:.3f} pu")
            parts.append(f"Max line loading: {e.max_line_loading_pct:.1f}%")
            parts.append(f"Gen: {e.total_gen_mw:.2f} MW / Load: {e.total_load_mw:.2f} MW")
            parts.append(f"Sim time: {e.elapsed_seconds:.2f}s")
            parts.append(f"Commands ({len(e.commands)}):")
            for cmd in e.commands:
                parts.append(f"  {json.dumps(cmd)}")
            parts.append(f"LLM reasoning: {e.llm_reasoning}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_json(self, path: Path) -> None:
        """Export the journal to a JSON file."""
        data = [asdict(e) for e in self._entries]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Journal exported to %s (%d entries)", path, len(self._entries))

    def export_csv(self, path: Path) -> None:
        """Export the journal to a CSV file."""
        if not self._entries:
            path.write_text("", encoding="utf-8")
            return

        fieldnames = [
            "iteration", "description", "commands", "objective_value",
            "feasible", "convergence_status", "violations_count",
            "voltage_min", "voltage_max", "max_line_loading_pct",
            "total_gen_mw", "total_load_mw", "llm_reasoning",
            "mode", "elapsed_seconds", "timestamp",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self._entries:
                row = asdict(e)
                row["commands"] = json.dumps(row["commands"])
                writer.writerow(row)

        logger.info("Journal CSV exported to %s (%d entries)", path, len(self._entries))

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary_stats(self) -> dict:
        """Compute summary statistics across all entries."""
        if not self._entries:
            return {
                "total_iterations": 0,
                "best_objective": None,
                "best_iteration": None,
                "feasible_count": 0,
                "infeasible_count": 0,
                "objective_trend": [],
                "voltage_range_trend": [],
            }

        feasible = [e for e in self._entries if e.feasible and e.objective_value is not None]
        infeasible_count = sum(1 for e in self._entries if not e.feasible)

        if feasible:
            best = min(feasible, key=lambda e: e.objective_value)  # type: ignore[arg-type]
            best_objective = best.objective_value
            best_iteration = best.iteration
        else:
            best_objective = None
            best_iteration = None

        return {
            "total_iterations": len(self._entries),
            "best_objective": best_objective,
            "best_iteration": best_iteration,
            "feasible_count": len(feasible),
            "infeasible_count": infeasible_count,
            "objective_trend": [e.objective_value for e in self._entries],
            "voltage_range_trend": [
                (e.voltage_min, e.voltage_max) for e in self._entries
            ],
        }
