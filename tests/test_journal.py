"""Tests for the Search Journal."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm_sim.engine.journal import JournalEntry, SearchJournal
from llm_sim.parsers.opflow_results import OPFLOWResult


def _make_entry(iteration: int, obj: float = 1000.0, feasible: bool = True, **kw) -> JournalEntry:
    """Helper to create a JournalEntry with sensible defaults."""
    defaults = dict(
        iteration=iteration,
        description=f"Iteration {iteration} changes",
        commands=[{"command": "scale_all_loads", "scale_factor": 1.0 + iteration * 0.1}],
        objective_value=obj if feasible else None,
        feasible=feasible,
        convergence_status="CONVERGED" if feasible else "DIVERGED",
        violations_count=0,
        voltage_min=1.05,
        voltage_max=1.10,
        max_line_loading_pct=50.0 + iteration,
        total_gen_mw=500.0,
        total_load_mw=480.0,
        llm_reasoning="Test reasoning",
        mode="accumulative",
        elapsed_seconds=0.5,
        timestamp="2026-01-01T00:00:00",
    )
    defaults.update(kw)
    return JournalEntry(**defaults)


def _make_opflow_result(**kw) -> OPFLOWResult:
    """Helper to create an OPFLOWResult with sensible defaults."""
    defaults = dict(
        converged=True,
        objective_value=27557.57,
        convergence_status="CONVERGED",
        solver="IPOPT",
        model="POWER_BALANCE_POLAR",
        objective_type="MIN_GEN_COST",
        num_iterations=23,
        solve_time=0.042,
        total_gen_mw=520.0,
        total_load_mw=490.0,
        total_gen_mvar=100.0,
        total_load_mvar=80.0,
        voltage_min=1.062,
        voltage_max=1.100,
        voltage_mean=1.080,
        max_line_loading_pct=52.3,
        num_violations=0,
    )
    defaults.update(kw)
    return OPFLOWResult(**defaults)


# ===========================================================================
# Basic add/access
# ===========================================================================

class TestBasicOperations:

    def test_empty_journal(self):
        j = SearchJournal()
        assert len(j) == 0
        assert j.latest is None
        assert j.entries == []

    def test_add_entry(self):
        j = SearchJournal()
        e1 = _make_entry(1)
        e2 = _make_entry(2, obj=900.0)
        j.add_entry(e1)
        j.add_entry(e2)
        assert len(j) == 2
        assert j.latest is e2
        assert j.entries[0] is e1

    def test_entries_returns_copy(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1))
        entries = j.entries
        entries.clear()
        assert len(j) == 1


# ===========================================================================
# add_from_results
# ===========================================================================

class TestAddFromResults:

    def test_with_opflow_result(self):
        j = SearchJournal()
        opf = _make_opflow_result(objective_value=28000.0, voltage_min=1.05)
        entry = j.add_from_results(
            iteration=1,
            description="Base case",
            commands=[],
            opflow_result=opf,
            sim_elapsed=1.2,
            llm_reasoning="No changes",
            mode="fresh",
        )
        assert len(j) == 1
        assert entry.feasible is True
        assert entry.objective_value == 28000.0
        assert entry.voltage_min == 1.05
        assert entry.convergence_status == "CONVERGED"
        assert entry.mode == "fresh"

    def test_with_none_result(self):
        j = SearchJournal()
        entry = j.add_from_results(
            iteration=2,
            description="Failed sim",
            commands=[{"command": "scale_all_loads", "scale_factor": 5.0}],
            opflow_result=None,
            sim_elapsed=0.1,
            llm_reasoning="Tried aggressive scaling",
            mode="accumulative",
        )
        assert entry.feasible is False
        assert entry.objective_value is None
        assert entry.convergence_status == "FAILED"
        assert entry.voltage_min == 0.0
        assert entry.voltage_max == 0.0


# ===========================================================================
# format_for_prompt
# ===========================================================================

class TestFormatForPrompt:

    def test_empty(self):
        j = SearchJournal()
        text = j.format_for_prompt()
        assert "no iterations" in text

    def test_basic_format(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=27557.57))
        j.add_entry(_make_entry(2, obj=30000.00))
        j.add_entry(_make_entry(3, feasible=False))

        text = j.format_for_prompt()
        assert "3 iterations" in text
        assert "27,557.57" in text
        assert "30,000.00" in text
        assert "N/A" in text
        assert "Yes" in text
        assert "No" in text

    def test_format_alignment(self):
        """All rows should have the same pipe positions."""
        j = SearchJournal()
        for i in range(1, 4):
            j.add_entry(_make_entry(i, obj=1000.0 * i))

        text = j.format_for_prompt()
        lines = text.strip().split("\n")
        # Data rows start after header (line 0), blank (line 1), column header (line 2), separator (line 3)
        data_lines = [l for l in lines[4:] if l.strip()]
        for line in data_lines:
            assert line.count("|") == 5

    def test_max_entries(self):
        j = SearchJournal()
        for i in range(1, 11):
            j.add_entry(_make_entry(i, obj=1000.0 * i))

        text = j.format_for_prompt(max_entries=5)
        # Should include entry 1
        assert "   1 |" in text
        # Should have ellipsis
        assert "..." in text
        # Should include entries 7-10
        assert "   7 |" in text
        assert "  10 |" in text
        # Should NOT include entry 3
        assert "   3 |" not in text

    def test_max_entries_larger_than_total(self):
        j = SearchJournal()
        for i in range(1, 4):
            j.add_entry(_make_entry(i))

        text = j.format_for_prompt(max_entries=10)
        assert "..." not in text


# ===========================================================================
# format_detailed
# ===========================================================================

class TestFormatDetailed:

    def test_empty(self):
        j = SearchJournal()
        assert "empty" in j.format_detailed()

    def test_detailed_content(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=27000.0))
        text = j.format_detailed()
        assert "Iteration 1" in text
        assert "27,000.00" in text
        assert "CONVERGED" in text
        assert "scale_all_loads" in text


# ===========================================================================
# Export
# ===========================================================================

class TestExportJSON:

    def test_export_json(self, tmp_path: Path):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=1000.0))
        j.add_entry(_make_entry(2, obj=2000.0))

        out = tmp_path / "journal.json"
        j.export_json(out)

        data = json.loads(out.read_text())
        assert len(data) == 2
        assert data[0]["iteration"] == 1
        assert data[0]["objective_value"] == 1000.0
        assert data[1]["iteration"] == 2
        assert isinstance(data[0]["commands"], list)

    def test_export_json_empty(self, tmp_path: Path):
        j = SearchJournal()
        out = tmp_path / "journal.json"
        j.export_json(out)
        data = json.loads(out.read_text())
        assert data == []


class TestExportCSV:

    def test_export_csv(self, tmp_path: Path):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=1000.0))
        j.add_entry(_make_entry(2, feasible=False))

        out = tmp_path / "journal.csv"
        j.export_csv(out)

        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["iteration"] == "1"
        assert rows[0]["feasible"] == "True"
        # commands column should be valid JSON
        cmds = json.loads(rows[0]["commands"])
        assert isinstance(cmds, list)

    def test_export_csv_empty(self, tmp_path: Path):
        j = SearchJournal()
        out = tmp_path / "journal.csv"
        j.export_csv(out)
        assert out.read_text() == ""


# ===========================================================================
# Summary statistics
# ===========================================================================

class TestSummaryStats:

    def test_empty(self):
        j = SearchJournal()
        stats = j.summary_stats()
        assert stats["total_iterations"] == 0
        assert stats["best_objective"] is None
        assert stats["best_iteration"] is None
        assert stats["feasible_count"] == 0

    def test_all_feasible(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=3000.0))
        j.add_entry(_make_entry(2, obj=2000.0))
        j.add_entry(_make_entry(3, obj=2500.0))

        stats = j.summary_stats()
        assert stats["total_iterations"] == 3
        assert stats["best_objective"] == 2000.0
        assert stats["best_iteration"] == 2
        assert stats["feasible_count"] == 3
        assert stats["infeasible_count"] == 0
        assert stats["objective_trend"] == [3000.0, 2000.0, 2500.0]
        assert len(stats["voltage_range_trend"]) == 3

    def test_mixed_feasibility(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=5000.0))
        j.add_entry(_make_entry(2, feasible=False))
        j.add_entry(_make_entry(3, obj=3000.0))

        stats = j.summary_stats()
        assert stats["best_objective"] == 3000.0
        assert stats["best_iteration"] == 3
        assert stats["feasible_count"] == 2
        assert stats["infeasible_count"] == 1

    def test_all_infeasible(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, feasible=False))
        j.add_entry(_make_entry(2, feasible=False))

        stats = j.summary_stats()
        assert stats["best_objective"] is None
        assert stats["best_iteration"] is None
        assert stats["feasible_count"] == 0
        assert stats["infeasible_count"] == 2

    def test_goal_type_included(self):
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=1000.0))

        stats = j.summary_stats(goal_type="feasibility_boundary")
        assert stats["goal_type"] == "feasibility_boundary"

        stats_none = j.summary_stats()
        assert stats_none["goal_type"] is None

    def test_best_iteration_override(self):
        """Override should select a specific iteration as best, not lowest cost."""
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=27000.0))  # base case, lowest cost
        j.add_entry(_make_entry(1, obj=30000.0))
        j.add_entry(_make_entry(2, obj=35000.0))
        j.add_entry(_make_entry(3, obj=43000.0))  # highest cost but best for boundary search

        # Default: lowest cost wins
        stats_default = j.summary_stats()
        assert stats_default["best_iteration"] == 0
        assert stats_default["best_objective"] == 27000.0

        # Override: iteration 3 is the best answer
        stats_override = j.summary_stats(best_iteration_override=3)
        assert stats_override["best_iteration"] == 3
        assert stats_override["best_objective"] == 43000.0

    def test_best_iteration_override_invalid(self):
        """Invalid override iteration falls back to default cost heuristic."""
        j = SearchJournal()
        j.add_entry(_make_entry(1, obj=5000.0))
        j.add_entry(_make_entry(2, obj=3000.0))

        stats = j.summary_stats(best_iteration_override=99)
        # Should fall back to default (lowest cost)
        assert stats["best_iteration"] == 2
        assert stats["best_objective"] == 3000.0

    def test_best_iteration_override_infeasible_entry(self):
        """Override can select an infeasible entry (for boundary searches)."""
        j = SearchJournal()
        j.add_entry(_make_entry(0, obj=27000.0))
        j.add_entry(_make_entry(1, obj=35000.0))
        j.add_entry(_make_entry(2, feasible=False))  # infeasible

        # Override selects the infeasible entry
        stats = j.summary_stats(best_iteration_override=2)
        assert stats["best_iteration"] == 2
        assert stats["best_objective"] is None  # infeasible has no objective
