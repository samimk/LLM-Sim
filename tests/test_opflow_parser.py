"""Tests for the OPFLOW results parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_sim.parsers.opflow_parser import parse_opflow_output, parse_simulation_result
from llm_sim.parsers.opflow_results import OPFLOWResult
from llm_sim.parsers.results_summary import results_summary

SAMPLE = Path(__file__).resolve().parent / "sample_opflow_output.txt"
_has_sample = SAMPLE.exists()


@pytest.fixture(scope="module")
def sample_text() -> str:
    if not _has_sample:
        pytest.skip("sample_opflow_output.txt not found")
    return SAMPLE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result(sample_text: str) -> OPFLOWResult:
    return parse_opflow_output(sample_text)


# ===========================================================================
# Parse tests
# ===========================================================================

class TestParseOPFLOW:

    def test_converged(self, result: OPFLOWResult):
        assert result.converged is True

    def test_convergence_status(self, result: OPFLOWResult):
        assert result.convergence_status == "CONVERGED"

    def test_objective_value(self, result: OPFLOWResult):
        assert abs(result.objective_value - 27557.57) < 0.01

    def test_solver(self, result: OPFLOWResult):
        assert result.solver == "IPOPT"

    def test_model(self, result: OPFLOWResult):
        assert result.model == "POWER_BALANCE_POLAR"

    def test_objective_type(self, result: OPFLOWResult):
        assert result.objective_type == "MIN_GEN_COST"

    def test_num_iterations(self, result: OPFLOWResult):
        assert result.num_iterations == 23

    def test_solve_time(self, result: OPFLOWResult):
        assert result.solve_time == pytest.approx(0.042, abs=0.001)

    def test_bus_count(self, result: OPFLOWResult):
        assert len(result.buses) == 200

    def test_branch_count(self, result: OPFLOWResult):
        assert len(result.branches) == 245

    def test_gen_count(self, result: OPFLOWResult):
        assert len(result.generators) == 49


# ===========================================================================
# Specific value tests
# ===========================================================================

class TestSpecificValues:

    def test_bus_189_slack(self, result: OPFLOWResult):
        """Bus 189 is the slack bus — Va should be 0.0."""
        bus189 = next(b for b in result.buses if b.bus_id == 189)
        assert bus189.Va == 0.0

    def test_bus_100_highest_vm(self, result: OPFLOWResult):
        """Bus with highest Vm."""
        max_bus = max(result.buses, key=lambda b: b.Vm)
        # Verify voltage_max matches
        assert abs(max_bus.Vm - result.voltage_max) < 1e-6

    def test_branch_187_189_highest_sf(self, result: OPFLOWResult):
        """Branch 187->189 should have the highest Sf."""
        max_br = max(result.branches, key=lambda br: br.Sf)
        assert max_br.from_bus == 187
        assert max_br.to_bus == 189
        assert abs(max_br.Sf - 387.19) < 0.01

    def test_gen_189(self, result: OPFLOWResult):
        """Gen at bus 189: Pg≈383.40, fuel=COAL."""
        gen189 = next(g for g in result.generators if g.bus == 189)
        assert abs(gen189.Pg - 383.40) < 0.01
        assert gen189.fuel == "COAL"


# ===========================================================================
# Derived metrics tests
# ===========================================================================

class TestDerivedMetrics:

    def test_voltage_min(self, result: OPFLOWResult):
        assert abs(result.voltage_min - 1.062) < 0.001

    def test_voltage_max(self, result: OPFLOWResult):
        assert abs(result.voltage_max - 1.100) < 0.001

    def test_gen_exceeds_load(self, result: OPFLOWResult):
        assert result.total_gen_mw > result.total_load_mw

    def test_max_line_loading(self, result: OPFLOWResult):
        # Highest loading is ~71.3% (bus 146->147: 93.07/130.60)
        assert result.max_line_loading_pct > 70.0
        assert result.max_line_loading_pct < 75.0


# ===========================================================================
# Summary test
# ===========================================================================

class TestResultsSummary:

    def test_summary_content(self, result: OPFLOWResult):
        text = results_summary(result)
        assert "CONVERGED" in text
        assert "27,557.57" in text
        assert "IPOPT" in text
        assert "Most loaded lines:" in text
        assert "146->147" in text
        assert "Generators:" in text
        assert "Violations:" in text


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_string(self):
        with pytest.raises(ValueError, match="not.*OPFLOW"):
            parse_opflow_output("")

    def test_random_text(self):
        with pytest.raises(ValueError, match="not.*OPFLOW"):
            parse_opflow_output("This is just random text\nwith no OPFLOW output")

    def test_non_converged(self, sample_text: str):
        """Replace EXIT message to simulate non-convergence."""
        modified = sample_text.replace(
            "EXIT: Optimal Solution Found.",
            "EXIT: Maximum Number of Iterations Exceeded.",
        ).replace(
            "Convergence status                  CONVERGED",
            "Convergence status                  DIVERGED",
        )
        result = parse_opflow_output(modified)
        assert result.converged is False
        assert result.convergence_status == "DIVERGED"

    def test_parse_simulation_result_failed(self):
        """parse_simulation_result returns None for failed simulations."""

        class FakeResult:
            success = False
            stdout = ""

        assert parse_simulation_result(FakeResult()) is None

    def test_parse_simulation_result_success(self, sample_text: str):
        """parse_simulation_result works for successful simulations."""

        class FakeResult:
            success = True
            stdout = sample_text

        result = parse_simulation_result(FakeResult())
        assert result is not None
        assert result.converged is True
