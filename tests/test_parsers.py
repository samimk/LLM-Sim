"""Tests for the MATPOWER parser, writer, and network summary."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from llm_sim.parsers import (
    Branch,
    Bus,
    GenCost,
    Generator,
    MATNetwork,
    network_summary,
    parse_matpower,
    write_matpower,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ACTIVSG200 = DATA_DIR / "case_ACTIVSg200.m"

_has_test_file = ACTIVSG200.exists()


# ===========================================================================
# Parse tests
# ===========================================================================

@pytest.mark.skipif(not _has_test_file, reason="case_ACTIVSg200.m not in data/")
class TestParseACTIVSg200:

    @pytest.fixture(scope="class")
    def net(self) -> MATNetwork:
        return parse_matpower(ACTIVSG200)

    def test_casename(self, net: MATNetwork):
        assert net.casename == "case_ACTIVSg200"

    def test_version(self, net: MATNetwork):
        assert net.version == "2"

    def test_basemva(self, net: MATNetwork):
        assert net.baseMVA == 100.0

    def test_bus_count(self, net: MATNetwork):
        assert len(net.buses) == 200

    def test_gen_count(self, net: MATNetwork):
        assert len(net.generators) == 49

    def test_branch_count(self, net: MATNetwork):
        assert len(net.branches) == 245

    def test_gencost_count(self, net: MATNetwork):
        assert len(net.gencost) == 49

    def test_slack_bus(self, net: MATNetwork):
        """Bus 189 should be the slack bus (type=3)."""
        bus189 = next(b for b in net.buses if b.bus_i == 189)
        assert bus189.type == 3

    def test_gen_at_bus189(self, net: MATNetwork):
        """Generator at bus 189 should have Pmax=569.15."""
        gen189 = next(g for g in net.generators if g.bus == 189)
        assert abs(gen189.Pmax - 569.15) < 0.01

    def test_bus_extended_fields(self, net: MATNetwork):
        """ACTIVSg200 has 17-column bus data (extended with lam_P etc.)."""
        bus1 = net.buses[0]
        assert bus1.lam_P is not None
        assert abs(bus1.lam_P - 6.87) < 0.01

    def test_gen_extra_columns(self, net: MATNetwork):
        """Generators should have extra columns (Pc1, Pc2, etc.)."""
        gen0 = net.generators[0]
        assert len(gen0.extra) > 0

    def test_branch_extra_columns(self, net: MATNetwork):
        """Branches should have extra columns (Pf, Qf, etc.)."""
        br0 = net.branches[0]
        assert len(br0.extra) > 0

    def test_extra_sections(self, net: MATNetwork):
        """Should have gentype, genfuel, bus_name as extra sections."""
        assert "gentype" in net.extra_sections
        assert "genfuel" in net.extra_sections
        assert "bus_name" in net.extra_sections


# ===========================================================================
# Round-trip test
# ===========================================================================

@pytest.mark.skipif(not _has_test_file, reason="case_ACTIVSg200.m not in data/")
class TestRoundTrip:

    def test_parse_write_parse(self):
        """Parse → write → parse must produce identical numerical data."""
        net1 = parse_matpower(ACTIVSG200)

        with tempfile.NamedTemporaryFile(suffix=".m", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            write_matpower(net1, tmp_path)
            net2 = parse_matpower(tmp_path)

            # Check counts
            assert len(net2.buses) == len(net1.buses)
            assert len(net2.generators) == len(net1.generators)
            assert len(net2.branches) == len(net1.branches)
            assert len(net2.gencost) == len(net1.gencost)

            # Check bus values
            for b1, b2 in zip(net1.buses, net2.buses):
                assert b1.bus_i == b2.bus_i
                assert abs(b1.Pd - b2.Pd) < 1e-6
                assert abs(b1.Qd - b2.Qd) < 1e-6
                assert abs(b1.Vm - b2.Vm) < 1e-6
                assert abs(b1.Va - b2.Va) < 1e-6

            # Check generator values
            for g1, g2 in zip(net1.generators, net2.generators):
                assert g1.bus == g2.bus
                assert abs(g1.Pg - g2.Pg) < 1e-6
                assert abs(g1.Pmax - g2.Pmax) < 1e-6
                assert abs(g1.Pmin - g2.Pmin) < 1e-6

            # Check branch values
            for br1, br2 in zip(net1.branches, net2.branches):
                assert br1.fbus == br2.fbus
                assert br1.tbus == br2.tbus
                assert abs(br1.r - br2.r) < 1e-6
                assert abs(br1.x - br2.x) < 1e-6

            # Check gencost
            for gc1, gc2 in zip(net1.gencost, net2.gencost):
                assert gc1.model == gc2.model
                assert gc1.ncost == gc2.ncost
                for c1, c2 in zip(gc1.coeffs, gc2.coeffs):
                    assert abs(c1 - c2) < 1e-6

        finally:
            tmp_path.unlink(missing_ok=True)


# ===========================================================================
# Summary test
# ===========================================================================

@pytest.mark.skipif(not _has_test_file, reason="case_ACTIVSg200.m not in data/")
class TestNetworkSummary:

    def test_summary_content(self):
        net = parse_matpower(ACTIVSG200)
        summary = network_summary(net)

        assert "case_ACTIVSg200" in summary
        assert "200" in summary  # bus count
        assert "49" in summary   # gen count
        assert "245" in summary  # branch count
        assert "Pd=" in summary
        assert "Area" in summary
        assert "Generators:" in summary


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_matpower(Path("/nonexistent/file.m"))

    def test_invalid_file(self, tmp_path: Path):
        bad_file = tmp_path / "bad.m"
        bad_file.write_text("this is not a matpower file\n")
        with pytest.raises(ValueError, match="Cannot find"):
            parse_matpower(bad_file)

    def test_minimal_case(self, tmp_path: Path):
        """A minimal valid MATPOWER file with only buses."""
        minimal = tmp_path / "minimal.m"
        minimal.write_text(
            "function mpc = minimal_case\n"
            "mpc.version = '2';\n"
            "mpc.baseMVA = 100;\n"
            "mpc.bus = [\n"
            "\t1\t3\t0\t0\t0\t0\t1\t1.0\t0\t115\t1\t1.1\t0.9;\n"
            "];\n"
            "mpc.gen = [\n"
            "];\n"
            "mpc.branch = [\n"
            "];\n"
        )
        net = parse_matpower(minimal)
        assert net.casename == "minimal_case"
        assert len(net.buses) == 1
        assert len(net.generators) == 0
        assert len(net.branches) == 0
