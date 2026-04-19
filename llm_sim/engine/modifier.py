"""Modification applicator — applies validated commands to a MATNetwork."""

from __future__ import annotations

import copy
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from llm_sim.engine.commands import (
    ModCommand,
    ScaleAllLoads,
    ScaleLoad,
    ScaleLoadProfile,
    SetAllBusVLimits,
    SetBranchRate,
    SetBranchStatus,
    SetBusVLimits,
    SetCostCoeffs,
    SetGenDispatch,
    SetGenStatus,
    SetGenVoltage,
    SetLoad,
)
from llm_sim.engine.validation import validate_command
from llm_sim.parsers.matpower_model import MATNetwork

logger = logging.getLogger("llm_sim.engine.modifier")


@dataclass
class ModificationReport:
    """Summary of all modifications applied in one iteration."""

    applied: list[tuple[ModCommand, str]] = field(default_factory=list)
    skipped: list[tuple[ModCommand, list[str]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    profile_paths: dict[str, Path] = field(default_factory=dict)


def _find_bus(net: MATNetwork, bus_id: int):
    for b in net.buses:
        if b.bus_i == bus_id:
            return b
    return None


def _find_gens_at_bus(net: MATNetwork, bus_id: int):
    return [g for g in net.generators if g.bus == bus_id]


def _find_branches(net: MATNetwork, fbus: int, tbus: int):
    return [
        br for br in net.branches
        if (br.fbus == fbus and br.tbus == tbus) or (br.fbus == tbus and br.tbus == fbus)
    ]


def _gen_index_in_network(net: MATNetwork, bus_id: int, gen_id: int | None) -> int:
    """Return the index into net.generators for the specified gen at bus."""
    gens_at_bus = []
    for i, g in enumerate(net.generators):
        if g.bus == bus_id:
            gens_at_bus.append(i)
    idx = gen_id if gen_id is not None else 0
    return gens_at_bus[idx]


def _branch_index_in_network(net: MATNetwork, fbus: int, tbus: int, ckt: int | None) -> int:
    """Return the index into net.branches for the specified branch."""
    matching = []
    for i, br in enumerate(net.branches):
        if (br.fbus == fbus and br.tbus == tbus) or (br.fbus == tbus and br.tbus == fbus):
            matching.append(i)
    idx = ckt if ckt is not None else 0
    return matching[idx]


def _apply_one(cmd: ModCommand, net: MATNetwork) -> str:
    """Apply a single command to *net* (mutating) and return a description."""

    if isinstance(cmd, SetLoad):
        bus = _find_bus(net, cmd.bus)
        parts = []
        if cmd.Pd is not None:
            bus.Pd = cmd.Pd
            parts.append(f"Pd={cmd.Pd} MW")
        if cmd.Qd is not None:
            bus.Qd = cmd.Qd
            parts.append(f"Qd={cmd.Qd} MVAr")
        return f"Set load at bus {cmd.bus}: {', '.join(parts)}"

    if isinstance(cmd, ScaleLoad):
        count = 0
        for bus in net.buses:
            match = False
            if cmd.bus is not None and bus.bus_i == cmd.bus:
                match = True
            elif cmd.area is not None and bus.area == cmd.area:
                match = True
            elif cmd.zone is not None and bus.zone == cmd.zone:
                match = True
            if match:
                bus.Pd *= cmd.factor
                bus.Qd *= cmd.factor
                count += 1
        scope = f"bus {cmd.bus}" if cmd.bus else f"area {cmd.area}" if cmd.area else f"zone {cmd.zone}"
        return f"Scaled load by factor {cmd.factor} at {scope} ({count} bus(es))"

    if isinstance(cmd, ScaleAllLoads):
        for bus in net.buses:
            bus.Pd *= cmd.factor
            bus.Qd *= cmd.factor
        return f"Scaled all loads by factor {cmd.factor}"

    if isinstance(cmd, SetGenStatus):
        gi = _gen_index_in_network(net, cmd.bus, cmd.gen_id)
        net.generators[gi].status = cmd.status
        status_str = "ON" if cmd.status == 1 else "OFF"
        return f"Set generator at bus {cmd.bus} status to {status_str}"

    if isinstance(cmd, SetGenDispatch):
        gi = _gen_index_in_network(net, cmd.bus, cmd.gen_id)
        net.generators[gi].Pg = cmd.Pg
        return f"Set generator at bus {cmd.bus} Pg={cmd.Pg} MW"

    if isinstance(cmd, SetGenVoltage):
        gi = _gen_index_in_network(net, cmd.bus, cmd.gen_id)
        net.generators[gi].Vg = cmd.Vg
        return f"Set generator at bus {cmd.bus} Vg={cmd.Vg} pu (initial guess only)"

    if isinstance(cmd, SetBranchStatus):
        bi = _branch_index_in_network(net, cmd.fbus, cmd.tbus, cmd.ckt)
        net.branches[bi].status = cmd.status
        status_str = "in-service" if cmd.status == 1 else "out-of-service"
        return f"Set branch {cmd.fbus}-{cmd.tbus} status to {status_str}"

    if isinstance(cmd, SetBranchRate):
        bi = _branch_index_in_network(net, cmd.fbus, cmd.tbus, cmd.ckt)
        net.branches[bi].rateA = cmd.rateA
        return f"Set branch {cmd.fbus}-{cmd.tbus} rateA={cmd.rateA} MVA"

    if isinstance(cmd, SetCostCoeffs):
        gi = _gen_index_in_network(net, cmd.bus, cmd.gen_id)
        # Find matching gencost (same index as generator)
        if gi < len(net.gencost):
            net.gencost[gi].coeffs = list(cmd.coeffs)
            net.gencost[gi].ncost = len(cmd.coeffs)
        return f"Set cost coefficients for generator at bus {cmd.bus}: {cmd.coeffs}"

    if isinstance(cmd, SetBusVLimits):
        bus = _find_bus(net, cmd.bus)
        parts = []
        if cmd.Vmin is not None:
            bus.Vmin = cmd.Vmin
            parts.append(f"Vmin={cmd.Vmin}")
        if cmd.Vmax is not None:
            bus.Vmax = cmd.Vmax
            parts.append(f"Vmax={cmd.Vmax}")
        return f"Set bus {cmd.bus} voltage limits: {', '.join(parts)}"

    if isinstance(cmd, SetAllBusVLimits):
        count = 0
        for bus in net.buses:
            if cmd.Vmin is not None:
                bus.Vmin = cmd.Vmin
            if cmd.Vmax is not None:
                bus.Vmax = cmd.Vmax
            count += 1
        parts = []
        if cmd.Vmin is not None:
            parts.append(f"Vmin={cmd.Vmin}")
        if cmd.Vmax is not None:
            parts.append(f"Vmax={cmd.Vmax}")
        return f"Set voltage limits on all {count} buses: {', '.join(parts)}"

    return f"Unknown command type: {type(cmd).__name__}"


_SET_GEN_VOLTAGE_OPF_WARNING = (
    "set_gen_voltage only sets the initial guess; OPFLOW will override it with "
    "the optimal voltage. Use set_bus_vlimits to enforce voltage constraints in OPF."
)

_DCOPFLOW_VOLTAGE_CMD_TYPES = (SetGenVoltage, SetBusVLimits, SetAllBusVLimits)

def _dcopflow_voltage_skip_warning(cmd: ModCommand) -> str:
    """Return the skip warning message for a voltage command in DCOPFLOW context."""
    action_name = type(cmd).__name__
    # Convert class name to action name for the message
    _name_map = {
        "SetGenVoltage": "set_gen_voltage",
        "SetBusVLimits": "set_bus_vlimits",
        "SetAllBusVLimits": "set_all_bus_vlimits",
    }
    action = _name_map.get(action_name, action_name)
    return (
        f"Command '{action}' skipped: has no effect in DCOPFLOW "
        "(DC approximation ignores voltage magnitude)."
    )


_SCALE_LOAD_PROFILE_NON_TCOPFLOW_WARNING = (
    "scale_load_profile skipped: only applies to TCOPFLOW. "
    "Use scale_all_loads or scale_load for other applications."
)


def scale_load_profile_csv(
    csv_path: Path,
    factor: float,
    output_dir: Path,
    suffix: str = "",
) -> Path:
    """Scale numeric values in a TCOPFLOW load profile CSV by a factor.

    The first column (Timestamp) is preserved. All other columns are
    multiplied by *factor*. The result is written to *output_dir* with
    the same filename.

    Args:
        csv_path: Path to the original profile CSV.
        factor: Scaling factor (e.g., 1.2 for +20%).
        output_dir: Directory to write the scaled CSV.
        suffix: Optional suffix appended to the filename (e.g., "_scaled").

    Returns:
        Path to the written scaled CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem + suffix
    out_path = output_dir / f"{stem}.csv"

    rows_out = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                rows_out.append(row)
                continue
            scaled = [row[0]]  # Timestamp column
            for val in row[1:]:
                try:
                    scaled.append(str(float(val) * factor))
                except ValueError:
                    scaled.append(val)
            rows_out.append(scaled)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    return out_path


def apply_modifications(
    net: MATNetwork,
    commands: list[ModCommand],
    application: str | None = None,
    pload_profile: Path | None = None,
    qload_profile: Path | None = None,
    profile_output_dir: Path | None = None,
) -> tuple[MATNetwork, ModificationReport]:
    """Apply a list of modification commands to a network.

    Creates a deep copy of the network before applying changes.
    Each command is validated first; invalid commands are skipped.

    Args:
        net: The network to modify (not mutated).
        commands: List of commands to apply.
        application: ExaGO application name (e.g. "opflow"). When provided and
            equal to "opflow", a warning is added for any SetGenVoltage commands
            because OPFLOW treats Vg as an initial guess, not a constraint.
        pload_profile: Path to active load profile CSV (required for ScaleLoadProfile).
        qload_profile: Path to reactive load profile CSV (required for ScaleLoadProfile).
        profile_output_dir: Directory for scaled profile output (required for ScaleLoadProfile).

    Returns:
        Tuple of (modified_network, report). The report.profile_paths dict
        contains "pload_profile" and "qload_profile" keys with paths to
        scaled profile files when ScaleLoadProfile was applied.
    """
    modified = copy.deepcopy(net)
    report = ModificationReport()

    for cmd in commands:
        # Handle ScaleLoadProfile separately — it modifies CSV files, not the network
        if isinstance(cmd, ScaleLoadProfile):
            if application != "tcopflow":
                report.warnings.append(_SCALE_LOAD_PROFILE_NON_TCOPFLOW_WARNING)
                logger.warning(_SCALE_LOAD_PROFILE_NON_TCOPFLOW_WARNING)
                report.skipped.append((cmd, ["Only applicable for TCOPFLOW"]))
                continue
            if pload_profile is None or qload_profile is None:
                err = "ScaleLoadProfile requires pload_profile and qload_profile paths"
                report.skipped.append((cmd, [err]))
                logger.warning(err)
                continue
            out_dir = profile_output_dir or pload_profile.parent
            effective_factor = cmd.factor
            # If profiles were already scaled, apply factor cumulatively
            if report.profile_paths.get("pload_profile"):
                base_p = report.profile_paths["pload_profile"]
                base_q = report.profile_paths["qload_profile"]
            else:
                base_p = pload_profile
                base_q = qload_profile
            scaled_p = scale_load_profile_csv(base_p, effective_factor, out_dir)
            scaled_q = scale_load_profile_csv(base_q, effective_factor, out_dir)
            report.profile_paths["pload_profile"] = scaled_p
            report.profile_paths["qload_profile"] = scaled_q
            desc = f"Scaled load profiles by factor {cmd.factor} (P: {scaled_p.name}, Q: {scaled_q.name})"
            report.applied.append((cmd, desc))
            logger.info("Applied: %s", desc)
            continue

        # Skip voltage commands entirely for DCOPFLOW (they have no effect)
        if application == "dcopflow" and isinstance(cmd, _DCOPFLOW_VOLTAGE_CMD_TYPES):
            warning = _dcopflow_voltage_skip_warning(cmd)
            report.warnings.append(warning)
            logger.warning(warning)
            continue

        result = validate_command(cmd, modified)
        report.warnings.extend(result.warnings)

        if not result.valid:
            report.skipped.append((cmd, result.errors))
            logger.warning("Skipped invalid command %s: %s", type(cmd).__name__, result.errors)
            continue

        # Emit OPF-specific warning for set_gen_voltage
        if isinstance(cmd, SetGenVoltage) and application == "opflow":
            report.warnings.append(_SET_GEN_VOLTAGE_OPF_WARNING)
            logger.warning(_SET_GEN_VOLTAGE_OPF_WARNING)

        desc = _apply_one(cmd, modified)
        report.applied.append((cmd, desc))
        logger.info("Applied: %s", desc)

    return modified, report
