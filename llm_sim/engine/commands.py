"""Modification command dataclasses and JSON parser."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

logger = logging.getLogger("llm_sim.engine.commands")


@dataclass
class SetLoad:
    """Set active/reactive load at a specific bus."""

    bus: int
    Pd: Optional[float] = None
    Qd: Optional[float] = None


@dataclass
class ScaleLoad:
    """Scale load at a bus, area, or zone by a factor."""

    factor: float
    bus: Optional[int] = None
    area: Optional[int] = None
    zone: Optional[int] = None


@dataclass
class ScaleAllLoads:
    """Scale all loads in the network uniformly."""

    factor: float


@dataclass
class SetGenStatus:
    """Turn a generator on or off."""

    bus: int
    status: int
    gen_id: Optional[int] = None


@dataclass
class SetGenDispatch:
    """Set generator active power output."""

    bus: int
    Pg: float
    gen_id: Optional[int] = None


@dataclass
class SetGenVoltage:
    """Set generator voltage setpoint."""

    bus: int
    Vg: float
    gen_id: Optional[int] = None


@dataclass
class SetBranchStatus:
    """Enable or disable a branch."""

    fbus: int
    tbus: int
    status: int
    ckt: Optional[int] = None


@dataclass
class SetBranchRate:
    """Modify branch thermal rating."""

    fbus: int
    tbus: int
    rateA: float
    ckt: Optional[int] = None


@dataclass
class SetCostCoeffs:
    """Modify generator cost curve coefficients."""

    bus: int
    coeffs: list[float] = field(default_factory=list)
    gen_id: Optional[int] = None


@dataclass
class SetBusVLimits:
    """Set bus voltage limits."""

    bus: int
    Vmin: Optional[float] = None
    Vmax: Optional[float] = None


ModCommand = Union[
    SetLoad, ScaleLoad, ScaleAllLoads, SetGenStatus, SetGenDispatch,
    SetGenVoltage, SetBranchStatus, SetBranchRate, SetCostCoeffs, SetBusVLimits,
]

# Map action names to command classes and their required fields
_COMMAND_MAP: dict[str, tuple[type, set[str]]] = {
    "set_load": (SetLoad, {"bus"}),
    "scale_load": (ScaleLoad, {"factor"}),
    "scale_all_loads": (ScaleAllLoads, {"factor"}),
    "set_gen_status": (SetGenStatus, {"bus", "status"}),
    "set_gen_dispatch": (SetGenDispatch, {"bus", "Pg"}),
    "set_gen_voltage": (SetGenVoltage, {"bus", "Vg"}),
    "set_branch_status": (SetBranchStatus, {"fbus", "tbus", "status"}),
    "set_branch_rate": (SetBranchRate, {"fbus", "tbus", "rateA"}),
    "set_cost_coeffs": (SetCostCoeffs, {"bus", "coeffs"}),
    "set_bus_vlimits": (SetBusVLimits, {"bus"}),
}


def parse_command(raw: dict) -> ModCommand:
    """Parse a raw JSON dict into a typed ModCommand.

    The dict must have an ``"action"`` key matching one of the command
    names (e.g., ``"set_load"``, ``"scale_load"``).

    Raises:
        ValueError: If the action is unknown or required fields are missing.
    """
    action = raw.get("action")
    if action is None:
        raise ValueError("Command dict missing 'action' key")

    entry = _COMMAND_MAP.get(action)
    if entry is None:
        raise ValueError(f"Unknown action '{action}'. Valid: {sorted(_COMMAND_MAP)}")

    cls, required = entry
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Action '{action}' missing required fields: {missing}")

    # Build kwargs from raw, excluding 'action'
    kwargs = {k: v for k, v in raw.items() if k != "action"}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid fields for '{action}': {exc}") from exc
