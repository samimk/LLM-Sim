"""YAML configuration loading, merging, and validation for LLM-Sim."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("llm_sim.config")

# ---------------------------------------------------------------------------
# Default values (mirrors configs/default_config.yaml)
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "exago": {
        "binary_dir": "./applications",
        "opflow_binary": None,
        "scopflow_binary": None,
        "tcopflow_binary": None,
        "sopflow_binary": None,
        "dcopflow_binary": None,
        "pflow_binary": None,
        "env_script": None,
        "timeout": 120,
    },
    "data": {
        "data_dir": "./data",
    },
    "llm": {
        "backend": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "openai_base_url": None,
        "ollama_host": "http://localhost:11434",
        "ollama_cloud_host": None,
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "search": {
        "max_iterations": 20,
        "default_mode": "accumulative",
        "base_case": None,
        "gic_file": None,
        "application": "opflow",
    },
    "output": {
        "workdir": "./workdir",
        "logs_dir": "./logs",
        "save_journal": True,
        "journal_format": "json",
        "save_modified_files": True,
        "verbose": False,
    },
}

# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExagoConfig:
    binary_dir: Path
    opflow_binary: Optional[Path]
    scopflow_binary: Optional[Path]
    tcopflow_binary: Optional[Path]
    sopflow_binary: Optional[Path]
    dcopflow_binary: Optional[Path]
    pflow_binary: Optional[Path]
    env_script: Optional[Path]
    timeout: int


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path


@dataclass(frozen=True)
class LLMConfig:
    backend: str
    model: str
    api_key_env: str
    openai_base_url: Optional[str]
    ollama_host: str
    ollama_cloud_host: Optional[str]
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class SearchConfig:
    max_iterations: int
    default_mode: str
    base_case: Optional[Path]
    gic_file: Optional[Path]
    application: str


@dataclass(frozen=True)
class OutputConfig:
    workdir: Path
    logs_dir: Path
    save_journal: bool
    journal_format: str
    save_modified_files: bool
    verbose: bool


@dataclass(frozen=True)
class AppConfig:
    """Top-level immutable configuration for LLM-Sim."""

    exago: ExagoConfig
    data: DataConfig
    llm: LLMConfig
    search: SearchConfig
    output: OutputConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Expand ${VAR} references in string values."""
    if not isinstance(value, str):
        return value
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), m.group(0)), value)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _resolve_path(value: Any, root: Path) -> Optional[Path]:
    """Resolve a path relative to *root*, or return None."""
    if value is None:
        return None
    p = Path(_expand_env_vars(str(value)))
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _build_section(cls: type, raw: dict, root: Path, path_fields: set[str]) -> Any:
    """Instantiate a frozen dataclass section, resolving paths."""
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        val = raw.get(f.name)
        val = _expand_env_vars(val)
        if f.name in path_fields:
            val = _resolve_path(val, root)
        kwargs[f.name] = val
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    path: Path | str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load configuration from a YAML file, merge with defaults and CLI overrides.

    Args:
        path: Path to the YAML config file. If *None*, only defaults are used.
        cli_overrides: Flat dict of CLI overrides. Keys use dot notation
            (e.g. ``"llm.backend"``).

    Returns:
        A frozen :class:`AppConfig` instance.
    """
    raw: dict[str, Any] = {}
    # Resolve relative paths against cwd (where the user invokes the tool),
    # not the config file's parent — the default YAML values like
    # "./applications" are written relative to the project root.
    config_root = Path.cwd()

    if path is not None:
        path = Path(path)
        if path.exists():
            with open(path) as fh:
                raw = yaml.safe_load(fh) or {}
            logger.debug("Loaded config from %s", path)
        else:
            logger.warning("Config file not found: %s — using defaults", path)

    # Merge: defaults <- file <- CLI overrides
    merged = _deep_merge(DEFAULTS, raw)

    if cli_overrides:
        for dotted_key, value in cli_overrides.items():
            parts = dotted_key.split(".")
            target = merged
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        logger.debug("Applied CLI overrides: %s", cli_overrides)

    # Build frozen dataclass sections
    exago_path_fields = {
        "binary_dir", "opflow_binary", "scopflow_binary", "tcopflow_binary",
        "sopflow_binary", "dcopflow_binary", "pflow_binary", "env_script",
    }
    exago = _build_section(ExagoConfig, merged["exago"], config_root, exago_path_fields)

    data = _build_section(DataConfig, merged["data"], config_root, {"data_dir"})

    llm = _build_section(LLMConfig, merged["llm"], config_root, set())

    search_path_fields = {"base_case", "gic_file"}
    search = _build_section(SearchConfig, merged["search"], config_root, search_path_fields)

    output = _build_section(OutputConfig, merged["output"], config_root, {"workdir", "logs_dir"})

    cfg = AppConfig(exago=exago, data=data, llm=llm, search=search, output=output)

    _validate(cfg)
    return cfg


def _validate(cfg: AppConfig) -> None:
    """Emit warnings for missing paths; does not raise."""
    if not cfg.exago.binary_dir.exists():
        logger.warning(
            "ExaGO binary_dir does not exist: %s (binaries may not be available)",
            cfg.exago.binary_dir,
        )
    if not cfg.data.data_dir.exists():
        logger.warning("Data directory does not exist: %s", cfg.data.data_dir)

    valid_backends = {"openai", "anthropic", "ollama", "ollama-cloud"}
    if cfg.llm.backend not in valid_backends:
        logger.warning("Unknown LLM backend '%s' (expected one of %s)", cfg.llm.backend, valid_backends)

    valid_apps = {"opflow", "scopflow", "tcopflow", "sopflow", "dcopflow", "pflow"}
    if cfg.search.application not in valid_apps:
        logger.warning("Unknown application '%s' (expected one of %s)", cfg.search.application, valid_apps)

    valid_modes = {"accumulative", "fresh"}
    if cfg.search.default_mode not in valid_modes:
        logger.warning("Unknown mode '%s' (expected one of %s)", cfg.search.default_mode, valid_modes)
