"""Configuration builder for the LLM-Sim Launcher.

Bridges Streamlit GUI widget values to the LLM-Sim configuration system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml


def get_project_root() -> Path:
    """Return the LLM-Sim project root directory.

    The project root is the parent of the launcher/ directory.
    This function works regardless of the current working directory.
    """
    return Path(__file__).resolve().parent.parent


def scan_data_files(data_dir: Path | None = None) -> list[Path]:
    """Find all MATPOWER .m files in the data directory.

    Args:
        data_dir: Path to data directory. Defaults to <project_root>/data.

    Returns:
        Sorted list of .m file paths (absolute).
    """
    if data_dir is None:
        data_dir = get_project_root() / "data"
    if not data_dir.is_dir():
        return []
    return sorted(data_dir.glob("*.m"))


def scan_config_files(configs_dir: Path | None = None) -> list[Path]:
    """Find all YAML config files in the configs directory.

    Args:
        configs_dir: Path to configs directory. Defaults to <project_root>/configs.

    Returns:
        Sorted list of .yaml file paths (absolute).
    """
    if configs_dir is None:
        configs_dir = get_project_root() / "configs"
    if not configs_dir.is_dir():
        return []
    # Include both .yaml and .yml, exclude templates
    files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))
    # Filter out template files
    files = [f for f in files if not f.name.endswith(".template")]
    return sorted(files)


def load_example_goals() -> list[dict[str, str]]:
    """Load preset search goals from assets/example_goals.yaml.

    Returns:
        List of dicts with 'label' and 'goal' keys.
        Returns empty list if file not found.
    """
    goals_path = Path(__file__).resolve().parent / "assets" / "example_goals.yaml"
    if not goals_path.exists():
        return []
    with open(goals_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("goals", [])


# Default model names per backend
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "ollama": "qwen2.5:7b",
    "ollama-cloud": "qwen2.5:7b",
}

# Available backends
BACKENDS: list[str] = ["anthropic", "openai", "ollama", "ollama-cloud"]

# Available applications (only opflow is fully supported for now)
APPLICATIONS: list[str] = ["opflow"]

# Future applications (shown as disabled in the UI)
FUTURE_APPLICATIONS: list[str] = ["scopflow", "tcopflow", "sopflow", "dcopflow", "pflow"]

# Available search modes
MODES: list[str] = ["accumulative", "fresh"]

# Available search mode types
SEARCH_MODES: list[str] = ["standard", "stress_test"]


def build_config_overrides(
    base_case: str | Path,
    backend: str,
    model: str,
    temperature: float,
    application: str,
    default_mode: str,
    max_iterations: int,
    gic_file: str | Path | None = None,
    ollama_host: str | None = None,
    ollama_cloud_host: str | None = None,
    openai_base_url: str | None = None,
    verbose: bool = False,
    search_mode: str = "standard",
) -> dict[str, Any]:
    """Build a CLI-style overrides dict from GUI widget values.

    The returned dict uses dot-notation keys compatible with
    ``llm_sim.config.load_config(cli_overrides=...)``.

    Args:
        base_case: Path to the .m base case file.
        backend: LLM backend name (anthropic, openai, ollama, ollama-cloud).
        model: Model name/identifier.
        temperature: LLM temperature (0.0-1.0).
        application: ExaGO application name (e.g., "opflow").
        default_mode: Search mode ("accumulative" or "fresh").
        max_iterations: Maximum search iterations.
        gic_file: Optional path to .gic file.
        ollama_host: Optional Ollama host URL.
        ollama_cloud_host: Optional Ollama cloud host URL.
        openai_base_url: Optional OpenAI base URL.
        verbose: Enable verbose output.

    Returns:
        Dict with dot-notation keys for load_config(cli_overrides=...).
    """
    overrides: dict[str, Any] = {
        "search.base_case": str(base_case),
        "search.application": application,
        "search.default_mode": default_mode,
        "search.max_iterations": max_iterations,
        "search.search_mode": search_mode,
        "llm.backend": backend,
        "llm.model": model,
        "llm.temperature": temperature,
        "output.verbose": verbose,
    }

    if gic_file:
        overrides["search.gic_file"] = str(gic_file)

    if ollama_host:
        overrides["llm.ollama_host"] = ollama_host

    if ollama_cloud_host:
        overrides["llm.ollama_cloud_host"] = ollama_cloud_host

    if openai_base_url:
        overrides["llm.openai_base_url"] = openai_base_url

    return overrides


def get_default_config_path() -> Path:
    """Return the default configuration file path.

    Returns:
        Absolute path to configs/default_config.yaml.
    """
    return get_project_root() / "configs" / "default_config.yaml"
