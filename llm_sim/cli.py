"""CLI entry point for LLM-Sim."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from llm_sim import __version__
from llm_sim.config import AppConfig, load_config
from llm_sim.logging_setup import setup_logging

logger = logging.getLogger("llm_sim.cli")


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llm-sim",
        description="LLM-Sim: LLM-driven iterative simulation for ExaGO",
    )
    parser.add_argument(
        "base_case",
        help="Path to MATPOWER .m base case file",
    )
    parser.add_argument(
        "goal",
        help="Search goal in natural language (quote the string)",
    )
    parser.add_argument(
        "--config",
        default="configs/default_config.yaml",
        help="Path to config YAML file (default: configs/default_config.yaml)",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "anthropic", "ollama", "ollama-cloud"],
        help="LLM backend: openai, anthropic, ollama, ollama-cloud",
    )
    parser.add_argument(
        "--model",
        help="Model name (overrides config)",
    )
    parser.add_argument(
        "--app",
        dest="application",
        choices=["opflow", "scopflow", "tcopflow", "sopflow", "dcopflow", "pflow"],
        help="ExaGO application: opflow, scopflow, tcopflow, sopflow, dcopflow, pflow",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        metavar="N",
        help="Maximum search iterations (default: 20)",
    )
    parser.add_argument(
        "--mode",
        choices=["accumulative", "fresh"],
        help="Iteration mode: accumulative, fresh",
    )
    parser.add_argument(
        "--gic",
        dest="gic_file",
        help="Path to .gic file (optional)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable detailed output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and validate without running (for testing setup)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI arguments into dot-notation config overrides."""
    overrides: dict[str, Any] = {}
    if args.backend is not None:
        overrides["llm.backend"] = args.backend
    if args.model is not None:
        overrides["llm.model"] = args.model
    if args.application is not None:
        overrides["search.application"] = args.application
    if args.max_iter is not None:
        overrides["search.max_iterations"] = args.max_iter
    if args.mode is not None:
        overrides["search.default_mode"] = args.mode
    if args.gic_file is not None:
        overrides["search.gic_file"] = args.gic_file
    if args.verbose is not None:
        overrides["output.verbose"] = args.verbose
    # base_case is set via positional arg
    overrides["search.base_case"] = args.base_case
    return overrides


def _print_config(cfg: AppConfig) -> None:
    """Pretty-print the resolved configuration."""
    def _section(name: str, obj: Any) -> None:
        print(f"\n  [{name}]")
        for f in fields(obj):
            print(f"    {f.name}: {getattr(obj, f.name)}")

    print("Resolved configuration:")
    _section("exago", cfg.exago)
    _section("data", cfg.data)
    _section("llm", cfg.llm)
    _section("search", cfg.search)
    _section("output", cfg.output)
    print()


def _print_banner(cfg: AppConfig, goal: str) -> None:
    """Print a startup banner with key settings."""
    print("=" * 60)
    print("  LLM-Sim — LLM-driven iterative simulation for ExaGO")
    print(f"  Version {__version__}")
    print("=" * 60)
    print(f"  Backend:        {cfg.llm.backend}")
    print(f"  Model:          {cfg.llm.model}")
    print(f"  Application:    {cfg.search.application}")
    print(f"  Base case:      {cfg.search.base_case}")
    print(f"  Goal:           {goal}")
    print(f"  Max iterations: {cfg.search.max_iterations}")
    print(f"  Mode:           {cfg.search.default_mode}")
    print("=" * 60)
    print()


def run_search(cfg: AppConfig, goal: str) -> None:
    """Run the LLM-driven search loop."""
    from llm_sim.engine.agent_loop import AgentLoopController

    controller = AgentLoopController(cfg)
    controller.run(cfg.search.base_case, goal)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = _cli_overrides(args)
    cfg = load_config(Path(args.config), cli_overrides=overrides)

    # Set up logging
    verbose = cfg.output.verbose
    setup_logging(cfg.output.logs_dir, verbose=verbose)

    logger.debug("CLI arguments: %s", args)

    # Validate base case
    if cfg.search.base_case and not cfg.search.base_case.exists():
        if args.dry_run:
            logger.warning("Base case file does not exist: %s (dry-run, continuing)", cfg.search.base_case)
        else:
            logger.error("Base case file does not exist: %s", cfg.search.base_case)
            sys.exit(1)

    if args.dry_run:
        _print_config(cfg)
        logger.info("Dry-run complete — exiting.")
        return

    _print_banner(cfg, args.goal)
    run_search(cfg, args.goal)


if __name__ == "__main__":
    main()
