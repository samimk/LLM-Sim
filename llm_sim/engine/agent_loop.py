"""Agent loop controller — central orchestrator for LLM-driven search."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from llm_sim.backends import create_backend
from llm_sim.backends.base import LLMBackend, LLMResponse
from llm_sim.config import AppConfig
from llm_sim.engine.commands import parse_command
from llm_sim.engine.executor import SimulationExecutor, SimulationResult
from llm_sim.engine.journal import JournalEntry, SearchJournal
from llm_sim.engine.modifier import apply_modifications
from llm_sim.engine.schema_description import command_schema_text
from llm_sim.parsers import (
    parse_matpower,
    network_summary,
    parse_simulation_result,
    results_summary,
)
from llm_sim.parsers.matpower_model import MATNetwork
from llm_sim.parsers.opflow_results import OPFLOWResult

logger = logging.getLogger("llm_sim.engine.agent_loop")

_MAX_CONSECUTIVE_PARSE_FAILURES = 3


@dataclass
class SearchSession:
    """Complete record of a search session."""

    goal: str
    application: str
    base_case_path: Path
    config: AppConfig
    journal: SearchJournal
    start_time: str
    end_time: Optional[str] = None
    termination_reason: str = ""
    final_findings: Optional[dict] = None


class AgentLoopController:
    """Drives the iterative LLM-driven search."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._backend: LLMBackend = create_backend(config.llm)
        self._executor = SimulationExecutor(config.exago, config.output)
        self._journal = SearchJournal()

        # State tracked across iterations
        self._base_network: Optional[MATNetwork] = None
        self._current_network: Optional[MATNetwork] = None
        self._latest_opflow: Optional[OPFLOWResult] = None
        self._latest_results_text: Optional[str] = None
        self._error_feedback: Optional[str] = None
        self._consecutive_parse_failures = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, base_case: Path, goal: str) -> SearchSession:
        """Execute the full search loop."""
        session = SearchSession(
            goal=goal,
            application=self._config.search.application,
            base_case_path=base_case,
            config=self._config,
            journal=self._journal,
            start_time=datetime.now().isoformat(),
        )

        # 1. Parse base case
        logger.info("Parsing base case: %s", base_case)
        self._base_network = parse_matpower(base_case)
        self._current_network = self._base_network
        net_summary_text = network_summary(self._base_network)

        # 2. Run base case simulation (iteration 0)
        print("[Iter 0] Running base case simulation...")
        sim_result = self._executor.run(
            self._base_network,
            self._config.search.application,
            iteration=0,
        )
        opflow = parse_simulation_result(sim_result)
        self._latest_opflow = opflow

        if opflow is not None:
            self._latest_results_text = results_summary(opflow)
            self._journal.add_from_results(
                iteration=0,
                description="Base case (no modifications)",
                commands=[],
                opflow_result=opflow,
                sim_elapsed=sim_result.elapsed_seconds,
                llm_reasoning="Baseline run",
                mode="fresh",
            )
            print(
                f"[Iter 0] Base case: {opflow.convergence_status}, "
                f"cost=${opflow.objective_value:,.2f}"
            )
        else:
            self._latest_results_text = None
            self._journal.add_from_results(
                iteration=0,
                description="Base case (no modifications)",
                commands=[],
                opflow_result=None,
                sim_elapsed=sim_result.elapsed_seconds,
                llm_reasoning="Baseline run",
                mode="fresh",
            )
            self._error_feedback = (
                f"Base case simulation failed: {sim_result.error_message or 'unknown error'}"
            )
            print(f"[Iter 0] Base case simulation FAILED: {sim_result.error_message}")

        # 3. Agent loop
        max_iter = self._config.search.max_iterations
        for iteration in range(1, max_iter + 1):
            action_type, should_continue = self._iteration(
                iteration, goal, net_summary_text
            )
            if not should_continue:
                if not session.termination_reason:
                    session.termination_reason = "completed"
                break
        else:
            session.termination_reason = "max_iterations"
            print(f"\nMax iterations ({max_iter}) reached.")

        if not session.termination_reason:
            session.termination_reason = "completed"

        session.end_time = datetime.now().isoformat()

        # 4. Finalize session
        self._finalize(session)
        return session

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def _iteration(
        self,
        iteration: int,
        goal: str,
        net_summary_text: str,
    ) -> tuple[str, bool]:
        """Execute one iteration. Returns (action_type, should_continue)."""
        print(
            f"\n[Iter {iteration}] Sending prompt to "
            f"{self._backend.name()} ({self._config.llm.model})..."
        )

        # Assemble and send prompt
        system_prompt, user_prompt = self._assemble_prompt(
            goal, net_summary_text, self._journal,
            self._latest_results_text, self._error_feedback,
        )
        self._error_feedback = None  # consumed

        response = self._backend.complete(system_prompt, user_prompt)
        logger.debug("LLM raw response: %s", response.raw_text[:500])

        # Parse JSON from response
        if response.json_data is None:
            self._consecutive_parse_failures += 1
            logger.warning(
                "Failed to parse JSON from LLM response (%d/%d): %s",
                self._consecutive_parse_failures,
                _MAX_CONSECUTIVE_PARSE_FAILURES,
                response.json_error,
            )
            print(f"[Iter {iteration}] Failed to parse LLM response as JSON")
            if self._consecutive_parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                print(f"[Iter {iteration}] Too many consecutive parse failures — aborting")
                return "error", False
            self._error_feedback = (
                "Failed to parse JSON from your response. "
                "Please respond with a valid JSON object."
            )
            return "error", True

        self._consecutive_parse_failures = 0
        data = response.json_data
        action = data.get("action", "").lower()

        # Dispatch action
        if action == "modify":
            return self._handle_modify(iteration, data)
        elif action == "complete":
            return self._handle_complete(iteration, data)
        elif action == "analyze":
            return self._handle_analyze(iteration, data)
        else:
            print(f"[Iter {iteration}] Unknown action: '{action}'")
            self._error_feedback = (
                f"Unknown action '{action}'. "
                f"Valid actions: modify, complete, analyze."
            )
            return "error", True

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_modify(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'modify' action from the LLM."""
        description = data.get("description", "No description")
        reasoning = data.get("reasoning", "")
        raw_commands = data.get("commands", [])
        mode = data.get("mode", self._config.search.default_mode)

        print(f'[Iter {iteration}] LLM action: modify — "{description}"')

        # Choose base network for modifications
        if mode == "fresh":
            base_net = self._base_network
        else:
            base_net = self._current_network

        # Parse and apply commands
        commands = []
        parse_errors = []
        for raw in raw_commands:
            try:
                commands.append(parse_command(raw))
            except ValueError as exc:
                logger.warning("Failed to parse command %s: %s", raw, exc)
                parse_errors.append(f"Invalid command {raw}: {exc}")

        if commands:
            modified_net, report = apply_modifications(base_net, commands)
            skipped_msgs = []
            for cmd, reasons in report.skipped:
                skipped_msgs.append(f"Skipped {cmd}: {'; '.join(reasons)}")
            applied_count = len(report.applied)
            skipped_count = len(report.skipped) + len(parse_errors)
        else:
            modified_net = base_net
            applied_count = 0
            skipped_count = len(parse_errors)
            skipped_msgs = []

        all_errors = parse_errors + skipped_msgs
        print(
            f"[Iter {iteration}] Applied {applied_count} command(s), "
            f"{skipped_count} skipped"
        )

        if all_errors:
            self._error_feedback = "Command errors:\n" + "\n".join(all_errors)

        # Run simulation
        print(f"[Iter {iteration}] Running {self._config.search.application} simulation...")
        sim_result = self._executor.run(
            modified_net,
            self._config.search.application,
            iteration=iteration,
        )

        # Parse results
        opflow = parse_simulation_result(sim_result)
        self._latest_opflow = opflow

        if opflow is not None:
            self._latest_results_text = results_summary(opflow)
            self._current_network = modified_net
            print(
                f"[Iter {iteration}] Simulation completed in "
                f"{sim_result.elapsed_seconds:.2f}s — "
                f"{opflow.convergence_status}, cost=${opflow.objective_value:,.2f}"
            )
        else:
            self._latest_results_text = None
            error_msg = sim_result.error_message or "unknown error"
            feedback = f"Simulation failed: {error_msg}"
            if self._error_feedback:
                self._error_feedback += "\n" + feedback
            else:
                self._error_feedback = feedback
            print(
                f"[Iter {iteration}] Simulation FAILED in "
                f"{sim_result.elapsed_seconds:.2f}s — {error_msg}"
            )

        # Update journal
        self._journal.add_from_results(
            iteration=iteration,
            description=description,
            commands=raw_commands,
            opflow_result=opflow,
            sim_elapsed=sim_result.elapsed_seconds,
            llm_reasoning=reasoning,
            mode=mode,
        )

        return "modify", True

    def _handle_complete(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle a 'complete' action from the LLM."""
        findings = data.get("findings", {})
        reasoning = data.get("reasoning", "")
        summary_text = findings.get("summary", reasoning)

        print(f"[Iter {iteration}] LLM action: complete")
        print(f'[Iter {iteration}] Search completed: "{summary_text}"')

        # Store findings for the session
        self._final_findings = findings
        return "complete", False

    def _handle_analyze(
        self, iteration: int, data: dict
    ) -> tuple[str, bool]:
        """Handle an 'analyze' action from the LLM."""
        query = data.get("query", "")
        reasoning = data.get("reasoning", "")

        print(f'[Iter {iteration}] LLM action: analyze — "{query}"')

        result_text = self._run_analysis_query(query)
        # Feed analysis results back as the latest results text
        self._latest_results_text = result_text
        logger.info("Analysis query result: %s", result_text[:300])

        return "analyze", True

    def _run_analysis_query(self, query: str) -> str:
        """Execute a basic analysis query against the latest OPFLOW results."""
        if self._latest_opflow is None:
            return "No simulation results available to analyze."

        opf = self._latest_opflow
        q = query.lower()

        # Buses with voltage below X
        m = re.search(r"voltage\s+below\s+([\d.]+)", q)
        if m:
            threshold = float(m.group(1))
            buses = [b for b in opf.buses if b.Vm < threshold]
            if not buses:
                return f"No buses with voltage below {threshold} pu."
            lines = [f"Buses with Vm < {threshold} pu:"]
            for b in sorted(buses, key=lambda b: b.Vm):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu")
            return "\n".join(lines)

        # Buses with voltage above X
        m = re.search(r"voltage\s+above\s+([\d.]+)", q)
        if m:
            threshold = float(m.group(1))
            buses = [b for b in opf.buses if b.Vm > threshold]
            if not buses:
                return f"No buses with voltage above {threshold} pu."
            lines = [f"Buses with Vm > {threshold} pu:"]
            for b in sorted(buses, key=lambda b: -b.Vm):
                lines.append(f"  Bus {b.bus_id}: Vm={b.Vm:.4f} pu")
            return "\n".join(lines)

        # Most loaded lines
        if "most loaded" in q or "loaded lines" in q or "line loading" in q:
            loaded = []
            for br in opf.branches:
                if br.Slim > 0:
                    pct = max(br.Sf, br.St) / br.Slim * 100
                    loaded.append((pct, br))
            loaded.sort(key=lambda x: -x[0])
            lines = ["Most loaded lines:"]
            for pct, br in loaded[:10]:
                flow = max(br.Sf, br.St)
                lines.append(
                    f"  {br.from_bus}->{br.to_bus}: {pct:.1f}% "
                    f"({flow:.2f}/{br.Slim:.2f} MVA)"
                )
            return "\n".join(lines)

        # Generators
        if "generator" in q:
            lines = ["Generators:"]
            for g in sorted(opf.generators, key=lambda g: -g.Pg):
                status = "ON" if g.status == 1 else "OFF"
                lines.append(
                    f"  Bus {g.bus}: {status} Pg={g.Pg:.2f} MW "
                    f"[{g.Pmin:.0f}-{g.Pmax:.0f}] fuel={g.fuel}"
                )
            return "\n".join(lines)

        return (
            "Query not understood. Available queries:\n"
            "  - 'buses with voltage below X'\n"
            "  - 'buses with voltage above X'\n"
            "  - 'most loaded lines'\n"
            "  - 'generators'"
        )

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _assemble_prompt(
        self,
        goal: str,
        network_summary_text: str,
        journal: SearchJournal,
        latest_results_text: Optional[str],
        error_feedback: Optional[str] = None,
    ) -> tuple[str, str]:
        """Assemble the system prompt and user prompt for the LLM."""

        # System prompt (static per session)
        system_parts = [
            "You are a power systems analysis agent. Your task is to iteratively "
            "modify a power grid network and run simulations to achieve a given goal.",
            "",
            "You have access to the following modification commands:",
            command_schema_text(),
            "",
            "Network information:",
            network_summary_text,
            "",
            "You must respond with a JSON object. Three possible actions:",
            "",
            '1. MODIFY the network:',
            '{',
            '  "action": "modify",',
            '  "reasoning": "Why I\'m making these changes...",',
            '  "mode": "fresh" or "accumulative",',
            '  "description": "Short description for the journal",',
            '  "commands": [{"action": "...", ...}]',
            '}',
            "",
            '2. COMPLETE the search (when goal is achieved or impossible):',
            '{',
            '  "action": "complete",',
            '  "reasoning": "Why I\'m stopping...",',
            '  "findings": {"summary": "...", "details": "..."}',
            '}',
            "",
            '3. ANALYZE results (request specific data):',
            '{',
            '  "action": "analyze",',
            '  "reasoning": "I need more information...",',
            '  "query": "Show all buses with voltage below 0.95 pu"',
            '}',
            "",
            "Rules:",
            "- Be systematic: try small changes first, then adjust based on results.",
            "- Explain your reasoning clearly in each response.",
            "- Respect physical bounds (voltage limits, generator capacities, etc.).",
            '- Use "fresh" mode to start from the base case, "accumulative" to build on last changes.',
            "- When the goal is achieved or you determine it is infeasible, use the complete action.",
        ]
        system_prompt = "\n".join(system_parts)

        # User prompt (refreshed each iteration)
        user_parts = [f"Goal: {goal}"]

        if len(journal) > 0:
            user_parts.append("")
            user_parts.append("=== Section C: Search Journal ===")
            user_parts.append(journal.format_for_prompt())

        if latest_results_text:
            user_parts.append("")
            user_parts.append("=== Section D: Latest Results ===")
            user_parts.append(latest_results_text)

        if error_feedback:
            user_parts.append("")
            user_parts.append("=== Error Feedback ===")
            user_parts.append(error_feedback)

        user_parts.append("")
        user_parts.append(
            "Based on the above, decide your next action. "
            "Respond with a single JSON object."
        )

        user_prompt = "\n".join(user_parts)
        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(self, session: SearchSession) -> None:
        """Print summary and save journal."""
        print()
        print("=" * 50)
        print("=== Search Session Summary ===")
        print("=" * 50)

        stats = self._journal.summary_stats()
        print(f"Iterations: {stats['total_iterations']}")
        if stats["best_objective"] is not None:
            print(
                f"Best objective: ${stats['best_objective']:,.2f} "
                f"(iteration {stats['best_iteration']})"
            )
        else:
            print("Best objective: N/A (no feasible solution)")
        print(
            f"Feasible: {stats['feasible_count']} / "
            f"Infeasible: {stats['infeasible_count']}"
        )
        print(f"Termination: {session.termination_reason}")

        # Print journal table
        print()
        print(self._journal.format_for_prompt())

        # Save journal if configured
        if self._config.output.save_journal:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fmt = self._config.output.journal_format
            journal_path = (
                self._config.output.workdir / f"journal_{timestamp}.{fmt}"
            )
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "csv":
                self._journal.export_csv(journal_path)
            else:
                self._journal.export_json(journal_path)
            print(f"\nJournal saved to: {journal_path}")

        # Print findings if complete
        findings = getattr(self, "_final_findings", None)
        if findings:
            session.final_findings = findings
            print(f"\nFindings: {findings.get('summary', '')}")
            details = findings.get("details", "")
            if details:
                print(f"Details: {details}")
