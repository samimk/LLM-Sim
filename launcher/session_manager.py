"""Session manager �� bridges Streamlit UI with the LLM-Sim AgentLoopController.

Manages a background thread running the search loop, collects iteration
updates via callbacks into a thread-safe queue, and stores key OPFLOWResult
objects for visualization.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
from pathlib import Path
from typing import Optional

from llm_sim.backends import create_backend
from llm_sim.config import load_config
from llm_sim.engine.agent_loop import AgentLoopController, SearchSession
from llm_sim.engine.journal import JournalEntry
from llm_sim.parsers.opflow_results import OPFLOWResult

try:
    from config_builder import get_default_config_path
except ModuleNotFoundError:
    from launcher.config_builder import get_default_config_path

logger = logging.getLogger("launcher.session_manager")


class SessionManager:
    """Bridges Streamlit UI with the LLM-Sim AgentLoopController.

    Manages a background thread running the search loop, collects
    iteration updates via callbacks into a thread-safe queue, and
    stores key OPFLOWResult objects for visualization.
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._controller: Optional[AgentLoopController] = None
        self._session: Optional[SearchSession] = None
        self._update_queue: queue.Queue = queue.Queue()
        self._error: Optional[str] = None
        self._base_opflow: Optional[OPFLOWResult] = None
        self._best_opflow: Optional[OPFLOWResult] = None
        self._best_objective: Optional[float] = None
        self._opflow_by_iteration: dict[int, OPFLOWResult] = {}
        self._goal_classification: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_search(
        self,
        config_overrides: dict,
        goal: str,
        config_path: str | Path | None = None,
    ) -> None:
        """Build config, create controller with callbacks, launch in background thread.

        Args:
            config_overrides: Dict from config_builder.build_config_overrides().
            goal: Natural-language search goal.
            config_path: Path to base config YAML. Defaults to configs/default_config.yaml.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("A search is already running.")
        # Reset controller reference from any previous run
        self._controller = None

        # Reset state
        self._session = None
        self._error = None
        self._base_opflow = None
        self._best_opflow = None
        self._best_objective = None
        self._opflow_by_iteration = {}
        self._goal_classification = None
        # Drain the queue
        while not self._update_queue.empty():
            try:
                self._update_queue.get_nowait()
            except queue.Empty:
                break

        # Build config
        if config_path is None:
            config_path = get_default_config_path()
        cfg = load_config(path=config_path, cli_overrides=config_overrides)

        # Create controller with callbacks
        controller = AgentLoopController(
            config=cfg,
            quiet=True,
            on_iteration=self._on_iteration_callback,
            on_phase=self._on_phase_callback,
        )
        self._controller = controller

        base_case_path = cfg.search.base_case

        # Launch background thread
        self._thread = threading.Thread(
            target=self._run_search,
            args=(controller, base_case_path, goal),
            daemon=True,
        )
        self._thread.start()

    def stop_search(self) -> None:
        """Request graceful stop of the running search."""
        if self._controller is not None:
            self._controller.request_stop()

    def poll_updates(self) -> list[dict]:
        """Non-blocking drain of the update queue.

        Returns all pending updates. Called by the Streamlit main loop
        during periodic reruns.

        Returns:
            List of update dicts (may be empty).
        """
        updates = []
        while True:
            try:
                updates.append(self._update_queue.get_nowait())
            except queue.Empty:
                break
        return updates

    def is_running(self) -> bool:
        """Check if the search thread is still alive."""
        return self._thread is not None and self._thread.is_alive()

    def get_session(self) -> SearchSession | None:
        """Get the completed SearchSession after search ends."""
        return self._session

    def get_error(self) -> str | None:
        """Get error message if the search thread crashed."""
        return self._error

    def get_base_opflow(self) -> OPFLOWResult | None:
        """Get the base case (iteration 0) OPFLOW result."""
        return self._base_opflow

    def get_best_opflow(self) -> OPFLOWResult | None:
        """Get the best feasible OPFLOW result found during the search.

        If a goal classification has been computed, returns the OPFLOW result
        for the LLM-classified best iteration. Otherwise falls back to
        the lowest-cost feasible result.
        """
        if self._goal_classification is not None:
            best_iter = self._goal_classification.get("best_iteration")
            if best_iter is not None and best_iter in self._opflow_by_iteration:
                return self._opflow_by_iteration[best_iter]
        return self._best_opflow

    def get_goal_classification(self) -> dict | None:
        """Get the LLM-determined goal classification, or None if not yet computed."""
        return self._goal_classification

    def get_opflow_by_iteration(self, iteration: int) -> OPFLOWResult | None:
        """Get the OPFLOW result for a specific iteration."""
        return self._opflow_by_iteration.get(iteration)

    def get_summary_analysis(self, session: SearchSession) -> str:
        """Generate an analytical summary of the completed search via a final LLM call.

        Also requests a structured goal classification (goal_type, best_iteration)
        from the LLM, parsed and stored in self._goal_classification.

        Args:
            session: The completed search session.

        Returns:
            Summary analysis text from the LLM.
        """
        try:
            backend = create_backend(session.config.llm)

            system_prompt = (
                "You are an expert power systems analyst reviewing the results of an "
                "LLM-driven optimization search performed using ExaGO's OPFLOW application. "
                "Provide a structured analytical summary of the search."
            )

            stats = session.journal.summary_stats()
            total_tokens = session.total_prompt_tokens + session.total_completion_tokens

            user_prompt = (
                f"Search goal: {session.goal}\n"
                f"Termination reason: {session.termination_reason}\n"
                f"Total iterations: {stats['total_iterations']}\n"
                f"Feasible: {stats['feasible_count']} / "
                f"Infeasible: {stats['infeasible_count']}\n"
                f"Lowest-cost feasible: {stats['best_objective']} "
                f"(iteration {stats['best_iteration']})\n"
                f"Tokens used: ~{total_tokens:,}\n"
                f"\n"
                f"=== Detailed Journal ===\n"
                f"{session.journal.format_detailed()}\n"
                f"\n"
                f"Please provide:\n"
                f"\n"
                f"1. A structured analysis covering:\n"
                f"   a. Overall assessment — was the goal achieved?\n"
                f"   b. Search strategy analysis — what approach was taken?\n"
                f"   c. Convergence behavior — monotonic improvement, exploration, plateaus?\n"
                f"   d. Key modifications that had the most impact\n"
                f"   e. Potential further improvements\n"
                f"   f. Recommendations\n"
                f"\n"
                f"2. At the END of your response, include a JSON block wrapped in\n"
                f"   ```json ... ``` with the following structure:\n"
                f"\n"
                f"```json\n"
                f'{{\n'
                f'  "goal_type": "<one of: cost_minimization | feasibility_boundary | constraint_satisfaction | parameter_exploration>",\n'
                f'  "best_iteration": <int — the iteration number that best answers the user\'s goal>,\n'
                f'  "best_iteration_rationale": "<one sentence explaining why this iteration is the best answer>"\n'
                f'}}\n'
                f"```\n"
                f"\n"
                f"Goal type definitions:\n"
                f"- cost_minimization: User wants to minimize generation cost. Best = lowest cost among feasible.\n"
                f"- feasibility_boundary: User wants to find the limit of a parameter before infeasibility. Best = feasible iteration closest to the boundary.\n"
                f"- constraint_satisfaction: User wants to satisfy specific constraints. Best = iteration that best satisfies them.\n"
                f"- parameter_exploration: User is exploring what-if scenarios. Best = most informative feasible iteration.\n"
            )

            response = backend.complete(system_prompt, user_prompt)
            raw_text = response.raw_text

            # Parse goal classification JSON from the response
            self._goal_classification = self._parse_goal_classification(
                raw_text, session
            )

            return raw_text

        except Exception as exc:
            logger.warning("Failed to generate summary analysis: %s", exc)
            return f"Summary analysis could not be generated: {exc}"

    def _parse_goal_classification(
        self, text: str, session: SearchSession
    ) -> dict | None:
        """Extract the goal classification JSON block from the LLM response.

        Falls back to cost_minimization defaults if parsing fails.
        """
        # Try to find ```json ... ``` block
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            # Try bare JSON object with goal_type key
            match = re.search(
                r'\{\s*"goal_type"\s*:.*?\}', text, re.DOTALL
            )

        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                data = json.loads(json_str)

                goal_type = data.get("goal_type", "cost_minimization")
                best_iter = data.get("best_iteration")
                rationale = data.get("best_iteration_rationale", "")

                # Validate best_iteration is a valid iteration number
                valid_iters = {e.iteration for e in session.journal.entries}
                if best_iter not in valid_iters:
                    logger.warning(
                        "LLM returned invalid best_iteration=%s, "
                        "falling back to cost heuristic",
                        best_iter,
                    )
                    return None

                logger.info(
                    "Goal classification: type=%s, best_iter=%s, rationale=%s",
                    goal_type, best_iter, rationale,
                )
                return {
                    "goal_type": goal_type,
                    "best_iteration": best_iter,
                    "best_iteration_rationale": rationale,
                }
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("Failed to parse goal classification JSON: %s", exc)

        logger.info("No goal classification found in analysis response; using defaults")
        return None

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _run_search(
        self, controller: AgentLoopController, base_case: Path, goal: str
    ) -> None:
        """Execute the search loop in a background thread."""
        try:
            session = controller.run(base_case, goal)
            self._session = session
        except Exception as exc:
            logger.exception("Search thread crashed: %s", exc)
            self._error = str(exc)
            self._update_queue.put({"type": "error", "message": str(exc)})
        finally:
            self._update_queue.put({"type": "search_finished"})

    def _on_iteration_callback(
        self,
        iteration: int,
        entry: JournalEntry,
        status: str,
        opflow_result: OPFLOWResult | None,
    ) -> None:
        """Called by AgentLoopController after each iteration."""
        # Store OPFLOW result indexed by iteration (for later lookup)
        if opflow_result is not None:
            self._opflow_by_iteration[iteration] = opflow_result

        # Store base case OPFLOW result
        if iteration == 0 and opflow_result is not None:
            self._base_opflow = opflow_result

        # Track best feasible OPFLOW result (by lowest cost — default heuristic)
        if entry.feasible and entry.objective_value is not None:
            if self._best_objective is None or entry.objective_value < self._best_objective:
                self._best_objective = entry.objective_value
                self._best_opflow = opflow_result

        # Enqueue update for the UI
        update = {
            "type": "iteration",
            "iteration": iteration,
            "timestamp": entry.timestamp,
            "description": entry.description,
            "status": status,
            "objective_value": entry.objective_value,
            "feasible": entry.feasible,
            "convergence_status": entry.convergence_status,
            "violations_count": entry.violations_count,
            "voltage_range": (entry.voltage_min, entry.voltage_max),
            "max_line_loading_pct": entry.max_line_loading_pct,
            "total_gen_mw": entry.total_gen_mw,
            "total_load_mw": entry.total_load_mw,
            "sim_elapsed": entry.elapsed_seconds,
            "llm_reasoning": entry.llm_reasoning,
            "commands_count": len(entry.commands),
            "commands": entry.commands,
            "mode": entry.mode,
        }
        self._update_queue.put(update)

    def _on_phase_callback(self, iteration: int, phase_name: str) -> None:
        """Called by AgentLoopController at phase transitions within an iteration."""
        self._update_queue.put({
            "type": "phase",
            "iteration": iteration,
            "phase": phase_name,
        })
