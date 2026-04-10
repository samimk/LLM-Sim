"""Shared post-search goal classification utilities.

Used by both the CLI path (AgentLoopController._finalize) and the GUI path
(SessionManager.get_summary_analysis) so the LLM prompt and JSON parsing logic
are defined in exactly one place.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("llm_sim.engine.goal_classifier")

_SYSTEM_PROMPT = (
    "You are an expert power systems analyst reviewing the results of an "
    "LLM-driven optimization search performed using ExaGO's OPFLOW application. "
    "Provide a structured analytical summary of the search."
)

_GOAL_TYPE_DEFS = (
    "Goal type definitions:\n"
    "- cost_minimization: User wants to minimize generation cost."
    " Best = lowest cost among feasible.\n"
    "- feasibility_boundary: User wants to find the limit of a parameter before infeasibility."
    " Best = feasible iteration closest to the boundary.\n"
    "- constraint_satisfaction: User wants to satisfy specific constraints."
    " Best = iteration that best satisfies them.\n"
    "- parameter_exploration: User is exploring what-if scenarios."
    " Best = most informative feasible iteration.\n"
)


def build_classification_prompts(
    goal: str,
    termination_reason: str,
    stats: dict[str, Any],
    journal_formatted: str,
    total_tokens: int,
) -> tuple[str, str]:
    """Build the (system_prompt, user_prompt) pair for post-search goal classification.

    The prompt requests both a structured analysis narrative *and* a JSON
    classification block, making it suitable for both CLI display and GUI
    reporting in a single LLM call.

    Args:
        goal: The original natural-language search goal.
        termination_reason: How the search ended (completed, max_iterations, etc.).
        stats: dict from ``SearchJournal.summary_stats()`` (before any override).
        journal_formatted: Output of ``SearchJournal.format_detailed()``.
        total_tokens: Total prompt + completion tokens used during the search.

    Returns:
        Tuple of (system_prompt, user_prompt) strings.
    """
    user_prompt = (
        f"Search goal: {goal}\n"
        f"Termination reason: {termination_reason}\n"
        f"Total iterations: {stats['total_iterations']}\n"
        f"Feasible: {stats['feasible_count']} / "
        f"Infeasible: {stats['infeasible_count']}\n"
        f"Lowest-cost feasible: {stats['best_objective']} "
        f"(iteration {stats['best_iteration']})\n"
        f"Tokens used: ~{total_tokens:,}\n"
        f"\n"
        f"=== Detailed Journal ===\n"
        f"{journal_formatted}\n"
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
        f'  "goal_type": "<one of: cost_minimization | feasibility_boundary'
        f' | constraint_satisfaction | parameter_exploration>",\n'
        f'  "best_iteration": <int — the iteration number that best answers'
        f" the user's goal>,\n"
        f'  "best_iteration_rationale": "<one sentence explaining why this'
        f' iteration is the best answer>"\n'
        f'}}\n'
        f"```\n"
        f"\n"
        f"{_GOAL_TYPE_DEFS}"
    )
    return _SYSTEM_PROMPT, user_prompt


def parse_goal_classification(
    text: str,
    valid_iteration_numbers: set[int],
) -> dict[str, Any] | None:
    """Extract and validate the goal classification JSON block from an LLM response.

    Tries a fenced ``\`\`\`json ... \`\`\`` block first, then falls back to a
    bare ``{"goal_type": ...}`` pattern.  Returns *None* if parsing or
    validation fails so callers can fall back to cost-heuristic defaults.

    Args:
        text: Raw LLM response text.
        valid_iteration_numbers: Set of iteration numbers present in the journal.
            Used to reject hallucinated iteration numbers.

    Returns:
        Dict with keys ``goal_type``, ``best_iteration``,
        ``best_iteration_rationale``, or *None*.
    """
    # Try ```json ... ``` fenced block first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        # Fall back to bare JSON object containing a "goal_type" key
        match = re.search(r'\{\s*"goal_type"\s*:.*?\}', text, re.DOTALL)

    if match:
        try:
            json_str = match.group(1) if match.lastindex else match.group(0)
            data = json.loads(json_str)

            goal_type = data.get("goal_type", "cost_minimization")
            best_iter = data.get("best_iteration")
            rationale = data.get("best_iteration_rationale", "")

            if best_iter not in valid_iteration_numbers:
                logger.warning(
                    "LLM returned invalid best_iteration=%s (valid: %s); "
                    "falling back to cost heuristic",
                    best_iter,
                    sorted(valid_iteration_numbers),
                )
                return None

            logger.info(
                "Goal classification: type=%s, best_iter=%s, rationale=%s",
                goal_type,
                best_iter,
                rationale,
            )
            return {
                "goal_type": goal_type,
                "best_iteration": best_iter,
                "best_iteration_rationale": rationale,
            }
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse goal classification JSON: %s", exc)

    logger.info("No goal classification found in LLM response; using cost heuristic")
    return None
