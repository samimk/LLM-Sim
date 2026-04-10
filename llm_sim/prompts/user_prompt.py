"""User prompt template for each iteration."""

from __future__ import annotations

from typing import Optional


def build_user_prompt(
    goal: str,
    journal_text: Optional[str],
    results_text: Optional[str],
    error_feedback: Optional[str] = None,
    steering_directives: list[dict] | None = None,
) -> str:
    """Build the user prompt for one iteration.

    Args:
        goal: Natural language search goal.
        journal_text: Output of journal.format_for_prompt(), or None.
        results_text: Output of results_summary(), or None.
        error_feedback: Error messages from previous iteration, if any.
        steering_directives: List of active steering directive dicts
            with keys "directive" and "mode", or None.

    Returns:
        Complete user prompt string.
    """
    parts = [f"Goal: {goal}"]

    if journal_text:
        parts.append("")
        parts.append("=== Section C: Search Journal ===")
        parts.append(journal_text)

    if results_text:
        parts.append("")
        parts.append("=== Section D: Latest Results ===")
        parts.append(results_text)

    if error_feedback:
        parts.append("")
        parts.append("=== Error Feedback ===")
        parts.append(error_feedback)

    if steering_directives:
        parts.append("")
        parts.append("=== Operator Directives ===")
        parts.append(
            "The user has provided the following steering instructions during the search.\n"
            "Incorporate these into your decision-making:"
        )
        for i, sd in enumerate(steering_directives, start=1):
            tag = "AUGMENT" if sd.get("mode") == "augment" else "REPLACE"
            parts.append(f'{i}. [{tag}] "{sd["directive"]}"')
        parts.append("")
        parts.append(
            "For AUGMENT directives, consider them alongside the original goal.\n"
            "For REPLACE directives, they supersede the original goal."
        )

    parts.append("")
    parts.append(
        "Based on the above, decide your next action. "
        "Respond with a single JSON object."
    )

    return "\n".join(parts)
