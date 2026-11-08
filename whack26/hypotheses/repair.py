"""Feedback prompts for hypothesis repair attempts."""

from __future__ import annotations

from typing import Sequence

_MAX_FEEDBACK_CHARS = 5000


def _clip(text: str, limit: int = _MAX_FEEDBACK_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def static_feedback(errors: Sequence[str]) -> str:
    joined = "\n".join(f"- {error}" for error in errors) or "- unknown static error"
    return _clip(
        "Your previous module failed static validation. Fix every violation and "
        "return the complete corrected module:\n" + joined
    )


def execution_feedback(
    status: str,
    error: str | None,
    traceback_text: str | None = None,
) -> str:
    details = [f"status: {status}", f"error: {error or 'unknown'}"]
    if traceback_text:
        details.append("traceback (last lines):\n" + traceback_text[-4000:])
    return _clip(
        "Your previous module failed when executed by the runner. Fix the bug and "
        "return the complete corrected module. Keep the model simple.\n"
        + "\n".join(details)
    )


def contract_feedback(reason: str) -> str:
    return _clip(
        "Your previous module did not satisfy the runner contract: "
        + reason
        + "\n`applies(df)` must return a real bool without raising, and "
        "`run(df, seed=0)` must return a dict with `predictions` and `target`."
    )


def duplicate_feedback() -> str:
    return (
        "Your previous module was too similar to another hypothesis in this run. "
        "Use a materially different model, feature set, or derivation and return "
        "the complete corrected module."
    )


def simpler_instruction(family: str) -> str:
    return (
        "This is a replacement attempt after repeated failures. Use the simplest "
        f"possible valid {family} template: no optional columns, no complex "
        "transformations, explicit missing-value handling, and a plain numpy/scipy "
        "implementation. Return the complete module."
    )
