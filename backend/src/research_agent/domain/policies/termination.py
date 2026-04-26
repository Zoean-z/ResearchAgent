"""Task termination policy helpers."""

from __future__ import annotations


def should_terminate_run(
    *,
    step_count: int,
    max_steps: int,
    has_final_answer: bool = False,
    ingest_completed: bool = False,
    unrecoverable_error: bool = False,
    no_recovery_path: bool = False,
) -> tuple[bool, str | None]:
    """Return whether the host runtime should finish a run and why."""

    if has_final_answer:
        return True, "final_answer_ready"
    if ingest_completed:
        return True, "ingest_completed"
    if unrecoverable_error:
        return True, "unrecoverable_error"
    if step_count >= max_steps:
        return True, "step_limit_reached"
    if no_recovery_path:
        return True, "insufficient_context_no_recovery_path"
    return False, None
