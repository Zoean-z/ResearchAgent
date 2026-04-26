"""Application service exceptions."""

from __future__ import annotations


class EntityNotFoundError(Exception):
    """Raised when a requested entity does not exist."""

    def __init__(self, entity_name: str, entity_id: str) -> None:
        self.entity_name = entity_name
        self.entity_id = entity_id
        super().__init__(f"{entity_name} not found: {entity_id}")


class InvalidTaskRunStateError(Exception):
    """Raised when a task run is not in the expected lifecycle state."""

    def __init__(self, run_id: str, expected_state: str, actual_state: str) -> None:
        self.run_id = run_id
        self.expected_state = expected_state
        self.actual_state = actual_state
        super().__init__(f"TaskRun {run_id} expected {expected_state}, got {actual_state}")


class InvalidIngestSourceError(Exception):
    """Raised when an ingest source cannot be read or parsed."""

    def __init__(self, source_type: str, source_value: str, reason: str) -> None:
        self.source_type = source_type
        self.source_value = source_value
        self.reason = reason
        super().__init__(f"Invalid {source_type} ingest source {source_value}: {reason}")
