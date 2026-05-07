"""Unit tests for recursive JSON-safe conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel

from research_agent.utils import to_json_safe


class SampleEnum(str, Enum):
    ALPHA = "alpha"
    BETA = "beta"


class NestedModel(BaseModel):
    value: int


@dataclass
class NestedData:
    when: datetime
    label: SampleEnum


def test_to_json_safe_converts_supported_types_recursively() -> None:
    when = datetime(2026, 4, 27, 15, 30, 0, tzinfo=timezone.utc)
    day = date(2026, 4, 27)
    identifier = uuid4()
    path = Path("C:/tmp/example.pdf")
    model = NestedModel(value=7)
    data = NestedData(when=when, label=SampleEnum.BETA)

    value = to_json_safe(
        {
            "datetime": when,
            "date": day,
            "enum": SampleEnum.ALPHA,
            "uuid": identifier,
            "path": path,
            "model": model,
            "dataclass": data,
            "nested": {
                "list": [when, SampleEnum.BETA],
                "tuple": (identifier, path),
                "set": {SampleEnum.ALPHA},
            },
        }
    )

    assert value["datetime"] == when.isoformat()
    assert value["date"] == day.isoformat()
    assert value["enum"] == "alpha"
    assert value["uuid"] == str(identifier)
    assert value["path"] == str(path)
    assert value["model"] == {"value": 7}
    assert value["dataclass"] == {"when": when.isoformat(), "label": "beta"}
    assert value["nested"]["list"] == [when.isoformat(), "beta"]
    assert value["nested"]["tuple"] == [str(identifier), str(path)]
    assert value["nested"]["set"] == ["alpha"]
