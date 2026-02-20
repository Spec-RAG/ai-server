from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class JsonIoError(ValueError):
    """Raised when JSON/JSONL cannot be parsed into an object."""


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonIoError(f"Invalid JSON at line {line_no}: {exc.msg}") from exc

            if not isinstance(payload, dict):
                raise JsonIoError(f"Line {line_no} is not a JSON object")

            yield line_no, payload


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any], indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)

