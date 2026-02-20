from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InputRow:
    row_index: int
    row_id: str
    content: str
    metadata: dict[str, Any]

