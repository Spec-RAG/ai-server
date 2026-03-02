from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    """Raised when required input fields are missing or invalid."""


def _require_non_empty_str(value: Any, field_name: str, line_no: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"line {line_no}: missing/invalid '{field_name}'")
    return value.strip()


def validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise ValidationError("batch_size must be > 0")


def validate_embedding_input_row(
    row: dict[str, Any], line_no: int
) -> tuple[str, str, dict[str, Any]]:
    row_id = _require_non_empty_str(row.get("id"), "id", line_no)
    content = _require_non_empty_str(row.get("content"), "content", line_no)

    metadata = row.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValidationError(f"line {line_no}: missing/invalid 'metadata'")

    return row_id, content, metadata


def validate_index_row_min(
    row: dict[str, Any], line_no: int
) -> tuple[str, list[Any], dict[str, Any]]:
    row_id = _require_non_empty_str(row.get("id"), "id", line_no)

    values = row.get("values")
    if not isinstance(values, list) or not values:
        raise ValidationError(f"line {line_no}: missing/invalid 'values'")

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValidationError(f"line {line_no}: missing/invalid 'metadata'")

    _require_non_empty_str(metadata.get("source_url"), "metadata.source_url", line_no)

    return row_id, values, metadata


def validate_vector_dim(expected_dim: int | None, values: list[Any], line_no: int) -> int:
    current_dim = len(values)
    if expected_dim is None:
        return current_dim
    if expected_dim != current_dim:
        raise ValidationError(
            f"line {line_no}: vector dimension mismatch: expected {expected_dim}, got {current_dim}"
        )
    return expected_dim

