from __future__ import annotations

from typing import Any


_ALLOWED_METADATA_FIELDS = (
    "project",
    "source_url",
    "heading",
    "content",
    "path",
    "content_hash",
    "title",
)


def _pick_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for field in _ALLOWED_METADATA_FIELDS:
        value = metadata.get(field)
        if value is None:
            continue

        if isinstance(value, str):
            if not value.strip():
                continue
            mapped[field] = value
            continue

        mapped[field] = value
    return mapped


def map_row_to_pinecone(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "id": row["id"],
        "values": row["values"],
        "metadata": _pick_metadata(metadata),
    }
