from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from data_pipeline.utils.json_io import JsonIoError, iter_jsonl
from data_pipeline.utils.record_mapper import map_row_to_pinecone
from data_pipeline.utils.validators import (
    ValidationError,
    validate_batch_size,
    validate_index_row_min,
    validate_vector_dim,
)


def _build_failure_example(
    *, line_no: int, row_id: str | None, error: str
) -> dict[str, Any]:
    return {"line": line_no, "id": row_id, "error": error}


def _build_batch_failure_examples(
    *, batch: list[dict[str, Any]], error_message: str | None
) -> list[dict[str, Any]]:
    return [
        {"id": item["id"], "error": f"upsert_failed: {error_message}"}
        for item in batch
    ]


def _flush_batch(
    *,
    index: Any,
    namespace: str,
    batch: list[dict[str, Any]],
    max_retries: int,
    backoff_sec: float,
) -> tuple[bool, str | None]:
    for attempt in range(max_retries + 1):
        try:
            index.upsert(vectors=batch, namespace=namespace)
            return True, None
        except Exception as exc:  
            if attempt == max_retries:
                return False, str(exc)
            time.sleep(backoff_sec * (2**attempt))
    return False, "unknown upsert error"


def _process_batch(
    *,
    index: Any,
    namespace: str,
    batch: list[dict[str, Any]],
    max_retries: int,
    backoff_sec: float,
) -> tuple[int, int, list[dict[str, Any]]]:
    ok, error_message = _flush_batch(
        index=index,
        namespace=namespace,
        batch=batch,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
    )
    if ok:
        return len(batch), 0, []
    return 0, len(batch), _build_batch_failure_examples(
        batch=batch, error_message=error_message
    )


def run_upsert(
    *,
    input_path: Path,
    index: Any,
    namespace: str,
    batch_size: int = 100,
    max_retries: int = 3,
    backoff_sec: float = 1.0,
) -> dict[str, Any]:
    validate_batch_size(batch_size)

    input_rows = 0
    upserted_rows = 0
    failed_rows = 0
    vector_dim: int | None = None
    failed_examples: list[dict[str, Any]] = []

    batch: list[dict[str, Any]] = []

    try:
        for line_no, row in iter_jsonl(input_path):
            input_rows += 1
            try:
                row_id, values, metadata = validate_index_row_min(row, line_no)
                vector_dim = validate_vector_dim(vector_dim, values, line_no)
            except ValidationError as exc:
                failed_rows += 1
                raw_id = row.get("id")
                failed_examples.append(
                    _build_failure_example(
                        line_no=line_no,
                        row_id=raw_id if isinstance(raw_id, str) else None,
                        error=str(exc),
                    )
                )
                continue

            mapped = map_row_to_pinecone(
                {"id": row_id, "values": values, "metadata": metadata}
            )
            batch.append(mapped)
            if len(batch) < batch_size:
                continue

            batch_upserted, batch_failed, batch_failures = _process_batch(
                index=index,
                namespace=namespace,
                batch=batch,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
            )
            upserted_rows += batch_upserted
            failed_rows += batch_failed
            failed_examples.extend(batch_failures)
            batch = []

        if batch:
            batch_upserted, batch_failed, batch_failures = _process_batch(
                index=index,
                namespace=namespace,
                batch=batch,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
            )
            upserted_rows += batch_upserted
            failed_rows += batch_failed
            failed_examples.extend(batch_failures)
    except JsonIoError as exc:
        raise RuntimeError(f"JSONL read failed: {exc}") from exc

    return {
        "input_rows": input_rows,
        "upserted_rows": upserted_rows,
        "failed_rows": failed_rows,
        "vector_dim": vector_dim,
        "failed_examples": failed_examples[:100],
    }
