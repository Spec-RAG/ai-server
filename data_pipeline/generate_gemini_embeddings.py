from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.utils.formatter import build_document  # noqa: E402
from data_pipeline.utils.gemini_api import embed_text  # noqa: E402
from data_pipeline.utils.json_io import JsonIoError, append_jsonl, iter_jsonl, write_json  # noqa: E402
from data_pipeline.utils.validators import ValidationError, validate_embedding_input_row  # noqa: E402

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/spring_docs_processed.jsonl")
    ap.add_argument("--out-dir", default="data/embeddings/gemini-embedding-001")
    ap.add_argument("--model", default="models/gemini-embedding-001")
    ap.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("[error] GEMINI_API_KEY is missing (.env)")

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "embedded_records.jsonl"
    out_stats = out_dir / "stats.json"
    if out_jsonl.exists():
        out_jsonl.unlink()

    total = 0
    succeeded = 0
    failed = 0
    vector_dim = None
    failures: list[dict] = []

    try:
        for line_no, row in iter_jsonl(in_path):
            if args.max_rows and total >= args.max_rows:
                break
            total += 1

            try:
                row_id, content, metadata = validate_embedding_input_row(row, line_no)
            except ValidationError as exc:
                failed += 1
                failures.append({"line": line_no, "id": row.get("id"), "error": str(exc)})
                continue

            document = build_document(content, metadata)
            try:
                values = embed_text(api_key, args.model, document)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failures.append({"line": line_no, "id": row_id, "error": str(exc)})
                continue

            if vector_dim is None:
                vector_dim = len(values)
            elif vector_dim != len(values):
                failed += 1
                failures.append(
                    {
                        "line": line_no,
                        "id": row_id,
                        "error": f"vector dimension mismatch: expected {vector_dim}, got {len(values)}",
                    }
                )
                continue

            append_jsonl(
                out_jsonl,
                {
                    "id": row_id,
                    "values": values,
                    "metadata": {
                        "project": metadata.get("project"),
                        "heading": metadata.get("heading"),
                        "path": metadata.get("path"),
                        "source_url": metadata.get("source_url"),
                        "content_hash": metadata.get("content_hash"),
                        "content": content,
                    },
                },
            )
            succeeded += 1
    except JsonIoError as exc:
        raise SystemExit(f"[error] invalid input jsonl: {exc}") from exc

    stats = {
        "model": args.model,
        "input": str(in_path),
        "output": str(out_jsonl),
        "total_rows": total,
        "succeeded_rows": succeeded,
        "failed_rows": failed,
        "vector_dim": vector_dim,
        "failed_examples": failures[:100],
    }
    write_json(out_stats, stats)

   
if __name__ == "__main__":
    main()
