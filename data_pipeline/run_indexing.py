from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.utils.json_io import write_json  # noqa: E402
from data_pipeline.utils.pinecone_client import build_pinecone_index, resolve_namespace  # noqa: E402
from data_pipeline.utils.upsert_runner import run_upsert  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upsert embedded JSONL into Pinecone")
    parser.add_argument(
        "--input",
        default="data/embeddings/gemini-embedding-001/embedded_records.jsonl",
        help="Path to embedded_records.jsonl",
    )
    parser.add_argument(
        "--report",
        default="data/embeddings/gemini-embedding-001/indexing_report.json",
        help="Path to indexing_report.json",
    )
    parser.add_argument("--index-name", default="", help="Pinecone index name override")
    parser.add_argument("--namespace", default="", help="Pinecone namespace override")
    parser.add_argument("--batch-size", type=int, default=100, help="Upsert batch size")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count per batch")
    parser.add_argument(
        "--backoff-sec",
        type=float,
        default=1.0,
        help="Initial backoff seconds (exponential)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    input_path = Path(args.input).resolve()
    report_path = Path(args.report).resolve()
    started_at = _utc_now_iso()

    if not input_path.exists():
        raise SystemExit(f"[error] input file not found: {input_path}")

    index, resolved_index_name = build_pinecone_index(args.index_name)
    namespace = resolve_namespace(args.namespace)

    result = run_upsert(
        input_path=input_path,
        index=index,
        namespace=namespace,
        batch_size=args.batch_size,
        max_retries=args.max_retries,
        backoff_sec=args.backoff_sec,
    )

    finished_at = _utc_now_iso()
    report = {
        "index_name": resolved_index_name,
        "namespace": namespace,
        "started_at": started_at,
        "finished_at": finished_at,
        "input_rows": result["input_rows"],
        "upserted_rows": result["upserted_rows"],
        "failed_rows": result["failed_rows"],
        "vector_dim": result["vector_dim"],
        "failed_examples": result["failed_examples"],
    }
    write_json(report_path, report)

    print(f"[index] {resolved_index_name}")
    print(f"[namespace] {namespace}")
    print(f"[input_rows] {report['input_rows']}")
    print(f"[upserted_rows] {report['upserted_rows']}")
    print(f"[failed_rows] {report['failed_rows']}")
    print(f"[report] {report_path}")

    if report["failed_rows"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
