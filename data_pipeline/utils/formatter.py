from __future__ import annotations


def build_document(content: str, metadata: dict) -> str:
    project = str(metadata.get("project", "")).strip()
    heading = str(metadata.get("heading", "")).strip()
    path = str(metadata.get("path", "")).strip()
    return (
        f"[project] {project}\n"
        f"[heading] {heading}\n"
        f"[path] {path}\n\n"
        f"{content.strip()}"
    )
    