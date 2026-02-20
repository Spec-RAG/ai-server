from __future__ import annotations

import os
from typing import Any


class PineconeConfigError(RuntimeError):
    """Raised when Pinecone configuration is missing or invalid."""


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise PineconeConfigError(f"Missing required environment variable: {name}")
    return value


def resolve_namespace(namespace: str | None = None) -> str:
    if namespace and namespace.strip():
        return namespace.strip()

    env_namespace = os.getenv("PINECONE_NAMESPACE", "").strip()
    if env_namespace:
        return env_namespace

    return "default"


def build_pinecone_index(index_name: str | None = None) -> tuple[Any, str]:
    api_key = _require_env("PINECONE_API_KEY")
    resolved_index_name = index_name.strip() if index_name and index_name.strip() else _require_env("PINECONE_INDEX_NAME")

    try:
        from pinecone import Pinecone
    except Exception as exc:  # noqa: BLE001
        raise PineconeConfigError(
            "pinecone package is not installed. Add dependency: pinecone"
        ) from exc

    client = Pinecone(api_key=api_key)
    try:
        client.describe_index(resolved_index_name)
    except Exception as exc:  # noqa: BLE001
        raise PineconeConfigError(
            f"Pinecone index '{resolved_index_name}' is not accessible."
        ) from exc

    return client.Index(resolved_index_name), resolved_index_name

