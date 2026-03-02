from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from app.core.config import settings


class RagOverloadedError(RuntimeError):
    """Raised when RAG admission control cannot acquire a slot in time."""


_rag_semaphore: asyncio.Semaphore | None = None


def _get_rag_semaphore() -> asyncio.Semaphore:
    global _rag_semaphore
    if _rag_semaphore is None:
        max_concurrency = max(1, int(getattr(settings, "RAG_MAX_CONCURRENCY", 16)))
        _rag_semaphore = asyncio.Semaphore(max_concurrency)
    return _rag_semaphore


def _get_wait_timeout_sec() -> float:
    return max(0.1, float(getattr(settings, "RAG_SEMAPHORE_WAIT_TIMEOUT_SEC", 1.0)))


def get_rag_retry_after_sec() -> int:
    return max(1, int(getattr(settings, "RAG_OVERLOAD_RETRY_AFTER_SEC", 1)))


@asynccontextmanager
async def rag_execution_slot():
    semaphore = _get_rag_semaphore()

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=_get_wait_timeout_sec())
    except asyncio.TimeoutError as exc:
        raise RagOverloadedError("RAG concurrency limit reached") from exc

    try:
        yield
    finally:
        semaphore.release()
