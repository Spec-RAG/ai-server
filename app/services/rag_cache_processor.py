from __future__ import annotations

import asyncio
import logging
import time

from app.services.history_mapper import build_history_messages
from app.services.query_processor import build_search_query
from app.services.rag_chain import get_rag_answer_async
from app.services.redis_service import (
    acquire_lock,
    build_answer_key,
    build_lock_key,
    build_qhash,
    get_answer_ttl_sec,
    get_cached_json,
    get_lock_poll_ms,
    get_lock_ttl_sec,
    get_lock_wait_ms,
    get_pipe_version,
    release_lock,
    set_cached_json,
)

logger = logging.getLogger(__name__)


async def get_rag_answer_cached(question: str, history: list = []) -> dict:
    history_messages = build_history_messages(history)
    raw_query = (question or "").strip()

    pipe_ver = get_pipe_version()

    # 1) 원문 해시 키 조회
    raw_qhash = build_qhash(raw_query)
    raw_ans_key = build_answer_key(pipe_ver, raw_qhash)
    cached_raw = await get_cached_json(raw_ans_key)
    if cached_raw is not None:
        logger.info("[CacheHit] phase=raw key=%s", raw_ans_key)
        return cached_raw

    # 2) canonical(rewrite+normalize) 해시 키 조회
    canonical_query = await build_search_query(raw_query, history_messages)
    canonical_qhash = build_qhash(canonical_query)
    canonical_ans_key = build_answer_key(pipe_ver, canonical_qhash)

    cached_canonical = await get_cached_json(canonical_ans_key)
    if cached_canonical is not None:
        logger.info("[CacheHit] phase=canonical key=%s", canonical_ans_key)
        return cached_canonical

    # 3) lock 기반 stampede 방지
    lock_key = build_lock_key(canonical_ans_key)
    lock_token = await acquire_lock(lock_key, get_lock_ttl_sec())

    if lock_token:
        try:
            # double-check
            cached_again = await get_cached_json(canonical_ans_key)
            if cached_again is not None:
                logger.info("[CacheHitAfterLock] key=%s", canonical_ans_key)
                return cached_again

            result = await get_rag_answer_async(raw_query, canonical_query, history_messages)

            # double-cache    
            ttl_sec = get_answer_ttl_sec()
            await set_cached_json(canonical_ans_key, result, ttl_sec)
            if raw_ans_key != canonical_ans_key:
                await set_cached_json(raw_ans_key, result, ttl_sec)

            logger.info("[CacheSet] raw_key=%s canonical_key=%s", raw_ans_key, canonical_ans_key)
            return result
        finally:
            await release_lock(lock_key, lock_token)

    # lock 실패 요청은 잠시 대기 후 재조회
    wait_sec = get_lock_wait_ms() / 1000.0
    poll_sec = max(get_lock_poll_ms(), 10) / 1000.0
    deadline = time.monotonic() + wait_sec

    while time.monotonic() < deadline:
        cached_wait = await get_cached_json(canonical_ans_key)
        if cached_wait is not None:
            logger.info("[CacheHitAfterWait] key=%s", canonical_ans_key)
            return cached_wait
        await asyncio.sleep(poll_sec)

    # 최종 fallback (fail-open)
    logger.info("[CacheBypass] key=%s", canonical_ans_key)
    return await get_rag_answer_async(raw_query, canonical_query, history_messages)
