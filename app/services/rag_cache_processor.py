from __future__ import annotations

import asyncio
import logging
import time

from app.services.history_mapper import build_history_messages
from app.services.query_processor import build_search_query
from app.services.rag_chain import get_rag_answer_async
from app.services.rag_concurrency import rag_execution_slot
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
    set_cached_json_if_lock_owner_and_release,
)

logger = logging.getLogger(__name__)

_running_map: dict[str, asyncio.Future[dict]] = {}
_running_map_lock = asyncio.Lock()


async def _cleanup_running_entry(running_key: str, future: asyncio.Future[dict]) -> None:
    await _running_map_lock.acquire()
    try:
        current = _running_map.get(running_key)
        if current is future:
            _running_map.pop(running_key, None)
            logger.info("[InFlightCleanup] key=%s action=removed_running_entry", running_key)
    finally:
        _running_map_lock.release()


async def get_rag_answer_cached(question: str, history: list = []) -> dict:
    history_messages = build_history_messages(history)
    raw_query = (question or "").strip()
    has_history = bool(history)

    pipe_ver = get_pipe_version()

    # 1) 원문 해시 키 조회 (히스토리가 없는 경우에만 안전)
    raw_qhash = build_qhash(raw_query)
    raw_ans_key = build_answer_key(pipe_ver, raw_qhash)
    if not has_history:
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
            raw_cache_key = raw_ans_key if (not has_history and raw_ans_key != canonical_ans_key) else None
            atomic_saved = await set_cached_json_if_lock_owner_and_release(
                lock_key=lock_key,
                expected_token=lock_token,
                canonical_key=canonical_ans_key,
                value=result,
                ttl_sec=ttl_sec,
                raw_key=raw_cache_key,
            )

            if atomic_saved is True:
                logger.info("[CacheSetAtomicSuccess] raw_key=%s canonical_key=%s", raw_ans_key, canonical_ans_key)
            elif atomic_saved is False:
                logger.info("[CacheSetAtomicSkippedOwnerMismatch] key=%s", canonical_ans_key)
            else:
                logger.warning("[CacheSetAtomicError] key=%s", canonical_ans_key)

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

    raise TimeoutError(f"cache wait timeout key={canonical_ans_key}")



async def get_rag_answer_cached_singleflight_in_process(question: str, history: list = []) -> dict:
    # 기존 레디스 분산락 로직 + 프로세스 내부 비동기 락 방식 추가
    history_messages = build_history_messages(history)
    raw_query = (question or "").strip()
    has_history = bool(history)

    pipe_ver = get_pipe_version()

    # 1-1) 원문 해시 키 조회 (히스토리가 없는 경우에만 안전)
    raw_qhash = build_qhash(raw_query)
    raw_ans_key = build_answer_key(pipe_ver, raw_qhash)
    if not has_history:
        cached_raw = await get_cached_json(raw_ans_key)
        if cached_raw is not None:
            logger.info("[CacheHit] phase=raw key=%s", raw_ans_key)
            return cached_raw

    # 1-2) canonical(rewrite+normalize) 해시 키 조회
    canonical_query = await build_search_query(raw_query, history_messages)
    canonical_qhash = build_qhash(canonical_query)
    canonical_ans_key = build_answer_key(pipe_ver, canonical_qhash)

    cached_canonical = await get_cached_json(canonical_ans_key)
    if cached_canonical is not None:
        logger.info("[CacheHit] phase=canonical key=%s", canonical_ans_key)
        return cached_canonical

    # 2~4) 실행 중 목록 조회 + 없으면 최초 요청으로 등록
    running_key = canonical_ans_key
    await _running_map_lock.acquire()
    try:
        existing_future = _running_map.get(running_key)
        if existing_future is not None:
            future = existing_future
            is_owner = False
        else:
            future = asyncio.get_running_loop().create_future()
            _running_map[running_key] = future
            is_owner = True
    finally:
        _running_map_lock.release()

    if not is_owner:
        logger.info(
            "[InFlightJoin] key=%s action=await_existing_future(skip_lock_and_polling)",
            running_key,
        )
        return await future

    logger.info("[InFlightRegister] key=%s action=owner_registered_future", running_key)

    # 5) 실행 중 목록 등록 후 owner가 Redis lock 시도
    lock_key = build_lock_key(canonical_ans_key)
    lock_token = None
    try:
        logger.info("[LockAttempt] key=%s owner=true", canonical_ans_key)
        lock_token = await acquire_lock(lock_key, get_lock_ttl_sec())

        if lock_token:
            logger.info("[LockAcquired] key=%s", canonical_ans_key)

            # double-check
            cached_again = await get_cached_json(canonical_ans_key)
            if cached_again is not None:
                logger.info("[CacheHitAfterLock] key=%s", canonical_ans_key)
                if not future.done():
                    future.set_result(cached_again)
                return cached_again

            result = await get_rag_answer_async(raw_query, canonical_query, history_messages)
            
            # double-cache
            ttl_sec = get_answer_ttl_sec()
            raw_cache_key = raw_ans_key if (not has_history and raw_ans_key != canonical_ans_key) else None
            atomic_saved = await set_cached_json_if_lock_owner_and_release(
                lock_key=lock_key,
                expected_token=lock_token,
                canonical_key=canonical_ans_key,
                value=result,
                ttl_sec=ttl_sec,
                raw_key=raw_cache_key,
            )

            if atomic_saved is True:
                logger.info("[CacheSetAtomicSuccess] raw_key=%s canonical_key=%s", raw_ans_key, canonical_ans_key)
            elif atomic_saved is False:
                logger.info("[CacheSetAtomicSkippedOwnerMismatch] key=%s", canonical_ans_key)
            else:
                logger.warning("[CacheSetAtomicError] key=%s", canonical_ans_key)

            if not future.done():
                future.set_result(result)
            return result

        # 6) lock 획득 실패: 다른 프로세스/서버 처리 결과를 캐시에서 polling
        logger.info("[LockMiss] key=%s action=poll_cache_until_ready", canonical_ans_key)
        wait_sec = get_lock_wait_ms() / 1000.0
        poll_sec = max(get_lock_poll_ms(), 10) / 1000.0
        deadline = time.monotonic() + wait_sec

        while time.monotonic() < deadline:
            cached_wait = await get_cached_json(canonical_ans_key)
            if cached_wait is not None:
                logger.info("[CacheHitAfterWait] key=%s", canonical_ans_key)
                if not future.done():
                    future.set_result(cached_wait)
                return cached_wait
            await asyncio.sleep(poll_sec)

        raise TimeoutError(f"cache wait timeout key={canonical_ans_key}")
    except BaseException as exc:
        # 9) owner 예외를 대기 중 요청에 전파하여 무한 대기 방지
        if not future.done():
            future.set_exception(exc)
        raise
    finally:
        if lock_token:
            await release_lock(lock_key, lock_token)

        # 8) cancellation이 발생해도 cleanup는 완료되도록 보호
        await asyncio.shield(_cleanup_running_entry(running_key, future))


async def get_rag_answer_cached_singleflight_in_process_with_semaphore(question: str, history: list = []) -> dict:
    # 기존 레디스 분산락 로직 + 프로세스 내부 비동기 락 방식 + RAG 세마포어 로직 추가

    history_messages = build_history_messages(history)
    raw_query = (question or "").strip()
    has_history = bool(history)

    pipe_ver = get_pipe_version()

    # 1-1) 원문 해시 키 조회 (히스토리가 없는 경우에만 안전)
    raw_qhash = build_qhash(raw_query)
    raw_ans_key = build_answer_key(pipe_ver, raw_qhash)
    if not has_history:
        cached_raw = await get_cached_json(raw_ans_key)
        if cached_raw is not None:
            logger.info("[CacheHit] phase=raw key=%s", raw_ans_key)
            return cached_raw

    # 1-2) canonical(rewrite+normalize) 해시 키 조회
    canonical_query = await build_search_query(raw_query, history_messages)
    canonical_qhash = build_qhash(canonical_query)
    canonical_ans_key = build_answer_key(pipe_ver, canonical_qhash)

    cached_canonical = await get_cached_json(canonical_ans_key)
    if cached_canonical is not None:
        logger.info("[CacheHit] phase=canonical key=%s", canonical_ans_key)
        return cached_canonical

    # 2~4) 실행 중 목록 조회 + 없으면 최초 요청으로 등록
    running_key = canonical_ans_key
    await _running_map_lock.acquire()
    try:
        existing_future = _running_map.get(running_key)
        if existing_future is not None:
            future = existing_future
            is_owner = False
        else:
            future = asyncio.get_running_loop().create_future()
            _running_map[running_key] = future
            is_owner = True
    finally:
        _running_map_lock.release()

    if not is_owner:
        logger.info(
            "[InFlightJoin] key=%s action=await_existing_future(skip_lock_and_polling)",
            running_key,
        )
        return await future

    logger.info("[InFlightRegister] key=%s action=owner_registered_future", running_key)

    # 5) 실행 중 목록 등록 후 owner가 Redis lock 시도
    lock_key = build_lock_key(canonical_ans_key)
    lock_token = None
    try:
        logger.info("[LockAttempt] key=%s owner=true", canonical_ans_key)
        lock_token = await acquire_lock(lock_key, get_lock_ttl_sec())

        if lock_token:
            logger.info("[LockAcquired] key=%s", canonical_ans_key)

            # double-check
            cached_again = await get_cached_json(canonical_ans_key)
            if cached_again is not None:
                logger.info("[CacheHitAfterLock] key=%s", canonical_ans_key)
                if not future.done():
                    future.set_result(cached_again)
                return cached_again

            async with rag_execution_slot():
                result = await get_rag_answer_async(raw_query, canonical_query, history_messages)
            
            # double-cache
            ttl_sec = get_answer_ttl_sec()
            raw_cache_key = raw_ans_key if (not has_history and raw_ans_key != canonical_ans_key) else None
            atomic_saved = await set_cached_json_if_lock_owner_and_release(
                lock_key=lock_key,
                expected_token=lock_token,
                canonical_key=canonical_ans_key,
                value=result,
                ttl_sec=ttl_sec,
                raw_key=raw_cache_key,
            )

            if atomic_saved is True:
                logger.info("[CacheSetAtomicSuccess] raw_key=%s canonical_key=%s", raw_ans_key, canonical_ans_key)
            elif atomic_saved is False:
                logger.info("[CacheSetAtomicSkippedOwnerMismatch] key=%s", canonical_ans_key)
            else:
                logger.warning("[CacheSetAtomicError] key=%s", canonical_ans_key)

            if not future.done():
                future.set_result(result)
            return result

        # 6) lock 획득 실패: 다른 프로세스/서버 처리 결과를 캐시에서 polling
        logger.info("[LockMiss] key=%s action=poll_cache_until_ready", canonical_ans_key)
        wait_sec = get_lock_wait_ms() / 1000.0
        poll_sec = max(get_lock_poll_ms(), 10) / 1000.0
        deadline = time.monotonic() + wait_sec

        while time.monotonic() < deadline:
            cached_wait = await get_cached_json(canonical_ans_key)
            if cached_wait is not None:
                logger.info("[CacheHitAfterWait] key=%s", canonical_ans_key)
                if not future.done():
                    future.set_result(cached_wait)
                return cached_wait
            await asyncio.sleep(poll_sec)

        raise TimeoutError(f"cache wait timeout key={canonical_ans_key}")
    except BaseException as exc:
        # 9) owner 예외를 대기 중 요청에 전파하여 무한 대기 방지
        if not future.done():
            future.set_exception(exc)
        raise
    finally:
        if lock_token:
            await release_lock(lock_key, lock_token)

        # 8) cancellation이 발생해도 cleanup는 완료되도록 보호
        await asyncio.shield(_cleanup_running_entry(running_key, future))
