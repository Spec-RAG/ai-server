from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis_client: Redis | None = None


def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        redis_url = getattr(settings, "REDIS_URL", "redis://localhost:6379")
        _redis_client = Redis.from_url(redis_url, decode_responses=True)
    return _redis_client


def build_qhash(text: str) -> str:
    payload = (text or "").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_answer_key(pipe_ver: str, qhash: str) -> str:
    return f"ans:p{pipe_ver}:{qhash}"


def build_lock_key(answer_key: str) -> str:
    return f"lock:{answer_key}"


async def get_cached_json(key: str) -> dict[str, Any] | None:
    try:
        raw = await get_redis_client().get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        return None
    except (RedisError, json.JSONDecodeError, TypeError):
        logger.exception("[RedisGetError] key=%s", key)
        return None


async def set_cached_json(key: str, value: dict[str, Any], ttl_sec: int) -> None:
    try:
        payload = json.dumps(value, ensure_ascii=False)
        await get_redis_client().set(key, payload, ex=ttl_sec)
    except RedisError:
        logger.exception("[RedisSetError] key=%s", key)


async def set_cached_json_if_lock_owner_and_release(
        # 현재 lock token이 내 token 과 일치할때만, 데이터 캐시 저장 및 lock 키 삭제 원자적으로 수행
    *,
    lock_key: str,
    expected_token: str,
    canonical_key: str,
    value: dict[str, Any],
    ttl_sec: int,
    raw_key: str | None = None,
) -> bool | None:
    """
    Returns:
      True  -> lock owner matches; cache commit + lock release succeeded
      False -> lock owner mismatch; no-op
      None  -> Redis error
    """
    script = """
local current = redis.call('GET', KEYS[1])
if current ~= ARGV[1] then
  return 0
end

redis.call('SET', KEYS[2], ARGV[2], 'EX', tonumber(ARGV[3]))

if ARGV[4] == '1' then
  redis.call('SET', KEYS[3], ARGV[2], 'EX', tonumber(ARGV[3]))
end

redis.call('DEL', KEYS[1])
return 1
"""
    payload = json.dumps(value, ensure_ascii=False)
    write_raw = "1" if raw_key else "0"
    raw_target = raw_key or canonical_key

    try:
        result = await get_redis_client().eval(
            script,
            3,
            lock_key,
            canonical_key,
            raw_target,
            expected_token,
            payload,
            str(ttl_sec),
            write_raw,
        )
        return bool(int(result))
    except RedisError:
        logger.exception(
            "[RedisAtomicCacheCommitError] lock_key=%s canonical_key=%s",
            lock_key,
            canonical_key,
        )
        return None


async def acquire_lock(lock_key: str, lock_ttl_sec: int) -> str | None:
    token = str(uuid.uuid4())
    try:
        ok = await get_redis_client().set(lock_key, token, nx=True, ex=lock_ttl_sec)
        if ok:
            return token
        return None
    except RedisError:
        logger.exception("[RedisLockAcquireError] key=%s", lock_key)
        return None


async def release_lock(lock_key: str, token: str) -> None:
    script = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
else
  return 0
end
"""
    try:
        await get_redis_client().eval(script, 1, lock_key, token)
    except RedisError:
        logger.exception("[RedisLockReleaseError] key=%s", lock_key)


def get_pipe_version() -> str:
    return str(getattr(settings, "PIPELINE_VERSION", "1"))


def get_answer_ttl_sec() -> int:
    return int(getattr(settings, "ANSWER_CACHE_TTL_SEC", 3600))


def get_lock_ttl_sec() -> int:
    return int(getattr(settings, "CACHE_LOCK_TTL_SEC", 120))


def get_lock_wait_ms() -> int:
    return int(getattr(settings, "CACHE_LOCK_WAIT_MS", 60000))


def get_lock_poll_ms() -> int:
    return int(getattr(settings, "CACHE_LOCK_POLL_MS", 100))
