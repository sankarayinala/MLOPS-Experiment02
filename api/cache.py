# api/cache.py

import json
import os
import redis


def _redis_port() -> int:
    raw = os.getenv("REDIS_PORT", "6379")
    if raw.startswith("tcp://"):
        return int(raw.rsplit(":", 1)[1])
    return int(raw)


cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=_redis_port(),
    db=0,
    decode_responses=True,
)


def get_cached(key: str):
    v = cache.get(key)
    return json.loads(v) if v else None


def set_cached(key: str, value, ttl: int = 300):
    cache.set(key, json.dumps(value), ex=ttl)


def list_keys(pattern="*"):
    return cache.keys(pattern)


def get_ttl(key: str):
    return cache.ttl(key)


def delete_key(key: str):
    return cache.delete(key)


def invalidate_user_cache(user_id: int):
    keys = list_keys(f"rec:{user_id}:*")
    for k in keys:
        delete_key(k)
    return len(keys)