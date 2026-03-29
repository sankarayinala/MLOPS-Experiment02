# api/cache.py

import json
import redis

cache = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

# ------------------------------
# ✅ Get cached value
# ------------------------------
def get_cached(key: str):
    try:
        v = cache.get(key)
        return json.loads(v) if v else None
    except Exception as e:
        print(f"[Cache Read Error] {e}")
        return None

# ------------------------------
# ✅ Set cached value
# ------------------------------
def set_cached(key: str, value, ttl: int = 300):
    try:
        cache.set(key, json.dumps(value, ensure_ascii=False), ex=ttl)
    except Exception as e:
        print(f"[Cache Write Error] {e}")

# ------------------------------
# ✅ List all redis keys
# ------------------------------
def list_keys(pattern="*"):
    try:
        return cache.keys(pattern)
    except Exception as e:
        print(f"[Cache Keys Error] {e}")
        return []

# ------------------------------
# ✅ Get TTL for a key
# ------------------------------
def get_ttl(key: str):
    try:
        return cache.ttl(key)
    except Exception as e:
        print(f"[Cache TTL Error] {e}")
        return None

# ------------------------------
# ✅ Delete a key
# ------------------------------
def delete_key(key: str):
    try:
        return cache.delete(key)
    except Exception as e:
        print(f"[Cache Delete Error] {e}")
        return 0

# ------------------------------
# ✅ Automatic per-user invalidation
# Example: invalidate all caches for user 11880
# ------------------------------
def invalidate_user_cache(user_id: int):
    pattern = f"rec:{user_id}:*"
    keys = list_keys(pattern)
    for k in keys:
        delete_key(k)
    return len(keys)