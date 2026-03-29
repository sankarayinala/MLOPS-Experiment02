# api/server.py

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth import authenticate, create_access_token
from api.cache import (
    get_cached,
    set_cached,
    list_keys,
    get_ttl,
    delete_key,
    invalidate_user_cache,
)
from pipeline.prediction_pipeline import hybrid_recommendation


# ------------------------------------------------------------
# ✅ Create FastAPI App + Global Rate Limiter
# ------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Anime Recommendation API",
    description="Hybrid Anime Recommendation API with explanations, caching, and admin tools",
    version="3.0.0"
)

app.state.limiter = limiter


# ------------------------------------------------------------
# ✅ AUTH ROUTES
# ------------------------------------------------------------
@app.get("/auth/login", tags=["Auth"])
def login_info():
    return {"message": "POST /auth/login with username & password"}


@app.post("/auth/login", tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    username = form.username
    password = form.password

    # Demo credentials (you can extend this later)
    if username != "demo" or password != "demo":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"username": username})
    return {"access_token": token, "token_type": "bearer"}


# ------------------------------------------------------------
# ✅ ROOT ROUTE
# ------------------------------------------------------------
@app.get("/", tags=["System"])
def root():
    return {"message": "✅ Anime Recommendation API is running!"}


# ------------------------------------------------------------
# ✅ MAIN RECOMMENDATION ROUTE (with caching & explanations)
# ------------------------------------------------------------
@app.get("/recommend/{user_id}", tags=["Recommend"])
@limiter.limit("10/minute")
async def recommend(
    request: Request,
    user_id: int,
    user_weight: float = 0.5,
    content_weight: float = 0.5,
    top_k: int = 10,
    user=Depends(authenticate)
):

    # Build deterministic cache key
    cache_key = f"rec:{user_id}:{user_weight}:{content_weight}:{top_k}"

    cached = get_cached(cache_key)
    if cached:
        return cached

    # Call hybrid recommender (returns dict with recommendations & explanations)
    result = hybrid_recommendation(
        user_id=user_id,
        user_weight=user_weight,
        content_weight=content_weight,
        top_k=top_k
    )

    # Cache for faster responses
    set_cached(cache_key, result)

    return result


# ------------------------------------------------------------
# ✅ ADMIN API: List Redis Keys with TTL
# ------------------------------------------------------------
@app.get("/admin/cache/keys", tags=["Admin"])
def admin_list_keys(user=Depends(authenticate)):
    keys = list_keys("*")
    return [
        {"key": k, "ttl": get_ttl(k)}
        for k in keys
    ]


# ------------------------------------------------------------
# ✅ ADMIN API: Delete a Single Key
# ------------------------------------------------------------
@app.delete("/admin/cache/key/{key}", tags=["Admin"])
def admin_delete_key(key: str, user=Depends(authenticate)):
    deleted = delete_key(key)
    return {"deleted": deleted, "key": key}


# ------------------------------------------------------------
# ✅ ADMIN API: Invalidate all cache entries for a user
# ------------------------------------------------------------
@app.delete("/admin/cache/user/{user_id}", tags=["Admin"])
def admin_delete_user_cache(user_id: int, user=Depends(authenticate)):
    removed = invalidate_user_cache(user_id)
    return {
        "user_id": user_id,
        "removed_keys": removed
    }