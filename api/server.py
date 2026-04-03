# api/server.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from api.auth import authenticate, create_access_token
from api.cache import (
    get_cached,
    set_cached,
    list_keys,
    get_ttl,
    delete_key,
    invalidate_user_cache,
)
from api.metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    INFLIGHT_REQUESTS,
)
from pipeline.prediction_pipeline import hybrid_recommendation, warmup_dataframes

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_dataframes()
    yield


app = FastAPI(
    title="Anime Recommendation API",
    description="Hybrid Anime Recommendation API with explanations, caching, and admin tools",
    version="3.2.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.get("/healthz", tags=["System"])
def healthz():
    return {"status": "ok"}


@app.get("/metrics", tags=["System"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["System"])
def root():
    return {"message": "✅ Anime Recommendation API is running!"}


@app.get("/auth/login", tags=["Auth"])
def login_info():
    return {"message": "POST /auth/login with username & password"}


@app.post("/auth/login", tags=["Auth"])
def login(form: OAuth2PasswordRequestForm = Depends()):
    username = form.username
    password = form.password

    if username != "demo" or password != "demo":
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"username": username})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/recommend/{user_id}", tags=["Recommend"])
@limiter.limit("10/minute")
async def recommend(
    request: Request,
    user_id: int,
    user_weight: float = 0.5,
    content_weight: float = 0.5,
    top_k: int = 10,
    user=Depends(authenticate),
):
    REQUEST_COUNT.inc()
    INFLIGHT_REQUESTS.inc()

    try:
        with REQUEST_LATENCY.time():
            cache_key = f"rec:{user_id}:{user_weight}:{content_weight}:{top_k}"

            cached = get_cached(cache_key)
            if cached:
                return cached

            result = hybrid_recommendation(
                user_id=user_id,
                user_weight=user_weight,
                content_weight=content_weight,
                top_k=top_k,
            )

            set_cached(cache_key, result)
            return result
    finally:
        INFLIGHT_REQUESTS.dec()


@app.get("/admin/cache/keys", tags=["Admin"])
def admin_list_keys(user=Depends(authenticate)):
    keys = list_keys("*")
    return [{"key": k, "ttl": get_ttl(k)} for k in keys]


@app.delete("/admin/cache/key/{key}", tags=["Admin"])
def admin_delete_key(key: str, user=Depends(authenticate)):
    deleted = delete_key(key)
    return {"deleted": deleted, "key": key}


@app.delete("/admin/cache/user/{user_id}", tags=["Admin"])
def admin_delete_user_cache(user_id: int, user=Depends(authenticate)):
    removed = invalidate_user_cache(user_id)
    return {"user_id": user_id, "removed_keys": removed}