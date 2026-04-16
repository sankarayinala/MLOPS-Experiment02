from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool

from api.auth import authenticate, create_access_token
from api.cache import (
    delete_key,
    get_cached,
    get_ttl,
    invalidate_user_cache,
    list_keys,
    set_cached,
)
from api.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    EMPTY_RESULTS,
    INFLIGHT_REQUESTS,
    RECOMMENDATION_COUNT,
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)
from pipeline.prediction_pipeline import hybrid_recommendation
from src.logger import get_logger

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from pipeline.prediction_pipeline import _load_dataframes
        _load_dataframes()
        logger.info("? Application warmup completed successfully")
    except Exception:
        logger.exception("? Warmup failed")
        raise
    yield


app = FastAPI(
    title="Anime Recommendation API",
    description="Hybrid Anime Recommendation API with explanations, caching, and admin tools",
    version="3.6.0",
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
    return {"message": "? Anime Recommendation API is running!"}


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
@limiter.limit("20/minute")
async def recommend(
    request: Request,
    user_id: int,
    user_weight: float = 0.6,
    content_weight: float = 0.4,
    top_k: int = 10,
    user=Depends(authenticate),
):
    REQUEST_COUNT.inc()
    INFLIGHT_REQUESTS.inc()
    start = perf_counter()

    try:
        with REQUEST_LATENCY.time():
            cache_key = f"rec:{user_id}:{user_weight:.2f}:{content_weight:.2f}:{top_k}"

            cached = get_cached(cache_key)
            if cached:
                CACHE_HITS.inc()
                recs = cached.get("recommendations", [])
                RECOMMENDATION_COUNT.observe(len(recs))
                if len(recs) == 0:
                    EMPTY_RESULTS.inc()
                logger.info(
                    f"Cache hit for user_id={user_id}, top_k={top_k}, "
                    f"elapsed={(perf_counter() - start):.2f}s"
                )
                return cached

            CACHE_MISSES.inc()

            result = await run_in_threadpool(
                hybrid_recommendation,
                user_id,
                user_weight,
                content_weight,
                top_k,
            )

            recs = result.get("recommendations", [])
            RECOMMENDATION_COUNT.observe(len(recs))
            if len(recs) == 0:
                EMPTY_RESULTS.inc()

            set_cached(cache_key, result, ttl=300)

            logger.info(
                f"Generated recommendations for user_id={user_id}, count={len(recs)}, "
                f"elapsed={(perf_counter() - start):.2f}s"
            )
            return result

    except HTTPException:
        REQUEST_ERRORS.inc()
        raise
    except Exception as exc:
        REQUEST_ERRORS.inc()
        logger.exception("Recommendation request failed")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {exc}") from exc
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