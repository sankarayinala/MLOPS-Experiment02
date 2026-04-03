
# api/metrics.py

from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter(
    "recommend_requests_total",
    "Total number of recommendation requests"
)

REQUEST_LATENCY = Histogram(
    "recommend_request_latency_seconds",
    "Latency of recommendation requests",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60)
)

FAISS_USER_LATENCY = Histogram(
    "faiss_user_search_latency_seconds",
    "Latency of FAISS user similarity search",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2)
)

FAISS_ANIME_LATENCY = Histogram(
    "faiss_anime_search_latency_seconds",
    "Latency of FAISS anime similarity search",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2)
)

PIPELINE_STAGE_LATENCY = Histogram(
    "recommend_pipeline_stage_latency_seconds",
    "Latency of pipeline stages",
    ["stage"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)

INFLIGHT_REQUESTS = Gauge(
    "recommend_inflight_requests",
    "Current inflight recommendation requests"
)
