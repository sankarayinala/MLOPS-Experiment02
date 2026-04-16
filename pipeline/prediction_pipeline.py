from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Dict, List, Tuple

import faiss
import joblib
import numpy as np
import pandas as pd

from api.metrics import (
    FAISS_ANIME_LATENCY,
    FAISS_USER_LATENCY,
    PIPELINE_STAGE_LATENCY,
)
from config.paths_config import (
    ANIME2ANIME_DECODED_PATH,
    ANIME2ANIME_ENCODED_PATH,
    ANIME_WEIGHTS_PATH,
    DF,
    RATING_DF,
    USER2USER_DECODED_PATH,
    USER2USER_ENCODED_PATH,
    USER_WEIGHTS_PATH,
    WEIGHTS_DIR,
)
from src.logger import get_logger

logger = get_logger(__name__)

# =============================================
# Global caches
# =============================================
_df_anime: pd.DataFrame | None = None
_df_ratings: pd.DataFrame | None = None
_ratings_by_user: pd.DataFrame | None = None

_user_row_ranges: Dict[int, Tuple[int, int]] = {}

_rating_user_ids: np.ndarray | None = None
_rating_anime_ids: np.ndarray | None = None
_rating_values: np.ndarray | None = None

anime_id_to_name: Dict[int, str] = {}
anime_id_to_members: Dict[int, float] = {}
anime_id_to_genres: Dict[int, str] = {}

# =============================================
# HNSW tuning
# =============================================
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 120
HNSW_EF_SEARCH = 64


def _read_table(path: str, columns=None, dtype_map=None) -> pd.DataFrame:
    """
    Read CSV or Parquet based on file extension.
    """
    lower = path.lower()

    if lower.endswith(".parquet"):
        df = pd.read_parquet(path, columns=columns, engine="pyarrow")
        if dtype_map:
            cast_map = {k: v for k, v in dtype_map.items() if k in df.columns}
            if cast_map:
                df = df.astype(cast_map)
        return df

    if lower.endswith(".csv"):
        return pd.read_csv(
            path,
            low_memory=True,
            usecols=columns,
            dtype=dtype_map,
        )

    raise ValueError(f"Unsupported file format for path: {path}")


def log_timing(stage: str, start_time: float) -> None:
    elapsed_ms = (perf_counter() - start_time) * 1000
    logger.info(f"⏱️ {stage} took {elapsed_ms:.2f} ms")


def _normalize_decoded_map(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list):
        return {i: v for i, v in enumerate(obj)}
    try:
        return dict(obj)
    except Exception:
        return obj


def _load_artifact_manifest() -> dict | None:
    manifest_path = os.path.join(WEIGHTS_DIR, "artifact_manifest.json")
    if not os.path.exists(manifest_path):
        logger.warning("artifact_manifest.json not found; skipping manifest validation")
        return None

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Could not read artifact manifest: {exc}")
        return None


def _validate_artifacts(
    user_emb: np.ndarray,
    anime_emb: np.ndarray,
    user2idx: dict,
    idx2user: dict,
    anime2idx: dict,
    idx2anime: dict,
) -> None:
    logger.info(
        f"✅ Loaded artifacts | "
        f"user_emb_rows={user_emb.shape[0]:,}, user2idx={len(user2idx):,}, idx2user={len(idx2user):,} | "
        f"anime_emb_rows={anime_emb.shape[0]:,}, anime2idx={len(anime2idx):,}, idx2anime={len(idx2anime):,}"
    )

    if user_emb.shape[0] != len(user2idx) or user_emb.shape[0] != len(idx2user):
        raise RuntimeError(
            "User artifact mismatch detected: "
            f"user_emb rows={user_emb.shape[0]}, "
            f"user2idx size={len(user2idx)}, "
            f"idx2user size={len(idx2user)}. "
            "These files must come from the same training run."
        )

    if anime_emb.shape[0] != len(anime2idx) or anime_emb.shape[0] != len(idx2anime):
        raise RuntimeError(
            "Anime artifact mismatch detected: "
            f"anime_emb rows={anime_emb.shape[0]}, "
            f"anime2idx size={len(anime2idx)}, "
            f"idx2anime size={len(idx2anime)}. "
            "These files must come from the same training run."
        )

    manifest = _load_artifact_manifest()
    if manifest:
        logger.info(f"Artifact manifest loaded | run_id={manifest.get('run_id')}")


def _load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    global _df_anime, _df_ratings, _ratings_by_user, _user_row_ranges
    global _rating_user_ids, _rating_anime_ids, _rating_values

    if _df_anime is None:
        start = perf_counter()
        logger.info("📊 Loading anime metadata into global cache...")

        _df_anime = _read_table(
            DF,
            columns=["anime_id", "eng_version", "Genres", "Members"],
            dtype_map={
                "anime_id": "int32",
                "eng_version": "string",
                "Genres": "string",
                "Members": "float32",
            },
        )

        anime_id_to_name.clear()
        anime_id_to_members.clear()
        anime_id_to_genres.clear()

        anime_id_to_name.update(
            dict(zip(_df_anime["anime_id"], _df_anime["eng_version"].fillna("")))
        )
        anime_id_to_members.update(dict(zip(_df_anime["anime_id"], _df_anime["Members"].astype(float))))
        anime_id_to_genres.update(dict(zip(_df_anime["anime_id"], _df_anime["Genres"])))

        logger.info(f"✅ Anime metadata cached | {len(_df_anime):,} rows")
        log_timing("Anime metadata warmup", start)

    if _df_ratings is None:
        start = perf_counter()
        logger.info("📊 Loading ratings into global cache...")

        _df_ratings = _read_table(
            RATING_DF,
            columns=["user_id", "anime_id", "rating"],
            dtype_map={
                "user_id": "int32",
                "anime_id": "int32",
                "rating": "float32",
            },
        )

        logger.info("Building sorted ratings arrays and user row index...")
        sorted_ratings = _df_ratings.sort_values(["user_id", "anime_id"]).reset_index(drop=True)
        _ratings_by_user = sorted_ratings

        _rating_user_ids = sorted_ratings["user_id"].to_numpy(dtype=np.int32, copy=True)
        _rating_anime_ids = sorted_ratings["anime_id"].to_numpy(dtype=np.int32, copy=True)
        _rating_values = sorted_ratings["rating"].to_numpy(dtype=np.float32, copy=True)

        unique_users, start_idx, counts = np.unique(
            _rating_user_ids, return_index=True, return_counts=True
        )

        _user_row_ranges.clear()
        for uid, s, c in zip(unique_users, start_idx, counts):
            _user_row_ranges[int(uid)] = (int(s), int(s + c))

        logger.info(
            f"✅ Ratings cached | {len(_df_ratings):,} rows | {len(_user_row_ranges):,} users indexed"
        )
        log_timing("Ratings warmup", start)

    return _df_anime, _df_ratings


def _get_user_slice(user_id: int) -> tuple[np.ndarray, np.ndarray]:
    if _rating_anime_ids is None or _rating_values is None:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    row_range = _user_row_ranges.get(int(user_id))
    if row_range is None:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    start, end = row_range
    return _rating_anime_ids[start:end], _rating_values[start:end]


def _get_user_ratings_fast(user_id: int) -> pd.DataFrame:
    anime_ids, ratings = _get_user_slice(user_id)
    if anime_ids.size == 0:
        return pd.DataFrame(columns=["user_id", "anime_id", "rating"])

    return pd.DataFrame(
        {
            "user_id": np.full(anime_ids.shape[0], int(user_id), dtype=np.int32),
            "anime_id": anime_ids,
            "rating": ratings,
        }
    )


logger.info("🔄 Loading user and anime embeddings...")

user_emb = joblib.load(USER_WEIGHTS_PATH).astype("float32")
anime_emb = joblib.load(ANIME_WEIGHTS_PATH).astype("float32")

user2idx = joblib.load(USER2USER_ENCODED_PATH)
idx2user = _normalize_decoded_map(joblib.load(USER2USER_DECODED_PATH))
anime2idx = joblib.load(ANIME2ANIME_ENCODED_PATH)
idx2anime = _normalize_decoded_map(joblib.load(ANIME2ANIME_DECODED_PATH))

_validate_artifacts(
    user_emb=user_emb,
    anime_emb=anime_emb,
    user2idx=user2idx,
    idx2user=idx2user,
    anime2idx=anime2idx,
    idx2anime=idx2anime,
)


def build_hnsw_index(vectors: np.ndarray, name: str = "default") -> faiss.IndexHNSWFlat:
    d = vectors.shape[1]
    logger.info(
        f"Building HNSW index for {name} | dim={d}, M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}"
    )

    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(vectors)

    logger.info(f"✅ HNSW index built for {name} | Total vectors: {vectors.shape[0]:,}")
    return index


FAISS_USER_INDEX = build_hnsw_index(user_emb, "User Embeddings")
FAISS_ANIME_INDEX = build_hnsw_index(anime_emb, "Anime Embeddings")
logger.info("✅ FAISS HNSW indexes ready")


def get_similar_users_faiss(user_id: int, top_k: int = 8) -> List[int]:
    with FAISS_USER_LATENCY.time():
        if user_id not in user2idx:
            logger.warning(f"User {user_id} not found in training mappings.")
            return []

        idx = user2idx[user_id]
        user_vec = user_emb[idx].reshape(1, -1)

        _, indices = FAISS_USER_INDEX.search(user_vec, max(top_k + 20, 25))

        similar_users: List[int] = []
        skipped_unmapped = 0

        for i in indices[0]:
            try:
                candidate = int(idx2user[int(i)])
                if candidate != user_id:
                    similar_users.append(candidate)
            except (KeyError, TypeError, ValueError, IndexError):
                skipped_unmapped += 1
                continue

            if len(similar_users) >= top_k:
                break

        if skipped_unmapped:
            logger.warning(
                f"Skipped {skipped_unmapped} FAISS neighbor indices missing from idx2user"
            )

        logger.info(f"Resolved {len(similar_users)} similar users for user_id={user_id}")
        return similar_users


def get_similar_animes_faiss(anime_id: int, n: int = 5) -> List[int]:
    with FAISS_ANIME_LATENCY.time():
        if anime_id not in anime2idx:
            return []

        idx = anime2idx[anime_id]
        vector = anime_emb[idx].reshape(1, -1)

        _, indices = FAISS_ANIME_INDEX.search(vector, max(n + 10, 15))

        results: List[int] = []
        for i in indices[0]:
            try:
                candidate = int(idx2anime[int(i)])
                if candidate != anime_id:
                    results.append(candidate)
            except (KeyError, TypeError, ValueError, IndexError):
                continue

            if len(results) >= n:
                break

        return results


def popularity_score(anime_id: int) -> float:
    return float(np.log1p(anime_id_to_members.get(anime_id, 0.0)))


def genre_overlap(anime_id_1: int, anime_id_2: int | None) -> float:
    if anime_id_2 is None:
        return 0.0

    g1 = {x.strip() for x in str(anime_id_to_genres.get(anime_id_1, "")).split(",") if x.strip()}
    g2 = {x.strip() for x in str(anime_id_to_genres.get(anime_id_2, "")).split(",") if x.strip()}
    union = g1 | g2
    if not union:
        return 0.0

    return len(g1 & g2) / len(union)


def mmr(candidates, embeddings, lambda_mmr: float = 0.7, top_k: int = 10):
    if not candidates:
        return []

    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        relevance = np.array([score for _, score in remaining], dtype=np.float32)

        diversity_scores = []
        for candidate_id, _ in remaining:
            sims = []
            if candidate_id in anime2idx:
                candidate_vec = embeddings[anime2idx[candidate_id]]
                for selected_id, _ in selected:
                    if selected_id in anime2idx:
                        selected_vec = embeddings[anime2idx[selected_id]]
                        sims.append(float(np.dot(candidate_vec, selected_vec)))
            diversity_scores.append(max(sims) if sims else 0.0)

        diversity = np.array(diversity_scores, dtype=np.float32)
        mmr_scores = lambda_mmr * relevance - (1.0 - lambda_mmr) * diversity
        best_idx = int(np.argmax(mmr_scores))
        selected.append(remaining.pop(best_idx))

    return selected


def _aggregate_user_scores(similar_users: List[int], watched: set[int]) -> Dict[int, float]:
    if not similar_users:
        return {}

    anime_chunks = []
    rating_chunks = []

    for uid in similar_users:
        anime_ids, ratings = _get_user_slice(uid)
        if anime_ids.size == 0:
            continue
        anime_chunks.append(anime_ids)
        rating_chunks.append(ratings)

    if not anime_chunks:
        return {}

    all_anime = np.concatenate(anime_chunks)
    all_ratings = np.concatenate(rating_chunks)

    if watched:
        watched_arr = np.fromiter(watched, dtype=np.int32, count=len(watched))
        keep_mask = ~np.isin(all_anime, watched_arr, assume_unique=False)
        all_anime = all_anime[keep_mask]
        all_ratings = all_ratings[keep_mask]

    if all_anime.size == 0:
        return {}

    unique_anime, inv = np.unique(all_anime, return_inverse=True)
    agg_scores = np.zeros(unique_anime.shape[0], dtype=np.float32)
    np.add.at(agg_scores, inv, all_ratings)

    return {int(aid): float(score) for aid, score in zip(unique_anime, agg_scores)}


def _aggregate_content_scores(top_watched: List[int], watched: set[int]) -> Dict[int, float]:
    if not top_watched:
        return {}

    chunks = []
    for seed_aid in top_watched:
        similar = get_similar_animes_faiss(seed_aid, n=8)
        if similar:
            chunks.append(np.asarray(similar, dtype=np.int32))

    if not chunks:
        return {}

    all_candidates = np.concatenate(chunks)

    if watched:
        watched_arr = np.fromiter(watched, dtype=np.int32, count=len(watched))
        keep_mask = ~np.isin(all_candidates, watched_arr, assume_unique=False)
        all_candidates = all_candidates[keep_mask]

    if all_candidates.size == 0:
        return {}

    unique_anime, inv = np.unique(all_candidates, return_inverse=True)
    counts = np.zeros(unique_anime.shape[0], dtype=np.float32)
    np.add.at(counts, inv, 1.0)

    return {int(aid): float(cnt) for aid, cnt in zip(unique_anime, counts)}


def hybrid_recommendation(
    user_id: int,
    user_weight: float = 0.6,
    content_weight: float = 0.4,
    top_k: int = 10,
    mmr_lambda: float = 0.7,
):
    total_start = perf_counter()
    logger.info(
        f"🎯 Generating recommendations for user {user_id} "
        f"(user_w={user_weight:.2f}, content_w={content_weight:.2f}, top_k={top_k})"
    )

    stage_start = perf_counter()
    _load_dataframes()
    PIPELINE_STAGE_LATENCY.labels(stage="load_dataframes").observe(perf_counter() - stage_start)

    stage_start = perf_counter()
    similar_users = get_similar_users_faiss(user_id, top_k=5)
    PIPELINE_STAGE_LATENCY.labels(stage="similar_users").observe(perf_counter() - stage_start)

    stage_start = perf_counter()
    user_anime_ids, user_ratings = _get_user_slice(user_id)
    PIPELINE_STAGE_LATENCY.labels(stage="load_user_history").observe(perf_counter() - stage_start)

    if user_anime_ids.size == 0:
        logger.warning(f"No rating history found for user {user_id}")
        PIPELINE_STAGE_LATENCY.labels(stage="total_pipeline").observe(perf_counter() - total_start)
        return {"recommendations": [], "explanations": {}}

    watched = set(user_anime_ids.tolist())

    stage_start = perf_counter()
    user_scores = _aggregate_user_scores(similar_users=similar_users, watched=watched)
    PIPELINE_STAGE_LATENCY.labels(stage="user_scoring").observe(perf_counter() - stage_start)
    logger.info(f"Found {len(user_scores)} candidate animes from similar users")

    stage_start = perf_counter()
    if user_ratings.size > 0:
        top_idx = np.argsort(user_ratings)[-8:][::-1]
        top_watched = user_anime_ids[top_idx].astype(np.int32).tolist()
    else:
        top_watched = []

    content_scores = _aggregate_content_scores(top_watched=top_watched, watched=watched)
    PIPELINE_STAGE_LATENCY.labels(stage="content_scoring").observe(perf_counter() - stage_start)

    stage_start = perf_counter()
    combined: Dict[int, float] = {}
    explanations: Dict[int, dict] = {}

    max_user = max(user_scores.values()) if user_scores else 1.0
    max_content = max(content_scores.values()) if content_scores else 1.0
    candidate_ids = set(user_scores.keys()) | set(content_scores.keys())

    top_seed_subset = top_watched[:3] if top_watched else []

    for aid in candidate_ids:
        u = user_scores.get(aid, 0.0) / max_user
        c = content_scores.get(aid, 0.0) / max_content

        pop = popularity_score(aid)
        genre_boost = 0.0
        if top_seed_subset:
            genre_boost = max(genre_overlap(aid, seed_id) for seed_id in top_seed_subset)

        score = (user_weight * u) + (content_weight * c) + (0.05 * pop) + (0.10 * genre_boost)

        combined[aid] = float(score)
        explanations[aid] = {
            "raw_user_score": float(u),
            "raw_content_score": float(c),
            "popularity": float(pop),
            "genre_overlap": float(genre_boost),
        }

    PIPELINE_STAGE_LATENCY.labels(stage="hybrid_scoring").observe(perf_counter() - stage_start)

    stage_start = perf_counter()
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[: max(top_k * 3, top_k)]
    reranked = mmr(ranked, anime_emb, lambda_mmr=mmr_lambda, top_k=top_k)
    PIPELINE_STAGE_LATENCY.labels(stage="ranking_mmr").observe(perf_counter() - stage_start)

    stage_start = perf_counter()
    recommendations: List[str] = []
    final_explanations: Dict[str, dict] = {}

    for aid, _score in reranked:
        anime_name = anime_id_to_name.get(aid)
        if not anime_name:
            continue
        recommendations.append(str(anime_name))
        final_explanations[str(aid)] = explanations.get(aid, {})

    PIPELINE_STAGE_LATENCY.labels(stage="format_response").observe(perf_counter() - stage_start)
    PIPELINE_STAGE_LATENCY.labels(stage="total_pipeline").observe(perf_counter() - total_start)

    logger.info(
        f"✅ Recommendation completed for user {user_id} | "
        f"count={len(recommendations)} | total={(perf_counter() - total_start):.2f}s"
    )

    return {
        "recommendations": recommendations,
        "explanations": final_explanations,
    }


def warmup_dataframes():
    _load_dataframes()
    logger.info("✅ Warmup completed via global cache")