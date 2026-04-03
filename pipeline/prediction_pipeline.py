"""
File: pipeline/prediction_pipeline.py
Purpose: Hybrid Anime Recommendation Engine - FAISS HNSW Optimized
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Dict, List

import faiss
import joblib
import numpy as np
import pandas as pd

from config.paths_config import (
    USER_WEIGHTS_PATH,
    ANIME_WEIGHTS_PATH,
    USER2USER_ENCODED_PATH,
    USER2USER_DECODED_PATH,
    ANIME2ANIME_ENCODED_PATH,
    ANIME2ANIME_DECODED_PATH,
    RATING_DF,
    DF,
)
from src.logger import get_logger

logger = get_logger(__name__)

# Parquet paths
RATING_PARQUET = "artifacts/processed/rating_df.parquet"
ANIME_PARQUET = "artifacts/processed/anime_df.parquet"

# Global caches
_df_anime: pd.DataFrame | None = None
_df_ratings: pd.DataFrame | None = None

# Lookup dictionaries for fast access
anime_id_to_name: Dict[int, str] = {}
anime_id_to_members: Dict[int, float] = {}
anime_id_to_genres: Dict[int, str] = {}

# ==================== FAISS HNSW Indexes (Upgraded) ====================

# Tunable HNSW settings
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_USER_EF_SEARCH = 128
HNSW_ANIME_EF_SEARCH = 64


def log_timing(stage: str, start_time: float) -> None:
    """Log elapsed time in milliseconds for a pipeline stage."""
    elapsed_ms = (perf_counter() - start_time) * 1000.0
    logger.info(f"⏱️ {stage} took {elapsed_ms:.2f} ms")


def build_hnsw_index(vectors: np.ndarray, index_name: str = "default") -> faiss.Index:
    """
    Build optimized HNSW index with good default parameters for recommendation systems.
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype("float32", copy=False)

    d = vectors.shape[1]  # dimension

    logger.info(
        f"Building HNSW index for {index_name} | dim={d}, M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}"
    )

    # Create HNSW index with Inner Product (best when vectors are normalized)
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)

    # Set construction parameters
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_ANIME_EF_SEARCH  # default search effort

    # Add vectors (HNSW requires training for some variants, but Flat works directly)
    index.add(vectors)

    logger.info(
        f"✅ HNSW index built successfully for {index_name} | Total vectors: {vectors.shape[0]:,}"
    )
    return index


# Load embeddings once at module level
logger.info("🔄 Loading user and anime embeddings...")
user_emb = joblib.load(USER_WEIGHTS_PATH).astype("float32")
anime_emb = joblib.load(ANIME_WEIGHTS_PATH).astype("float32")

user2idx = joblib.load(USER2USER_ENCODED_PATH)
idx2user = joblib.load(USER2USER_DECODED_PATH)
anime2idx = joblib.load(ANIME2ANIME_ENCODED_PATH)
idx2anime = joblib.load(ANIME2ANIME_DECODED_PATH)

# Build optimized HNSW indexes
FAISS_USER_INDEX = build_hnsw_index(user_emb, "User Embeddings")
FAISS_ANIME_INDEX = build_hnsw_index(anime_emb, "Anime Embeddings")

logger.info("✅ FAISS HNSW indexes ready")


def get_similar_users_faiss(user_id: int, top_k: int = 5) -> List[int]:
    """Find similar users using HNSW"""
    start = perf_counter()

    if user_id not in user2idx:
        logger.warning(f"User {user_id} not found in training data.")
        return []

    idx = user2idx[user_id]
    user_vec = user_emb[idx].reshape(1, -1)

    # Increase efSearch for better recall during inference
    FAISS_USER_INDEX.hnsw.efSearch = HNSW_USER_EF_SEARCH

    _, indices = FAISS_USER_INDEX.search(user_vec, top_k + 1)

    similar_users: List[int] = []
    for i in indices[0]:
        try:
            candidate = int(idx2user[int(i)])
            if candidate != user_id:
                similar_users.append(candidate)
        except (KeyError, TypeError, ValueError):
            continue

    log_timing("FAISS user similarity search", start)
    return similar_users[:top_k]


def get_similar_animes_faiss(anime_id: int, n: int = 8) -> List[int]:
    """Find similar animes using HNSW"""
    start = perf_counter()

    if anime_id not in anime2idx:
        return []

    idx = anime2idx[anime_id]
    vector = anime_emb[idx].reshape(1, -1)

    FAISS_ANIME_INDEX.hnsw.efSearch = HNSW_ANIME_EF_SEARCH

    _, indices = FAISS_ANIME_INDEX.search(vector, n + 1)

    results: List[int] = []
    for i in indices[0]:
        try:
            candidate_aid = int(idx2anime[int(i)])
            if candidate_aid != anime_id:
                results.append(candidate_aid)
        except (KeyError, TypeError, ValueError):
            continue

    log_timing("FAISS anime similarity search", start)
    return results[:n]


# ====================== Data Loading & Lookup ======================

def _load_anime_df() -> pd.DataFrame:
    if os.path.exists(ANIME_PARQUET):
        logger.info(f"📦 Loading anime metadata from Parquet: {ANIME_PARQUET}")
        df = pd.read_parquet(
            ANIME_PARQUET,
            columns=["anime_id", "eng_version", "Genres", "Members"],
        )
    else:
        logger.info(f"📄 Loading anime metadata from CSV: {DF}")
        df = pd.read_csv(
            DF,
            usecols=["anime_id", "eng_version", "Genres", "Members"],
            dtype={
                "anime_id": "int32",
                "eng_version": "string",
                "Genres": "string",
                "Members": "float32",
            },
            low_memory=True,
        )

    df["anime_id"] = pd.to_numeric(df["anime_id"], errors="coerce").fillna(0).astype("int32")
    df["eng_version"] = df["eng_version"].fillna("").astype("string")
    df["Genres"] = df["Genres"].fillna("").astype("string")
    df["Members"] = pd.to_numeric(df["Members"], errors="coerce").fillna(0).astype("float32")

    return df


def _load_ratings_df() -> pd.DataFrame:
    if os.path.exists(RATING_PARQUET):
        logger.info(f"📦 Loading ratings from Parquet: {RATING_PARQUET}")
        df = pd.read_parquet(
            RATING_PARQUET,
            columns=["user_id", "anime_id", "rating"],
        )
    else:
        logger.info(f"📄 Loading ratings from CSV: {RATING_DF}")
        df = pd.read_csv(
            RATING_DF,
            usecols=["user_id", "anime_id", "rating"],
            dtype={
                "user_id": "int32",
                "anime_id": "int32",
                "rating": "float32",
            },
            low_memory=True,
        )

    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(0).astype("int32")
    df["anime_id"] = pd.to_numeric(df["anime_id"], errors="coerce").fillna(0).astype("int32")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype("float32")

    return df


def warmup_dataframes() -> None:
    global _df_anime, _df_ratings
    global anime_id_to_name, anime_id_to_members, anime_id_to_genres

    if _df_anime is None:
        start = perf_counter()
        _df_anime = _load_anime_df()

        anime_id_to_name = dict(zip(_df_anime["anime_id"], _df_anime["eng_version"]))
        anime_id_to_members = dict(zip(_df_anime["anime_id"], _df_anime["Members"].astype(float)))
        anime_id_to_genres = dict(zip(_df_anime["anime_id"], _df_anime["Genres"]))

        logger.info(f"✅ Anime metadata cached | {len(_df_anime):,} rows")
        log_timing("Anime metadata warmup", start)

    if _df_ratings is None:
        start = perf_counter()
        _df_ratings = _load_ratings_df()
        logger.info(f"✅ Ratings cached | {len(_df_ratings):,} rows")
        log_timing("Ratings warmup", start)


def _load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    warmup_dataframes()
    return _df_anime, _df_ratings


# ====================== Helper Functions ======================

def popularity_score(anime_id: int) -> float:
    members = float(anime_id_to_members.get(anime_id, 0.0))
    return float(np.log1p(members)) if members > 0 else 0.0


def genre_overlap(anime_id_1: int, anime_id_2: int | None) -> float:
    if anime_id_2 is None:
        return 0.0

    g1_raw = anime_id_to_genres.get(anime_id_1, "")
    g2_raw = anime_id_to_genres.get(anime_id_2, "")

    g1 = {x.strip() for x in str(g1_raw).split(",") if x.strip()}
    g2 = {x.strip() for x in str(g2_raw).split(",") if x.strip()}

    union = g1 | g2
    if not union:
        return 0.0

    return float(len(g1 & g2) / len(union))


def mmr(candidates, embeddings, lambda_mmr: float = 0.7, top_k: int = 10):
    """Maximal Marginal Relevance re-ranking"""
    start = perf_counter()

    if not candidates:
        log_timing("MMR reranking", start)
        return []

    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        last_idx = selected[-1][0]
        last_vec = embeddings[last_idx]

        relevance = np.array([score for _, score in remaining], dtype=np.float32)
        diversity = np.array(
            [np.dot(last_vec, embeddings[idx]) for idx, _ in remaining],
            dtype=np.float32,
        )

        mmr_scores = lambda_mmr * relevance - (1.0 - lambda_mmr) * diversity
        best_pos = int(np.argmax(mmr_scores))
        selected.append(remaining.pop(best_pos))

    log_timing("MMR reranking", start)
    return selected


# ====================== Recommendation Logic ======================

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

    # ================= LOAD DATA =================
    t0 = perf_counter()
    _, df_ratings = _load_dataframes()
    log_timing("Dataframe load", t0)

    # ================= USER HISTORY =================
    t0 = perf_counter()
    target_user_rows = df_ratings[df_ratings["user_id"] == user_id]
    log_timing("User history fetch", t0)

    if target_user_rows.empty:
        logger.warning(f"No rating history found for user {user_id}")
        log_timing("TOTAL recommendation pipeline", total_start)
        return {"recommendations": [], "explanations": {}}

    watched_anime_ids = set(target_user_rows["anime_id"].astype("int32").tolist())

    # 1. User-based Collaborative Filtering
    t0 = perf_counter()
    similar_users = get_similar_users_faiss(user_id, top_k=5)
    log_timing("Similar users lookup", t0)

    # ================= USER-BASED SCORING =================
    t0 = perf_counter()
    user_scores: Dict[int, float] = {}
    if similar_users:
        similar_rows = df_ratings[df_ratings["user_id"].isin(similar_users)]
        similar_rows = similar_rows[~similar_rows["anime_id"].isin(watched_anime_ids)]

        grouped = (
            similar_rows.groupby("anime_id", observed=True)["rating"]
            .sum()
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            user_scores[int(row.anime_id)] = float(row.rating)

    log_timing("User-based scoring", t0)
    logger.info(f"Found {len(user_scores)} candidate animes from similar users")

    # ================= CONTENT-BASED =================
    t0 = perf_counter()
    content_scores: Dict[int, float] = {}
    top_user_items = (
        target_user_rows.sort_values("rating", ascending=False)
        .head(10)["anime_id"]
        .astype("int32")
        .tolist()
    )

    for seed_aid in top_user_items:
        similar_animes = get_similar_animes_faiss(seed_aid, n=8)
        for similar_aid in similar_animes:
            if similar_aid in watched_anime_ids:
                continue
            content_scores[similar_aid] = content_scores.get(similar_aid, 0.0) + 1.0

    log_timing("Content-based scoring", t0)
    logger.info(f"Found {len(content_scores)} candidate animes from content similarity")

    # ================= HYBRID =================
    t0 = perf_counter()
    combined: Dict[int, float] = {}
    explanations: Dict[int, dict] = {}

    max_user = max(user_scores.values()) if user_scores else 1.0
    max_content = max(content_scores.values()) if content_scores else 1.0

    for aid in set(user_scores.keys()) | set(content_scores.keys()):
        u_score = float(user_scores.get(aid, 0.0) / max_user) if max_user else 0.0
        c_score = float(content_scores.get(aid, 0.0) / max_content) if max_content else 0.0
        score = (user_weight * u_score) + (content_weight * c_score)

        combined[aid] = score
        explanations[aid] = {
            "raw_user_score": u_score,
            "raw_content_score": c_score,
        }

    log_timing("Hybrid scoring", t0)

    if not combined:
        logger.warning(f"No combined recommendation candidates found for user {user_id}")
        log_timing("TOTAL recommendation pipeline", total_start)
        return {"recommendations": [], "explanations": {}}

    # 4. Boosters
    t0 = perf_counter()
    for aid in list(combined.keys()):
        pop = popularity_score(aid)
        explanations[aid]["popularity"] = pop
        combined[aid] += 0.1 * pop

    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    top_aid = sorted_items[0][0] if sorted_items else None

    for aid in combined:
        explanations[aid]["genre_overlap"] = genre_overlap(aid, top_aid)

    log_timing("Boosters and genre overlap", t0)

    # 5. MMR Re-ranking
    t0 = perf_counter()
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    mmr_ready = [
        (anime2idx[aid], score)
        for aid, score in ranked
        if aid in anime2idx
    ]
    log_timing("MMR candidate preparation", t0)

    final = mmr(
        candidates=mmr_ready,
        embeddings=anime_emb,
        lambda_mmr=mmr_lambda,
        top_k=top_k,
    )

    # 6. Convert final anime IDs to names
    t0 = perf_counter()
    result_names: List[str] = []
    result_explanations: Dict[int, dict] = {}

    for idx, _ in final:
        try:
            aid = int(idx2anime[int(idx)])
        except (KeyError, TypeError, ValueError):
            continue

        name = anime_id_to_name.get(aid, "")
        if not name:
            continue

        result_names.append(name)
        result_explanations[aid] = explanations.get(aid, {})
        result_explanations[aid]["anime_name"] = name

    log_timing("Final mapping", t0)

    logger.info(f"✅ Successfully generated {len(result_names)} recommendations for user {user_id}")
    log_timing("TOTAL recommendation pipeline", total_start)

    return {
        "recommendations": result_names,
        "explanations": result_explanations,
    }