# pipeline/prediction_pipeline.py

import numpy as np
import pandas as pd
import joblib
import faiss

from config.paths_config import (
    USER_WEIGHTS_PATH,
    ANIME_WEIGHTS_PATH,
    USER2USER_ENCODED_PATH,
    USER2USER_DECODED_PATH,
    ANIME2ANIME_ENCODED_PATH,
    ANIME2ANIME_DECODED_PATH,
    RATING_DF,
    DF,
    SYNOPSIS_DF
)

###############################################
# ✅ Load Embeddings + ID Mappings
###############################################
print("Loading file:", USER2USER_ENCODED_PATH)

def load_embeddings():
    user_emb = joblib.load(USER_WEIGHTS_PATH).astype("float32")
    anime_emb = joblib.load(ANIME_WEIGHTS_PATH).astype("float32")
    
    user2idx = joblib.load(USER2USER_ENCODED_PATH)
    idx2user = joblib.load(USER2USER_DECODED_PATH)

    anime2idx = joblib.load(ANIME2ANIME_ENCODED_PATH)
    idx2anime = joblib.load(ANIME2ANIME_DECODED_PATH)

    return user_emb, anime_emb, user2idx, idx2user, anime2idx, idx2anime

USER_EMB, ANIME_EMB, USER2IDX, IDX2USER, ANIME2IDX, IDX2ANIME = load_embeddings()


###############################################
# ✅ Build FAISS Indexes
###############################################
def build_faiss_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

FAISS_USER_INDEX = build_faiss_index(USER_EMB)
FAISS_ANIME_INDEX = build_faiss_index(ANIME_EMB)


###############################################
# ✅ Similarity Functions (FAISS)
###############################################
def get_similar_users_faiss(user_id, top_k=20):
    if user_id not in USER2IDX:
        return []

    idx = USER2IDX[user_id]
    user_vec = USER_EMB[idx].reshape(1, -1)
    distances, indices = FAISS_USER_INDEX.search(user_vec, top_k)

    similar_users = [IDX2USER[i] for i in indices[0]]
    return similar_users


def get_similar_animes_faiss(anime_name, n=10):
    df_anime = pd.read_csv(DF)

    row = df_anime[df_anime["eng_version"] == anime_name]
    if row.empty:
        return []

    anime_id = int(row["anime_id"].values[0])

    if anime_id not in ANIME2IDX:
        return []

    idx = ANIME2IDX[anime_id]
    vector = ANIME_EMB[idx].reshape(1, -1)

    distances, indices = FAISS_ANIME_INDEX.search(vector, n)

    results = [IDX2ANIME[i] for i in indices[0]]
    return results


###############################################
# ✅ Popularity Boost
###############################################
def popularity_score(anime_id):
    df = pd.read_csv(DF)
    row = df[df["anime_id"] == anime_id]

    if row.empty:
        return 0

    members = row["Members"].values[0]
    return np.log1p(members)


###############################################
# ✅ Genre Overlap Score
###############################################
def genre_overlap(anime_id_1, anime_id_2):
    df = pd.read_csv(DF)
    a = df[df["anime_id"] == anime_id_1]
    b = df[df["anime_id"] == anime_id_2]

    if a.empty or b.empty:
        return 0

    g1 = set(str(a["Genres"].values[0]).split(","))
    g2 = set(str(b["Genres"].values[0]).split(","))

    if len(g1 | g2) == 0:
        return 0

    return len(g1 & g2) / len(g1 | g2)


###############################################
# ✅ MMR Re-ranking
###############################################
def mmr(candidates, embeddings, lambda_mmr=0.7, top_k=10):
    selected = []
    remaining = candidates.copy()

    while remaining and len(selected) < top_k:
        if not selected:
            selected.append(remaining.pop(0))
            continue

        last_idx = selected[-1][0]
        last_vec = embeddings[last_idx]

        relevance = np.array([score for _, score in remaining])
        diversity = np.array([
            np.dot(last_vec, embeddings[idx]) for idx, _ in remaining
        ])

        mmr_scores = lambda_mmr * relevance - (1 - lambda_mmr) * diversity
        best = remaining.pop(np.argmax(mmr_scores))
        selected.append(best)

    return selected


###############################################
# ✅ Load User Preferences
###############################################
def load_user_preferences(user_id):
    df_ratings = pd.read_csv(RATING_DF)
    df_anime = pd.read_csv(DF)

    user_data = df_ratings[df_ratings["user_id"] == user_id]
    merged = pd.merge(user_data, df_anime, on="anime_id", how="left")

    return merged


###############################################
# ✅ FINAL HYBRID MODEL
###############################################
def hybrid_recommendation(
    user_id,
    user_weight=0.5,
    content_weight=0.5,
    top_k=10
):

    # -------------------------------
    # STEP 1: USER-BASED RECOMMENDER
    # -------------------------------
    similar_users = get_similar_users_faiss(user_id, top_k=20)
    user_pref = load_user_preferences(user_id)

    if user_pref.empty:
        return {
            "recommendations": [],
            "explanations": {}
        }

    user_scores = {}
    for uid in similar_users:
        rows = user_pref[user_pref["user_id"] == uid]
        for _, r in rows.iterrows():
            anime_id = r["anime_id"]
            user_scores[anime_id] = user_scores.get(anime_id, 0) + r["rating"]

    # -------------------------------
    # STEP 2: CONTENT-BASED
    # -------------------------------
    df_anime = pd.read_csv(DF)
    df_syn = pd.read_csv(SYNOPSIS_DF)

    content_scores = {}

    for anime_id in list(user_scores.keys()):
        row = df_anime[df_anime["anime_id"] == anime_id]
        if row.empty:
            continue
        name = row["eng_version"].values[0]

        similar_ids = get_similar_animes_faiss(name, n=15)

        for sid in similar_ids:
            content_scores[sid] = content_scores.get(sid, 0) + 1

    # -------------------------------
    # STEP 3: COMBINE SCORES
    # -------------------------------
    combined = {}
    explanations = {}

    max_user = max(user_scores.values()) if user_scores else 1
    max_content = max(content_scores.values()) if content_scores else 1

    # add normalized user and content scores
    for aid in set(list(user_scores.keys()) + list(content_scores.keys())):

        u_score = user_scores.get(aid, 0) / max_user
        c_score = content_scores.get(aid, 0) / max_content

        combined_score = user_weight * u_score + content_weight * c_score

        explanations[aid] = {
            "raw_user_score": float(u_score),
            "raw_content_score": float(c_score)
        }

        combined[aid] = combined_score

    # -------------------------------
    # STEP 4: BOOSTERS
    # -------------------------------
    for aid in combined:
        pop = popularity_score(aid)
        explanations[aid]["popularity"] = float(pop)
        combined[aid] += 0.1 * pop

    # genre overlap using the top candidate as baseline
    sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    top_aid = sorted_items[0][0] if sorted_items else None

    for aid in combined:
        if top_aid:
            g = genre_overlap(aid, top_aid)
            explanations[aid]["genre_overlap"] = float(g)
        else:
            explanations[aid]["genre_overlap"] = 0.0

    # -------------------------------
    # STEP 5: RANK
    # -------------------------------
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # for MMR we need embedding indexes
    mmr_ready = [(ANIME2IDX[aid], score) for aid, score in ranked if aid in ANIME2IDX]

    final = mmr(
        candidates=mmr_ready,
        embeddings=ANIME_EMB,
        lambda_mmr=0.7,
        top_k=top_k
    )

    result_anime_ids = [IDX2ANIME[idx] for idx, _ in final]

    # Map anime IDs → names
    result_names = []
    for aid in result_anime_ids:
        row = df_anime[df_anime["anime_id"] == aid]
        if not row.empty:
            result_names.append(row["eng_version"].values[0])

        explanations[aid]["anime_name"] = row["eng_version"].values[0]
        explanations[aid]["combined_final_score"] = float(combined.get(aid, 0))
        explanations[aid]["neighbors_considered"] = 15

    return {
        "recommendations": result_names,
        "explanations": explanations
    }