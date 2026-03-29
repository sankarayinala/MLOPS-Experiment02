# utils/helpers.py

import pandas as pd
import numpy as np
import joblib

from config.paths_config import (
    DF,
    SYNOPSIS_DF,
    RATING_DF,
    USER_WEIGHTS_PATH,
    ANIME_WEIGHTS_PATH,
    USER2USER_ENCODED_PATH,
    USER2USER_DECODED_PATH,
    ANIME2ANIME_ENCODED_PATH,
    ANIME2ANIME_DECODED_PATH
)

###########################################################
# ✅ Load Anime + Synopsis + Ratings Data
###########################################################

def load_anime_df():
    try:
        return pd.read_csv(DF)
    except Exception as e:
        print(f"❌ Error loading anime_df: {e}")
        return pd.DataFrame()

def load_synopsis_df():
    try:
        return pd.read_csv(SYNOPSIS_DF)
    except Exception as e:
        print(f"❌ Error loading synopsis_df: {e}")
        return pd.DataFrame()

def load_rating_df():
    try:
        return pd.read_csv(RATING_DF)
    except Exception as e:
        print(f"❌ Error loading rating_df: {e}")
        return pd.DataFrame()


###########################################################
# ✅ Generic Helpers
###########################################################

def detect_anime_id_column(df: pd.DataFrame):
    """Detect whether anime dataframe uses anime_id or MAL_ID."""
    if "anime_id" in df.columns:
        return "anime_id"
    if "MAL_ID" in df.columns:
        return "MAL_ID"
    raise KeyError("No anime_id or MAL_ID column found in anime dataframe")


###########################################################
# ✅ getAnimeFrame
###########################################################

def getAnimeFrame(anime_id, anime_csv_path=DF):
    df = pd.read_csv(anime_csv_path)
    col = "anime_id" if "anime_id" in df.columns else "MAL_ID"
    return df[df[col] == anime_id]


###########################################################
# ✅ getSynopsis
###########################################################

def getSynopsis(anime_id, synopsis_csv_path=SYNOPSIS_DF):
    df = pd.read_csv(synopsis_csv_path)

    id_col = "MAL_ID" if "MAL_ID" in df.columns else "anime_id"

    row = df[df[id_col] == anime_id]
    if row.empty:
        return "Synopsis not available."

    syn_col = "sypnopsis" if "sypnopsis" in row.columns else "synopsis"
    return row[syn_col].iloc[0]


###########################################################
# ✅ find_similar_users (FAISS-based pipeline uses engine directly)
###########################################################

def find_similar_users(
    item_input,
    path_user_weights=USER_WEIGHTS_PATH,
    path_user2user_encoded=USER2USER_ENCODED_PATH,
    path_user2user_decoded=USER2USER_DECODED_PATH,
    n=10
):
    """
    Compatibility wrapper – legacy structure.
    The real similarity is computed in prediction_pipeline.py.
    """
    try:
        embeddings = joblib.load(path_user_weights)
        encoded = joblib.load(path_user2user_encoded)
        decoded = joblib.load(path_user2user_decoded)

        if item_input not in encoded:
            print(f"⚠️ User {item_input} not found in encoded mapping.")
            return []

        idx = encoded[item_input]

        # Dummy fallback logic (FAISS is handled in pipeline)
        # Here we return top-n nearest vector indices
        sims = np.dot(embeddings, embeddings[idx])
        top = sims.argsort()[-n:][::-1]

        return [decoded[i] for i in top]

    except Exception as e:
        print(f"⚠️ Error in find_similar_users(): {e}")
        return []


###########################################################
# ✅ find_similar_animes (FAISS-based pipeline computes real similarity)
###########################################################

def find_similar_animes(
    name,
    path_anime_weights=ANIME_WEIGHTS_PATH,
    path_anime2anime_encoded=ANIME2ANIME_ENCODED_PATH,
    path_anime2anime_decoded=ANIME2ANIME_DECODED_PATH,
    path_anime_df=DF,
    n=10
):
    """
    Compatibility wrapper – legacy structure.
    Real FAISS similarity is handled inside prediction_pipeline.py.
    """
    try:
        df = pd.read_csv(path_anime_df)
        row = df[df["eng_version"] == name]

        if row.empty:
            print(f"⚠️ Anime '{name}' not found.")
            return pd.DataFrame()

        anime_id = row["anime_id"].iloc[0]

        encoded = joblib.load(path_anime2anime_encoded)
        decoded = joblib.load(path_anime2anime_decoded)
        emb = joblib.load(path_anime_weights)

        if anime_id not in encoded:
            print(f"⚠️ Anime id {anime_id} not in encoded mapping.")
            return pd.DataFrame()

        idx = encoded[anime_id]

        sims = np.dot(emb, emb[idx])
        top = sims.argsort()[-n:][::-1]

        result_ids = [decoded[i] for i in top]
        return df[df["anime_id"].isin(result_ids]]

    except Exception as e:
        print(f"⚠️ find_similar_animes() failed: {e}")
        return pd.DataFrame()


###########################################################
# ✅ get_user_preferences
###########################################################

def get_user_preferences(user_id, path_rating_df=RATING_DF, path_anime_df=DF):
    ratings = pd.read_csv(path_rating_df)
    anime = pd.read_csv(path_anime_df)

    user_df = ratings[ratings["user_id"] == user_id]
    if user_df.empty:
        return pd.DataFrame()

    anime_id_col = detect_anime_id_column(anime)

    # Normalize column names
    if anime_id_col == "MAL_ID" and "anime_id" in user_df.columns:
        user_df = user_df.rename(columns={"anime_id": "MAL_ID"})
    if anime_id_col == "anime_id" and "MAL_ID" in user_df.columns:
        user_df = user_df.rename(columns={"MAL_ID": "anime_id"})

    merged = pd.merge(user_df, anime, on=anime_id_col, how="left")
    return merged


###########################################################
# ✅ get_user_recommendations (legacy wrapper)
###########################################################

def get_user_recommendations(
    similar_users,
    user_pref,
    path_anime_df=DF,
    path_synopsis_df=SYNOPSIS_DF,
    path_rating_df=RATING_DF,
    n=10
):
    """
    Provides basic user-based recommendations.
    True hybrid recommendations come from prediction_pipeline.py.
    """
    if user_pref.empty:
        return pd.DataFrame()

    anime_df = pd.read_csv(path_anime_df)
    rating_df = pd.read_csv(path_rating_df)

    recommended = {}

    for uid in similar_users:
        rows = rating_df[rating_df["user_id"] == uid]
        for _, r in rows.iterrows():
            recommended[r["anime_id"]] = recommended.get(r["anime_id"], 0) + r["rating"]

    if not recommended:
        return pd.DataFrame()

    # Rank & return top n
    sorted_ids = sorted(recommended.items(), key=lambda x: x[1], reverse=True)
    top_ids = [aid for aid, _ in sorted_ids[:n]]

    return anime_df[anime_df["anime_id"].isin(top_ids)].copy()