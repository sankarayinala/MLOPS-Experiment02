# test.py – Expanded testing for helpers.py

import random
import pandas as pd
import numpy as np
import joblib

from config.paths_config import *

from utils.helpers import (
    getAnimeFrame,
    getSynopsis,
    find_similar_animes,
    find_similar_users,
    get_user_preferences,
    get_user_recommendations
)

# -----------------------------
# ✅ UTILITY
# -----------------------------

def print_header(title):
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)


# -----------------------------
# ✅ TESTS
# -----------------------------

def test_find_similar_animes():
    print_header("TEST: find_similar_animes (extended coverage)")

    for _ in range(3):
        test_id = random.randint(1, 60000)
        print(f"\nTesting with anime_id = {test_id}")

        result = find_similar_animes(
            name=test_id,
            path_anime_weights=ANIME_WEIGHTS_PATH,
            path_anime2anime_encoded=ANIME_WEIGHTS_PATH,
            path_anime2anime_decoded=ANIME_WEIGHTS_PATH,
            path_anime_df=ANIME_CSV,
            n=5,
        )

        print(type(result))
        print(result)


def test_find_similar_users():
    print_header("TEST: find_similar_users (extended coverage)")

    for _ in range(3):
        user_id = random.randint(1, 20000)
        print(f"\nTesting with user_id = {user_id}")

        result = find_similar_users(
            item_input=user_id,
            path_user_weights=USER_WEIGHTS_PATH,
            path_user2user_encoded=USER_WEIGHTS_PATH,
            path_user2user_decoded=USER_WEIGHTS_PATH,
            n=7,
        )

        print(type(result))
        print(result)


def test_get_user_preferences():
    print_header("TEST: get_user_preferences (extended coverage)")

    for _ in range(3):
        user_id = random.randint(1, 20000)
        print(f"\nTesting with user_id = {user_id}")

        prefs = get_user_preferences(
            user_id=user_id,
            path_rating_df=RATING_DF,
            path_anime_df=ANIME_CSV,
        )

        print("Returned type:", type(prefs))
        print(prefs.head())


def test_get_user_recommendations():
    print_header("TEST: get_user_recommendations (extended coverage)")

    for _ in range(3):
        user_id = random.randint(1, 20000)
        print(f"\nTesting with user_id = {user_id}")

        # Step 1: Similar users
        similar_users = find_similar_users(
            item_input=user_id,
            path_user_weights=USER_WEIGHTS_PATH,
            path_user2user_encoded=USER_WEIGHTS_PATH,
            path_user2user_decoded=USER_WEIGHTS_PATH,
            n=5,
        )

        # Step 2: User preferences
        user_pref = get_user_preferences(
            user_id=user_id,
            path_rating_df=RATING_DF,
            path_anime_df=ANIME_CSV,
        )

        # Step 3: Recommendations
        recs = get_user_recommendations(
            similar_users=similar_users,
            user_pref=user_pref,
            path_anime_df=ANIME_CSV,
            path_synopsis_df=ANIMESYNOPSIS_CSV,
            path_rating_df=RATING_DF,
            n=5,
        )

        print("Returned type:", type(recs))
        print(recs.head())


# -----------------------------
# ✅ RUN ALL TESTS
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running Extended Helper Tests...\n")

    test_find_similar_animes()
    test_find_similar_users()
    test_get_user_preferences()
    test_get_user_recommendations()

    print("\n✅ All tests completed.")