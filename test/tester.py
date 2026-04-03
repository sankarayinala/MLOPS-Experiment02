# tester.py

from utils.helpers import *
from config.paths_config import *
from pipeline.prediction_pipeline import hybrid_recommendation

print("\n===== TEST: find_similar_animes =====")
print(find_similar_animes(
    'Fairy Tail',
    ANIME_WEIGHTS_PATH,
    ANIME2ANIME_ENCODED_PATH,
    ANIME2ANIME_DECODED_PATH,
    DF
))

print("\n===== TEST: find_similar_users =====")
similar_users = find_similar_users(
    11880,
    USER_WEIGHTS_PATH,
    USER2USER_ENCODED_PATH,
    USER2USER_DECODED_PATH
)
print(similar_users)

print("\n===== TEST: get_user_preferences =====")
pref = get_user_preferences(
    11880,
    RATING_DF,
    DF
)
print(pref.head())

print("\n===== TEST: get_user_recommendations =====")
recs = get_user_recommendations(
    similar_users=similar_users,
    user_pref=pref,
    path_anime_df=DF,
    path_synopsis_df=SYNOPSIS_DF,
    path_rating_df=RATING_DF,
    n=10
)
print(recs)

print("\n===== TEST: HYBRID RECOMMENDATION =====")
print(hybrid_recommendation(11880))