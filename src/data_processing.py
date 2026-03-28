import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Optional

# ====================== CRITICAL PATH FIX ======================
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from logger import get_logger
from custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, input_file: str = None, output_dir: str = None):
        self.input_file = input_file or ANIMELIST_CSV
        self.output_dir = output_dir or PROCESSED_DIR
        
        self.rating_df: Optional[pd.DataFrame] = None
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None
        
        self.user2user_encoded = {}
        self.user2user_decoded = {}
        self.anime2anime_encoded = {}
        self.anime2anime_decoded = {}
        
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"✅ DataProcessor initialized | Input: {os.path.basename(self.input_file)} | Output: {self.output_dir}")

    # ... [All other methods remain the same as previous version] ...

    def load_data(self):
        try:
            self.rating_df = pd.read_csv(self.input_file, low_memory=True, usecols=["user_id", "anime_id", "rating"])
            logger.info(f"✅ Loaded {len(self.rating_df):,} ratings")
        except Exception as e:
            raise CustomException(f"Failed to load data: {e}") from e

    def filter_users(self, min_ratings: int = 400):
        try:
            user_counts = self.rating_df["user_id"].value_counts()
            active_users = user_counts[user_counts >= min_ratings].index
            self.rating_df = self.rating_df[self.rating_df["user_id"].isin(active_users)].copy()
            logger.info(f"✅ Filtered to {len(active_users):,} active users")
        except Exception as e:
            raise CustomException(f"Failed to filter users: {e}") from e

    def scale_ratings(self):
        try:
            min_r = self.rating_df["rating"].min()
            max_r = self.rating_df["rating"].max()
            self.rating_df["rating"] = ((self.rating_df["rating"] - min_r) / (max_r - min_r)).astype(np.float64)
            logger.info("✅ Ratings scaled to [0, 1]")
        except Exception as e:
            raise CustomException(f"Failed to scale ratings: {e}") from e

    def encode_data(self):
        try:
            user_ids = self.rating_df["user_id"].unique().tolist()
            self.user2user_encoded = {uid: i for i, uid in enumerate(user_ids)}
            self.user2user_decoded = {i: uid for i, uid in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.user2user_encoded)

            anime_ids = self.rating_df["anime_id"].unique().tolist()
            self.anime2anime_encoded = {aid: i for i, aid in enumerate(anime_ids)}
            self.anime2anime_decoded = {i: aid for i, aid in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.anime2anime_encoded)

            logger.info(f"✅ Encoding completed - Users: {len(user_ids)} | Anime: {len(anime_ids)}")
        except Exception as e:
            raise CustomException(f"Failed to encode data: {e}") from e

    def split_data(self, test_size: int = 1000, random_state: int = 43):
        try:
            self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            X = self.rating_df[["user", "anime"]].values
            y = self.rating_df["rating"].values
            train_end = len(self.rating_df) - test_size

            self.X_train_array = [X[:train_end, 0], X[:train_end, 1]]
            self.X_test_array = [X[train_end:, 0], X[train_end:, 1]]
            self.y_train = y[:train_end]
            self.y_test = y[train_end:]
            logger.info(f"✅ Split done → Train: {len(self.y_train):,} | Test: {len(self.y_test):,}")
        except Exception as e:
            raise CustomException(f"Failed to split data: {e}") from e

    def save_artifacts(self):
        try:
            joblib.dump(self.user2user_encoded, os.path.join(self.output_dir, "user2user_encoded.pkl"))
            joblib.dump(self.user2user_decoded, os.path.join(self.output_dir, "user2user_decoded.pkl"))
            joblib.dump(self.anime2anime_encoded, os.path.join(self.output_dir, "anime2anime_encoded.pkl"))
            joblib.dump(self.anime2anime_decoded, os.path.join(self.output_dir, "anime2anime_decoded.pkl"))

            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)
            self.rating_df.to_csv(RATING_DF, index=False)

            logger.info("✅ All artifacts saved successfully")
        except Exception as e:
            raise CustomException(f"Failed to save artifacts: {e}") from e

    def process_anime_data(self):
        try:
            anime_df = pd.read_csv(ANIME_CSV)
            synopsis_df = pd.read_csv(ANIMESYNOPSIS_CSV, usecols=["MAL_ID", "Name", "Genres", "sypnopsis"])

            anime_df = anime_df.replace("Unknown", np.nan)

            def get_anime_name(mal_id):
                row = anime_df[anime_df["MAL_ID"] == mal_id]
                if not row.empty:
                    eng = row["English name"].iloc[0]
                    return eng if pd.notna(eng) else row["Name"].iloc[0]
                return "Unknown"

            anime_df["anime_id"] = anime_df["MAL_ID"]
            anime_df["eng_version"] = anime_df["MAL_ID"].apply(get_anime_name)
            anime_df = anime_df.sort_values("Score", ascending=False, na_position="last")

            final_df = anime_df[["anime_id", "eng_version", "Score", "Genres", "Episodes", "Type", "Premiered", "Members"]]
            final_df.to_csv(DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)

            logger.info("✅ Anime metadata processed")
        except Exception as e:
            raise CustomException(f"Failed to process anime data: {e}") from e

    def run(self):
        try:
            logger.info("🚀 Starting Data Processing Pipeline...")
            self.load_data()
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()
            logger.info("🎉 Data Processing Pipeline completed successfully!")
        except CustomException as ce:
            logger.error(f"❌ Pipeline failed: {ce}")
            raise


if __name__ == "__main__":
    try:
        processor = DataProcessor()
        processor.run()
    except Exception as e:
        logger.error(f"Failed to start pipeline: {e}")
        sys.exit(1)