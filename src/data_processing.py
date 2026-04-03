"""
File: src/data_processor.py
Purpose: Processes Anime rating and metadata datasets for model training.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Tuple

# ====================== FIX PATHS FOR RELATIVE IMPORTS ======================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from logger import get_logger
from custom_exception import CustomException
from config.paths_config import (
    ANIMELIST_CSV,
    ANIME_CSV,
    ANIMESYNOPSIS_CSV,
    PROCESSED_DIR,
    X_TRAIN_ARRAY,
    X_TEST_ARRAY,
    Y_TRAIN,
    Y_TEST,
    RATING_DF,
    DF,
    SYNOPSIS_DF,
)

logger = get_logger(__name__)


class DataProcessor:
    """
    Handles rating preprocessing, train-test splitting,
    encoding mappings, and anime metadata processing.
    """

    def __init__(self, input_file: Optional[str] = None, output_dir: Optional[str] = None):
        self.input_file = input_file or ANIMELIST_CSV
        self.output_dir = output_dir or PROCESSED_DIR

        os.makedirs(self.output_dir, exist_ok=True)

        # Internal containers
        self.rating_df: Optional[pd.DataFrame] = None
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None

        self.user2user_encoded: Dict[int, int] = {}
        self.user2user_decoded: Dict[int, int] = {}
        self.anime2anime_encoded: Dict[int, int] = {}
        self.anime2anime_decoded: Dict[int, int] = {}

        logger.info(f"✅ DataProcessor initialized")
        logger.info(f"   Input file     : {self.input_file}")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Processed dir exists: {os.path.exists(self.output_dir)}")


    def load_data(self):
        """Loads rating data from animelist.csv."""
        try:
            logger.info(f"📥 Loading ratings from: {self.input_file}")
            self.rating_df = pd.read_csv(
                self.input_file,
                low_memory=True,
                usecols=["user_id", "anime_id", "rating"]
            )
            logger.info(f"✅ Loaded {len(self.rating_df):,} rating records successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load rating data from {self.input_file}")
            raise CustomException(f"Failed to load rating data: {e}") from e


    def filter_users(self, min_ratings: int = 400):
        """Removes users with insufficient interactions."""
        try:
            logger.info(f"🔍 Filtering users with minimum {min_ratings} ratings...")
            user_counts = self.rating_df["user_id"].value_counts()
            active_users = user_counts[user_counts >= min_ratings].index

            before = len(self.rating_df)
            self.rating_df = self.rating_df[self.rating_df["user_id"].isin(active_users)].copy()
            after = len(self.rating_df)

            logger.info(f"✅ User filtering completed")
            logger.info(f"   Active users   : {len(active_users):,}")
            logger.info(f"   Rows before    : {before:,}")
            logger.info(f"   Rows after     : {after:,} ({(after/before)*100:.1f}% retained)")
        except Exception as e:
            raise CustomException(f"Failed to filter users: {e}") from e


    def scale_ratings(self):
        """Normalizes ratings to [0, 1] range."""
        try:
            logger.info("📊 Scaling ratings to [0, 1] range...")
            min_r = self.rating_df["rating"].min()
            max_r = self.rating_df["rating"].max()

            if max_r == min_r:
                raise CustomException("All ratings have identical values. Cannot scale.")

            self.rating_df["rating"] = (
                (self.rating_df["rating"] - min_r) / (max_r - min_r)
            ).astype(np.float64)

            logger.info("✅ Ratings successfully scaled to [0, 1]")
        except Exception as e:
            raise CustomException(f"Failed to scale ratings: {e}") from e


    def encode_data(self):
        """Creates integer encoding for users and anime IDs."""
        try:
            logger.info("🔢 Creating user and anime ID encodings...")

            # User encoding
            user_ids = sorted(self.rating_df["user_id"].unique())
            self.user2user_encoded = {uid: idx for idx, uid in enumerate(user_ids)}
            self.user2user_decoded = {idx: uid for idx, uid in enumerate(user_ids)}
            self.rating_df["user"] = self.rating_df["user_id"].map(self.user2user_encoded)

            # Anime encoding
            anime_ids = sorted(self.rating_df["anime_id"].unique())
            self.anime2anime_encoded = {aid: idx for idx, aid in enumerate(anime_ids)}
            self.anime2anime_decoded = {idx: aid for idx, aid in enumerate(anime_ids)}
            self.rating_df["anime"] = self.rating_df["anime_id"].map(self.anime2anime_encoded)

            logger.info(f"✅ Encoding completed successfully")
            logger.info(f"   Unique users   : {len(user_ids):,}")
            logger.info(f"   Unique animes  : {len(anime_ids):,}")
        except Exception as e:
            raise CustomException(f"Encoding failed: {e}") from e


    def split_data(self, test_size: int = 1000, random_state: int = 42):
        """Creates train-test splits from encoded rating data."""
        try:
            logger.info(f"✂️  Splitting data into train/test (test_size={test_size})...")
            self.rating_df = self.rating_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

            X = self.rating_df[["user", "anime"]].values
            y = self.rating_df["rating"].values

            train_end = len(self.rating_df) - test_size

            self.X_train_array = [X[:train_end, 0], X[:train_end, 1]]
            self.X_test_array = [X[train_end:, 0], X[train_end:, 1]]
            self.y_train = y[:train_end]
            self.y_test = y[train_end:]

            logger.info(f"✅ Data split completed")
            logger.info(f"   Training samples : {len(self.y_train):,}")
            logger.info(f"   Test samples     : {len(self.y_test):,}")
        except Exception as e:
            raise CustomException(f"Train-test split failed: {e}") from e


    def save_artifacts(self):
        """Persists processed artifacts + encoding mappings."""
        try:
            logger.info("💾 Saving processed artifacts...")

            # Save encoding mappings
            joblib.dump(self.user2user_encoded, os.path.join(self.output_dir, "user2user_encoded.pkl"))
            joblib.dump(self.user2user_decoded, os.path.join(self.output_dir, "user2user_decoded.pkl"))
            joblib.dump(self.anime2anime_encoded, os.path.join(self.output_dir, "anime2anime_encoded.pkl"))
            joblib.dump(self.anime2anime_decoded, os.path.join(self.output_dir, "anime2anime_decoded.pkl"))

            # Save arrays
            joblib.dump(self.X_train_array, X_TRAIN_ARRAY)
            joblib.dump(self.X_test_array, X_TEST_ARRAY)
            joblib.dump(self.y_train, Y_TRAIN)
            joblib.dump(self.y_test, Y_TEST)

            # Save rating dataframe
            self.rating_df.to_csv(RATING_DF, index=False)

            logger.info("✅ All artifacts saved successfully!")
            logger.info(f"   Location: {self.output_dir}")
            logger.info(f"   Files saved:")
            logger.info(f"     - rating_df.csv")
            logger.info(f"     - user2user_encoded/decoded.pkl")
            logger.info(f"     - anime2anime_encoded/decoded.pkl")
            logger.info(f"     - X_train_array.pkl, y_train.pkl, etc.")

        except Exception as e:
            logger.error(f"❌ Failed to save artifacts to {self.output_dir}")
            raise CustomException(f"Failed to save artifacts: {e}") from e


    def process_anime_data(self):
        """Cleans metadata and merges with synopsis."""
        try:
            logger.info("🎬 Processing anime metadata and synopsis...")

            anime_df = pd.read_csv(ANIME_CSV)
            synopsis_df = pd.read_csv(ANIMESYNOPSIS_CSV)

            logger.info(f"   Loaded anime.csv: {len(anime_df):,} rows")
            logger.info(f"   Loaded anime_with_synopsis.csv: {len(synopsis_df):,} rows")

            anime_df = anime_df.replace({"Unknown": np.nan})

            def get_name(mal_id):
                row = anime_df.loc[anime_df["MAL_ID"] == mal_id]
                if row.empty:
                    return "Unknown"
                eng = row["English name"].iloc[0]
                return eng if pd.notna(eng) else row["Name"].iloc[0]

            anime_df["anime_id"] = anime_df["MAL_ID"]
            anime_df["eng_version"] = anime_df["MAL_ID"].apply(get_name)

            refined = anime_df.sort_values("Score", ascending=False)

            clean_cols = ["anime_id", "eng_version", "Score", "Genres", "Episodes", "Type", "Premiered", "Members"]
            refined[clean_cols].to_csv(DF, index=False)
            synopsis_df.to_csv(SYNOPSIS_DF, index=False)

            logger.info("✅ Anime metadata processing completed")
            logger.info(f"   Saved: {DF}")
            logger.info(f"   Saved: {SYNOPSIS_DF}")

        except Exception as e:
            logger.error("❌ Failed to process anime metadata")
            raise CustomException(f"Failed to process anime metadata: {e}")


    def run(self):
        """End-to-end processing pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("🚀 STARTING FULL DATA PROCESSING PIPELINE")
            logger.info("=" * 60)

            self.load_data()
            self.filter_users()
            self.scale_ratings()
            self.encode_data()
            self.split_data()
            self.save_artifacts()
            self.process_anime_data()

            logger.info("=" * 60)
            logger.info("🎉 DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"   All files saved in: {PROCESSED_DIR}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("❌ Pipeline failed!")
            logger.error(str(e))
            raise CustomException(e) from e


# =============================================================================
# ✅ Runner
# =============================================================================
if __name__ == "__main__":
    try:
        processor = DataProcessor()
        processor.run()
    except Exception as e:
        logger.error(f"💥 Fatal pipeline failure: {e}")
        sys.exit(1)