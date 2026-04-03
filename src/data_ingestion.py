"""
### Data Ingestion Module
# This module is responsible for downloading the raw data files from Google Cloud Storage,
# and preparing the train/test splits for the animelist dataset.
# It is designed to be flexible, allowing for configuration of GCP credentials, file names, and whether to truncate the animelist for faster experimentation.

### File: src/data_ingestion.py 
"""

### File: src/data_ingestion.py

import os
import sys
import warnings
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import RAW_DIR, TRAIN_FILE_PATH, TEST_FILE_PATH
from utils.common_functions import read_yaml_file

# Silence Google auth warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth")

logger = get_logger("data_ingestion")


class DataIngestion:
    """
    Handles data ingestion from GCS and prepares train/test datasets.
    Always processes the FULL dataset (no truncation).
    """

    def __init__(self, config):
        try:
            # Load config if path provided
            if isinstance(config, (str, os.PathLike)):
                logger.info(f"Loading config from: {config}")
                config = read_yaml_file(config)

            self.config = config
            ingestion_cfg = self.config["data_ingestion"]

            # Core parameters
            self.bucket_name = ingestion_cfg["bucket_name"]
            self.bucket_file_names = ingestion_cfg["bucket_file_names"]
            self.train_ratio = ingestion_cfg.get("train_ratio", 0.8)

            # Optional credentials
            self.gcp_credentials = ingestion_cfg.get("gcp_credentials")

            # Ensure raw directory exists
            os.makedirs(RAW_DIR, exist_ok=True)

            logger.info("DataIngestion initialized")
            logger.info(f"Bucket: {self.bucket_name}")
            logger.info(f"Files: {self.bucket_file_names}")
            logger.info(f"Train ratio: {self.train_ratio}")
            logger.info("Mode: FULL DATASET (no truncation)")

        except Exception as e:
            raise CustomException(f"Initialization failed: {e}") from e

    def _get_gcp_client(self):
        """Initialize GCP Storage client."""
        try:
            if self.gcp_credentials and os.path.exists(self.gcp_credentials):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.gcp_credentials
                logger.info("Using provided GCP credentials")

            return storage.Client()

        except Exception as e:
            raise CustomException(f"GCP client init failed: {e}") from e

    def download_data_from_gcs(self):
        """Download files from GCS bucket."""
        try:
            logger.info("Connecting to GCS...")
            client = self._get_gcp_client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.bucket_file_names:
                local_path = os.path.join(RAW_DIR, file_name)

                logger.info(f"Downloading: {file_name}")
                blob = bucket.blob(file_name)
                blob.download_to_filename(local_path)

                logger.info(f"Downloaded: {file_name}")

        except Exception as e:
            raise CustomException(f"GCS download failed: {e}") from e

    def split_data_into_train_test(self):
        """Split dataset into train/test."""
        try:
            raw_file = os.path.join(RAW_DIR, "animelist.csv")

            if not os.path.exists(raw_file):
                raise FileNotFoundError(f"Missing file: {raw_file}")

            logger.info("Reading FULL dataset...")
            df = pd.read_csv(raw_file)

            logger.info(f"Total rows loaded: {len(df):,}")

            train_df, test_df = train_test_split(
                df,
                test_size=1 - self.train_ratio,
                random_state=42
            )

            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train rows: {len(train_df):,}")
            logger.info(f"Test rows: {len(test_df):,}")

        except Exception as e:
            raise CustomException(f"Train/test split failed: {e}") from e

    def run(self):
        """Execute full pipeline."""
        try:
            logger.info("Starting Data Ingestion Pipeline")
            self.download_data_from_gcs()
            self.split_data_into_train_test()
            logger.info("Pipeline completed successfully")

        except Exception as e:
            raise CustomException(f"Pipeline failed: {e}") from e


if __name__ == "__main__":
    try:
        ingestion = DataIngestion(config="config/config.yaml")
        ingestion.run()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)