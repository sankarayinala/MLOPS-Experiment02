import os
import sys
import warnings

# Suppress Google auth warning (optional but clean)
warnings.filterwarnings("ignore", category=UserWarning, module="google.auth")

# === CRITICAL FIX FOR YOUR SETUP ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Imports
from logger import get_logger
from custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml_file

import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split


logger = get_logger(__name__)


class DataIngestion:
    def __init__(self, config):
        try:
            if isinstance(config, (str, os.PathLike)):
                logger.info(f"Loading config from: {config}")
                config = read_yaml_file(config)

            self.config = config
            self.bucket_name = self.config['data_ingestion']['bucket_name']
            self.bucket_file_names = self.config['data_ingestion']['bucket_file_names']
            self.train_ratio = self.config['data_ingestion']['train_ratio']

            os.makedirs(RAW_DIR, exist_ok=True)

            logger.info("✅ DataIngestion class initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing DataIngestion: {e}")
            raise CustomException(f"Error initializing DataIngestion: {e}") from e


    def download_data_from_gcs(self):
        try:
            logger.info("🌐 Connecting to Google Cloud Storage...")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.bucket_file_names:
                file_path = os.path.join(RAW_DIR, file_name)

                logger.info(f"Downloading {file_name} from bucket '{self.bucket_name}'...")
                blob = bucket.blob(file_name)
                blob.download_to_filename(file_path)

                if file_name == "animelist.csv":
                    logger.info(f"Truncating animelist.csv to 5,000,000 rows...")
                    data = pd.read_csv(file_path, nrows=5_000_000)
                    data.to_csv(file_path, index=False)
                    logger.info(f"✅ Successfully downloaded and truncated {file_name} to {len(data):,} rows")
                else:
                    logger.info(f"✅ Successfully downloaded {file_name}")

        except Exception as e:
            logger.error(f"Error downloading data from GCS: {e}")
            raise CustomException(f"Error downloading data from GCS: {e}") from e


    def split_data_into_train_test(self):
        try:
            raw_file_path = os.path.join(RAW_DIR, "animelist.csv")

            if not os.path.exists(raw_file_path):
                raise FileNotFoundError(f"Raw file not found: {raw_file_path}")

            logger.info(f"Reading {raw_file_path} for train-test split...")

            df = pd.read_csv(raw_file_path)

            train_df, test_df = train_test_split(
                df,
                test_size=1 - self.train_ratio,
                random_state=42
            )

            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"✅ Train-Test split completed!")
            logger.info(f"   Train set: {len(train_df):,} rows → {TRAIN_FILE_PATH}")
            logger.info(f"   Test set : {len(test_df):,} rows → {TEST_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error during train-test split: {e}")
            raise CustomException(f"Error during train-test split: {e}") from e


if __name__ == "__main__":
    try:
        config_path = "config/config.yaml"

        logger.info("🚀 Starting Data Ingestion Pipeline...")

        ingestion = DataIngestion(config=config_path)

        ingestion.download_data_from_gcs()
        ingestion.split_data_into_train_test()

        logger.info("🎉 Data Ingestion Pipeline completed successfully!")

    except CustomException as ce:
        logger.error(f"❌ Pipeline failed with CustomException: {ce}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise