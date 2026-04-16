#!/usr/bin/env python

import sys
import time
from datetime import datetime

from src.custom_exception import CustomException
from src.data_processing import DataProcessor
from logger import get_logger
from pipeline.model_training import ModelTrainer
from config.paths_config import ANIMELIST_CSV

logger = get_logger(__name__)


def run_full_pipeline():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 80)
    logger.info(f"🚀 Starting Full MLOps Pipeline - {timestamp}")
    logger.info("=" * 80)

    try:
        logger.info("📊 STAGE 1: Data Processing")
        data_processor = DataProcessor(input_file=ANIMELIST_CSV)
        data_processor.run()

        logger.info("🧠 STAGE 2: Model Training")
        model_trainer = ModelTrainer(
            config_path="config/config.yaml",
            experiment_name=f"anime-recommender-{datetime.now().strftime('%Y%m%d-%H%M')}",
        )
        model_trainer.run(epochs=12, batch_size=8192)

        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)

        logger.info("=" * 80)
        logger.info("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total Execution Time : {minutes} min {seconds} sec")
        logger.info("=" * 80)

    except CustomException as exc:
        logger.error(f"❌ Pipeline failed with CustomException: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"❌ Unexpected error in pipeline: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except KeyboardInterrupt:
        logger.info("⛔ Pipeline interrupted by user")
        sys.exit(1)