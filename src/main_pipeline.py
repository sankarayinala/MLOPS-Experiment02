#!/usr/bin/env python
"""
Main Pipeline for Anime Recommendation System
Runs the full MLOps pipeline:
1. Data Processing
2. Model Training
"""

import sys
import time
from datetime import datetime

from logger import get_logger
from custom_exception import CustomException

# Import pipeline components
from data_processing import DataProcessor
from model_training import ModelTrainer
from config.paths_config import ANIMELIST_CSV

logger = get_logger(__name__)


def run_full_pipeline():
    """Run the complete end-to-end pipeline"""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info("=" * 80)
    logger.info(f"🚀 Starting Full MLOps Pipeline - {timestamp}")
    logger.info("=" * 80)

    try:
        # ==================== STAGE 1: DATA PROCESSING ====================
        logger.info("\n📊 STAGE 1: Data Processing")
        logger.info("-" * 50)

        data_processor = DataProcessor(input_file=ANIMELIST_CSV)
        data_processor.run()

        logger.info("✅ Stage 1 (Data Processing) completed successfully")

        # ==================== STAGE 2: MODEL TRAINING ====================
        logger.info("\n🧠 STAGE 2: Model Training")
        logger.info("-" * 50)

        model_trainer = ModelTrainer(
            config_path="config/config.yaml",
            experiment_name=f"anime-recommender-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        
        # You can adjust epochs and batch_size here
        model_trainer.run(epochs=12, batch_size=8192)

        logger.info("✅ Stage 2 (Model Training) completed successfully")

        # ==================== PIPELINE SUMMARY ====================
        total_time = time.time() - start_time
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)

        logger.info("=" * 80)
        logger.info("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total Execution Time : {minutes} min {seconds} sec")
        logger.info(f"Started at           : {timestamp}")
        logger.info(f"Finished at          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)

    except CustomException as ce:
        logger.error(f"❌ Pipeline failed with CustomException: {ce}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error in pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        run_full_pipeline()
    except KeyboardInterrupt:
        logger.info("⛔ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical failure: {e}")
        sys.exit(1)