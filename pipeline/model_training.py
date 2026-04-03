"""
File: pipeline/model_training.py
Purpose: Trains the Anime Recommendation Neural Collaborative Filtering model with enhanced logging and config support.
"""

import os
import sys

# Suppress TensorFlow oneDNN warning (must be before importing tensorflow)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import comet_ml
import joblib
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from src.logger import get_logger
from src.custom_exception import CustomException
from src.base_model import BaseModel
from utils.common_functions import read_yaml_file
from config.paths_config import *

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml", experiment_name: str = None):
        self.config_path = config_path
        self.experiment = None
        self.config = None
        self.comet_config = None

        # Default hyperparameters (will be overridden by config if available)
        self.epochs = 12
        self.batch_size = 8192
        self.patience = 4

        try:
            # Load configuration
            logger.info(f"📄 Loading configuration from: {config_path}")
            self.config = read_yaml_file(config_path)

            # Extract model configuration
            model_config = self.config.get("model", {})
            self.epochs = model_config.get("epochs", self.epochs)
            self.batch_size = model_config.get("batch_size", self.batch_size)
            self.patience = model_config.get("patience", self.patience)

            self.comet_config = self.config.get("comet_ml", {})

            # Determine experiment name
            exp_name = experiment_name or self.comet_config.get("experiment_name", "anime-recommender-v2")

            # Initialize Comet ML (optional)
            if self.comet_config.get("api_key") and self.comet_config.get("enabled", True):
                self.experiment = comet_ml.Experiment(
                    api_key=self.comet_config.get("api_key"),
                    project_name=self.comet_config.get("project_name", "anime-recommender"),
                    workspace=self.comet_config.get("workspace"),
                )
                self.experiment.set_name(exp_name)
                logger.info(f"✅ Comet ML experiment started: {exp_name}")
                logger.info(f"🔗 Experiment URL: {self.experiment.url}")
            else:
                logger.warning("⚠️ Comet ML tracking is disabled or API key not found.")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize Comet ML or load config: {e}. Continuing without tracking.")
            self.experiment = None

        # Data attributes
        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None
        self.n_users = None
        self.n_anime = None
        self.model = None
        self.history = None

        logger.info("✅ ModelTrainer initialized successfully")
        logger.info(f"   Config path     : {config_path}")
        logger.info(f"   Epochs          : {self.epochs}")
        logger.info(f"   Batch size      : {self.batch_size}")
        logger.info(f"   Patience        : {self.patience}")


    def load_data(self):
        """Load preprocessed training data"""
        try:
            logger.info("📥 Loading processed training data...")

            self.X_train_array = joblib.load(X_TRAIN_ARRAY)
            self.X_test_array = joblib.load(X_TEST_ARRAY)
            self.y_train = joblib.load(Y_TRAIN)
            self.y_test = joblib.load(Y_TEST)

            # Load encoders to get vocabulary sizes
            user_encoded = joblib.load(os.path.join(PROCESSED_DIR, "user2user_encoded.pkl"))
            anime_encoded = joblib.load(os.path.join(PROCESSED_DIR, "anime2anime_encoded.pkl"))

            self.n_users = len(user_encoded)
            self.n_anime = len(anime_encoded)

            logger.info("✅ Training data loaded successfully")
            logger.info(f"   Number of users : {self.n_users:,}")
            logger.info(f"   Number of animes: {self.n_anime:,}")
            logger.info(f"   X_train shape   : {self.X_train_array[0].shape}")

        except FileNotFoundError as e:
            logger.error(f"❌ Missing processed file: {e}")
            raise CustomException(f"Data file not found. Please run data processing first: {e}") from e
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {e}")
            raise CustomException(f"Data loading failed: {e}") from e


    def get_lr_scheduler(self):
        """Learning rate scheduler with warm-up and decay"""
        def lrfn(epoch):
            start_lr = 1e-5
            max_lr = 5e-4
            min_lr = 1e-6
            rampup = 5
            sustain = 2
            exp_decay = 0.8

            if epoch < rampup:
                return (max_lr - start_lr) / rampup * epoch + start_lr
            elif epoch < rampup + sustain:
                return max_lr
            else:
                return (max_lr - min_lr) * exp_decay ** (epoch - rampup - sustain) + min_lr

        return LearningRateScheduler(lrfn, verbose=0)


    def train_model(self, epochs: int = None, batch_size: int = None, patience: int = None):
        """Main training logic"""
        # Override defaults with passed values or config
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        patience = patience or self.patience

        try:
            self.load_data()

            logger.info("🏗️ Building RecommenderNet model...")
            base_model = BaseModel(config_path=self.config_path)
            self.model = base_model.RecommenderNet(n_users=self.n_users, n_anime=self.n_anime)

            logger.info(f"🚀 Starting model training:")
            logger.info(f"   Epochs      : {epochs}")
            logger.info(f"   Batch size  : {batch_size}")
            logger.info(f"   Patience    : {patience}")

            callbacks = [
                ModelCheckpoint(
                    filepath=CHECKPOINT_FILE_PATH,
                    save_weights_only=True,
                    monitor="val_loss",
                    mode="min",
                    save_best_only=True,
                    verbose=1
                ),
                EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                self.get_lr_scheduler()
            ]

            self.history = self.model.fit(
                x=self.X_train_array,
                y=self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(self.X_test_array, self.y_test),
                callbacks=callbacks
            )

            # Load best weights
            self.model.load_weights(CHECKPOINT_FILE_PATH)
            logger.info("✅ Best model weights loaded successfully")

            self._log_to_comet()
            self.save_model_and_weights()

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise CustomException(f"Model training failed: {e}") from e


    def _log_to_comet(self):
        """Log training metrics to Comet ML"""
        if self.experiment is None or self.history is None:
            return

        try:
            logger.info("📊 Logging metrics to Comet ML...")
            for epoch in range(len(self.history.history.get('loss', []))):
                self.experiment.log_metric("train_loss", self.history.history["loss"][epoch], step=epoch)
                self.experiment.log_metric("val_loss", self.history.history.get("val_loss", [0])[epoch], step=epoch)
                self.experiment.log_metric("train_mae", self.history.history.get("mae", [0])[epoch], step=epoch)
                if "mse" in self.history.history:
                    self.experiment.log_metric("train_mse", self.history.history["mse"][epoch], step=epoch)

            logger.info("✅ All training metrics successfully logged to Comet ML")
        except Exception as e:
            logger.warning(f"⚠️ Failed to log metrics to Comet ML: {e}")


    def extract_weights(self, layer_name: str):
        """Extract and L2-normalize embedding weights"""
        try:
            weights = self.model.get_layer(layer_name).get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
            logger.info(f"✅ Extracted and normalized weights for layer: {layer_name}")
            return weights
        except Exception as e:
            raise CustomException(f"Failed to extract {layer_name} weights") from e


    def save_model_and_weights(self):
        """Save model and embedding weights"""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            self.model.save(MODEL_PATH)
            logger.info(f"💾 Full model saved: {MODEL_PATH}")

            user_weights = self.extract_weights("user_embedding")
            anime_weights = self.extract_weights("anime_embedding")

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            logger.info("✅ Embedding weights saved successfully:")
            logger.info(f"   → {USER_WEIGHTS_PATH}")
            logger.info(f"   → {ANIME_WEIGHTS_PATH}")

            if self.experiment:
                self.experiment.log_asset(MODEL_PATH)
                self.experiment.log_asset(USER_WEIGHTS_PATH)
                self.experiment.log_asset(ANIME_WEIGHTS_PATH)
                logger.info("📤 Model artifacts uploaded to Comet ML")

        except Exception as e:
            logger.error(f"❌ Failed to save model and weights: {e}")
            raise CustomException(f"Model saving failed: {e}") from e


    def run(self, epochs: int = None, batch_size: int = None, patience: int = None):
        """Run the complete training pipeline"""
        try:
            logger.info("=" * 80)
            logger.info("🚀 STARTING FULL MODEL TRAINING PIPELINE")
            logger.info("=" * 80)

            self.train_model(epochs=epochs, batch_size=batch_size, patience=patience)

            logger.info("=" * 80)
            logger.info("🎉 MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)

            if self.experiment:
                self.experiment.end()
                logger.info(f"✅ Comet ML experiment ended. View results at: {self.experiment.url}")

        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            if self.experiment:
                self.experiment.end()
            raise


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        trainer.run(epochs=None, batch_size=None, patience=None)   # Use values from config
    except Exception as e:
        logger.error(f"💥 Failed to start ModelTrainer: {e}")
        sys.exit(1)