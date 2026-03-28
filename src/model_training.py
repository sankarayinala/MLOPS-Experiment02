import os
import sys

# === MUST BE IMPORTED FIRST ===
import comet_ml

import joblib
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from logger import get_logger
from custom_exception import CustomException
from base_model import BaseModel
from config.paths_config import *

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml", experiment_name: str = "anime-recommender-v2"):
        self.config_path = config_path
        self.experiment = None

        try:
            self.experiment = comet_ml.Experiment(
                api_key="kA1fB80gcoco5B1tsjnM18zSf",
                project_name="mlops-course-2",
                workspace="sankar-bmc"
            )
            self.experiment.set_name(experiment_name)
            logger.info(f"✅ Comet ML experiment started: {experiment_name}")
            logger.info(f"🔗 Live at: {self.experiment.url}")
        except Exception as e:
            logger.warning(f"Could not initialize Comet ML: {e}")
            self.experiment = None

        self.X_train_array = None
        self.X_test_array = None
        self.y_train = None
        self.y_test = None
        self.n_users = None
        self.n_anime = None
        self.model = None
        self.history = None

        logger.info("✅ ModelTrainer initialized")

    def load_data(self):
        try:
            logger.info("Loading processed training data...")
            self.X_train_array = joblib.load(X_TRAIN_ARRAY)
            self.X_test_array = joblib.load(X_TEST_ARRAY)
            self.y_train = joblib.load(Y_TRAIN)
            self.y_test = joblib.load(Y_TEST)

            user_path = os.path.join(PROCESSED_DIR, "user2user_encoded.pkl")
            anime_path = os.path.join(PROCESSED_DIR, "anime2anime_encoded.pkl")

            user_encoded = joblib.load(user_path)
            anime_encoded = joblib.load(anime_path)

            self.n_users = len(user_encoded)
            self.n_anime = len(anime_encoded)

            logger.info(f"✅ Data loaded | Users: {self.n_users:,} | Anime: {self.n_anime:,}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise CustomException(f"Data loading failed: {e}") from e

    def get_lr_scheduler(self):
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

    def train_model(self, epochs: int = 12, batch_size: int = 8192, patience: int = 4):
        try:
            self.load_data()

            logger.info("Building model using BaseModel...")
            base_model = BaseModel(config_path=self.config_path)
            self.model = base_model.RecommenderNet(n_users=self.n_users, n_anime=self.n_anime)

            logger.info(f"🚀 Starting training for {epochs} epochs (batch_size={batch_size})...")

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

            self.model.load_weights(CHECKPOINT_FILE_PATH)
            logger.info("✅ Best weights loaded.")

            self._log_to_comet()
            self.save_model_and_weights()

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise CustomException(f"Model training failed: {e}") from e

    def _log_to_comet(self):
        if self.experiment is None or self.history is None:
            return
        try:
            for epoch in range(len(self.history.history.get('loss', []))):
                self.experiment.log_metric("train_loss", self.history.history["loss"][epoch], step=epoch)
                self.experiment.log_metric("val_loss", self.history.history.get("val_loss", [0])[epoch], step=epoch)
                self.experiment.log_metric("train_mae", self.history.history.get("mae", [0])[epoch], step=epoch)
            logger.info("✅ Metrics logged to Comet ML")
        except Exception as e:
            logger.warning(f"Comet ML logging failed: {e}")

    def extract_weights(self, layer_name: str):
        try:
            weights = self.model.get_layer(layer_name).get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
            return weights
        except Exception as e:
            raise CustomException(f"Failed to extract {layer_name} weights") from e

    def save_model_and_weights(self):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            self.model.save(MODEL_PATH)
            logger.info(f"✅ Model saved: {MODEL_PATH}")

            user_weights = self.extract_weights("user_embedding")
            anime_weights = self.extract_weights("anime_embedding")

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            logger.info(f"✅ Weights saved → User: {USER_WEIGHTS_PATH} | Anime: {ANIME_WEIGHTS_PATH}")

            if self.experiment:
                self.experiment.log_asset(MODEL_PATH)
                self.experiment.log_asset(USER_WEIGHTS_PATH)
                self.experiment.log_asset(ANIME_WEIGHTS_PATH)

        except Exception as e:
            logger.error(f"Failed to save model/weights: {e}")
            raise CustomException(f"Failed to save model: {e}") from e

    def run(self, epochs: int = 12, batch_size: int = 8192):
        try:
            logger.info("🚀 Starting full Model Training Pipeline...")
            self.train_model(epochs=epochs, batch_size=batch_size)
            logger.info("🎉 Model Training Pipeline completed successfully!")
            
            if self.experiment:
                self.experiment.end()
                logger.info(f"✅ Experiment ended. View final results at: {self.experiment.url}")
                
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if self.experiment:
                self.experiment.end()
            raise
            raise


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        trainer.run(epochs=10, batch_size=8192)
    except Exception as e:
        logger.error(f"Failed to start ModelTrainer: {e}")
        sys.exit(1)