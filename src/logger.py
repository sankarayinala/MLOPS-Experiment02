import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Environment-driven configs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
MAX_LOG_SIZE = int(os.getenv("LOG_MAX_SIZE", 5 * 1024 * 1024))  # 5 MB
BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

# Create logs directory
os.makedirs(LOG_DIR, exist_ok=True)

# Timestamped log file (per run)
LOG_FILE_NAME = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Improved format (adds file + line number)
LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a production-grade logger.

    Features:
    - Console + Rotating file logging
    - Environment-based configuration
    - Idempotent handler setup
    - Detailed traceability (file + line number)
    """

    logger = logging.getLogger(name)

    # Prevent duplicate handlers (critical in pipelines / notebooks)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    # ---------------------------
    # File Handler (Rotating)
    # ---------------------------
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(LOG_LEVEL)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"[LOGGER WARNING] File handler failed: {e}")

    # ---------------------------
    # Console Handler
    # ---------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(LOG_LEVEL)
    logger.addHandler(console_handler)

    # Avoid propagation to root logger (prevents duplicate logs in frameworks)
    logger.propagate = False

    return logger