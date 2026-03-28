import os

# ==================== DIRECTORIES ====================
ARTIFACTS_DIR = "artifacts"
RAW_DIR = os.path.join(ARTIFACTS_DIR, "raw")
PROCESSED_DIR = os.path.join(ARTIFACTS_DIR, "processed")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "model")
WEIGHTS_DIR = os.path.join(ARTIFACTS_DIR, "weights")

# Create all directories
for directory in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, WEIGHTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== INPUT FILES ====================
ANIMELIST_CSV = os.path.join(RAW_DIR, "animelist.csv")
ANIME_CSV = os.path.join(RAW_DIR, "anime.csv")
ANIMESYNOPSIS_CSV = os.path.join(RAW_DIR, "anime_with_synopsis.csv")

# ==================== PROCESSED DATA ====================
X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, "X_train_array.pkl")
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, "X_test_array.pkl")
Y_TRAIN = os.path.join(PROCESSED_DIR, "y_train.pkl")
Y_TEST = os.path.join(PROCESSED_DIR, "y_test.pkl")
RATING_DF = os.path.join(PROCESSED_DIR, "rating_df.csv")

# ==================== MODEL & WEIGHTS PATHS ====================
MODEL_PATH = os.path.join(MODEL_DIR, "recommender_model.h5")
CHECKPOINT_FILE_PATH = os.path.join(WEIGHTS_DIR, "best_model.weights.h5")

USER_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "user_weights.pkl")
ANIME_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "anime_weights.pkl")

# ==================== METADATA ====================
DF = os.path.join(PROCESSED_DIR, "anime_df.csv")
SYNOPSIS_DF = os.path.join(PROCESSED_DIR, "synopsis_df.csv")

print("✅ All paths configured successfully!")
print(f"   PROCESSED_DIR : {PROCESSED_DIR}")
print(f"   MODEL_PATH    : {MODEL_PATH}")