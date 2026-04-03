# pipeline/faiss_index.py

"""
?? Experimental / fallback FAISS index builder.
Not used in production pipeline (HNSW is used instead).
"""

import faiss
import joblib
from config.paths_config import ANIME_WEIGHTS_PATH


def build_flat_index(path_anime_weights):
    vectors = joblib.load(path_anime_weights).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # brute-force cosine
    index.add(vectors)
    return index


if __name__ == "__main__":
    print("?? Building Flat FAISS index (for testing only)")
    index = build_flat_index(ANIME_WEIGHTS_PATH)
    print(f"Index size: {index.ntotal}")