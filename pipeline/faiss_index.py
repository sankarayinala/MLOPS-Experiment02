# pipeline/faiss_index.py

import faiss
import numpy as np
import joblib
from config.paths_config import *

def build_faiss_index(path_anime_weights):
    vectors = joblib.load(path_anime_weights).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity
    index.add(vectors)
    return index

FAISS_ANIME_INDEX = build_faiss_index(ANIME_WEIGHTS_PATH)