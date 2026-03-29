import os

# ✅ FastAPI Backend Config
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"

# ✅ Streamlit Config
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "localhost")

# ✅ Login Credentials (Demo Only, Replace for Prod)
LOGIN_USER = os.getenv("LOGIN_USER", "demo")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "demo")

# ✅ Redis Config
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# ✅ FAISS Config
FAISS_DIM = int(os.getenv("FAISS_DIM", "128"))  # adjust based on embeddings

# ✅ Paths (Loaded from paths_config)
# These use config.paths_config so kept separate