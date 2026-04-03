import os

# =============================================
# ✅ FastAPI Backend Configuration (for UI)
# =============================================
# This is used by the Streamlit frontend to call the FastAPI backend
API_HOST = os.getenv("API_HOST", "backend-service")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"

# =============================================
# ✅ Streamlit / UI Configuration
# =============================================
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))
STREAMLIT_HOST = os.getenv("STREAMLIT_HOST", "0.0.0.0")   # Changed to 0.0.0.0 for Kubernetes

# =============================================
# ✅ Authentication (Demo credentials)
# =============================================
LOGIN_USER = os.getenv("LOGIN_USER", "demo")
LOGIN_PASSWORD = os.getenv("LOGIN_PASSWORD", "demo")

# =============================================
# ✅ Redis Configuration
# =============================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")           # Use service name in K8s
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# =============================================
# ✅ FAISS Configuration
# =============================================
FAISS_DIM = int(os.getenv("FAISS_DIM", "128"))

# =============================================
# ✅ Debug / Environment Info
# =============================================
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# Optional: Print config on startup (helpful for debugging in K8s)
if __name__ == "__main__" or os.getenv("DEBUG_CONFIG", "false").lower() == "true":
    print("=== App Configuration Loaded ===")
    print(f"API_URL          : {API_URL}")
    print(f"REDIS_HOST       : {REDIS_HOST}:{REDIS_PORT}")
    print(f"STREAMLIT_HOST   : {STREAMLIT_HOST}:{STREAMLIT_PORT}")
    print(f"ENVIRONMENT      : {ENVIRONMENT}")
    print("===============================")