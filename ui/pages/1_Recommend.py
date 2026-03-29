import sys
from pathlib import Path

# ✅ Auto-detect the project root (folder that contains "ui", "api", "pipeline", etc.)
CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parents[2]    # MLProject2

# ✅ Add root to sys.path for imports like "ui.xxx" and "config.xxx"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
import pandas as pd
import requests
from ui.jikan_client import get_poster_url
from ui.utils_ui import anime_card, explanation_card
from config.paths_config import DF

API_URL = "http://localhost:8000"

# ---------------------------------------------------
# ✅ Load Metadata
# ---------------------------------------------------
@st.cache_data
def load_anime_df():
    return pd.read_csv(DF)

anime_df = load_anime_df()


# ---------------------------------------------------
# ✅ Token Retrieval
# ---------------------------------------------------
def get_token():
    try:
        r = requests.post(
            f"{API_URL}/auth/login",
            data={"username": "demo", "password": "demo"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return r.json().get("access_token")
    except:
        return None


# ---------------------------------------------------
# ✅ API Call for Recommendations
# ---------------------------------------------------
def get_recommendations(user_id, token, user_weight, content_weight, top_k):
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "user_weight": user_weight,
        "content_weight": content_weight,
        "top_k": top_k
    }

    r = requests.get(f"{API_URL}/recommend/{user_id}", headers=headers, params=params)

    if r.status_code == 200:
        return r.json()
    else:
        st.error(f"❌ API Error: {r.text}")
        return {}


# ---------------------------------------------------
# ✅ PAGE UI
# ---------------------------------------------------
st.title("🎯 Anime Recommendations")
st.caption("FAISS • Hybrid • Embeddings • MMR • Explanations")

st.sidebar.header("⚙️ Recommendation Settings")
user_weight = st.sidebar.slider("User-based Weight", 0.0, 1.0, 0.6)
content_weight = st.sidebar.slider("Content-based Weight", 0.0, 1.0, 0.4)
top_k = st.sidebar.slider("Top-K", 5, 25, 10)

st.write("### Enter User ID")
user_id = st.number_input("User ID:", min_value=1, value=11880)

if st.button("🎌 Get Recommendations", use_container_width=True):

    token = get_token()
    if not token:
        st.error("❌ Authentication failed. Cannot access API.")
        st.stop()

    with st.spinner("Fetching recommendations..."):
        result = get_recommendations(
            user_id, token, user_weight, content_weight, top_k
        )

    if not result:
        st.warning("No results returned.")
        st.stop()

    recs = result.get("recommendations", [])
    explanations = result.get("explanations", {})

    st.subheader(f"✅ Top {len(recs)} Recommendations")
    st.write("---")

    cols = st.columns(3)
    for i, anime_name in enumerate(recs):
        with cols[i % 3]:
            anime_card(anime_name, anime_df)

            # Find the anime_id
            try:
                anime_id = int(
                    anime_df[anime_df["eng_version"] == anime_name]["anime_id"].values[0]
                )
            except:
                anime_id = None

            exp = explanations.get(str(anime_id), explanations.get(anime_id, {}))
            explanation_card(exp)