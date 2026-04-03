""" File: ui/pages/1_Recommend.py"""
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

CURRENT = Path(__file__).resolve()
APP_ROOT = CURRENT.parent.parent  # /app inside container

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from utils_ui import anime_card, explanation_card

##API_URL = "http://backend-service:8000"
#API_URL = "http://backend-service.default.svc.cluster.local:8000"
API_URL = "http://10.108.90.164:8000"

DF = APP_ROOT / "data" / "anime_metadata.csv"


@st.cache_data
def load_anime_df():
    if not DF.exists():
        st.error(f"❌ Metadata file not found: {DF}")
        st.stop()
    return pd.read_csv(DF)


anime_df = load_anime_df()


def get_token():
    try:
        r = requests.post(
            f"{API_URL}/auth/login",
            data={"username": "demo", "password": "demo"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        st.error(f"❌ Authentication failed: {e}")
        return None


def get_recommendations(user_id, token, user_weight, content_weight, top_k):
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "user_weight": user_weight,
        "content_weight": content_weight,
        "top_k": top_k,
    }

    try:
        r = requests.get(
            f"{API_URL}/recommend/{user_id}",
            headers=headers,
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ API Error: {e}")
        return {}


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
        st.stop()

    with st.spinner("Fetching recommendations..."):
        result = get_recommendations(
            user_id=user_id,
            token=token,
            user_weight=user_weight,
            content_weight=content_weight,
            top_k=top_k,
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

            try:
                anime_id = int(
                    anime_df[anime_df["eng_version"] == anime_name]["anime_id"].values[0]
                )
            except Exception:
                anime_id = None

            exp = explanations.get(str(anime_id), explanations.get(anime_id, {}))
            explanation_card(exp)