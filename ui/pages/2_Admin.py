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

API_URL = "http://localhost:8000"


# ---------------------------------------------------
# ✅ Require Admin Authentication (checked in app.py)
# ---------------------------------------------------
if "admin_logged_in" not in st.session_state or not st.session_state["admin_logged_in"]:
    st.error("🔒 Admin access only. Log in using the sidebar.")
    st.stop()


st.title("🧰 Cache Administration Dashboard")
st.caption("Inspect Redis Keys • TTL Monitoring • Delete Keys • Invalidate User Cache")

# ---------------------------------------------------
# ✅ API Helper Functions
# ---------------------------------------------------
def get_token():
    r = requests.post(
        f"{API_URL}/auth/login",
        data={"username": "demo", "password": "demo"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return r.json().get("access_token")


def list_keys(token):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{API_URL}/admin/cache/keys", headers=headers)
    return r.json()


def delete_key(token, key):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.delete(f"{API_URL}/admin/cache/key/{key}", headers=headers)
    return r.json()


def invalidate_user_cache(token, user_id):
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.delete(f"{API_URL}/admin/cache/user/{user_id}", headers=headers)
    return r.json()


token = get_token()

# ---------------------------------------------------
# ✅ View Cache Keys
# ---------------------------------------------------
st.subheader("📌 Redis Keys")

keys = list_keys(token)

if keys:
    df = pd.DataFrame(keys)
    df.columns = ["Key", "TTL (seconds)"]
    st.dataframe(df, use_container_width=True)
else:
    st.info("No cached keys found.")

# ---------------------------------------------------
# ✅ Delete Individual Key
# ---------------------------------------------------
if keys:
    st.subheader("❌ Delete a specific key")

    key_choices = [k["key"] for k in keys]
    key_to_delete = st.selectbox("Choose a key to delete:", key_choices)

    if st.button("Delete this Key"):
        result = delete_key(token, key_to_delete)
        st.success(result)
        st.rerun()

# ---------------------------------------------------
# ✅ Invalidate Cache for Specific User
# ---------------------------------------------------
st.write("---")
st.subheader("🧹 Invalidate cached recommendations for a user")

user_id = st.number_input("User ID", min_value=1, value=11880)

if st.button("Invalidate User Cache"):
    result = invalidate_user_cache(token, user_id)
    st.success(result)
    st.rerun()