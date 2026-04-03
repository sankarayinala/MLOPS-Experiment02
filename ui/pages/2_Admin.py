import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import requests

CURRENT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT.parent.parent  # /app inside container

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

##API_URL = "http://backend-service:8000"
##API_URL = "http://backend-service.default.svc.cluster.local:8000"
API_URL = "http://10.108.90.164:8000"

# ---------------------------------------------------
# Require Admin Authentication
# ---------------------------------------------------
if "admin_logged_in" not in st.session_state or not st.session_state["admin_logged_in"]:
    st.error("🔒 Admin access only. Log in using the sidebar.")
    st.stop()

st.title("🧰 Cache Administration Dashboard")
st.caption("Inspect Redis Keys • TTL Monitoring • Delete Keys • Invalidate User Cache")


# ---------------------------------------------------
# API Helper Functions
# ---------------------------------------------------
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
        st.error(f"❌ Failed to authenticate with backend API: {e}")
        return None


def list_keys(token):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(f"{API_URL}/admin/cache/keys", headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Failed to fetch cache keys: {e}")
        return []


def delete_key(token, key):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.delete(f"{API_URL}/admin/cache/key/{key}", headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def invalidate_user_cache(token, user_id):
    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.delete(f"{API_URL}/admin/cache/user/{user_id}", headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


token = get_token()
if not token:
    st.stop()


# ---------------------------------------------------
# View Cache Keys
# ---------------------------------------------------
st.subheader("📌 Redis Keys")

keys = list_keys(token)

if keys:
    df = pd.DataFrame(keys)
    if not df.empty:
        df.columns = ["Key", "TTL (seconds)"]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No cached keys found.")
else:
    st.info("No cached keys found.")


# ---------------------------------------------------
# Delete Individual Key
# ---------------------------------------------------
if keys:
    st.subheader("❌ Delete a specific key")

    key_choices = [k["key"] for k in keys if "key" in k]
    if key_choices:
        key_to_delete = st.selectbox("Choose a key to delete:", key_choices)

        if st.button("Delete this Key"):
            result = delete_key(token, key_to_delete)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result)
                st.rerun()


# ---------------------------------------------------
# Invalidate Cache for Specific User
# ---------------------------------------------------
st.write("---")
st.subheader("🧹 Invalidate cached recommendations for a user")

user_id = st.number_input("User ID", min_value=1, value=11880)

if st.button("Invalidate User Cache"):
    result = invalidate_user_cache(token, user_id)
    if "error" in result:
        st.error(result["error"])
    else:
        st.success(result)
        st.rerun()