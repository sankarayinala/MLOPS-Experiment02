import streamlit as st
import requests
import pandas as pd

#API_URL = "http://backend-service:8000"
#API_URL = "http://backend-service.default.svc.cluster.local:8000"
API_URL = "http://10.108.90.164:8000"
##############################################
# ✅ Get JWT token (same as main UI)
##############################################
def get_token(username="demo", password="demo"):
    try:
        resp = requests.post(
            f"{API_URL}/auth/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json()["access_token"]
        else:
            st.error(f"Login failed: {resp.text}")
            return None
    except Exception:
        st.error("❌ Could not reach API.")
        return None


##############################################
# ✅ API wrappers
##############################################
def fetch_cache_keys(token):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{API_URL}/admin/cache/keys", headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        st.error(resp.text)
        return []


def delete_key(token, key):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.delete(f"{API_URL}/admin/cache/key/{key}", headers=headers)
    return resp.json()


def invalidate_user_cache(token, user_id):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.delete(f"{API_URL}/admin/cache/user/{user_id}", headers=headers)
    return resp.json()


##############################################
# ✅ Streamlit Layout
##############################################
st.set_page_config(
    page_title="Cache Admin Dashboard",
    page_icon="🧰",
    layout="wide"
)

st.title("🧰 Cache Administration Dashboard")
st.caption("Monitor Redis, inspect TTL, delete keys, and invalidate user cache.")

##############################################
# ✅ Auth Section
##############################################
st.subheader("🔐 Admin Login")

if "token" not in st.session_state:
    username = st.text_input("Username", value="demo")
    password = st.text_input("Password", type="password", value="demo")

    if st.button("Login"):
        token = get_token(username, password)
        if token:
            st.session_state["token"] = token
            st.success("✅ Logged in successfully!")
            st.rerun()
    st.stop()

token = st.session_state["token"]


##############################################
# ✅ View All Redis Keys
##############################################
st.subheader("📌 Redis Cache Keys")

if st.button("🔄 Refresh Keys"):
    st.rerun()

keys = fetch_cache_keys(token)

if keys:
    df = pd.DataFrame(keys)
    df.columns = ["Key", "TTL (seconds)"]
    st.dataframe(df, use_container_width=True)
else:
    st.info("No keys found in Redis.")


##############################################
# ✅ Delete Individual Key
##############################################
st.write("---")
st.subheader("❌ Delete an Individual Key")

if keys:
    key_list = [k["key"] for k in keys]
    selected_key = st.selectbox("Select Redis Key to Delete:", key_list)

    if st.button("Delete Selected Key"):
        response = delete_key(token, selected_key)
        st.success(f"✅ Deleted: {response}")
        st.rerun()
else:
    st.info("No keys available to delete.")


##############################################
# ✅ Invalidate a User's Cached Recommendations
##############################################
st.write("---")
st.subheader("🧹 Invalidate Cache for a Specific User")

user_id = st.number_input("Enter User ID to clear cache for:", min_value=1, value=11880)

if st.button("Invalidate User Cache"):
    resp = invalidate_user_cache(token, user_id)
    st.success(f"✅ Removed {resp['removed_keys']} keys for user {user_id}")
    st.rerun()
