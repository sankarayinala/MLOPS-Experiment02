import streamlit as st
from ui.jikan_client import get_poster_url

def anime_card(anime_name, anime_df):
    row = anime_df[anime_df["eng_version"] == anime_name]
    if row.empty:
        st.error("Anime metadata missing.")
        return

    aid = int(row["anime_id"].iloc[0])
    poster = get_poster_url(aid)

    st.markdown("#### " + anime_name)
    if poster:
        st.image(poster, width=180)
    else:
        st.write("No image available.")

    st.write(f"**Genres:** {row['Genres'].iloc[0]}")
    st.write(f"**Score:** {row['Score'].iloc[0]}")
    st.write(f"**Episodes:** {row['Episodes'].iloc[0]}")
    st.write("---")


def explanation_card(e):
    if not e:
        st.info("No explanation available.")
        return

    st.write("🔎 **Why this anime?**")

    st.write(f"- ⭐ **Model Score:** `{e.get('raw_score', 0):.3f}`")
    st.write(f"- 📈 **Popularity Boost:** `{e.get('popularity', 0):.3f}`")
    st.write(f"- 🎭 **Genre Alignment:** `{e.get('genre_alignment', 0):.3f}`")
    st.write(f"- 👥 **Similar Users Count:** `{e.get('user_signal_count', 0)}`")
    st.write(f"- 🔍 **FAISS Neighbor Count:** `{e.get('faiss_neighbors', 0)}`")

    st.write("---")