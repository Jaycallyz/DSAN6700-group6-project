import sys
from pathlib import Path
import streamlit as st

# Ensure backend is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.search_service import get_engine 

# Page Config
st.set_page_config(
    page_title="Media → Destination Recommender",
    page_icon="🌍",
    layout="centered",
)

st.title("Media → Destination Recommender")
st.markdown(
    """
Enter a **movie**, **book**, or **music** title,  
and discover **destination cities** with similar semantic vibes 🌎  
_(powered by FAISS + Sentence-BERT embeddings)_
"""
)


engine = get_engine()

# UI Controls 
media_type = st.selectbox("Choose media type", ["movie", "book", "music"])
title = st.text_input("Enter title", placeholder="e.g. Inception, Blinding Lights, Harry Potter ...")
top_k = st.slider("Number of recommendations", 3, 10, 5)

col1, col2 = st.columns(2)
with col1:
    run_btn = st.button("🔍 Recommend")
with col2:
    suggest_btn = st.button("💡 Suggest similar titles")

# Suggest Titles
if suggest_btn and title.strip():
    suggestions = engine.suggest_titles(media_type, title, max_suggestions=5)
    if suggestions:
        st.markdown("**Did you mean:**")
        for s in suggestions:
            st.write(f"- {s}")
    else:
        st.info("No similar titles found. Try another keyword.")

# Main Recommendation
if run_btn:
    if not title.strip():
        st.warning("Please enter a valid title.")
    else:
        with st.spinner("Searching..."):
            results = engine.recommend_from_media(media_type, title, top_k=top_k)

        if not results:
            st.error("No exact match found. Try using the suggestion button.")
        else:
            st.subheader("Top Recommendations")
            st.dataframe(results, use_container_width=True)
            st.caption("Similarity = FAISS inner product over normalized embeddings.")
