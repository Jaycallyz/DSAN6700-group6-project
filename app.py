"""
Cultural Media Travel Recommender
DSAN 6700 Group 6 Final Project

Team: Yiqin Zhou, Jiaqi Wei, Jiachen Gao, Zihao Huang
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from search_service import get_engine

# Page config
st.set_page_config(
    page_title="Cultural Media Travel Recommender",
    page_icon="🌍",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .destination-card {
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
        margin: 0.5rem 0;
    }
    .score-badge {
        background-color: #667eea;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌍 Cultural Media Travel Recommender</h1>
    <p>Discover destinations through movies, books, and music</p>
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
    **Enter your favorite media titles:**
    - 🎬 **Movie**: Full movie title (e.g., "The Devil's Advocate")
    - 📚 **Book**: Complete book title (e.g., "The Hunger Games")
    - 🎵 **Music**: Track name (e.g., "Die For You")
    
    **Tips:**
    - All three fields are required
    - Use exact titles for best results
    - Check suggestions if no match found
    """)

# Main input section
st.markdown("### 📝 Enter Your Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    movie_input = st.text_input(
        "🎬 Movie Title",
        placeholder="e.g., Inception",
        help="Enter the full movie title"
    )

with col2:
    book_input = st.text_input(
        "📚 Book Title",
        placeholder="e.g., 1984",
        help="Enter the complete book title"
    )

with col3:
    music_input = st.text_input(
        "🎵 Music/Track",
        placeholder="e.g., As It Was",
        help="Enter the track name"
    )

# Search button
st.markdown("<br>", unsafe_allow_html=True)
search_button = st.button("🔍 Get Recommendations", use_container_width=True, type="primary")

# Results section
if search_button:
    # Validation
    if not movie_input or not book_input or not music_input:
        st.error("⚠️ Please fill in all three fields!")
    else:
        with st.spinner("🔄 Searching for your perfect destinations..."):
            try:
                # Load engine
                engine = get_engine()
                
                # Get recommendations
                results, status = engine.recommend_from_combined_media(
                    movie_title=movie_input.strip(),
                    book_title=book_input.strip(),
                    music_title=music_input.strip(),
                    top_k=5
                )
                
                # Display status
                st.markdown("---")
                st.markdown("### 📊 Search Status")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if "✅" in status.get("movie", ""):
                        st.markdown(f'<p class="status-success">✅ Movie Found</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-error">❌ Movie Not Found</p>', unsafe_allow_html=True)
                
                with col2:
                    if "✅" in status.get("book", ""):
                        st.markdown(f'<p class="status-success">✅ Book Found</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-error">❌ Book Not Found</p>', unsafe_allow_html=True)
                
                with col3:
                    if "✅" in status.get("music", ""):
                        st.markdown(f'<p class="status-success">✅ Music Found</p>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-error">❌ Music Not Found</p>', unsafe_allow_html=True)
                
                # Display results
                if results:
                    st.markdown("---")
                    st.markdown("### 🌍 Top 5 Recommended Destinations")
                    
                    for result in results:
                        rank = result['rank']
                        name = result.get('name', 'Unknown')
                        country = result.get('country', 'Unknown')
                        score = result['score']
                        
                        # Emoji for top 3
                        emoji = ""
                        if rank == 1:
                            emoji = "🏆"
                        elif rank == 2:
                            emoji = "🥈"
                        elif rank == 3:
                            emoji = "🥉"
                        
                        # Display destination card
                        st.markdown(f"""
                        <div class="destination-card">
                            <h4>{emoji} {rank}. {name}, {country}</h4>
                            <span class="score-badge">Similarity: {score:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Additional info if partial match
                    not_found_count = sum(1 for msg in status.values() if "❌" in msg)
                    if not_found_count > 0:
                        st.info(f"ℹ️ {not_found_count} item(s) not found. Results based on available matches.")
                
                else:
                    st.warning("⚠️ No recommendations found. Please check your inputs.")
                
                # Show suggestions for not found items
                has_not_found = any("❌" in msg for msg in status.values())
                if has_not_found:
                    st.markdown("---")
                    st.markdown("### 💡 Did you mean?")
                    
                    if "❌" in status.get("movie", ""):
                        movie_suggestions = engine.suggest_titles("movie", movie_input, 5)
                        if movie_suggestions:
                            st.markdown("**🎬 Movie Suggestions:**")
                            for sugg in movie_suggestions:
                                st.write(f"  • {sugg}")
                    
                    if "❌" in status.get("book", ""):
                        book_suggestions = engine.suggest_titles("book", book_input, 5)
                        if book_suggestions:
                            st.markdown("**📚 Book Suggestions:**")
                            for sugg in book_suggestions:
                                st.write(f"  • {sugg}")
                    
                    if "❌" in status.get("music", ""):
                        music_suggestions = engine.suggest_titles("music", music_input, 5)
                        if music_suggestions:
                            st.markdown("**🎵 Music Suggestions:**")
                            for sugg in music_suggestions:
                                st.write(f"  • {sugg}")
            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please make sure all data files are properly loaded.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p><strong>DSAN 6700 Final Project - Group 6</strong></p>
    <p>Team: Yiqin Zhou, Jiaqi Wei, Jiachen Gao, Zihao Huang</p>
    <p style="font-size: 0.85rem;">Georgetown University | Fall 2025</p>
</div>
""", unsafe_allow_html=True)