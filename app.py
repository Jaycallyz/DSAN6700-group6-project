"""
Cultural Media Travel Recommender with Pexels Images
DSAN 6700 Group 6 Final Project
"""

import streamlit as st
import sys
from pathlib import Path
import requests
from urllib.parse import quote
import json
from datetime import datetime

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from search_service import get_engine

# Page config
st.set_page_config(
    page_title="Cultural Media Travel Recommender",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto;}
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .destination-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .destination-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    .destination-content {padding: 1.5rem;}
    .destination-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .city-image {
        width: 100%;
        height: 280px;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_city_image_pexels(city_name, country_code):
    """
    从 Pexels 获取城市图片（带缓存）
    """
    try:
        # 从 Streamlit secrets 读取 API key
        api_key = st.secrets.get("PEXELS_API_KEY", None)
        
        if not api_key:
            # 如果没有 API key，使用占位图
            return get_placeholder_image(city_name)
        
        # 搜索查询
        query = f"{city_name} {country_code} city landmark"
        url = "https://api.pexels.com/v1/search"
        
        headers = {"Authorization": api_key}
        params = {
            "query": query,
            "per_page": 1,
            "orientation": "landscape"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('photos') and len(data['photos']) > 0:
                # 返回大图 URL
                return data['photos'][0]['src']['large']
        
        # 失败时返回占位图
        return get_placeholder_image(city_name)
    
    except Exception as e:
        return get_placeholder_image(city_name)

def get_placeholder_image(city_name):
    """生成占位图（备用方案）"""
    city_id = sum(ord(c) for c in city_name) % 1000
    return f"https://picsum.photos/seed/{city_id}/800/280"

def get_wikipedia_excerpt(city_name):
    """获取维基百科简介"""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city_name.replace(' ', '_')}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            excerpt = data.get('extract', '')
            if excerpt and len(excerpt) > 50:
                if len(excerpt) > 300:
                    excerpt = excerpt[:297] + "..."
                return excerpt
        return None
    except:
        return None

def format_population(pop):
    """格式化人口"""
    if not pop or pop <= 0:
        return ""
    if pop >= 1_000_000:
        return f"🏙️ Major city ({pop/1_000_000:.1f}M people)"
    elif pop >= 500_000:
        return f"🌆 Large city ({pop:,} people)"
    elif pop >= 100_000:
        return f"🏘️ Medium city ({pop:,} people)"
    else:
        return f"🏡 Small town ({pop:,} people)"

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 style="font-size: 2.5rem; margin: 0;">🌍 Cultural Media Travel Recommender</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.95;">
        Discover your next destination through movies, books, and music
    </p>
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. Enter a movie title you love
    2. Enter a book title you enjoyed
    3. Enter a music track you like
    4. Click "Get Recommendations"
    5. Explore personalized destinations with real photos!
    """)

# Input
st.markdown("### 📝 Enter Your Preferences")
col1, col2, col3 = st.columns(3)

with col1:
    movie_input = st.text_input("🎬 Movie", placeholder="e.g., Inception")
with col2:
    book_input = st.text_input("📚 Book", placeholder="e.g., 1984")
with col3:
    music_input = st.text_input("🎵 Music", placeholder="e.g., Imagine")

# Search Button
st.markdown("<br>", unsafe_allow_html=True)
search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
with search_col2:
    search_button = st.button("🔍 Discover Destinations", use_container_width=True, type="primary")

# Handle Search
if search_button:
    if not movie_input or not book_input or not music_input:
        st.error("⚠️ Please fill in all three fields!")
    else:
        with st.spinner("🔄 Finding destinations..."):
            try:
                engine = get_engine()
                results, status = engine.recommend_from_combined_media(
                    movie_title=movie_input.strip(),
                    book_title=book_input.strip(),
                    music_title=music_input.strip(),
                    top_k=5
                )
                
                st.session_state['results'] = results
                st.session_state['status'] = status
                st.session_state['inputs'] = {
                    'movie': movie_input,
                    'book': book_input,
                    'music': music_input
                }
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display Results
if 'results' in st.session_state:
    results = st.session_state['results']
    status = st.session_state['status']
    inputs = st.session_state['inputs']
    
    # Status
    st.markdown("---")
    st.markdown("### 📊 Search Status")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if "✅" in status.get("movie", ""):
            st.success("✅ Movie Found")
            st.caption(f"_{inputs['movie']}_")
        else:
            st.error("❌ Not Found")
    with col2:
        if "✅" in status.get("book", ""):
            st.success("✅ Book Found")
            st.caption(f"_{inputs['book']}_")
        else:
            st.error("❌ Not Found")
    with col3:
        if "✅" in status.get("music", ""):
            st.success("✅ Music Found")
            st.caption(f"_{inputs['music']}_")
        else:
            st.error("❌ Not Found")
    
    # Results
    if results:
        st.markdown("---")
        st.markdown("### 🌍 Your Personalized Destinations")
        
        for r in results:
            rank = r['rank']
            name = r.get('name', 'Unknown')
            country = r.get('country', 'Unknown')
            score = r['score']
            population = r.get('population', None)
            
            emoji = {1: "🏆", 2: "🥈", 3: "🥉"}.get(rank, "")
            
            # Get image
            with st.spinner(f"Loading image for {name}..."):
                image_url = get_city_image_pexels(name, country)
            
            # Display card
            st.markdown(f"""
            <div class="destination-card">
                <img src="{image_url}" class="city-image" alt="{name}">
                <div class="destination-content">
                    <h3 class="destination-title">{emoji} {rank}. {name}, {country}</h3>
                    <span class="score-badge">Match: {score:.4f}</span>
            """, unsafe_allow_html=True)
            
            # Population
            pop_str = format_population(population)
            if pop_str:
                st.markdown(f"<p style='color: #6c757d; margin-top: 0.5rem;'>{pop_str}</p>", unsafe_allow_html=True)
            
            # Wikipedia
            wiki_excerpt = get_wikipedia_excerpt(name)
            if wiki_excerpt:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #667eea; margin: 1rem 0;'>
                    <strong>📚 About {name}:</strong><br>
                    <span style='line-height: 1.6;'>{wiki_excerpt}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Links
            wiki_link = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
            maps_link = f"https://www.google.com/maps/search/{quote(name)}+{country}"
            
            link_col1, link_col2 = st.columns(2)
            with link_col1:
                st.markdown(f"[📖 Wikipedia]({wiki_link})")
            with link_col2:
                st.markdown(f"[🗺️ Google Maps]({maps_link})")
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Feedback
        st.markdown("---")
        st.markdown("### 📊 Rate Your Experience")
        
        with st.form("feedback", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                rating = st.radio(
                    "Satisfaction:",
                    ["😍 Love them!", "😊 Pretty good", "😐 It's okay", "😞 Not satisfied"],
                    label_visibility="collapsed"
                )
            
            with col2:
                stars = st.slider("Stars:", 1, 5, 4, label_visibility="collapsed")
                st.markdown(f"**{'⭐' * stars}**")
            
            comments = st.text_area("Comments (optional):", height=100, label_visibility="collapsed")
            
            submitted = st.form_submit_button("📤 Submit", type="primary")
            
            if submitted:
                feedback_dir = Path("data/feedback")
                feedback_dir.mkdir(parents=True, exist_ok=True)
                
                feedback_data = {
                    "timestamp": datetime.now().isoformat(),
                    "inputs": inputs,
                    "status": status,
                    "recommendations": [
                        {"rank": r["rank"], "name": r.get("name"), 
                         "country": r.get("country"), "score": r["score"]}
                        for r in results
                    ],
                    "feedback": {
                        "satisfaction": rating,
                        "stars": stars,
                        "comments": comments or None
                    }
                }
                
                feedback_file = feedback_dir / "user_feedback.jsonl"
                with open(feedback_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
                
                st.success("✅ Thank you!")
                st.balloons()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p style="font-weight: 600;">DSAN 6700 Final Project - Group 6</p>
    <p>Yiqin Zhou, Jiaqi Wei, Jiachen Gao, Zihao Huang</p>
    <p style="font-size: 0.9rem; color: #999;">Georgetown University | Fall 2025</p>
</div>
""", unsafe_allow_html=True)