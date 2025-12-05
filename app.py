"""
Cultural Media Travel Recommender
DSAN 6700 Group 6 Final Project

Team: Yiqin Zhou, Jiaqi Wei, Jiachen Gao, Zihao Huang
"""

import streamlit as st
import sys
from pathlib import Path
import requests
from urllib.parse import quote

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from search_service import get_engine

# Page config
st.set_page_config(
    page_title="Cultural Media Travel Recommender",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 主容器 */
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header 样式 */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Hero 图片 */
    .hero-section {
        position: relative;
        height: 250px;
        background: url('https://images.unsplash.com/photo-1488646953014-85cb44e25828?w=1200&h=250&fit=crop') center/cover;
        border-radius: 10px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.7);
    }
    
    .hero-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.3);
        border-radius: 10px;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
    }
    
    /* 状态标签 */
    .status-success {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* 目的地卡片 */
    .destination-card {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .destination-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .destination-image {
        width: 100%;
        height: 200px;
        object-fit: cover;
    }
    
    .destination-content {
        padding: 1.2rem;
    }
    
    .destination-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.8rem;
    }
    
    .destination-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin: 0;
    }
    
    .score-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
    }
    
    .city-info {
        color: #6c757d;
        font-size: 0.95rem;
        margin: 0.5rem 0;
    }
    
    .wiki-excerpt {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #333;
    }
    
    /* 链接样式 */
    .destination-links {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .custom-link {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #f0f0f0;
        border-radius: 6px;
        text-decoration: none;
        color: #333;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    
    .custom-link:hover {
        background: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

def get_city_image_url(city_name, country_code):
    """
    获取城市图片 URL (使用 Unsplash)
    """
    # Unsplash Source - 免费，无需 API key
    query = f"{city_name} {country_code} landmark"
    encoded_query = quote(query)
    return f"https://source.unsplash.com/800x400/?{encoded_query}"

def get_wikipedia_excerpt(city_name, country_code):
    """
    获取维基百科简介 - 多种尝试策略
    """
    try:
        # 尝试1: 直接用城市名
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city_name.replace(' ', '_')}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            excerpt = data.get('extract', '')
            
            if excerpt and len(excerpt) > 50:
                if len(excerpt) > 300:
                    excerpt = excerpt[:297] + "..."
                return excerpt
        
        # 尝试2: 搜索 API
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            'action': 'query',
            'list': 'search',
            'srsearch': f"{city_name} {country_code}",
            'format': 'json',
            'srlimit': 1
        }
        
        search_response = requests.get(search_url, params=search_params, timeout=3)
        if search_response.status_code == 200:
            search_data = search_response.json()
            results = search_data.get('query', {}).get('search', [])
            
            if results:
                page_title = results[0]['title']
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
                summary_response = requests.get(summary_url, timeout=3)
                
                if summary_response.status_code == 200:
                    summary_data = summary_response.json()
                    excerpt = summary_data.get('extract', '')
                    
                    if excerpt and len(excerpt) > 50:
                        if len(excerpt) > 300:
                            excerpt = excerpt[:297] + "..."
                        return excerpt
        
        return None
    except Exception as e:
        return None

def format_population(pop):
    """格式化人口数字"""
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
    <div class="hero-overlay"></div>
    <div class="hero-content">
        <h1 style="font-size: 3rem; margin: 0;">🌍 Cultural Media Travel Recommender</h1>
        <p style="font-size: 1.3rem; margin-top: 1rem;">Discover your next destination through movies, books, and music</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ How to use this recommender", expanded=False):
    st.markdown("""
    ### 📝 Enter your favorite media:
    - 🎬 **Movie**: Full title (e.g., "The Shawshank Redemption", "Inception")
    - 📚 **Book**: Complete title (e.g., "To Kill a Mockingbird", "1984")
    - 🎵 **Music**: Track name (e.g., "Imagine", "Bohemian Rhapsody")
    
    ### 💡 Tips:
    - All three fields are required for best results
    - Use exact titles for accurate matching
    - Check suggestions if no match found
    - Explore each destination to see photos and descriptions
    """)

# Main input section
st.markdown("### 📝 Enter Your Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    movie_input = st.text_input(
        "🎬 Movie Title",
        placeholder="e.g., Inception",
        help="Enter the full movie title",
        key="movie"
    )

with col2:
    book_input = st.text_input(
        "📚 Book Title",
        placeholder="e.g., 1984",
        help="Enter the complete book title",
        key="book"
    )

with col3:
    music_input = st.text_input(
        "🎵 Music/Track",
        placeholder="e.g., Imagine",
        help="Enter the track name",
        key="music"
    )

# Search button
st.markdown("<br>", unsafe_allow_html=True)
search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
with search_col2:
    search_button = st.button("🔍 Discover Destinations", use_container_width=True, type="primary")

# Results section
if search_button:
    if not movie_input or not book_input or not music_input:
        st.error("⚠️ Please fill in all three fields to get personalized recommendations!")
    else:
        with st.spinner("🔄 Finding your perfect destinations..."):
            try:
                engine = get_engine()
                
                results, status = engine.recommend_from_combined_media(
                    movie_title=movie_input.strip(),
                    book_title=book_input.strip(),
                    music_title=music_input.strip(),
                    top_k=5
                )
                
                # Display status
                st.markdown("---")
                st.markdown("### 📊 Search Status")
                
                status_col1, status_col2, status_col3 = st.columns(3)
                
                with status_col1:
                    if "✅" in status.get("movie", ""):
                        st.markdown(f'<p class="status-success">✅ Movie Found</p>', unsafe_allow_html=True)
                        st.caption(f"_{movie_input}_")
                    else:
                        st.markdown(f'<p class="status-error">❌ Movie Not Found</p>', unsafe_allow_html=True)
                
                with status_col2:
                    if "✅" in status.get("book", ""):
                        st.markdown(f'<p class="status-success">✅ Book Found</p>', unsafe_allow_html=True)
                        st.caption(f"_{book_input}_")
                    else:
                        st.markdown(f'<p class="status-error">❌ Book Not Found</p>', unsafe_allow_html=True)
                
                with status_col3:
                    if "✅" in status.get("music", ""):
                        st.markdown(f'<p class="status-success">✅ Music Found</p>', unsafe_allow_html=True)
                        st.caption(f"_{music_input}_")
                    else:
                        st.markdown(f'<p class="status-error">❌ Music Not Found</p>', unsafe_allow_html=True)
                
                # Display results
                if results:
                    st.markdown("---")
                    st.markdown("### 🌍 Your Personalized Destination Recommendations")
                    
                    for result in results:
                        rank = result['rank']
                        name = result.get('name', 'Unknown')
                        country = result.get('country', 'Unknown')
                        score = result['score']
                        population = result.get('population', None)
                        
                        # Rank emoji
                        rank_emoji = {1: "🏆", 2: "🥈", 3: "🥉"}.get(rank, "")
                        
                        # Get city image
                        image_url = get_city_image_url(name, country)
                        
                        # Create destination card
                        with st.container():
                            st.markdown(f"""
                            <div class="destination-card">
                                <img src="{image_url}" class="destination-image" alt="{name}">
                                <div class="destination-content">
                                    <div class="destination-header">
                                        <h3 class="destination-title">{rank_emoji} {rank}. {name}, {country}</h3>
                                        <span class="score-badge">Match: {score:.4f}</span>
                                    </div>
                            """, unsafe_allow_html=True)
                            
                            # Population info
                            pop_str = format_population(population)
                            if pop_str:
                                st.markdown(f'<p class="city-info">{pop_str}</p>', unsafe_allow_html=True)
                            
                            # Wikipedia excerpt
                            wiki_excerpt = get_wikipedia_excerpt(name, country)
                            
                            if wiki_excerpt:
                                st.markdown(f"""
                                <div class="wiki-excerpt">
                                    <strong>📚 About {name}:</strong><br>
                                    {wiki_excerpt}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Links
                            wiki_link = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
                            maps_link = f"https://www.google.com/maps/search/{quote(name)}+{country}"
                            
                            link_col1, link_col2 = st.columns(2)
                            with link_col1:
                                st.markdown(f"[📖 Wikipedia]({wiki_link})")
                            with link_col2:
                                st.markdown(f"[🗺️ Google Maps]({maps_link})")
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Partial match info
                    not_found_count = sum(1 for msg in status.values() if "❌" in msg)
                    if not_found_count > 0:
                        st.info(f"ℹ️ {not_found_count} item(s) not found in our database. Results based on {3-not_found_count} available match(es).")
                
                else:
                    st.warning("⚠️ No recommendations found. Please try different titles.")
                
                # Suggestions
                has_not_found = any("❌" in msg for msg in status.values())
                if has_not_found:
                    st.markdown("---")
                    st.markdown("### 💡 Did you mean?")
                    
                    sugg_col1, sugg_col2, sugg_col3 = st.columns(3)
                    
                    with sugg_col1:
                        if "❌" in status.get("movie", ""):
                            movie_sugg = engine.suggest_titles("movie", movie_input, 5)
                            if movie_sugg:
                                st.markdown("**🎬 Movies:**")
                                for sugg in movie_sugg:
                                    st.write(f"• {sugg}")
                    
                    with sugg_col2:
                        if "❌" in status.get("book", ""):
                            book_sugg = engine.suggest_titles("book", book_input, 5)
                            if book_sugg:
                                st.markdown("**📚 Books:**")
                                for sugg in book_sugg:
                                    st.write(f"• {sugg}")
                    
                    with sugg_col3:
                        if "❌" in status.get("music", ""):
                            music_sugg = engine.suggest_titles("music", music_input, 5)
                            if music_sugg:
                                st.markdown("**🎵 Music:**")
                                for sugg in music_sugg:
                                    st.write(f"• {sugg}")
            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem 1rem;">
    <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">DSAN 6700 Final Project - Group 6</p>
    <p style="margin: 0.3rem 0;">Team: Yiqin Zhou, Jiaqi Wei, Jiachen Gao, Zihao Huang</p>
    <p style="font-size: 0.9rem; color: #999; margin-top: 0.5rem;">Georgetown University | Fall 2025</p>
</div>
""", unsafe_allow_html=True)