# 🌍 Cultural Media Travel Recommender

**DSAN 6700 Final Project - Group 6**

Discover your next travel destination through the cultural media you love - movies, books, and music!

## Team Members

- **Yiqin Zhou** 
- **Jiaqi Wei**  
- **Jiachen Gao**
- **Zihao Huang** 

## Project Overview

This project builds a **semantic-based travel recommendation system** that suggests destinations based on users' preferences in movies, books, and music. By analyzing the cultural themes, settings, and atmospheres embedded in media content, our system identifies travel destinations that resonate with users' tastes.


## System Architecture
**User Input → Embeddings → Fusion → Similarity Search → Recommendations**

- Frontend: Streamlit
- Backend: RecommendationEngine class
- Data Layer: preprocessed CSV + NPZ embeddings
- APIs: Wikipedia (descriptions), GeoNames (coords), Pexels (images)

```text
┌───────────────────────────────────────────────┐
│                 User Input                    │
│        (Movie • Book • Music selections)      │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│               Text Processing                 │
│     - Clean text                              │
│     - Merge metadata fields                   │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│         Sentence-BERT Embedding (384d)        │
│  v = w_m · v_m + w_b · v_b + w_s · v_s        │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│             Multimodal Fusion                 │
│      (FAISS index locally / NumPy fallback)   │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│               Top-K Retrieval                 │
│         (Rank destinations by similarity)     │
└───────────────────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────┐
│                Display Results                │
│  - City name                                  │
│  - Explanation (semantic match)               │
│  - Image card (Pexels API)                    │
└───────────────────────────────────────────────┘
```

This diagram summarizes the end-to-end flow from user cultural preferences
to destination recommendations.


## Dataset
Data is available on Google Drive: https://drive.google.com/drive/folders/1_zfbeevmwvxwBrZBVJR-y3mmF0MMEPMX?usp=drive_link

### Dataset Summary
- Movies: 1,000 titles
- Books: 2,000
- Music: 2,000
- Destinations: 1,000

### Preprocessing techniques:
-stratified sampling
- deduplication
- text cleaning
- text concat for embedding


##  Quick Start
### Installation & Running the App
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
src/
  search_service.py
  generate_embeddings.py
data/
  clean_data/
  embeddings/
app.py
```

## How to Use
```python
from src.search_service import get_engine
engine = get_engine()
results = engine.recommend_from_combined_media(
    movie_title="Inception",
    book_title="1984",
    music_title="Imagine",
    top_k=5
)
```

## Deployment
Live demo: https://huggingface.co/spaces/yz1395/destination_recommender

- Used Hugging Face for container-based deployment
- Automatic rebuild via GitHub Actions

## Limitations
- Dataset size limited → many works not included
- Destination list too broad → similarity scores dispersed
- Pexels API images sometimes inaccurate
- Must type exact media title (no fuzzy search yet)
- Only returns cities, no full travel guides
- Feedback system works only locally (HF is read-only)

## Future Improvements
- Expand media datasets
- Add destination safety filtering
- Improve image relevance (CLIP / search reranking)
- Add fuzzy matching for user inputs
- Connect LLMs to generate travel guides
- Use cloud DB to collect online user feedback

## Baseline Improvement
### Baseline
The baseline version of our system performs single-modality recommendations, where each media type (movie, book, or music) independently produces a destination recommendation.

### Improvement Over Baseline
Our final system introduces a multi-modal fusion strategy: 
- Combines user preferences from movies, books, and music
- Produces a more stable and culturally coherent embedding
- Reduces noise or bias from relying on a single media source
- Provides richer semantic signals for retrieval

Because this is a retrieval-based system, we do not evaluate improvements with numerical metrics such as accuracy or RMSE. Instead, the improvement lies in the quality and diversity of user experience, where fused inputs generate more contextually consistent recommendations compared to single-modality baselines.

## Note on ML Experiment Tracking Tools

Because our system performs semantic retrieval using pretrained embeddings 
(without training or evaluation steps), ML experiment tracking tools are not 
necessary or applicable.

## Reference
### Dateset
- Music Dataset: Spotify Tracks Dataset (Kaggle)
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- Books Dataset: GoodBooks-10k Dataset (Kaggle)
https://www.kaggle.com/datasets/zygmunt/goodbooks-10k
- Movies Dataset: Retrieved via TMDB API, using movie titles, genres, and overviews
https://www.themoviedb.org/
- Destinations Dataset: City summaries from Wikipedia API and geographic metadata from GeoNames API
https://www.wikipedia.org/ and https://www.geonames.org/
