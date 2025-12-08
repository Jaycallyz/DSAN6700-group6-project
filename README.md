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

### Key Features

- **Multi-Modal Fusion**: Intelligently combines preferences from three media types
- **Semantic Search**: Uses Sentence-BERT embeddings (all-MiniLM-L6-v2, 384 dimensions)
- **Fast Retrieval**: Numpy-based cosine similarity for efficient search
- **Interactive UI**: Beautiful Streamlit web application
- **Real Images**: City photos via Picsum API
- **User Feedback System**: Collects ratings to enable future improvements
- **Production Ready**: Deployed on Hugging Face Spaces with CI/CD

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
│                   (Streamlit App)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │  Movie   │  │   Book   │  │  Music   │  Input          │
│  │  Input   │  │  Input   │  │  Input   │                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│       │             │             │                         │
│       └─────────────┴─────────────┘                         │
│                     │                                       │
└─────────────────────┼───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────────────────────────────────────────────┐
│              Recommendation Engine                          │
│              (search_service.py)                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Embedding   │  │   Metadata   │  │  Similarity  │    │
│  │   Lookup     │→ │   Retrieval  │→ │   Compute    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                                      │            │
│         ↓                                      ↓            │
│  ┌──────────────┐                    ┌──────────────┐    │
│  │   Fusion     │                    │   Ranking    │    │
│  │  Strategy    │                    │   & Filter   │    │
│  └──────────────┘                    └──────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Embeddings  │  │     CSVs     │  │   Feedback   │    │
│  │   (.npz)     │  │ (Metadata)   │  │   (JSONL)    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

**1. User Interface (Streamlit)**
- Multi-input form for movies, books, and music
- Visual result cards with city images
- Interactive feedback collection system
- Responsive design with gradient styling

**2. Recommendation Engine**
- Pre-computed embedding lookup from .npz files
- Configurable fusion strategies (averaging, weighted)
- Cosine similarity calculation using numpy
- Result ranking and metadata enrichment

**3. Data Layer**
- **Embeddings**: 384-dim vectors stored in compressed .npz format
  - Movies: 1000 samples
  - Books: 2000 samples
  - Music: 2000 tracks
  - Destinations: 1000 cities
- **Metadata**: CSV files with titles, descriptions, populations
- **Feedback**: User ratings stored in JSONL for future analysis


## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) | Semantic representation |
| **Similarity Search** | Numpy (cosine similarity) | Fast retrieval |
| **Backend** | Python 3.9+ | Core logic |
| **Frontend** | Streamlit | Web interface |
| **Images** | Picsum Photos API | City imagery |
| **Data Format** | NPZ, CSV, JSONL | Storage |
| **Testing** | pytest | Quality assurance |
| **CI/CD** | GitHub Actions | Automation |
| **Deployment** | Hugging Face Spaces | Production hosting |
| **Containerization** | Docker + Docker Compose | Local development |


## Dataset

### Media Collections
- **Movies**: 1,000 titles from IMDb (stratified sampling by genre)
- **Books**: 2,000 titles from Goodreads (stratified sampling by category)
- **Music**: 2,000 tracks from Spotify (stratified sampling by genre)
- **Destinations**: 1,000 cities worldwide (stratified sampling by region)

### Data Processing Pipeline
1. **Collection**: Gathered from public APIs and datasets
2. **Cleaning**: Removed duplicates, handled missing values
3. **Stratification**: Ensured diverse genre/category representation
4. **Text Combination**: Merged titles, descriptions, genres for embedding
5. **Embedding Generation**: Used Sentence-BERT to create 384-dim vectors
6. **Normalization**: L2-normalized for cosine similarity


##  Quick Start

### Prerequisites

```bash
Python 3.9+
pip or conda
```

### Installation

```bash
# Clone repository
git clone https://github.com/Jaycallyz/DSAN6700-group6-project.git
cd DSAN6700-group6-project

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the API

```python
from src.search_service import get_engine

# Initialize engine (cached singleton)
engine = get_engine()

# Get recommendations from a single media type
results = engine.recommend_from_media(
    media_type="movie",  # or "book", "music"
    media_title="Inception",
    top_k=5
)

# Get combined recommendations
results, status = engine.recommend_from_combined_media(
    movie_title="Inception",
    book_title="The Hobbit", 
    music_title="Bohemian Rhapsody",
    top_k=5
)

# Results format
for r in results:
    print(f"{r['rank']}. {r['name']}, {r['country']}")
    print(f"   Score: {r['score']:.4f}")

# Get title suggestions (fuzzy matching)
suggestions = engine.suggest_titles(
    media_type="movie",
    query="incep",
    max_suggestions=5
)
# Returns: ['Inception', 'Inception: The Cobol Job', ...]
```

## Project Structure

```
DSAN6700-group6-project/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Modern Python packaging
├── README.md                       # This file
├── Dockerfile                      # Container definition
├── docker-compose.yml              # Service orchestration
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── clean_data/                 # Processed CSV files
│   │   ├── movie_processed.csv
│   │   ├── book_processed.csv
│   │   ├── music_processed.csv
│   │   └── destination_processed.csv
│   │
│   ├── embeddings/                 # Pre-computed embeddings
│   │   ├── movie_embeddings.npz
│   │   ├── book_embeddings.npz
│   │   ├── music_embeddings.npz
│   │   └── destination_embeddings.npz
│   │
│   └── feedback/                   # User feedback data
│       └── user_feedback.jsonl
│
├── src/
│   ├── __init__.py
│   ├── search_service.py           # Core recommendation engine
│   └── generate_embeddings.py      # Embedding generation utility
│   ├── process_data_with_tfidf.py  # Data sampling
│   └── view_feedback.py            # Feedback analysis
│   └── build_faiss.py              # Feedback analysis
├── tests/
│   ├── __init__.py
│   └── test_search_service.py      # Unit tests
│
└── .github/
    └── workflows/
        └── ci.yml                  # CI/CD pipeline
```

---

## Reproducing Experiments

### Step 1: Data Preparation

```bash
# If starting from raw data
python scr/process_data_with_tfidf.py
```

### Step 2: Generate Embeddings

```bash
# Generate all embeddings
python src/generate_embeddings.py

# Embeddings will be saved in data/embeddings/
```

### Step 3: Run Tests

```bash
# Run test suite
pytest tests/ -v --cov=src

# Expected output: All tests pass
```

### Step 4: Launch Application

```bash
streamlit run app.py
```

### Step 5: Test Recommendations

Example test case:
   - Movie: The Devil's Advocate
   - Book: The Hunger Games
   - Music: Die For You

## 🐳 Docker Deployment

### Using Docker Compose 

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f frontend

# Stop services
docker-compose down
```

Access at `http://localhost:8501`

### Manual Docker Build

```bash
# Build image
docker build -t travel-recommender:latest .

# Run container
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  --name travel-recommender \
  travel-recommender:latest

# View logs
docker logs -f travel-recommender
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

---

## Cloud Deployment

### Current Deployment: Hugging Face Spaces

**Live Demo**: https://huggingface.co/spaces/yz1395/destination_recommender

We chose Hugging Face Spaces for deployment because:
- ✅ Built-in Docker containerization
- ✅ Automatic CI/CD pipeline
- ✅ Free hosting for academic projects
- ✅ Easy collaboration and version control
- ✅ Professional-grade infrastructure

## 💡 How It Works

### 1. Embedding Generation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Combine text fields
text = f"{title}. {description}. {genres}"

# Generate normalized embeddings
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)
```

### 2. Similarity Search

```python
# Load user input embedding
query_emb = load_embedding(media_type, title)

# Normalize for cosine similarity
query_emb = query_emb / np.linalg.norm(query_emb)

# Compute similarity with all destinations
scores = np.dot(destination_embeddings, query_emb)

# Rank and return top-K
top_k_indices = np.argsort(scores)[::-1][:k]
```

### 3. Multi-Modal Fusion

Simple averaging strategy:

```python
# Get embeddings for each media type
movie_emb = get_embedding("movie", movie_title)
book_emb = get_embedding("book", book_title)
music_emb = get_embedding("music", music_title)

# Average (equal weights)
combined_emb = (movie_emb + book_emb + music_emb) / 3

# Search with combined embedding
results = search(combined_emb, top_k=5)
```


## Evaluation
**System Performance**
- Query latency: <100ms
- Memory usage: ~2GB
- Embedding load time: <5s


## CI/CD Pipeline

GitHub Actions workflow runs on every push:

1. ✅ Set up Python environment
2. ✅ Install dependencies
3. ✅ Run linting (ruff)
4. ✅ Run tests (pytest)
5. ✅ Check formatting

View workflow: [.github/workflows/ci.yml](.github/workflows/ci.yml)

## Future Improvements
- Expand Cultural Database
- Destination Filtering & Safety
- Better Image Acquisition
- Input Fuzzy Matching
- AI-Generated Travel Guides
- Online Feedback System
- Learned Fusion Weights

##  References
### Datasets
- IMDb for movie data
- Goodreads for book data
- Spotify for music data
- GeoNames for destination data

### Tools & Libraries
- [Sentence-Transformers](https://www.sbert.net/) by UKPLab
- [Streamlit](https://streamlit.io/) for web framework
- [Hugging Face](https://huggingface.co/) for deployment
