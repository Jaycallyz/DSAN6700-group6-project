# 🌍 Cultural Media Travel Recommender

**DSAN 6700 Final Project - Group 6**

Discover your next travel destination through the cultural media you love - movies, books, and music!

---

## 👥 Team Members

- **Yiqin Zhou**
- **Jiaqi Wei**
- **Jiachen Gao**
- **Zihao Huang**

---

## 📖 Project Overview

This project uses **semantic embeddings** and **FAISS similarity search** to recommend travel destinations based on your favorite cultural media. By analyzing the themes, settings, and atmospheres of movies, books, and music you enjoy, our system suggests destinations that align with your preferences.

### 🎯 Key Features

- **Multi-Modal Fusion**: Combines movie, book, and music preferences
- **Semantic Search**: Uses Sentence-BERT embeddings (all-MiniLM-L6-v2)
- **Fast Retrieval**: FAISS IndexFlatIP for efficient similarity search
- **Real Images**: Pexels API for authentic city photos
- **Interactive UI**: Built with Streamlit
- **User Feedback**: Collects ratings to improve recommendations

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Search** | FAISS |
| **Backend** | Python 3.9+ |
| **Frontend** | Streamlit |
| **Images** | Pexels API |
| **Deployment** | Hugging Face Spaces |

---

## 📊 Dataset

- **Movies**: 1,000 titles (stratified sampling)
- **Books**: 2,000 titles (stratified sampling)
- **Music**: 2,000 tracks (stratified sampling)
- **Destinations**: 1,000 cities worldwide (stratified sampling)

---

## 🚀 How to Use

### Local Installation

```bash
# Clone repository
git clone https://github.com/Jaycallyz/DSAN6700-group6-project.git
cd DSAN6700-group6-project

# Install dependencies
pip install -r requirements.txt

# Set up API key (optional)
mkdir -p .streamlit
echo 'PEXELS_API_KEY = "YOUR_KEY"' > .streamlit/secrets.toml

# Run
streamlit run app.py
```

---

## 💡 How It Works

1. **Input**: User provides movie, book, and music titles
2. **Embedding Lookup**: Retrieve pre-computed embeddings
3. **Fusion**: Average the three embeddings
4. **Search**: Find similar destinations using FAISS
5. **Display**: Show results with photos and descriptions

---

## 📁 Project Structure

```
├── app.py                  # Main application
├── requirements.txt        # Dependencies
├── data/
│   ├── clean_data/        # Processed CSVs
│   └── embeddings/        # .npy files + FAISS index
├── src/
│   └── search_service.py  # Recommendation engine
└── scripts/               # Data processing scripts
```

---

## 🔮 Future Improvements

- Learned fusion weights via user feedback
- Multi-language support
- Seasonal recommendations
- Price integration
