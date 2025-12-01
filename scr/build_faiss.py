# src/build_faiss.py

import os
import numpy as np
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDING_PATH = os.path.join(BASE_DIR, "data", "embeddings", "destination_embeddings.npy")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "destinations_faiss_ip.index")


def load_destination_embeddings() -> np.ndarray:
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(f"Destination embeddings not found at {EMBEDDING_PATH}")
    emb = np.load(EMBEDDING_PATH)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D array for embeddings, got shape {emb.shape}")
    return emb.astype("float32")


def build_ip_index(embeddings: np.ndarray) -> faiss.Index:
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str = INDEX_PATH) -> None:
    faiss.write_index(index, path)
    print(f"Saved FAISS index with {index.ntotal} vectors to {path}")


def main():
    print("Loading destination embeddings...")
    dest_embeddings = load_destination_embeddings()

    print("Building FAISS inner-product index...")
    index = build_ip_index(dest_embeddings)

    print("Saving index...")
    save_index(index)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Full data processing pipeline + TF-IDF reweighting
Based on DSAN 6700 Final Project plan.

Features:
- Process Movie, Book, Music, Destination datasets
- TF-IDF re-weighting to downweight popular/common words
- Target dataset sizes (configurable via CLI):
    Movie:        500–1000
    Book:         1000–2000
    Music:        1000–2000
    Destination:  500–1000
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


class DataProcessorWithTFIDF:
    """
    Data processor with optional TF-IDF-based text reweighting.
    """
    
    def __init__(
        self,
        raw_data_dir: str = "data/raw_data",
        output_dir: str = "data/clean_data",
        movie_count: int = 1000,
        book_count: int = 2000,
        music_count: int = 2000,
        destination_count: int = 1000,
        use_tfidf: bool = True
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.movie_count = movie_count
        self.book_count = book_count
        self.music_count = music_count
        self.destination_count = destination_count
        self.use_tfidf = use_tfidf
        
        print(f"\n{'='*60}")
        print(f"Data Processing Configuration")
        print(f"{'='*60}")
        print(f"Movie count: {movie_count}")
        print(f"Book count: {book_count}")
        print(f"Music count: {music_count}")
        print(f"Destination count: {destination_count}")
        print(f"Use TF-IDF reweighting: {use_tfidf}")
        print(f"{'='*60}\n")
    
    # ============================================================
    # TF-IDF Utility Functions
    # ============================================================
    
    def apply_tfidf_reweighting(
        self, 
        texts: list, 
        column_name: str = "description"
    ) -> list:
        """
        Apply TF-IDF-based downweighting / highlighting.

        Goal:
            - Downweight common, generic words (e.g., "beautiful", "amazing")
            - Emphasize more distinctive phrases
              (e.g., "Gothic architecture", "coastal tranquility")
        
        Args:
            texts: list of raw text strings
            column_name: column label used only for logging

        Returns:
            List of reweighted text strings (original text + key TF-IDF terms)
        """
        if not self.use_tfidf:
            return texts
        
        print(f"🔄 Applying TF-IDF reweighting to {column_name}...")
        
        # Compute TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  # unigrams + bigrams
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Rebuild each document by emphasizing high TF-IDF terms
            reweighted_texts = []
            for i, text in enumerate(texts):
                # Get TF-IDF scores for this document
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                
                # Select top-k important terms
                top_indices = tfidf_scores.argsort()[-20:][::-1]  # top 20
                top_words = [feature_names[idx] for idx in top_indices if tfidf_scores[idx] > 0]
                
                # Rebuild text: original text + emphasized keywords
                emphasized = " ".join(top_words[:10])  # top 10
                reweighted = f"{text} {emphasized}"
                
                reweighted_texts.append(reweighted)
            
            print(f"   ✅ TF-IDF reweighting applied")
            return reweighted_texts
        
        except Exception as e:
            print(f"   ⚠️ TF-IDF failed: {e}, using original texts")
            return texts
    
    def create_text_representation(
        self,
        df: pd.DataFrame,
        columns: list,
        apply_tfidf: bool = False,
        column_name: str = "text"
    ) -> list:
        """
        Create concatenated text representation from one or more columns,
        with optional TF-IDF reweighting.
        """
        texts = []
        for _, row in df.iterrows():
            parts = []
            for col in columns:
                if col in df.columns:
                    val = str(row[col])
                    if val and val != 'nan' and val.strip():
                        parts.append(val.strip())
            
            text = ". ".join(parts) if parts else "Unknown"
            texts.append(text)
        
        # Apply TF-IDF reweighting if requested
        if apply_tfidf and self.use_tfidf:
            texts = self.apply_tfidf_reweighting(texts, column_name)
        
        return texts
    
    # ============================================================
    # MOVIE Processing
    # ============================================================
    
    def process_movies(self):
        """Process movie data from TMDB CSV."""
        print("\n" + "="*60)
        print("🎬 Processing Movies...")
        print("="*60)
        
        INPUT_PATH = self.raw_data_dir / "tmdb_5000_movies.csv"
        OUTPUT_CSV = self.output_dir / "movie_processed.csv"
        
        if not INPUT_PATH.exists():
            print(f"❌ File not found: {INPUT_PATH}")
            return
        
        df = pd.read_csv(INPUT_PATH)
        print(f"📊 Loaded: {len(df)} movies")
        
        # Cast numeric columns
        for col in ["runtime", "vote_average", "vote_count", "popularity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["overview"] = df["overview"].fillna("").astype(str)
        df["title"] = df["title"].fillna("").astype(str)
        
        # Basic filtering
        mask = (
            (df["title"].str.len() > 0) &
            (df["overview"].str.len() > 20)
        )
        
        if "status" in df.columns:
            mask &= (df["status"].fillna("") == "Released")
        
        df = df[mask].copy()
        
        # Drop duplicates
        if "original_title" in df.columns:
            df = df.drop_duplicates(subset=["original_title"], keep="first")
        
        # Weighted rating (Bayesian-style)
        df_votes = df[df["vote_count"].notna() & (df["vote_count"] > 0)].copy()
        if len(df_votes) > 0:
            C = df_votes["vote_average"].mean()
            m = df_votes["vote_count"].quantile(0.25)
            v = df_votes["vote_count"]
            R = df_votes["vote_average"]
            df_votes["weighted_rating"] = (v/(v+m))*R + (m/(v+m))*C
            
            df = df.merge(
                df_votes[["original_title", "weighted_rating"]], 
                on="original_title", 
                how="left"
            )
        
        df = df.sort_values(
            ["weighted_rating", "vote_count"], 
            ascending=False,
            na_position="last"
        ).head(self.movie_count)
        
        # Create text representation (with TF-IDF)
        print("📝 Creating text representations...")
        text_cols = ["title", "overview"]
        if "keywords" in df.columns:
            text_cols.append("keywords")
        
        df["text_representation"] = self.create_text_representation(
            df, 
            text_cols,
            apply_tfidf=True,
            column_name="movie"
        )
        
        # Save
        keep_cols = [
            "title", "original_title", "overview", "genres", "keywords",
            "release_date", "runtime", "vote_average", "vote_count",
            "popularity", "weighted_rating", "text_representation"
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        df[keep_cols].reset_index(drop=True).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"✅ Saved {len(df)} movies → {OUTPUT_CSV}")
    
    # ============================================================
    # BOOK Processing
    # ============================================================
    
    def process_books(self):
        """Process book data (Goodreads-style dataset)."""
        print("\n" + "="*60)
        print("📚 Processing Books...")
        print("="*60)
        
        BASE = self.raw_data_dir / "books"
        OUTPUT_CSV = self.output_dir / "book_processed.csv"
        
        books_path = BASE / "books.csv"
        book_tags_path = BASE / "book_tags.csv"
        tags_path = BASE / "tags.csv"
        
        if not books_path.exists():
            print(f"❌ File not found: {books_path}")
            return
        
        # Load datasets
        books = pd.read_csv(books_path)
        book_tags = pd.read_csv(book_tags_path)
        tags = pd.read_csv(tags_path)
        
        print(f"📖 Loaded books: {books.shape}")
        
        # Normalize join keys for tag info
        if "book_id" in book_tags.columns:
            book_tags = book_tags.rename(columns={"book_id": "goodreads_book_id"})
        
        if "book_id" not in books.columns and "id" in books.columns:
            books = books.rename(columns={"id": "book_id"})
        
        book_tags = book_tags.merge(tags, on="tag_id", how="left")
        
        book_tag_grouped = (
            book_tags.groupby("goodreads_book_id")["tag_name"]
            .apply(lambda x: ", ".join(x.head(15)))
            .reset_index()
            .rename(columns={"tag_name": "tag_list"})
        )
        
        books = books.merge(
            book_tag_grouped,
            left_on="book_id",
            right_on="goodreads_book_id",
            how="left"
        )
        
        # Cleaning
        books["original_title"] = books["original_title"].fillna(books["title"])
        books["authors"] = books["authors"].fillna("Unknown")
        books["average_rating"] = pd.to_numeric(books["average_rating"], errors="coerce")
        books["ratings_count"] = pd.to_numeric(books["ratings_count"], errors="coerce")
        books["tag_list"] = books["tag_list"].fillna("")
        
        # Filter by rating, popularity, and year
        books_filt = books[
            (books["average_rating"] >= 3.5) &
            (books["ratings_count"] >= 100) &
            (books["original_publication_year"].between(1800, 2025))
        ].copy()
        
        # Weighted rating
        C = books_filt["average_rating"].mean()
        m = books_filt["ratings_count"].quantile(0.50)
        v = books_filt["ratings_count"]
        R = books_filt["average_rating"]
        books_filt["weighted_score"] = (v/(v+m))*R + (m/(v+m))*C
        
        books_filt = books_filt.sort_values("weighted_score", ascending=False).head(self.book_count)
        
        # Create text representation (with TF-IDF)
        print("📝 Creating text representations...")
        books_filt["text_representation"] = self.create_text_representation(
            books_filt,
            ["title", "original_title", "authors", "tag_list"],
            apply_tfidf=True,
            column_name="book"
        )
        
        # Save
        keep_cols = [
            "book_id", "title", "original_title", "authors",
            "average_rating", "ratings_count", "tag_list",
            "weighted_score", "text_representation"
        ]
        keep_cols = [c for c in keep_cols if c in books_filt.columns]
        
        books_filt[keep_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"✅ Saved {len(books_filt)} books → {OUTPUT_CSV}")
    
    # ============================================================
    # MUSIC Processing
    # ============================================================
    
    def process_music(self):
        """Process music data (tracks + audio features)."""
        print("\n" + "="*60)
        print("🎵 Processing Music...")
        print("="*60)
        
        INPUT_PATH = self.raw_data_dir / "music_dataset.csv"
        OUTPUT_CSV = self.output_dir / "music_processed.csv"
        
        if not INPUT_PATH.exists():
            print(f"❌ File not found: {INPUT_PATH}")
            return
        
        df = pd.read_csv(INPUT_PATH)
        print(f"🎼 Loaded: {len(df)} tracks")
        
        # Numeric columns
        num_cols = ["popularity", "duration_ms", "danceability", "energy", "valence"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        # Drop duplicate tracks
        if "track_id" in df.columns:
            df = df.drop_duplicates(subset=["track_id"], keep="first")
        
        # Basic filtering
        mask = pd.Series(True, index=df.index)
        if "duration_ms" in df.columns:
            mask &= df["duration_ms"].between(30_000, 600_000)
        if "track_name" in df.columns:
            mask &= (df["track_name"].str.len() > 0)
        
        df = df[mask].copy()
        
        # Sort and limit by popularity
        if "popularity" in df.columns:
            df = df.sort_values("popularity", ascending=False)
        
        df = df.head(self.music_count)
        
        # Create textual audio description
        print("📝 Creating text representations with audio features...")
        
        def describe_audio_features(row):
            """Convert audio features into a short descriptive string."""
            parts = []
            
            if "track_name" in row and pd.notna(row["track_name"]):
                parts.append(str(row["track_name"]))
            
            if "artists" in row and pd.notna(row["artists"]):
                parts.append(f"by {row['artists']}")
            
            if "track_genre" in row and pd.notna(row["track_genre"]):
                parts.append(f"{row['track_genre']} genre")
            
            # Audio feature descriptions
            if pd.notna(row.get("energy", None)):
                energy = row["energy"]
                if energy > 0.7:
                    parts.append("high energy")
                elif energy < 0.3:
                    parts.append("low energy calm")
            
            if pd.notna(row.get("valence", None)):
                valence = row["valence"]
                if valence > 0.7:
                    parts.append("joyful upbeat mood")
                elif valence < 0.3:
                    parts.append("melancholic sad mood")
            
            if pd.notna(row.get("danceability", None)):
                dance = row["danceability"]
                if dance > 0.7:
                    parts.append("highly danceable")
            
            return ". ".join(parts)
        
        df["audio_description"] = df.apply(describe_audio_features, axis=1)
        
        # Apply TF-IDF reweighting
        if self.use_tfidf:
            df["text_representation"] = self.apply_tfidf_reweighting(
                df["audio_description"].tolist(),
                "music"
            )
        else:
            df["text_representation"] = df["audio_description"]
        
        # Save
        keep_cols = [
            "track_id", "track_name", "artists", "track_genre",
            "popularity", "duration_ms", "danceability", "energy", "valence",
            "audio_description", "text_representation"
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        df[keep_cols].reset_index(drop=True).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"✅ Saved {len(df)} tracks → {OUTPUT_CSV}")
    
    # ============================================================
    # DESTINATION Processing
    # ============================================================
    
    def process_destinations(self):
        """Process destination (city) data."""
        print("\n" + "="*60)
        print("🌍 Processing Destinations...")
        print("="*60)
        
        INPUT_PATH = self.raw_data_dir / "worldcitiespop.csv"
        OUTPUT_CSV = self.output_dir / "destination_processed.csv"
        
        if not INPUT_PATH.exists():
            print(f"❌ File not found: {INPUT_PATH}")
            return
        
        usecols = ["Country", "City", "AccentCity", "Region", "Population", "Latitude", "Longitude"]
        df = pd.read_csv(INPUT_PATH, encoding="latin-1", usecols=usecols, low_memory=False)
        print(f"🗺️  Loaded: {len(df):,} cities")
        
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"country": "country_code", "accentcity": "city_display"})
        
        # Basic cleaning
        for c in ["city", "city_display", "country_code", "region"]:
            df[c] = df[c].astype(str).str.strip()
        
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        
        df = df[(df["city"].str.len() > 0) & df["latitude"].notna()].copy()
        
        df["name"] = np.where(
            df["city_display"].str.len() > 0,
            df["city_display"],
            df["city"].str.title()
        )
        df["country"] = df["country_code"].str.upper()
        
        # Sort and deduplicate
        df = df.sort_values("population", ascending=False, na_position="last")
        df = df.drop_duplicates(subset=["name", "country"], keep="first")
        df = df.head(self.destination_count)
        
        # Create simple city description
        print("📝 Creating descriptions...")
        
        def make_description(row):
            parts = []
            parts.append(f"{row['name']} is a city in {row['country']}")
            if pd.notna(row["population"]) and row["population"] > 0:
                pop = int(row["population"])
                if pop > 1_000_000:
                    parts.append("major metropolitan area")
                elif pop > 500_000:
                    parts.append("large city")
                elif pop > 100_000:
                    parts.append("medium-sized city")
                parts.append(f"population {pop:,}")
            if isinstance(row.get("region", ""), str) and row["region"].strip():
                parts.append(f"in {row['region']} region")
            return ". ".join(parts) + "."
        
        df["description"] = df.apply(make_description, axis=1)
        
        # Apply TF-IDF reweighting
        if self.use_tfidf:
            df["text_representation"] = self.apply_tfidf_reweighting(
                df["description"].tolist(),
                "destination"
            )
        else:
            df["text_representation"] = df["description"]
        
        # Save
        keep_cols = ["name", "country", "city", "region", "population", 
                     "latitude", "longitude", "description", "text_representation"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        df[keep_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"✅ Saved {len(df):,} destinations → {OUTPUT_CSV}")
    
    # ============================================================
    # Main Pipeline
    # ============================================================
    
    def process_all(self):
        """Run the full processing pipeline for all datasets."""
        print("\n" + "="*60)
        print("🚀 Starting Complete Data Processing Pipeline")
        print("   with TF-IDF Reweighting")
        print("="*60)
        
        self.process_movies()
        self.process_books()
        self.process_music()
        self.process_destinations()
        
        print("\n" + "="*60)
        print("✅ All Data Processing Complete!")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob("*_processed.csv")):
            df = pd.read_csv(file)
            print(f"  ✓ {file.name}: {len(df):,} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Process data with TF-IDF reweighting (DSAN 6700 Final Project)"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw_data",
        help="Directory containing raw data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/clean_data",
        help="Output directory for processed CSVs"
    )
    parser.add_argument(
        "--movie-count",
        type=int,
        default=1000,
        help="Number of movies to keep after filtering"
    )
    parser.add_argument(
        "--book-count",
        type=int,
        default=2000,
        help="Number of books to keep after filtering"
    )
    parser.add_argument(
        "--music-count",
        type=int,
        default=2000,
        help="Number of music tracks to keep after filtering"
    )
    parser.add_argument(
        "--destination-count",
        type=int,
        default=1000,
        help="Number of destinations to keep after filtering"
    )
    parser.add_argument(
        "--no-tfidf",
        action="store_true",
        help="Disable TF-IDF reweighting (use raw text only)"
    )
    
    args = parser.parse_args()
    
    processor = DataProcessorWithTFIDF(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        movie_count=args.movie_count,
        book_count=args.book_count,
        music_count=args.music_count,
        destination_count=args.destination_count,
        use_tfidf=not args.no_tfidf
    )
    
    processor.process_all()


if __name__ == "__main__":
    main()
