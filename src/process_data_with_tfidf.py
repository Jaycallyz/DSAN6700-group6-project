#!/usr/bin/env python
"""
Stratified sampling data processing script + TF-IDF reweighting.

Stratified Sampling with TF-IDF Reweighting

According to DSAN 6700 Final Project requirements:
- Ensure content diversity
- Mitigate popularity bias
- Balance quality and serendipity

Sampling strategy:
- 40%: High-quality classics (ensure overall quality)
- 30%: Strong but unique items (balance quality and diversity)
- 20%: Genre/topic diversity (coverage)
- 10%: Random niche items (serendipity)
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
import json


class StratifiedDataProcessor:
    """
    Stratified sampling data processor with optional TF-IDF reweighting.
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
        print(f"Stratified Sampling Data Processor")
        print(f"{'='*60}")
        print(f"Strategy: 40% High-Quality + 30% Unique + 20% Diverse + 10% Serendipity")
        print(f"Movie: {movie_count} | Book: {book_count} | Music: {music_count} | Dest: {destination_count}")
        print(f"TF-IDF: {'Enabled' if use_tfidf else 'Disabled'}")
        print(f"{'='*60}\n")
    
    # ============================================================
    # TF-IDF Utility
    # ============================================================
    
    def apply_tfidf_reweighting(self, texts: list, column_name: str = "text") -> list:
        """
        Apply TF-IDF-based reweighting to a list of texts.
        Emphasizes distinctive terms and downweights very common ones.
        """
        if not self.use_tfidf or len(texts) == 0:
            return texts
        
        print(f"  🔄 Applying TF-IDF to {column_name}...")
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2  # must appear in at least 2 documents
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            reweighted_texts = []
            for i, text in enumerate(texts):
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                top_indices = tfidf_scores.argsort()[-15:][::-1]
                top_words = [feature_names[idx] for idx in top_indices if tfidf_scores[idx] > 0]
                
                emphasized = " ".join(top_words[:10])
                reweighted = f"{text} {emphasized}"
                reweighted_texts.append(reweighted)
            
            print(f"     ✅ TF-IDF applied")
            return reweighted_texts
        
        except Exception as e:
            print(f"     ⚠️  TF-IDF failed: {e}, using original texts")
            return texts
    
    # ============================================================
    # Stratified Sampling Utilities
    # ============================================================
    
    def stratified_sample(
        self, 
        df: pd.DataFrame, 
        total_n: int,
        strata_config: dict,
        sort_column: str = None
    ) -> pd.DataFrame:
        """
        Core stratified sampling function.
        
        Args:
            df: input DataFrame
            total_n: target total number of samples
            strata_config: mapping {stratum_name: (filter_func, ratio)}
            sort_column: optional column used for sorting within each stratum
            
        Returns:
            DataFrame after stratified sampling
        """
        samples = []
        
        for stratum_name, (filter_func, ratio) in strata_config.items():
            n_samples = int(total_n * ratio)
            
            # Apply filter condition
            stratum_df = df[filter_func(df)].copy()
            
            if len(stratum_df) == 0:
                print(f"     ⚠️  {stratum_name}: no data, skipping")
                continue
            
            # Sort within stratum if requested
            if sort_column and sort_column in stratum_df.columns:
                stratum_df = stratum_df.sort_values(sort_column, ascending=False)
            
            # Sample
            n_available = len(stratum_df)
            n_to_sample = min(n_samples, n_available)
            
            sampled = stratum_df.head(n_to_sample)
            sampled["_stratum"] = stratum_name
            samples.append(sampled)
            
            print(f"     ✓ {stratum_name}: {n_to_sample}/{n_samples} (available: {n_available})")
        
        # Combine all strata
        result = pd.concat(samples, ignore_index=True)
        
        # If total is still below target, randomly fill from leftovers
        if len(result) < total_n:
            remaining = total_n - len(result)
            leftover = df[~df.index.isin(result.index)]
            if len(leftover) > 0:
                extra = leftover.sample(n=min(remaining, len(leftover)), random_state=42)
                extra["_stratum"] = "Extra"
                result = pd.concat([result, extra], ignore_index=True)
                print(f"     ✓ Extra fill: {len(extra)} samples")
        
        return result
    
    # ============================================================
    # MOVIE - Stratified Sampling
    # ============================================================
    
    def process_movies(self):
        """Process movies with stratified sampling."""
        print("\n" + "="*60)
        print("🎬 Processing Movies (Stratified Sampling)")
        print("="*60)
        
        INPUT_PATH = self.raw_data_dir / "tmdb_5000_movies.csv"
        OUTPUT_CSV = self.output_dir / "movie_processed.csv"
        
        if not INPUT_PATH.exists():
            print(f"❌ File not found: {INPUT_PATH}")
            return
        
        df = pd.read_csv(INPUT_PATH)
        print(f"📊 Loaded: {len(df)} movies")
        
        # Basic cleaning
        for col in ["runtime", "vote_average", "vote_count", "popularity"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df["overview"] = df["overview"].fillna("").astype(str)
        df["title"] = df["title"].fillna("").astype(str)
        
        # Base filtering
        mask = (
            (df["title"].str.len() > 0) &
            (df["overview"].str.len() > 20)
        )
        if "status" in df.columns:
            mask &= (df["status"].fillna("") == "Released")
        
        df = df[mask].copy()
        
        # Deduplicate
        if "original_title" in df.columns:
            df = df.drop_duplicates(subset=["original_title"], keep="first")
        
        # Compute weighted rating
        df_votes = df[df["vote_count"].notna() & (df["vote_count"] > 0)].copy()
        if len(df_votes) > 0:
            C = df_votes["vote_average"].mean()
            m = df_votes["vote_count"].quantile(0.25)
            v = df_votes["vote_count"]
            R = df_votes["vote_average"]
            df_votes["weighted_rating"] = (v/(v+m))*R + (m/(v+m))*C
            df = df.merge(df_votes[["original_title", "weighted_rating"]], on="original_title", how="left")
        else:
            df["weighted_rating"] = df.get("vote_average", 0)
        
        print(f"\n  📋 Stratified Sampling Strategy:")
        
        # Strata config
        strata = {
            "High-Score Classics (40%)": (
                lambda d: (d["weighted_rating"] >= 8.0) & (d["vote_count"] >= 10000),
                0.40
            ),
            "Strong but Less Mainstream (30%)": (
                lambda d: (d["weighted_rating"] >= 7.0) & (d["weighted_rating"] < 8.0) & (d["vote_count"] >= 1000),
                0.30
            ),
            "Genre/Theme Variety (20%)": (
                lambda d: (d["weighted_rating"] >= 6.5) & (d["vote_count"] >= 500),
                0.20
            ),
            "Random Niche (10%)": (
                lambda d: (d["weighted_rating"] >= 6.0) & (d["vote_count"] >= 100),
                0.10
            )
        }
        
        sampled = self.stratified_sample(df, self.movie_count, strata, "weighted_rating")
        
        # Create text representation
        print(f"\n  📝 Creating text representations...")
        text_parts = []
        for _, row in sampled.iterrows():
            parts = [str(row["title"])]
            if "overview" in row and row["overview"]:
                parts.append(str(row["overview"]))
            if "genres" in row and row["genres"]:
                parts.append(str(row["genres"]))
            text_parts.append(". ".join(parts))
        
        sampled["text_representation"] = self.apply_tfidf_reweighting(text_parts, "movie")
        
        # Save
        keep_cols = [
            "title", "original_title", "overview", "genres", "release_date",
            "runtime", "vote_average", "vote_count", "popularity",
            "weighted_rating", "_stratum", "text_representation"
        ]
        keep_cols = [c for c in keep_cols if c in sampled.columns]
        
        sampled[keep_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"\n✅ Saved {len(sampled)} movies → {OUTPUT_CSV}")
        print(f"\n  📊 Stratum distribution:")
        print(sampled["_stratum"].value_counts().to_string())
    
    # ============================================================
    # BOOK - Stratified Sampling
    # ============================================================
    
    def process_books(self):
        """Process books with stratified sampling."""
        print("\n" + "="*60)
        print("📚 Processing Books (Stratified Sampling)")
        print("="*60)
        
        BASE = self.raw_data_dir / "books"
        OUTPUT_CSV = self.output_dir / "book_processed.csv"
        
        books_path = BASE / "books.csv"
        book_tags_path = BASE / "book_tags.csv"
        tags_path = BASE / "tags.csv"
        
        if not books_path.exists():
            print(f"❌ File not found: {books_path}")
            return
        
        # Load data
        books = pd.read_csv(books_path)
        book_tags = pd.read_csv(book_tags_path)
        tags = pd.read_csv(tags_path)
        
        print(f"📖 Loaded: {books.shape[0]} books")
        
        # Merge tags
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
        
        books = books.merge(book_tag_grouped, left_on="book_id", right_on="goodreads_book_id", how="left")
        
        # Cleaning
        books["original_title"] = books["original_title"].fillna(books["title"])
        books["authors"] = books["authors"].fillna("Unknown")
        books["average_rating"] = pd.to_numeric(books["average_rating"], errors="coerce")
        books["ratings_count"] = pd.to_numeric(books["ratings_count"], errors="coerce")
        books["original_publication_year"] = pd.to_numeric(books["original_publication_year"], errors="coerce")
        books["tag_list"] = books["tag_list"].fillna("")
        
        # Base filter
        books = books[
            (books["average_rating"] >= 3.0) &
            (books["ratings_count"] >= 50) &
            (books["original_publication_year"].between(1800, 2025))
        ].copy()
        
        # Weighted score
        C = books["average_rating"].mean()
        m = books["ratings_count"].quantile(0.40)
        v = books["ratings_count"]
        R = books["average_rating"]
        books["weighted_score"] = (v/(v+m))*R + (m/(v+m))*C
        
        print(f"\n  📋 Stratified Sampling Strategy:")
        
        # Strata config
        strata = {
            "Bestselling Classics (40%)": (
                lambda d: (d["average_rating"] >= 4.3) & (d["ratings_count"] >= 10000),
                0.40
            ),
            "High-Quality Works (30%)": (
                lambda d: (d["average_rating"] >= 4.0) & (d["average_rating"] < 4.3) & (d["ratings_count"] >= 1000),
                0.30
            ),
            "Thematically Diverse (20%)": (
                lambda d: (d["average_rating"] >= 3.8) & (d["ratings_count"] >= 500),
                0.20
            ),
            "Across Time Periods (10%)": (
                lambda d: (d["average_rating"] >= 3.5) & (d["ratings_count"] >= 100),
                0.10
            )
        }
        
        sampled = self.stratified_sample(books, self.book_count, strata, "weighted_score")
        
        # Create text representation
        print(f"\n  📝 Creating text representations...")
        text_parts = []
        for _, row in sampled.iterrows():
            parts = [str(row["title"]), str(row["authors"])]
            if row["tag_list"]:
                parts.append(str(row["tag_list"]))
            text_parts.append(". ".join(parts))
        
        sampled["text_representation"] = self.apply_tfidf_reweighting(text_parts, "book")
        
        # Save
        keep_cols = [
            "book_id", "title", "original_title", "authors",
            "original_publication_year", "average_rating", "ratings_count",
            "tag_list", "weighted_score", "_stratum", "text_representation"
        ]
        keep_cols = [c for c in keep_cols if c in sampled.columns]
        
        sampled[keep_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"\n✅ Saved {len(sampled)} books → {OUTPUT_CSV}")
        print(f"\n  📊 Stratum distribution:")
        print(sampled["_stratum"].value_counts().to_string())
    
    # ============================================================
    # MUSIC - Stratified Sampling
    # ============================================================
    
    def process_music(self):
        """Process music tracks with stratified sampling."""
        print("\n" + "="*60)
        print("🎵 Processing Music (Stratified Sampling)")
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
        
        # Deduplicate
        if "track_id" in df.columns:
            df = df.drop_duplicates(subset=["track_id"], keep="first")
        
        # Filter
        mask = pd.Series(True, index=df.index)
        if "duration_ms" in df.columns:
            mask &= df["duration_ms"].between(30_000, 600_000)
        if "track_name" in df.columns:
            mask &= (df["track_name"].str.len() > 0)
        
        df = df[mask].copy()
        df["popularity"] = df["popularity"].fillna(0)
        
        print(f"\n  📋 Stratified Sampling Strategy:")
        
        # Strata configuration (based on popularity + audio features)
        strata = {
            "Highly Popular (30%)": (
                lambda d: d["popularity"] >= 70,
                0.30
            ),
            "Moderately Popular (30%)": (
                lambda d: (d["popularity"] >= 40) & (d["popularity"] < 70),
                0.30
            ),
            "Emotionally Varied (20%)": (
                lambda d: (d["popularity"] >= 20) & (d["popularity"] < 40),
                0.20
            ),
            "Independent / Niche (20%)": (
                lambda d: d["popularity"] < 20,
                0.20
            )
        }
        
        sampled = self.stratified_sample(df, self.music_count, strata, "popularity")
        
        # Audio features -> text
        print(f"\n  📝 Creating audio feature descriptions...")
        
        def describe_audio(row):
            parts = []
            if "track_name" in row and pd.notna(row["track_name"]):
                parts.append(str(row["track_name"]))
            if "artists" in row and pd.notna(row["artists"]):
                parts.append(f"by {row['artists']}")
            if "track_genre" in row and pd.notna(row["track_genre"]):
                parts.append(f"{row['track_genre']} genre")
            
            # Audio features
            if pd.notna(row.get("energy", None)):
                if row["energy"] > 0.7:
                    parts.append("high energy")
                elif row["energy"] < 0.3:
                    parts.append("calm low energy")
            
            if pd.notna(row.get("valence", None)):
                if row["valence"] > 0.7:
                    parts.append("joyful upbeat")
                elif row["valence"] < 0.3:
                    parts.append("melancholic sad")
            
            if pd.notna(row.get("danceability", None)) and row["danceability"] > 0.7:
                parts.append("danceable")
            
            return ". ".join(parts)
        
        sampled["audio_description"] = sampled.apply(describe_audio, axis=1)
        sampled["text_representation"] = self.apply_tfidf_reweighting(
            sampled["audio_description"].tolist(), "music"
        )
        
        # Save
        keep_cols = [
            "track_id", "track_name", "artists", "track_genre",
            "popularity", "danceability", "energy", "valence",
            "audio_description", "_stratum", "text_representation"
        ]
        keep_cols = [c for c in keep_cols if c in sampled.columns]
        
        sampled[keep_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"\n✅ Saved {len(sampled)} tracks → {OUTPUT_CSV}")
        print(f"\n  📊 Stratum distribution:")
        print(sampled["_stratum"].value_counts().to_string())
    
    # ============================================================
    # DESTINATION - Stratified Sampling
    # ============================================================
    
    def process_destinations(self):
        """Process destinations (cities) with stratified sampling."""
        print("\n" + "="*60)
        print("🌍 Processing Destinations (Stratified Sampling)")
        print("="*60)
        
        INPUT_PATH = self.raw_data_dir / "worldcitiespop.csv"
        OUTPUT_CSV = self.output_dir / "destination_processed.csv"
        
        if not INPUT_PATH.exists():
            print(f"❌ File not found: {INPUT_PATH}")
            return
        
        usecols = ["Country", "City", "AccentCity", "Region", "Population", "Latitude", "Longitude"]
        df = pd.read_csv(INPUT_PATH, encoding="latin-1", usecols=usecols, low_memory=False)
        print(f"🗺️  Loaded: {len(df):,} cities")
        
        # Cleaning
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"country": "country_code", "accentcity": "city_display"})
        
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
        df["population"] = df["population"].fillna(0)
        
        # Deduplicate
        df = df.sort_values("population", ascending=False)
        df = df.drop_duplicates(subset=["name", "country"], keep="first")
        
        print(f"\n  📋 Stratified Sampling Strategy:")
        
        # Strata configuration (based on population)
        strata = {
            "Metropolitan (30%)": (
                lambda d: d["population"] > 1_000_000,
                0.30
            ),
            "Large Cities (30%)": (
                lambda d: (d["population"] > 500_000) & (d["population"] <= 1_000_000),
                0.30
            ),
            "Medium Cities (20%)": (
                lambda d: (d["population"] > 100_000) & (d["population"] <= 500_000),
                0.20
            ),
            "Small Towns (20%)": (
                lambda d: d["population"] <= 100_000,
                0.20
            )
        }
        
        sampled = self.stratified_sample(df, self.destination_count, strata, "population")
        
        # Generate descriptions
        print(f"\n  📝 Creating descriptions...")
        
        def make_description(row):
            parts = []
            parts.append(f"{row['name']} is a city in {row['country']}")
            
            pop = row["population"]
            if pop > 1_000_000:
                parts.append("major metropolitan area")
            elif pop > 500_000:
                parts.append("large city")
            elif pop > 100_000:
                parts.append("medium-sized city")
            else:
                parts.append("charming town")
            
            if pop > 0:
                parts.append(f"population {int(pop):,}")
            
            if isinstance(row.get("region", ""), str) and row["region"].strip():
                parts.append(f"in {row['region']} region")
            
            return ". ".join(parts) + "."
        
        sampled["description"] = sampled.apply(make_description, axis=1)
        sampled["text_representation"] = self.apply_tfidf_reweighting(
            sampled["description"].tolist(), "destination"
        )
        
        # Save
        keep_cols = [
            "name", "country", "city", "region", "population",
            "latitude", "longitude", "description", "_stratum", "text_representation"
        ]
        keep_cols = [c for c in keep_cols if c in sampled.columns]
        
        sampled[keep_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        
        print(f"\n✅ Saved {len(sampled):,} destinations → {OUTPUT_CSV}")
        print(f"\n  📊 Stratum distribution:")
        print(sampled["_stratum"].value_counts().to_string())
    
    # ============================================================
    # Main Pipeline
    # ============================================================
    
    def process_all(self):
        """Run the full stratified sampling pipeline for all datasets."""
        print("\n" + "="*60)
        print("🚀 Starting Stratified Sampling Pipeline")
        print("="*60)
        
        self.process_movies()
        self.process_books()
        self.process_music()
        self.process_destinations()
        
        print("\n" + "="*60)
        print("✅ All Processing Complete!")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print("\nGenerated files:")
        
        total_rows = 0
        for file in sorted(self.output_dir.glob("*_processed.csv")):
            df = pd.read_csv(file)
            total_rows += len(df)
            print(f"  ✓ {file.name}: {len(df):,} rows")
            
            # Show stratification summary if applicable
            if "_stratum" in df.columns:
                print(f"     Strata: {df['_stratum'].nunique()} categories")
        
        print(f"\nTotal: {total_rows:,} records")


def main():
    parser = argparse.ArgumentParser(
        description="Stratified Sampling + TF-IDF (DSAN 6700)"
    )
    parser.add_argument("--raw-data-dir", type=str, default="data/raw_data")
    parser.add_argument("--output-dir", type=str, default="data/clean_data")
    parser.add_argument("--movie-count", type=int, default=1000)
    parser.add_argument("--book-count", type=int, default=2000)
    parser.add_argument("--music-count", type=int, default=2000)
    parser.add_argument("--destination-count", type=int, default=1000)
    parser.add_argument("--no-tfidf", action="store_true", help="Disable TF-IDF reweighting")
    
    args = parser.parse_args()
    
    processor = StratifiedDataProcessor(
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
