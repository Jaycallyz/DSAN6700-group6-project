# src/search_service.py

import os
from functools import lru_cache
from typing import List, Dict, Literal, Tuple, Optional

import numpy as np
import pandas as pd
import faiss

MediaType = Literal["movie", "book", "music"]

# ---- Paths ----

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
CLEAN_DIR = os.path.join(DATA_DIR, "clean_data")

# Embeddings
MOVIE_EMB_PATH = os.path.join(EMB_DIR, "movie_embeddings.npy")
BOOK_EMB_PATH = os.path.join(EMB_DIR, "book_embeddings.npy")
MUSIC_EMB_PATH = os.path.join(EMB_DIR, "music_embeddings.npy")
DEST_EMB_PATH = os.path.join(EMB_DIR, "destination_embeddings.npy")

# FAISS index for destinations
INDEX_PATH = os.path.join(EMB_DIR, "destinations_faiss_ip.index")

# Cleaned metadata CSVs
MOVIE_META_PATH = os.path.join(CLEAN_DIR, "movie_sample_50.csv")
BOOK_META_PATH = os.path.join(CLEAN_DIR, "book_sample_100.csv")
MUSIC_META_PATH = os.path.join(CLEAN_DIR, "music_sample_100.csv")
DEST_META_PATH = os.path.join(CLEAN_DIR, "destination_sample_wikipedia.csv")


class RecommendationEngine:
    """
    Core recommendation engine.
    
    Option A: Combined mode
    - The user must input movie + book + music (all three)
    - The system combines the three embeddings
    - Outputs a unified list of recommended destinations

    Frontend usage:

        from src.search_service import get_engine
        engine = get_engine()
        
        # Combined recommendation (all three inputs are required)
        results, status = engine.recommend_from_combined_media(
            movie_title="Inception",
            book_title="1984",
            music_title="Bohemian Rhapsody",
            top_k=5
        )

    Returns: list of dicts with destination info + similarity score.
    """

    def __init__(self) -> None:
        # ---- Load embeddings ----
        self.dest_embeddings = self._load_embeddings(DEST_EMB_PATH, allow_missing=False)
        self.movie_embeddings = self._load_embeddings(MOVIE_EMB_PATH, allow_missing=True)
        self.book_embeddings = self._load_embeddings(BOOK_EMB_PATH, allow_missing=True)
        self.music_embeddings = self._load_embeddings(MUSIC_EMB_PATH, allow_missing=True)

        # ---- Load metadata ----
        # Destinations metadata is required: used to label FAISS results
        self.dest_meta = self._load_metadata(DEST_META_PATH, required=True)

        # Media metadata
        self.movie_meta = self._load_metadata(MOVIE_META_PATH)
        self.book_meta = self._load_metadata(BOOK_META_PATH)
        self.music_meta = self._load_metadata(MUSIC_META_PATH)

        # ---- Load FAISS index ----
        self.index = self._load_index()

        # ---- Build lookup tables ----
        # Movies: title / original_title
        self.movie_lookup = self._build_lookup(
            self.movie_meta,
            ["title", "original_title"],
        )

        # Books: title (and original_title if present)
        self.book_lookup = self._build_lookup(
            self.book_meta,
            ["title", "original_title"],
        )

        # Music
        self.music_lookup = self._build_lookup(
            self.music_meta,
            ["track_name", "track_id"],
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_embeddings(path: str, allow_missing: bool) -> np.ndarray:
        if not os.path.exists(path):
            if allow_missing:
                return np.empty((0, 0), dtype="float32")
            raise FileNotFoundError(f"Embeddings not found at {path}")
        emb = np.load(path)
        if emb.ndim != 2:
            raise ValueError(f"Expected 2D embeddings at {path}, got shape {emb.shape}")
        return emb.astype("float32")

    @staticmethod
    def _load_metadata(path: str, required: bool = False) -> pd.DataFrame:
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(f"Metadata CSV not found at {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    def _load_index(self) -> faiss.Index:
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                f"Run `python src/build_faiss.py` to create it."
            )
        index = faiss.read_index(INDEX_PATH)

        # sanity check
        if self.dest_embeddings.size > 0 and index.ntotal != self.dest_embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between FAISS index size ({index.ntotal}) "
                f"and destination_embeddings rows ({self.dest_embeddings.shape[0]})."
            )
        return index

    # ------------------------------------------------------------------
    # Lookup construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_lookup(df: pd.DataFrame, title_columns: List[str]) -> Dict[str, int]:
        """
        Build case-insensitive mapping from one or more columns to row index.

        For each row, for each of the given columns that exist:
            key = lowercased string value
            value = row index

        Later we resolve user input by lowercasing and looking up here.
        """
        if df.empty:
            return {}

        valid_cols = [c for c in title_columns if c in df.columns]
        if not valid_cols:
            return {}

        lookup: Dict[str, int] = {}
        for i, row in df.iterrows():
            for col in valid_cols:
                val = str(row[col]).strip()
                if val:
                    lookup[val.lower()] = i
        return lookup

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend_from_combined_media(
        self,
        movie_title: str,
        book_title: str,
        music_title: str,
        weights: Optional[Tuple[float, float, float]] = None,
        top_k: int = 5,
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Combined recommendation: fuse embeddings from movie, book, and music.
        
        Option A: all three media titles are required.
        
        Args:
            movie_title: movie title (required)
            book_title: book title (required)
            music_title: music title (required)
            weights: (movie_weight, book_weight, music_weight), default (1.0, 1.0, 1.0).
                     Used to adjust the importance of each media type.
            top_k: number of recommendations to return.
            
        Returns:
            (recommendations, status_info)
            - recommendations: list of destination recommendations
            - status_info: status of each media lookup
            
        Example:
            results, status = engine.recommend_from_combined_media(
                movie_title="Inception",
                book_title="1984",
                music_title="Bohemian Rhapsody",
                weights=(1.0, 1.0, 1.0),  # equal weights
                top_k=5
            )
        """
        # Validate inputs: all three must be non-empty
        if not movie_title or not movie_title.strip():
            raise ValueError("movie_title cannot be empty (Option A requires all three media inputs).")
        if not book_title or not book_title.strip():
            raise ValueError("book_title cannot be empty (Option A requires all three media inputs).")
        if not music_title or not music_title.strip():
            raise ValueError("music_title cannot be empty (Option A requires all three media inputs).")
        
        if weights is None:
            weights = (1.0, 1.0, 1.0)
        
        # Collect embeddings and status
        vectors = []
        status_info: Dict[str, str] = {}
        
        # Movie
        movie_vec = self._get_media_vector("movie", movie_title)
        if movie_vec is not None:
            vectors.append(movie_vec)
            status_info["movie"] = f"✅ Found: {movie_title}"
        else:
            status_info["movie"] = f"❌ Not found: {movie_title}"
        
        # Book
        book_vec = self._get_media_vector("book", book_title)
        if book_vec is not None:
            vectors.append(book_vec)
            status_info["book"] = f"✅ Found: {book_title}"
        else:
            status_info["book"] = f"❌ Not found: {book_title}"
        
        # Music
        music_vec = self._get_media_vector("music", music_title)
        if music_vec is not None:
            vectors.append(music_vec)
            status_info["music"] = f"✅ Found: {music_title}"
        else:
            status_info["music"] = f"❌ Not found: {music_title}"
        
        # If none were found, return empty
        if len(vectors) == 0:
            return [], status_info
        
        # If only some were found, still proceed using the available ones.
        # Missing ones are indicated in status_info.
        
        # Combine embeddings (weighted average).
        # Only use embeddings that were actually found.
        if len(vectors) == 3:
            # All three found: use full weights
            weights_to_use = weights
        elif len(vectors) == 2:
            # Only two found: adjust weights automatically
            # Determine which one is missing
            if movie_vec is None:
                weights_to_use = (weights[1], weights[2])  # book, music
            elif book_vec is None:
                weights_to_use = (weights[0], weights[2])  # movie, music
            else:  # music_vec is None
                weights_to_use = (weights[0], weights[1])  # movie, book
        else:
            # Only one found: just use that vector (weight is effectively 1)
            weights_to_use = (1.0,)
        
        vectors_array = np.array(vectors, dtype="float32")
        weights_array = np.array(weights_to_use, dtype="float32")
        
        # Normalize weights
        weights_array = weights_array / weights_array.sum()
        
        # Weighted average
        combined_vec = np.average(vectors_array, axis=0, weights=weights_array)
        
        # Normalize for cosine similarity
        q = combined_vec.reshape(1, -1)
        faiss.normalize_L2(q)
        
        # Search
        scores, indices = self.index.search(q, top_k)
        results = self._format_results(indices[0], scores[0])
        
        return results, status_info

    def suggest_titles(
        self,
        media_type: MediaType,
        query: str,
        max_suggestions: int = 5,
    ) -> List[str]:
        """
        Fuzzy suggestions when no exact match is found.
        Looks for partial matches (substring containment) in appropriate columns.
        """
        query_l = query.strip().lower()
        if not query_l:
            return []

        if media_type == "movie":
            df = self.movie_meta
            cols = ["title", "original_title"]
        elif media_type == "book":
            df = self.book_meta
            cols = ["title", "original_title"]
        elif media_type == "music":
            df = self.music_meta
            cols = ["track_name", "track_id"]
        else:
            return []

        if df.empty:
            return []

        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return []

        mask = False
        for col in valid_cols:
            col_mask = df[col].astype(str).str.lower().str.contains(query_l)
            mask = col_mask if isinstance(mask, bool) else (mask | col_mask)

        if isinstance(mask, bool):
            return []

        # Use the first valid column for display
        display_col = valid_cols[0]
        matches = df.loc[mask, display_col].astype(str).drop_duplicates().head(max_suggestions)
        return matches.tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_media_vector(
        self,
        media_type: MediaType,
        media_title: str,
    ) -> Optional[np.ndarray]:
        """
        Map user-provided title -> embedding row.

        Movies:
          - match by title / original_title
        Books:
          - match by title / (and original_title if present)
        Music:
          - match by track_name (preferred) or track_id (fallback)
        """
        key = media_title.strip().lower()
        if not key:
            return None

        if media_type == "movie":
            idx = self.movie_lookup.get(key)
            return None if idx is None or self.movie_embeddings.size == 0 else self.movie_embeddings[idx]

        if media_type == "book":
            idx = self.book_lookup.get(key)
            return None if idx is None or self.book_embeddings.size == 0 else self.book_embeddings[idx]

        if media_type == "music":
            idx = self.music_lookup.get(key)
            return None if idx is None or self.music_embeddings.size == 0 else self.music_embeddings[idx]

        raise ValueError(f"Unsupported media_type: {media_type}")

    def _format_results(
        self,
        indices: np.ndarray,
        scores: np.ndarray,
    ) -> List[Dict]:
        """
        Turn FAISS output into a list of rich destination records.
        Assumes DEST_META_PATH = destination_sample_wikipedia.csv
        with columns: name, country, city, region, description, ...
        """
        results: List[Dict] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            idx = int(idx)
            if idx < 0 or idx >= len(self.dest_meta):
                continue

            row = self.dest_meta.iloc[idx].to_dict()
            row_out = {
                "rank": rank,
                "score": float(score),
                **row,
            }
            results.append(row_out)

        return results


# ----------------------------------------------------------------------
# Global accessor for reuse (esp. in Streamlit / Hugging Face Spaces)
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    """
    Cached singleton engine.

    Ensures heavy resources (embeddings, FAISS index) load only once
    per process.
    """
    return RecommendationEngine()


# ----------------------------------------------------------------------
# Simple CLI demo for local testing
# ----------------------------------------------------------------------

def _demo() -> None:
    engine = get_engine()
    print("=" * 60)
    print("Combined Recommendation System - Option A")
    print("You must enter: Movie + Book + Music (all three)")
    print("=" * 60)
    
    print("\nPlease enter three media titles:\n")
    
    movie = input("🎬 Movie title: ").strip()
    book = input("📚 Book title: ").strip()
    music = input("🎵 Music title: ").strip()
    
    # Validate input
    if not movie or not book or not music:
        print("\n❌ Error: all three titles are required!")
        return
    
    try:
        results, status = engine.recommend_from_combined_media(
            movie_title=movie,
            book_title=book,
            music_title=music,
            top_k=5
        )
        
        print("\n" + "=" * 60)
        print("Lookup status:")
        print(f"  🎬 Movie: {status['movie']}")
        print(f"  📚 Book:  {status['book']}")
        print(f"  🎵 Music: {status['music']}")
        print("=" * 60)
        
        if not results:
            print("\n❌ No valid media found, cannot generate recommendations.")
            print("\n💡 Tip: please check that the titles are correct. You can try:")
            
            # Suggestions
            if "❌" in status["movie"]:
                suggestions = engine.suggest_titles("movie", movie, max_suggestions=3)
                if suggestions:
                    print(f"\n  Movie suggestions: {', '.join(suggestions)}")
            
            if "❌" in status["book"]:
                suggestions = engine.suggest_titles("book", book, max_suggestions=3)
                if suggestions:
                    print(f"  Book suggestions: {', '.join(suggestions)}")
            
            if "❌" in status["music"]:
                suggestions = engine.suggest_titles("music", music, max_suggestions=3)
                if suggestions:
                    print(f"  Music suggestions: {', '.join(suggestions)}")
            
            return
        
        print(f"\n🌍 Combined recommendation results (Top {len(results)}):\n")
        for r in results:
            name = r.get("name", "")
            city = r.get("city", "")
            country = r.get("country", "")
            score = r["score"]
            loc = ", ".join([x for x in [city, country] if x])
            print(f"  {r['rank']}. {name} ({loc})  similarity={score:.4f}")
    
    except ValueError as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    _demo()

