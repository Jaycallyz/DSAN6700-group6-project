# src/search_service.py

import os
from functools import lru_cache
from typing import List, Dict, Literal, Tuple, Optional

import numpy as np
import pandas as pd

MediaType = Literal["movie", "book", "music"]

# ---- Paths ----

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
CLEAN_DIR = os.path.join(DATA_DIR, "clean_data")

# Embeddings (using .npz format for HF Spaces compatibility)
MOVIE_EMB_PATH = os.path.join(EMB_DIR, "movie_embeddings.npz")
BOOK_EMB_PATH = os.path.join(EMB_DIR, "book_embeddings.npz")
MUSIC_EMB_PATH = os.path.join(EMB_DIR, "music_embeddings.npz")
DEST_EMB_PATH = os.path.join(EMB_DIR, "destination_embeddings.npz")

# Cleaned metadata CSVs
MOVIE_META_PATH = os.path.join(CLEAN_DIR, "movie_processed.csv")
BOOK_META_PATH = os.path.join(CLEAN_DIR, "book_processed.csv")
MUSIC_META_PATH = os.path.join(CLEAN_DIR, "music_processed.csv")
DEST_META_PATH = os.path.join(CLEAN_DIR, "destination_processed.csv")


class RecommendationEngine:
    """
    Core recommendation engine.

    Frontend usage:

        from src.search_service import get_engine
        engine = get_engine()
        results = engine.recommend_from_media("movie", "Inception", top_k=5)

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

        # Normalize destination embeddings for cosine similarity
        self._normalize_embeddings()

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
        # Load from .npz format
        loaded = np.load(path)
        emb = loaded['embeddings']
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

    def _normalize_embeddings(self) -> None:
        """Normalize destination embeddings for cosine similarity (L2 norm)"""
        if self.dest_embeddings.size > 0:
            norms = np.linalg.norm(self.dest_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)  # Avoid division by zero
            self.dest_embeddings = self.dest_embeddings / norms

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

    def recommend_from_media(
        self,
        media_type: MediaType,
        media_title: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Given:
            media_type: "movie" | "book" | "music"
            media_title: user-facing title (movie name, book title, or track_name)
        Return:
            top_k recommended destinations as list of dicts:
            [
              {
                 "rank": 1,
                 "score": 0.17,
                 "name": ...,
                 "city": ...,
                 "country": ...,
                 "region": ...,
                 "description": ...,
                 ...
              },
              ...
            ]
        """
        media_vec = self._get_media_vector(media_type, media_title)
        if media_vec is None:
            return []

        # Normalize query vector
        q = media_vec.astype("float32").reshape(1, -1)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Compute cosine similarity with all destinations
        scores = np.dot(self.dest_embeddings, q.T).flatten()
        
        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        
        return self._format_results(top_indices, top_scores)

    def suggest_titles(
        self,
        media_type: MediaType,
        query: str,
        max_suggestions: int = 5,
    ) -> List[str]:
        """
        Fuzzy suggestions when no exact match is found.
        Looks for partial matches (contains) in appropriate columns.
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

        # Use first valid column for display
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
    print("Simple demo. Example:")
    print("   media_type = movie | book | music")

    while True:
        media_type = input("\nEnter media type (movie/book/music or 'q' to quit): ").strip().lower()
        if media_type == "q":
            break
        if media_type not in ("movie", "book", "music"):
            print("Invalid media type. Try again.")
            continue

        title = input("Enter title: ").strip()
        if not title:
            print("Empty title, try again.")
            continue

        results = engine.recommend_from_media(media_type, title, top_k=5)
        if not results:
            print("No exact match found for that title.")
            suggestions = engine.suggest_titles(media_type, title, max_suggestions=5)
            if suggestions:
                print("Did you mean:")
                for s in suggestions:
                    print(f" - {s}")
            else:
                print("No similar titles found in the sample dataset. Try another query.")
            continue

        print()
        for r in results:
            # destination_sample_wikipedia.csv: name, city, country, region, description
            name = r.get("name", "")
            city = r.get("city", "")
            country = r.get("country", "")
            score = r["score"]
            loc = ", ".join([x for x in [city, country] if x])
            print(f"{r['rank']}. {name} ({loc})  score={score:.4f}")


if __name__ == "__main__":
    _demo()