"""
Search & Recommendation Service for Cultural Media Travel Recommender.

This module loads pre-computed embeddings for:
    - movies
    - books
    - music tracks
    - travel destinations

It then builds a similarity index (FAISS if available, otherwise
sklearn NearestNeighbors) and exposes a `RecommendationEngine`
with the following key methods:

    - recommend_from_combined_media(movie_title, book_title, music_title, ...)
    - suggest_titles(media_type, query, max_suggestions)

The engine is used by `app.py` via:

    from search_service import get_engine
    engine = get_engine()

"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional FAISS import with sklearn fallback
# ---------------------------------------------------------------------
FAISS_AVAILABLE = True
try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - only used when FAISS is missing
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors  # type: ignore

    logger.warning(
        "FAISS is not available. Falling back to sklearn.NearestNeighbors "
        "(slower but functional)."
    )

MediaType = Literal["movie", "book", "music"]

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EMB_DIR = DATA_DIR / "embeddings"
CLEAN_DIR = DATA_DIR / "clean_data"

# Embedding files – we primarily use NPZ, but keep NPY as a fallback
DEST_EMB_NPZ = EMB_DIR / "destination_embeddings.npz"
DEST_EMB_NPY = EMB_DIR / "destination_embeddings.npy"

MOVIE_EMB_NPZ = EMB_DIR / "movie_embeddings.npz"
MOVIE_EMB_NPY = EMB_DIR / "movie_embeddings.npy"

BOOK_EMB_NPZ = EMB_DIR / "book_embeddings.npz"
BOOK_EMB_NPY = EMB_DIR / "book_embeddings.npy"

MUSIC_EMB_NPZ = EMB_DIR / "music_embeddings.npz"
MUSIC_EMB_NPY = EMB_DIR / "music_embeddings.npy"

FAISS_INDEX_PATH = EMB_DIR / "destinations_faiss_ip.index"

# Metadata CSVs
MOVIE_META_PATH = CLEAN_DIR / "movie_processed.csv"
BOOK_META_PATH = CLEAN_DIR / "book_processed.csv"
MUSIC_META_PATH = CLEAN_DIR / "music_processed.csv"
DEST_META_PATH = CLEAN_DIR / "destination_processed.csv"


# =====================================================================
#                         Recommendation Engine
# =====================================================================
class RecommendationEngine:
    """
    Core engine that powers the fused travel recommendation logic.

    Responsibilities:
        * Load embeddings (destinations + media items)
        * Load metadata tables
        * Build a similarity index over destinations
        * Provide APIs for:
            - fused media → destination recommendations
            - fuzzy title suggestions

    The engine is designed to be stateless after initialization so that
    it can be cached via `get_engine()` and reused across Streamlit
    interactions.
    """

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    def __init__(self) -> None:
        logger.info("Initializing RecommendationEngine...")

        # ---- Load embeddings ----
        self.dest_embeddings = self._load_embeddings_npz_first(
            DEST_EMB_NPZ, DEST_EMB_NPY, name="destination", allow_missing=False
        )
        self.movie_embeddings = self._load_embeddings_npz_first(
            MOVIE_EMB_NPZ, MOVIE_EMB_NPY, name="movie", allow_missing=True
        )
        self.book_embeddings = self._load_embeddings_npz_first(
            BOOK_EMB_NPZ, BOOK_EMB_NPY, name="book", allow_missing=True
        )
        self.music_embeddings = self._load_embeddings_npz_first(
            MUSIC_EMB_NPZ, MUSIC_EMB_NPY, name="music", allow_missing=True
        )

        # ---- Load metadata ----
        self.dest_meta = self._load_metadata(DEST_META_PATH, required=True)
        self.movie_meta = self._load_metadata(MOVIE_META_PATH)
        self.book_meta = self._load_metadata(BOOK_META_PATH)
        self.music_meta = self._load_metadata(MUSIC_META_PATH)

        # ---- Build similarity index over destinations ----
        self.index = self._load_or_build_index()

        # ---- Build title lookup tables ----
        self.movie_lookup = self._build_lookup(self.movie_meta, ["title", "original_title"])
        self.book_lookup = self._build_lookup(self.book_meta, ["title", "original_title"])
        self.music_lookup = self._build_lookup(self.music_meta, ["track_name", "track_id"])

        logger.info("RecommendationEngine initialized successfully.")

    # =================================================================
    #                         Loading helpers
    # =================================================================
    def _load_embeddings_npz_first(
        self,
        npz_path: Path,
        npy_path: Path,
        name: str,
        allow_missing: bool,
    ) -> np.ndarray:
        """
        Load an embedding matrix, preferring `.npz` over `.npy`.

        The `.npz` format is non-pickled and is safer for distribution
        (e.g., on Hugging Face Hub). This function will:

        1. Try to load from NPZ
        2. If missing, fall back to NPY
        3. If still missing:
            - return empty array if `allow_missing=True`
            - raise FileNotFoundError otherwise

        Args:
            npz_path: Path to the `.npz` file.
            npy_path: Path to the `.npy` fallback file.
            name: Human-readable name for logs (e.g., "movie").
            allow_missing: If True, missing file returns an empty array.

        Returns:
            A 2D numpy array of dtype float32.
        """
        # 1) Try NPZ
        if npz_path.exists():
            logger.info("Loading %s embeddings from NPZ: %s", name, npz_path)
            data = np.load(npz_path)
            # We accept any key; if you saved with `embeddings=arr` we
            # use that, otherwise we just take the first key.
            key = "embeddings" if "embeddings" in data.files else data.files[0]
            arr = np.asarray(data[key], dtype="float32")
            if arr.ndim != 2:
                raise ValueError(f"{name} embeddings must be 2D, got shape {arr.shape}")
            return arr

        # 2) Fallback to NPY
        if npy_path.exists():
            logger.warning(
                "%s NPZ not found (%s); falling back to NPY: %s",
                name,
                npz_path,
                npy_path,
            )
            arr = np.load(npy_path).astype("float32")
            if arr.ndim != 2:
                raise ValueError(f"{name} embeddings must be 2D, got shape {arr.shape}")
            return arr

        # 3) Missing
        msg = f"{name.capitalize()} embeddings missing: {npz_path} and {npy_path}."
        if allow_missing:
            logger.warning(msg + " Continuing with empty embeddings.")
            return np.empty((0, 0), dtype="float32")

        logger.error(msg)
        raise FileNotFoundError(msg)

    @staticmethod
    def _load_metadata(path: Path, required: bool = False) -> pd.DataFrame:
        """
        Load a processed metadata CSV file.

        Args:
            path: Path to the CSV file.
            required: If True, missing file raises FileNotFoundError.

        Returns:
            DataFrame containing metadata. May be empty if not required
            and missing.
        """
        if not path.exists():
            msg = f"Metadata file not found at {path}"
            if required:
                logger.error(msg)
                raise FileNotFoundError(msg)
            logger.warning(msg + " (optional, continuing with empty DataFrame).")
            return pd.DataFrame()

        df = pd.read_csv(path)
        logger.info("Loaded metadata from %s (rows=%d)", path, len(df))
        return df

    # =================================================================
    #                         Index construction
    # =================================================================
    def _load_or_build_index(self):
        """
        Load a FAISS index from disk if possible, otherwise build an
        in-memory FAISS or sklearn index.

        Returns:
            - `faiss.Index` if FAISS is available, or
            - `sklearn.neighbors.NearestNeighbors` instance otherwise.
        """
        emb = self.dest_embeddings
        if emb.size == 0:
            raise ValueError("Destination embeddings are empty — cannot build index.")

        if FAISS_AVAILABLE:
            # Try loading existing FAISS index
            if FAISS_INDEX_PATH.exists():
                try:
                    logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
                    index = faiss.read_index(str(FAISS_INDEX_PATH))
                    if index.ntotal != emb.shape[0]:
                        logger.warning(
                            "FAISS index size (%d) does not match embeddings (%d). "
                            "Rebuilding index in memory.",
                            index.ntotal,
                            emb.shape[0],
                        )
                        index = self._build_faiss_index(emb)
                    return index
                except Exception as e:  # pragma: no cover - defensive
                    logger.error("Failed to read FAISS index: %s. Rebuilding.", e)

            # No index saved, or failed to read → build a fresh one
            logger.info("Building FAISS index in memory...")
            return self._build_faiss_index(emb)

        # sklearn fallback
        logger.info("Building sklearn NearestNeighbors index (cosine distance)...")
        nn = NearestNeighbors(metric="cosine")  # type: ignore
        nn.fit(emb)
        return nn

    @staticmethod
    def _build_faiss_index(emb: np.ndarray):
        """
        Build an in-memory FAISS index using inner-product (cosine) similarity.

        Args:
            emb: Destination embedding matrix (num_destinations, dim).

        Returns:
            A FAISS `IndexFlatIP` instance.
        """
        emb = emb.astype("float32").copy()
        faiss.normalize_L2(emb)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        return index

    # =================================================================
    #                         Lookup construction
    # =================================================================
    @staticmethod
    def _build_lookup(df: pd.DataFrame, title_columns: List[str]) -> Dict[str, int]:
        """
        Build a case-insensitive lookup from title to row index.

        Args:
            df: Metadata DataFrame.
            title_columns: Candidate columns to use as titles.

        Returns:
            A dictionary mapping normalized title strings to row indices.
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
                if val and val.lower() != "nan":
                    lookup[val.lower()] = i

        logger.info("Built lookup with %d entries from columns: %s", len(lookup), valid_cols)
        return lookup

    # =================================================================
    #                         Public API: Recommend
    # =================================================================
    def recommend_from_combined_media(
        self,
        movie_title: str,
        book_title: str,
        music_title: str,
        weights: Optional[Tuple[float, float, float]] = None,
        top_k: int = 5,
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Recommend destinations using a fused representation of
        movie, book, and music embeddings.

        Args:
            movie_title: Movie title as typed by the user.
            book_title: Book title as typed by the user.
            music_title: Music/track title as typed by the user.
            weights: Optional (movie, book, music) weights. If None,
                all modalities are equally weighted. When only a subset
                of modalities is found, the corresponding subset of
                weights is renormalized.
            top_k: Number of destinations to return.

        Returns:
            A tuple of:
                - results: list of dicts, each containing destination
                  metadata plus `rank` and `score`.
                - status: dict with keys {"movie","book","music"} and
                  human-readable status messages.
        """
        if weights is None:
            weights = (1.0, 1.0, 1.0)

        if not movie_title or not movie_title.strip():
            raise ValueError("Movie title cannot be empty.")
        if not book_title or not book_title.strip():
            raise ValueError("Book title cannot be empty.")
        if not music_title or not music_title.strip():
            raise ValueError("Music title cannot be empty.")

        logger.info(
            "Received fused query: movie=%r, book=%r, music=%r",
            movie_title,
            book_title,
            music_title,
        )

        vectors: List[np.ndarray] = []
        status: Dict[str, str] = {}
        present_flags: List[bool] = []

        # Movie
        movie_vec = self._get_media_vector("movie", movie_title)
        if movie_vec is not None:
            vectors.append(movie_vec)
            present_flags.append(True)
            status["movie"] = f"✅ Found: {movie_title}"
        else:
            present_flags.append(False)
            status["movie"] = f"❌ Not found: {movie_title}"

        # Book
        book_vec = self._get_media_vector("book", book_title)
        if book_vec is not None:
            vectors.append(book_vec)
            present_flags.append(True)
            status["book"] = f"✅ Found: {book_title}"
        else:
            present_flags.append(False)
            status["book"] = f"❌ Not found: {book_title}"

        # Music
        music_vec = self._get_media_vector("music", music_title)
        if music_vec is not None:
            vectors.append(music_vec)
            present_flags.append(True)
            status["music"] = f"✅ Found: {music_title}"
        else:
            present_flags.append(False)
            status["music"] = f"❌ Not found: {music_title}"

        if not vectors:
            logger.warning("None of the provided titles were found in the metadata.")
            return [], status

        # Use only weights corresponding to present modalities
        used_weights = [w for w, ok in zip(weights, present_flags) if ok]
        used_weights_arr = np.asarray(used_weights, dtype="float32")
        used_weights_arr = used_weights_arr / used_weights_arr.sum()

        vectors_arr = np.vstack(vectors).astype("float32")
        combined_vec = np.average(vectors_arr, axis=0, weights=used_weights_arr)

        # Search the index
        q = combined_vec.reshape(1, -1).astype("float32")

        if FAISS_AVAILABLE:
            faiss.normalize_L2(q)
            scores, indices = self.index.search(q, top_k)  # type: ignore
            scores = scores[0]
            indices = indices[0]
        else:
            # sklearn NearestNeighbors uses distances → convert to similarity
            distances, indices = self.index.kneighbors(q, n_neighbors=top_k)  # type: ignore
            scores = 1.0 - distances[0]
            indices = indices[0]

        results = self._format_results(indices, scores)
        logger.info("Generated %d destination recommendations.", len(results))
        return results, status

    # =================================================================
    #                         Public API: Suggestions
    # =================================================================
    def suggest_titles(
        self,
        media_type: MediaType,
        query: str,
        max_suggestions: int = 5,
    ) -> List[str]:
        """
        Fuzzy search for titles when the user input is not found exactly.

        Args:
            media_type: One of {"movie", "book", "music"}.
            query: User's raw query string.
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            A list of candidate titles, possibly empty.
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
            logger.warning("Unknown media_type %r in suggest_titles", media_type)
            return []

        if df.empty:
            return []

        valid_cols = [c for c in cols if c in df.columns]
        if not valid_cols:
            return []

        mask = False
        for col in valid_cols:
            col_mask = df[col].astype(str).str.lower().str.contains(query_l, na=False)
            mask = col_mask if isinstance(mask, bool) else (mask | col_mask)

        if isinstance(mask, bool):
            return []

        display_col = valid_cols[0]
        matches = df.loc[mask, display_col].astype(str).drop_duplicates().head(max_suggestions)
        return matches.tolist()

    # =================================================================
    #                         Internal helpers
    # =================================================================
    def _get_media_vector(self, media_type: MediaType, title: str) -> Optional[np.ndarray]:
        """
        Map a user-provided title to its embedding vector for the given
        media type.

        Args:
            media_type: "movie", "book", or "music".
            title: Raw title string as entered by the user.

        Returns:
            A 1D numpy array of shape (embedding_dim,) or None if no
            match is found or embeddings are missing.
        """
        key = title.strip().lower()
        if not key:
            return None

        if media_type == "movie":
            idx = self.movie_lookup.get(key)
            if idx is None or self.movie_embeddings.size == 0:
                return None
            return self.movie_embeddings[idx]

        if media_type == "book":
            idx = self.book_lookup.get(key)
            if idx is None or self.book_embeddings.size == 0:
                return None
            return self.book_embeddings[idx]

        if media_type == "music":
            idx = self.music_lookup.get(key)
            if idx is None or self.music_embeddings.size == 0:
                return None
            return self.music_embeddings[idx]

        raise ValueError(f"Unsupported media_type: {media_type}")

    def _format_results(self, indices: np.ndarray, scores: np.ndarray) -> List[Dict]:
        """
        Combine FAISS/sklearn search results with destination metadata.

        Args:
            indices: 1D array of row indices in `dest_meta`.
            scores: 1D array of similarity scores.

        Returns:
            List of dictionaries, each containing all destination fields
            plus:
                - "rank": 1-based rank
                - "score": similarity score as float
        """
        results: List[Dict] = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            i = int(idx)
            if i < 0 or i >= len(self.dest_meta):
                continue

            row = self.dest_meta.iloc[i].to_dict()
            result = {
                "rank": rank,
                "score": float(score),
                **row,
            }
            results.append(result)

        return results


# =====================================================================
#                          Global accessor
# =====================================================================
@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    """
    Return a singleton instance of `RecommendationEngine`.

    Using an LRU cache avoids repeated loading of embeddings and
    metadata across Streamlit reruns, which keeps the demo responsive.
    """
    logger.info("get_engine() called — returning cached RecommendationEngine instance.")
    return RecommendationEngine()
