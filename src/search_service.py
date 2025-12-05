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

# Cleaned metadata CSVs (✅ Updated to _processed files)
MOVIE_META_PATH = os.path.join(CLEAN_DIR, "movie_processed.csv")
BOOK_META_PATH = os.path.join(CLEAN_DIR, "book_processed.csv")
MUSIC_META_PATH = os.path.join(CLEAN_DIR, "music_processed.csv")
DEST_META_PATH = os.path.join(CLEAN_DIR, "destination_processed.csv")


class RecommendationEngine:
    """
    融合推荐引擎 (Fusion Recommendation Engine)
    
    用户输入：Movie + Book + Music (三个都要)
    系统：融合三个 embeddings，推荐目的地
    
    Usage:
        from src.search_service import get_engine
        engine = get_engine()
        
        results, status = engine.recommend_from_combined_media(
            movie_title="Inception",
            book_title="Harry Potter and the Philosopher's Stone",
            music_title="Bohemian Rhapsody",
            top_k=5
        )
        
        # status = {"movie": "✅ 找到: Inception", "book": "✅ 找到: ...", ...}
        # results = [{"rank": 1, "score": 0.85, "name": "Paris", ...}, ...]
    """

    def __init__(self) -> None:
        # ---- Load embeddings ----
        self.dest_embeddings = self._load_embeddings(DEST_EMB_PATH, allow_missing=False)
        self.movie_embeddings = self._load_embeddings(MOVIE_EMB_PATH, allow_missing=True)
        self.book_embeddings = self._load_embeddings(BOOK_EMB_PATH, allow_missing=True)
        self.music_embeddings = self._load_embeddings(MUSIC_EMB_PATH, allow_missing=True)

        # ---- Load metadata ----
        self.dest_meta = self._load_metadata(DEST_META_PATH, required=True)
        self.movie_meta = self._load_metadata(MOVIE_META_PATH)
        self.book_meta = self._load_metadata(BOOK_META_PATH)
        self.music_meta = self._load_metadata(MUSIC_META_PATH)

        # ---- Load FAISS index ----
        self.index = self._load_index()

        # ---- Build lookup tables ----
        self.movie_lookup = self._build_lookup(self.movie_meta, ["title", "original_title"])
        self.book_lookup = self._build_lookup(self.book_meta, ["title", "original_title"])
        self.music_lookup = self._build_lookup(self.music_meta, ["track_name", "track_id"])

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
        """Build case-insensitive mapping from columns to row index."""
        if df.empty:
            return {}

        valid_cols = [c for c in title_columns if c in df.columns]
        if not valid_cols:
            return {}

        lookup: Dict[str, int] = {}
        for i, row in df.iterrows():
            for col in valid_cols:
                val = str(row[col]).strip()
                if val and val.lower() != 'nan':
                    lookup[val.lower()] = i
        return lookup

    # ------------------------------------------------------------------
    # 核心方法: 融合推荐
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
        融合推荐：同时使用 movie + book + music
        
        Args:
            movie_title: 电影标题
            book_title: 书籍标题
            music_title: 音乐标题
            weights: 权重 (movie_weight, book_weight, music_weight)，默认 (1.0, 1.0, 1.0)
            top_k: 返回前 k 个推荐
            
        Returns:
            (results, status_info)
            - results: 推荐目的地列表
            - status_info: 每个媒体的查找状态
        """
        if weights is None:
            weights = (1.0, 1.0, 1.0)
        
        # 验证输入
        if not movie_title or not movie_title.strip():
            raise ValueError("Movie title cannot be empty")
        if not book_title or not book_title.strip():
            raise ValueError("Book title cannot be empty")
        if not music_title or not music_title.strip():
            raise ValueError("Music title cannot be empty")
        
        vectors = []
        status_info = {}
        
        # 获取 movie embedding
        movie_vec = self._get_media_vector("movie", movie_title)
        if movie_vec is not None:
            vectors.append(movie_vec)
            status_info["movie"] = f"✅ Found: {movie_title}"
        else:
            status_info["movie"] = f"❌ Not found: {movie_title}"
        
        # 获取 book embedding
        book_vec = self._get_media_vector("book", book_title)
        if book_vec is not None:
            vectors.append(book_vec)
            status_info["book"] = f"✅ Found: {book_title}"
        else:
            status_info["book"] = f"❌ Not found: {book_title}"
        
        # 获取 music embedding
        music_vec = self._get_media_vector("music", music_title)
        if music_vec is not None:
            vectors.append(music_vec)
            status_info["music"] = f"✅ Found: {music_title}"
        else:
            status_info["music"] = f"❌ Not found: {music_title}"
        
        # 如果都没找到，返回空
        if len(vectors) == 0:
            return [], status_info
        
        # 智能权重调整（如果只找到部分）
        if len(vectors) == 3:
            weights_to_use = weights
        elif len(vectors) == 2:
            # 只找到2个，调整权重
            if movie_vec is None:
                weights_to_use = (weights[1], weights[2])  # book, music
            elif book_vec is None:
                weights_to_use = (weights[0], weights[2])  # movie, music
            else:  # music_vec is None
                weights_to_use = (weights[0], weights[1])  # movie, book
        else:
            # 只找到1个
            weights_to_use = (1.0,)
        
        # 融合 embeddings（加权平均）
        vectors_array = np.array(vectors, dtype="float32")
        weights_array = np.array(weights_to_use, dtype="float32")
        
        # 归一化权重
        weights_array = weights_array / weights_array.sum()
        
        # 加权平均
        combined_vec = np.average(vectors_array, axis=0, weights=weights_array)
        
        # Normalize for cosine similarity
        q = combined_vec.reshape(1, -1)
        faiss.normalize_L2(q)
        
        # 搜索
        scores, indices = self.index.search(q, top_k)
        results = self._format_results(indices[0], scores[0])
        
        return results, status_info

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def suggest_titles(
        self,
        media_type: MediaType,
        query: str,
        max_suggestions: int = 5,
    ) -> List[str]:
        """模糊搜索建议"""
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
            col_mask = df[col].astype(str).str.lower().str.contains(query_l, na=False)
            mask = col_mask if isinstance(mask, bool) else (mask | col_mask)

        if isinstance(mask, bool):
            return []

        display_col = valid_cols[0]
        matches = df.loc[mask, display_col].astype(str).drop_duplicates().head(max_suggestions)
        return matches.tolist()

    def _get_media_vector(
        self,
        media_type: MediaType,
        media_title: str,
    ) -> Optional[np.ndarray]:
        """Map title to embedding."""
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
        """Format FAISS results."""
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
# Global accessor
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_engine() -> RecommendationEngine:
    """Cached singleton engine."""
    return RecommendationEngine()


# ----------------------------------------------------------------------
# CLI demo
# ----------------------------------------------------------------------

def _show_available_titles(engine: RecommendationEngine, n: int = 10) -> None:
    """显示可用的标题（前 n 个）"""
    print("\n" + "="*60)
    print(f"Available Titles (showing first {n} from each category)")
    print("="*60)
    
    # Movies
    if not engine.movie_meta.empty and "title" in engine.movie_meta.columns:
        print(f"\n🎬 Movies ({len(engine.movie_meta)} total):")
        for i, title in enumerate(engine.movie_meta["title"].head(n), 1):
            print(f"  {i}. {title}")
    
    # Books
    if not engine.book_meta.empty and "title" in engine.book_meta.columns:
        print(f"\n📚 Books ({len(engine.book_meta)} total):")
        for i, title in enumerate(engine.book_meta["title"].head(n), 1):
            print(f"  {i}. {title}")
    
    # Music
    if not engine.music_meta.empty and "track_name" in engine.music_meta.columns:
        print(f"\n🎵 Music ({len(engine.music_meta)} total):")
        for i, track in enumerate(engine.music_meta["track_name"].head(n), 1):
            artist = engine.music_meta.iloc[i-1].get("artists", "")
            if artist:
                print(f"  {i}. {track} - {artist}")
            else:
                print(f"  {i}. {track}")
    
    print("\n" + "="*60)


def _demo() -> None:
    engine = get_engine()
    print("\n" + "="*60)
    print("Fusion Recommendation Demo")
    print("="*60)
    print("Enter Movie + Book + Music titles to get destination recommendations")
    print()
    print("💡 Tips:")
    print("  - Type 'list' to see available titles")
    print("  - Type 'q' to quit")
    print()

    while True:
        print("\n" + "-"*60)
        movie = input("🎬 Movie title (or 'list'/'q'): ").strip()
        
        if movie.lower() == 'q':
            break
        
        if movie.lower() == 'list':
            _show_available_titles(engine)
            continue
        
        book = input("📚 Book title (or 'list'): ").strip()
        if book.lower() == 'list':
            _show_available_titles(engine)
            continue
            
        music = input("🎵 Music/Track name (or 'list'): ").strip()
        if music.lower() == 'list':
            _show_available_titles(engine)
            continue
        
        if not movie or not book or not music:
            print("❌ All three inputs are required!")
            continue
        
        try:
            results, status = engine.recommend_from_combined_media(
                movie_title=movie,
                book_title=book,
                music_title=music,
                top_k=5
            )
            
            print("\n📊 Status:")
            for media_type, msg in status.items():
                print(f"  {media_type.capitalize()}: {msg}")
            
            # 显示推荐结果
            if results:
                print(f"\n🌍 Top {len(results)} Recommended Destinations:")
                for r in results:
                    name = r.get("name", "")
                    country = r.get("country", "")
                    score = r["score"]
                    print(f"  {r['rank']}. {name}, {country} (score: {score:.4f})")
            else:
                print("\n⚠️  Only 1 media found, recommendations may be less accurate.")
            
            # 为未找到的标题提供建议（无论有没有推荐结果）
            has_not_found = any("❌" in msg for msg in status.values())
            if has_not_found:
                print("\n💡 Suggestions for titles not found:")
                
                # Movie suggestions
                if "❌" in status.get("movie", ""):
                    movie_sugg = engine.suggest_titles("movie", movie, 5)
                    if movie_sugg:
                        print(f"  🎬 Movies: {', '.join(movie_sugg)}")
                    else:
                        print(f"  🎬 Movies: No matches found for '{movie}'")
                
                # Book suggestions
                if "❌" in status.get("book", ""):
                    book_sugg = engine.suggest_titles("book", book, 5)
                    if book_sugg:
                        print(f"  📚 Books: {', '.join(book_sugg)}")
                    else:
                        print(f"  📚 Books: No matches found for '{book}'")
                
                # Music suggestions
                if "❌" in status.get("music", ""):
                    music_sugg = engine.suggest_titles("music", music, 5)
                    if music_sugg:
                        print(f"  🎵 Music: {', '.join(music_sugg)}")
                    else:
                        print(f"  🎵 Music: No matches found for '{music}'")
        
        except ValueError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    _demo()