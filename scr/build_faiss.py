# src/build_faiss.py

import os
import numpy as np
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDING_PATH = os.path.join(BASE_DIR, "data", "embeddings", "destination_embeddings.npy")
INDEX_PATH = os.path.join(BASE_DIR, "data", "embeddings", "destinations_faiss_ip.index")


def load_destination_embeddings() -> np.ndarray:
    """destination embeddings"""
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(f"Destination embeddings not found at {EMBEDDING_PATH}")
    emb = np.load(EMBEDDING_PATH)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D array for embeddings, got shape {emb.shape}")
    return emb.astype("float32")


def build_ip_index(embeddings: np.ndarray) -> faiss.Index:
    """"build FAISS inner product index (for cosine similarity)"""
    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str = INDEX_PATH) -> None:
    """save as FAISS index"""
    faiss.write_index(index, path)
    print(f"✅ Saved FAISS index with {index.ntotal} vectors to {path}")


def main():
    print("\n" + "="*60)
    print("Building FAISS Index for Destinations")
    print("="*60)
    
    print("\n📥 Loading destination embeddings...")
    dest_embeddings = load_destination_embeddings()
    print(f"   Shape: {dest_embeddings.shape}")

    print("\n🔨 Building FAISS inner-product index...")
    index = build_ip_index(dest_embeddings)
    print(f"   Index type: {type(index).__name__}")
    print(f"   Total vectors: {index.ntotal}")

    print("\n💾 Saving index...")
    save_index(index)
    
    print("\n" + "="*60)
    print("✅ FAISS Index Built Successfully!")
    print("="*60)


if __name__ == "__main__":
    main()