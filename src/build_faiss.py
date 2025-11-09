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
