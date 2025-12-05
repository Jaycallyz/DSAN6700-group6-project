#!/usr/bin/env python
"""
Embedding generation script (simplified version).

This script:
- Generates only .npy embedding files (no metadata, no FAISS index)
- Uses Sentence-BERT (all-MiniLM-L6-v2) to produce 384-dimensional vectors
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time


class EmbeddingGenerator:
    """Sentence-BERT embedding generator."""
    
    def __init__(
        self,
        data_dir: str = "data/clean_data",
        output_dir: str = "data/embeddings",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"\n{'='*60}")
        print(f"Embedding Generator")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
        
        # Load model
        print("📥 Loading Sentence-BERT model...")
        self.model = SentenceTransformer(model_name)
        embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded | Embedding dimension: {embedding_dim}")
    
    def generate_embeddings(self, texts: list, desc: str = "Encoding") -> np.ndarray:
        """Generate embeddings for a list of texts."""
        print(f"\n🔄 {desc}...")
        print(f"   Total texts: {len(texts)}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization
        )
        
        print(f"✅ Embeddings generated | Shape: {embeddings.shape}")
        
        return embeddings.astype("float32")
    
    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """Save embeddings to a .npy file."""
        output_path = self.output_dir / filename
        np.save(output_path, embeddings)
        
        print(f"💾 Saved: {output_path}")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    def process_movies(self):
        """Generate movie embeddings."""
        print("\n" + "="*60)
        print("🎬 Processing Movie Embeddings")
        print("="*60)
        
        input_file = self.data_dir / "movie_processed.csv"
        output_file = "movie_embeddings.npy"
        
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            return
        
        df = pd.read_csv(input_file)
        print(f"📊 Loaded: {len(df)} movies")
        
        if "text_representation" not in df.columns:
            print("❌ Missing 'text_representation' column")
            return
        
        texts = df["text_representation"].fillna("").astype(str).tolist()
        embeddings = self.generate_embeddings(texts, "Encoding movies")
        self.save_embeddings(embeddings, output_file)
        
        print(f"\n✓ Sample titles:")
        for i in range(min(3, len(df))):
            print(f"  {i+1}. {df.iloc[i]['title']}")
    
    def process_books(self):
        """Generate book embeddings."""
        print("\n" + "="*60)
        print("📚 Processing Book Embeddings")
        print("="*60)
        
        input_file = self.data_dir / "book_processed.csv"
        output_file = "book_embeddings.npy"
        
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            return
        
        df = pd.read_csv(input_file)
        print(f"📊 Loaded: {len(df)} books")
        
        if "text_representation" not in df.columns:
            print("❌ Missing 'text_representation' column")
            return
        
        texts = df["text_representation"].fillna("").astype(str).tolist()
        embeddings = self.generate_embeddings(texts, "Encoding books")
        self.save_embeddings(embeddings, output_file)
        
        print(f"\n✓ Sample titles:")
        for i in range(min(3, len(df))):
            print(f"  {i+1}. {df.iloc[i]['title']}")
    
    def process_music(self):
        """Generate music embeddings."""
        print("\n" + "="*60)
        print("🎵 Processing Music Embeddings")
        print("="*60)
        
        input_file = self.data_dir / "music_processed.csv"
        output_file = "music_embeddings.npy"
        
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            return
        
        df = pd.read_csv(input_file)
        print(f"📊 Loaded: {len(df)} tracks")
        
        if "text_representation" not in df.columns:
            print("❌ Missing 'text_representation' column")
            return
        
        texts = df["text_representation"].fillna("").astype(str).tolist()
        embeddings = self.generate_embeddings(texts, "Encoding music")
        self.save_embeddings(embeddings, output_file)
        
        print(f"\n✓ Sample tracks:")
        for i in range(min(3, len(df))):
            if "track_name" in df.columns:
                print(f"  {i+1}. {df.iloc[i]['track_name']}")
    
    def process_destinations(self):
        """Generate destination embeddings."""
        print("\n" + "="*60)
        print("🌍 Processing Destination Embeddings")
        print("="*60)
        
        input_file = self.data_dir / "destination_processed.csv"
        output_file = "destination_embeddings.npy"
        
        if not input_file.exists():
            print(f"❌ File not found: {input_file}")
            return
        
        df = pd.read_csv(input_file)
        print(f"📊 Loaded: {len(df)} destinations")
        
        if "text_representation" not in df.columns:
            print("❌ Missing 'text_representation' column")
            return
        
        texts = df["text_representation"].fillna("").astype(str).tolist()
        embeddings = self.generate_embeddings(texts, "Encoding destinations")
        self.save_embeddings(embeddings, output_file)
        
        print(f"\n✓ Sample destinations:")
        for i in range(min(3, len(df))):
            print(f"  {i+1}. {df.iloc[i]['name']}, {df.iloc[i]['country']}")
    
    def process_all(self):
        """Run the full embedding generation pipeline for all data types."""
        print("\n" + "="*60)
        print("🚀 Starting Embedding Generation Pipeline")
        print("="*60)
        
        start_time = time.time()
        
        self.process_movies()
        self.process_books()
        self.process_music()
        self.process_destinations()
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ All Embeddings Generated!")
        print("="*60)
        print(f"\nTotal time: {elapsed/60:.2f} minutes")
        print(f"Output directory: {self.output_dir.absolute()}")
        
        print("\n📊 Generated files:")
        total_size = 0
        for file in sorted(self.output_dir.glob("*.npy")):
            size_mb = file.stat().st_size / 1024 / 1024
            total_size += size_mb
            
            emb = np.load(file)
            print(f"  ✓ {file.name}")
            print(f"     Shape: {emb.shape} | Size: {size_mb:.2f} MB")
        
        print(f"\nTotal size: {total_size:.2f} MB")
        
        # Validate embedding dimensionality consistency
        print("\n🔍 Validation:")
        files = list(self.output_dir.glob("*.npy"))
        if files:
            dims = [np.load(f).shape[1] for f in files]
            if len(set(dims)) == 1:
                print(f"  ✅ All embeddings have the same dimension: {dims[0]}")
            else:
                print(f"  ⚠️  Dimension mismatch across files: {set(dims)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Sentence-BERT embeddings (only .npy files)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/clean_data",
        help="Directory containing processed CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["movie", "book", "music", "destination", "all"],
        default="all",
        help="Which data type to process"
    )
    
    args = parser.parse_args()
    
    generator = EmbeddingGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size
    )
    
    if args.data_type == "all":
        generator.process_all()
    elif args.data_type == "movie":
        generator.process_movies()
    elif args.data_type == "book":
        generator.process_books()
    elif args.data_type == "music":
        generator.process_music()
    elif args.data_type == "destination":
        generator.process_destinations()


if __name__ == "__main__":
    main()
