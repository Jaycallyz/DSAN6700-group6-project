"""
Tests for search_service.py
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from search_service import RecommendationEngine, get_engine


class TestRecommendationEngine:
    """Test suite for RecommendationEngine"""
    
    def test_engine_initialization(self):
        """Test that engine initializes without errors"""
        engine = get_engine()
        assert engine is not None
        assert hasattr(engine, 'dest_embeddings')
        assert hasattr(engine, 'movie_embeddings')
        
    def test_embeddings_loaded(self):
        """Test that embeddings are loaded correctly"""
        engine = get_engine()
        assert engine.dest_embeddings.shape[1] == 384  # Embedding dimension
        assert len(engine.dest_embeddings) > 0
        
    def test_metadata_loaded(self):
        """Test that metadata CSVs are loaded"""
        engine = get_engine()
        assert not engine.dest_meta.empty
        assert 'name' in engine.dest_meta.columns
        assert 'country' in engine.dest_meta.columns
        
    def test_recommend_from_media_valid(self):
        """Test recommendation with valid input"""
        engine = get_engine()
        
        # Get any valid title from the data
        if not engine.movie_meta.empty:
            sample_title = engine.movie_meta['title'].iloc[0]
            results = engine.recommend_from_media("movie", sample_title, top_k=5)
            
            assert len(results) <= 5
            assert all('rank' in r for r in results)
            assert all('score' in r for r in results)
            assert all('name' in r for r in results)
    
    def test_recommend_from_media_invalid(self):
        """Test recommendation with invalid input"""
        engine = get_engine()
        results = engine.recommend_from_media("movie", "NonexistentMovie12345", top_k=5)
        
        # Should return empty list for non-existent title
        assert isinstance(results, list)
        
    def test_suggest_titles(self):
        """Test title suggestion functionality"""
        engine = get_engine()
        
        if not engine.movie_meta.empty:
            # Use partial match
            first_title = engine.movie_meta['title'].iloc[0]
            partial = first_title[:5]
            
            suggestions = engine.suggest_titles("movie", partial, max_suggestions=5)
            
            assert isinstance(suggestions, list)
            assert len(suggestions) <= 5
            
    def test_result_format(self):
        """Test that results have correct format"""
        engine = get_engine()
        
        if not engine.movie_meta.empty:
            sample_title = engine.movie_meta['title'].iloc[0]
            results = engine.recommend_from_media("movie", sample_title, top_k=3)
            
            if results:
                result = results[0]
                assert 'rank' in result
                assert 'score' in result
                assert 'name' in result
                assert isinstance(result['rank'], int)
                assert isinstance(result['score'], (float, np.floating))
                
    def test_embeddings_normalized(self):
        """Test that embeddings are L2 normalized"""
        engine = get_engine()
        
        # Check if destination embeddings are normalized
        norms = np.linalg.norm(engine.dest_embeddings, axis=1)
        
        # All norms should be close to 1.0
        assert np.allclose(norms, 1.0, atol=1e-5)
        
    def test_top_k_parameter(self):
        """Test that top_k parameter works correctly"""
        engine = get_engine()
        
        if not engine.movie_meta.empty:
            sample_title = engine.movie_meta['title'].iloc[0]
            
            results_3 = engine.recommend_from_media("movie", sample_title, top_k=3)
            results_5 = engine.recommend_from_media("movie", sample_title, top_k=5)
            
            assert len(results_3) <= 3
            assert len(results_5) <= 5


class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    def test_csv_files_exist(self):
        """Test that all required CSV files exist"""
        base_dir = Path(__file__).parent.parent
        
        required_files = [
            'data/clean_data/movie_processed.csv',
            'data/clean_data/book_processed.csv',
            'data/clean_data/music_processed.csv',
            'data/clean_data/destination_processed.csv'
        ]
        
        for file_path in required_files:
            full_path = base_dir / file_path
            assert full_path.exists(), f"Missing required file: {file_path}"
            
    def test_embedding_files_exist(self):
        """Test that all embedding files exist"""
        base_dir = Path(__file__).parent.parent
        
        required_files = [
            'data/embeddings/movie_embeddings.npz',
            'data/embeddings/book_embeddings.npz',
            'data/embeddings/music_embeddings.npz',
            'data/embeddings/destination_embeddings.npz'
        ]
        
        for file_path in required_files:
            full_path = base_dir / file_path
            assert full_path.exists(), f"Missing required file: {file_path}"
            
    def test_csv_columns(self):
        """Test that CSVs have required columns"""
        base_dir = Path(__file__).parent.parent
        
        dest_df = pd.read_csv(base_dir / 'data/clean_data/destination_processed.csv')
        
        required_cols = ['name', 'country']
        for col in required_cols:
            assert col in dest_df.columns, f"Missing column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])