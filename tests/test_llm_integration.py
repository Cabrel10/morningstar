import pytest
from pathlib import Path
import json
import numpy as np
from unittest.mock import patch, MagicMock
from utils.llm_integration import LLMIntegration

class TestLLMIntegration:
    @pytest.fixture
    def llm(self):
        return LLMIntegration()

    @pytest.fixture
    def mock_embedding_file(self, tmp_path):
        """Create a mock embedding file for testing"""
        cache_dir = tmp_path / "data" / "llm_cache" / "instruments"
        cache_dir.mkdir(parents=True)
        
        embedding_data = {
            "symbol": "BTC",
            "name": "Bitcoin",
            "description": "Cryptocurrency description",
            "embedding": [0.1, 0.2, 0.3],
            "dimensions": 3
        }
        
        file_path = cache_dir / "BTC_description.json"
        with open(file_path, "w") as f:
            json.dump(embedding_data, f)
            
        return file_path

    def test_get_cached_embedding_found(self, llm, mock_embedding_file):
        """Test retrieving an existing cached embedding"""
        with patch("utils.llm_integration.INSTRUMENTS_CACHE_DIR", mock_embedding_file.parent):
            embedding = llm.get_cached_embedding("BTC", "", "instrument")
            
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 3
        assert embedding.tolist() == [0.1, 0.2, 0.3]

    def test_get_cached_embedding_not_found(self, llm):
        """Test behavior when embedding is not found"""
        embedding = llm.get_cached_embedding("UNKNOWN", "", "instrument")
        assert embedding is None

    def test_get_cached_embedding_invalid_file(self, llm, tmp_path):
        """Test handling of invalid/corrupted cache files"""
        cache_dir = tmp_path / "data" / "llm_cache" / "instruments"
        cache_dir.mkdir(parents=True)
        
        # Create invalid JSON file
        file_path = cache_dir / "INVALID.json"
        with open(file_path, "w") as f:
            f.write("{invalid json}")
            
        with patch("utils.llm_integration.INSTRUMENTS_CACHE_DIR", cache_dir):
            embedding = llm.get_cached_embedding("INVALID", "", "instrument")
            assert embedding is None

    def test_get_cached_embedding_fallback(self, llm, monkeypatch):
        """Test fallback behavior when cache directory doesn't exist"""
        monkeypatch.setattr("utils.llm_integration.INSTRUMENTS_CACHE_DIR", Path("/nonexistent/path"))
        embedding = llm.get_cached_embedding("BTC", "", "instrument")
        assert embedding is None
