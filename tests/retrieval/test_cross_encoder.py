"""Tests for cross-encoder reranker."""

from unittest.mock import MagicMock, patch

import pytest


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    @pytest.mark.asyncio
    async def test_initialization_without_model(self):
        """Test initialization when sentence-transformers not available."""
        from src.retrieval.cross_encoder import CrossEncoderReranker

        with patch("src.retrieval.cross_encoder._import_sentence_transformers") as mock_import:
            mock_import.return_value = None

            reranker = CrossEncoderReranker()
            assert reranker.model is None
            assert reranker.cache_enabled is False

    @pytest.mark.asyncio
    async def test_rerank_without_model(self):
        """Test reranking fallback when model not available."""
        from src.retrieval.cross_encoder import CrossEncoderReranker

        with patch("src.retrieval.cross_encoder._import_sentence_transformers") as mock_import:
            mock_import.return_value = None

            reranker = CrossEncoderReranker()
            results = await reranker.rerank(
                query="test query",
                documents=["doc1", "doc2", "doc3"],
                top_k=2,
            )

            assert len(results) == 2
            assert all("text" in r for r in results)
            assert all("score" in r for r in results)
            assert all(r["score"] == 0.0 for r in results)

    @pytest.mark.asyncio
    async def test_rerank_with_mock_model(self):
        """Test reranking with mock model."""
        from src.retrieval.cross_encoder import CrossEncoderReranker

        with patch("src.retrieval.cross_encoder._import_sentence_transformers") as mock_import:
            # Mock the CrossEncoder class
            mock_cross_encoder_class = MagicMock()
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9, 0.7, 0.5]
            mock_cross_encoder_class.return_value = mock_model
            mock_import.return_value = mock_cross_encoder_class

            reranker = CrossEncoderReranker(cache_enabled=True)
            assert reranker.model is not None

            results = await reranker.rerank(
                query="test query",
                documents=["doc1", "doc2", "doc3"],
                top_k=2,
            )

            assert len(results) == 2
            # Should be sorted by score descending
            assert results[0]["score"] >= results[1]["score"]
            assert mock_model.predict.call_count == 1

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test that reranker caches scores."""
        from src.retrieval.cross_encoder import CrossEncoderReranker

        with patch("src.retrieval.cross_encoder._import_sentence_transformers") as mock_import:
            mock_cross_encoder_class = MagicMock()
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9]
            mock_cross_encoder_class.return_value = mock_model
            mock_import.return_value = mock_cross_encoder_class

            reranker = CrossEncoderReranker(cache_enabled=True)

            # First call
            await reranker.rerank(
                query="test",
                documents=["doc1"],
            )
            assert mock_model.predict.call_count == 1

            # Second call with same query/doc should use cache
            await reranker.rerank(
                query="test",
                documents=["doc1"],
            )
            # Should still be 1 (cached)
            assert mock_model.predict.call_count == 1

            # Different query should call model again
            await reranker.rerank(
                query="different query",
                documents=["doc1"],
            )
            assert mock_model.predict.call_count == 2

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test cache clearing."""
        from src.retrieval.cross_encoder import CrossEncoderReranker

        with patch("src.retrieval.cross_encoder._import_sentence_transformers") as mock_import:
            mock_cross_encoder_class = MagicMock()
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.9]
            mock_cross_encoder_class.return_value = mock_model
            mock_import.return_value = mock_cross_encoder_class

            reranker = CrossEncoderReranker(cache_enabled=True)

            await reranker.rerank(query="test", documents=["doc1"])
            assert len(reranker._cache) > 0

            reranker.clear_cache()
            assert len(reranker._cache) == 0

    def test_get_cache_stats(self):
        """Test cache statistics."""
        from src.retrieval.cross_encoder import CrossEncoderReranker

        with patch("src.retrieval.cross_encoder._import_sentence_transformers") as mock_import:
            mock_import.return_value = None

            reranker = CrossEncoderReranker(cache_enabled=False)
            stats = reranker.get_cache_stats()

            assert "enabled" in stats
            assert "size" in stats
            assert stats["enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
