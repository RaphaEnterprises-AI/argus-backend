"""Tests for the Corrective RAG Agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCorrectiveRAGDataClasses:
    """Tests for Corrective RAG data classes."""

    def test_retrieved_document_creation(self, mock_env_vars):
        """Test RetrievedDocument dataclass."""
        from src.agents.corrective_rag_agent import RetrievedDocument, RelevanceLevel

        doc = RetrievedDocument(
            doc_id="DOC-001",
            content="Authentication flow documentation...",
            source="docs/auth.md",
            relevance_score=0.85,
            relevance_level=RelevanceLevel.HIGH,
            metadata={"type": "documentation", "version": "2.0"},
        )

        assert doc.doc_id == "DOC-001"
        assert doc.relevance_level == RelevanceLevel.HIGH
        assert doc.relevance_score == 0.85

    def test_crag_result_creation(self, mock_env_vars):
        """Test CRAGResult dataclass."""
        from src.agents.corrective_rag_agent import CRAGResult, RetrievedDocument, RelevanceLevel, RetrievalAction

        result = CRAGResult(
            query="How does authentication work?",
            documents=[
                RetrievedDocument(
                    doc_id="DOC-001",
                    content="Auth docs...",
                    source="docs/auth.md",
                    relevance_score=0.9,
                    relevance_level=RelevanceLevel.HIGH,
                )
            ],
            action_taken=RetrievalAction.USE_RETRIEVED,
            confidence=0.88,
            reformulated_query=None,
            answer="Authentication uses JWT tokens...",
        )

        assert result.confidence == 0.88
        assert result.action_taken == RetrievalAction.USE_RETRIEVED
        assert len(result.documents) == 1


class TestCorrectiveRAGEnums:
    """Tests for Corrective RAG enums."""

    def test_relevance_level_values(self, mock_env_vars):
        """Test RelevanceLevel enum."""
        from src.agents.corrective_rag_agent import RelevanceLevel

        assert RelevanceLevel.HIGH.value == "high"
        assert RelevanceLevel.MEDIUM.value == "medium"
        assert RelevanceLevel.LOW.value == "low"
        assert RelevanceLevel.IRRELEVANT.value == "irrelevant"

    def test_retrieval_action_values(self, mock_env_vars):
        """Test RetrievalAction enum."""
        from src.agents.corrective_rag_agent import RetrievalAction

        assert RetrievalAction.USE_RETRIEVED.value == "use_retrieved"
        assert RetrievalAction.REFORMULATE.value == "reformulate"
        assert RetrievalAction.WEB_SEARCH.value == "web_search"
        assert RetrievalAction.COMBINE.value == "combine"


class TestCorrectiveRAGAgent:
    """Tests for CorrectiveRAGAgent class."""

    def test_crag_agent_initialization(self, mock_env_vars):
        """Test CorrectiveRAGAgent initialization."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        assert agent is not None
        assert hasattr(agent, 'query')
        assert hasattr(agent, '_assess_relevance')
        assert hasattr(agent, '_reformulate_query')

    @pytest.mark.asyncio
    async def test_query_with_high_relevance(self, mock_env_vars):
        """Test query when documents are highly relevant."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        with patch.object(agent, '_retrieve_documents', new_callable=AsyncMock) as mock_retrieve:
            with patch.object(agent, '_assess_relevance', new_callable=AsyncMock) as mock_assess:
                with patch.object(agent, '_generate_answer', new_callable=AsyncMock) as mock_generate:
                    mock_retrieve.return_value = [
                        {"doc_id": "1", "content": "Relevant content", "score": 0.9}
                    ]
                    mock_assess.return_value = {"level": "high", "confidence": 0.9}
                    mock_generate.return_value = "Generated answer"

                    result = await agent.query(
                        query="How does the login work?",
                        org_id="test_org",
                        project_id="test_project",
                    )

                    assert result is not None

    @pytest.mark.asyncio
    async def test_query_with_low_relevance_triggers_reformulation(self, mock_env_vars):
        """Test query reformulation when relevance is low."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "documents": [],
                    "confidence": 0.3,
                    "action": "reformulate",
                    "reformulated_query": "authentication process steps"
                }'''
            )

            # This tests the reformulation path
            result = await agent.query(
                query="auth?",  # Vague query
                org_id="test_org",
                project_id="test_project",
            )

            assert result is not None


class TestAdaptiveRAGRouter:
    """Tests for AdaptiveRAGRouter class."""

    def test_adaptive_router_initialization(self, mock_env_vars):
        """Test AdaptiveRAGRouter initialization."""
        from src.agents.corrective_rag_agent import AdaptiveRAGRouter

        router = AdaptiveRAGRouter()

        assert router is not None
        assert hasattr(router, 'route')
        assert hasattr(router, '_classify_query_complexity')

    @pytest.mark.asyncio
    async def test_route_simple_query(self, mock_env_vars):
        """Test routing a simple query."""
        from src.agents.corrective_rag_agent import AdaptiveRAGRouter, AdaptiveRAGStrategy

        router = AdaptiveRAGRouter()

        with patch.object(router, '_classify_query_complexity', new_callable=AsyncMock) as mock_classify:
            mock_classify.return_value = AdaptiveRAGStrategy.SIMPLE

            strategy = await router.route("What is the API endpoint?")

            assert strategy == AdaptiveRAGStrategy.SIMPLE

    @pytest.mark.asyncio
    async def test_route_complex_query(self, mock_env_vars):
        """Test routing a complex query."""
        from src.agents.corrective_rag_agent import AdaptiveRAGRouter, AdaptiveRAGStrategy

        router = AdaptiveRAGRouter()

        with patch.object(router, '_classify_query_complexity', new_callable=AsyncMock) as mock_classify:
            mock_classify.return_value = AdaptiveRAGStrategy.COMPLEX

            strategy = await router.route(
                "Explain the authentication flow including OAuth, JWT validation, and session management"
            )

            assert strategy == AdaptiveRAGStrategy.COMPLEX


class TestAdaptiveRAGStrategy:
    """Tests for AdaptiveRAGStrategy enum."""

    def test_strategy_values(self, mock_env_vars):
        """Test AdaptiveRAGStrategy enum values."""
        from src.agents.corrective_rag_agent import AdaptiveRAGStrategy

        assert AdaptiveRAGStrategy.SIMPLE.value == "simple"
        assert AdaptiveRAGStrategy.MODERATE.value == "moderate"
        assert AdaptiveRAGStrategy.COMPLEX.value == "complex"


class TestCRAGIntegration:
    """Integration tests for CRAG workflow."""

    @pytest.mark.asyncio
    async def test_full_crag_workflow(self, mock_env_vars):
        """Test full CRAG workflow: retrieve -> assess -> correct -> answer."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            # Mock the full workflow
            mock_ai.return_value = MagicMock(
                content='''{
                    "documents": [
                        {
                            "doc_id": "DOC-001",
                            "content": "Authentication uses JWT...",
                            "source": "docs/auth.md",
                            "relevance_score": 0.92
                        }
                    ],
                    "confidence": 0.92,
                    "action": "use_retrieved",
                    "answer": "The system uses JWT-based authentication with refresh tokens."
                }'''
            )

            result = await agent.query(
                query="How does authentication work in this system?",
                org_id="org_123",
                project_id="proj_456",
            )

            assert result is not None
            mock_ai.assert_called()

    @pytest.mark.asyncio
    async def test_crag_web_search_fallback(self, mock_env_vars):
        """Test CRAG falls back to web search when needed."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            with patch.object(agent, '_web_search_fallback', new_callable=AsyncMock) as mock_web:
                mock_ai.return_value = MagicMock(
                    content='''{
                        "documents": [],
                        "confidence": 0.1,
                        "action": "web_search"
                    }'''
                )
                mock_web.return_value = {"results": ["Web search result"]}

                result = await agent.query(
                    query="Latest Python 3.13 features",  # Query about recent info
                    org_id="org_123",
                    project_id="proj_456",
                )

                assert result is not None
