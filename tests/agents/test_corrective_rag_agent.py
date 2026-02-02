"""Tests for the Corrective RAG Agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCorrectiveRAGDataClasses:
    """Tests for Corrective RAG data classes."""

    def test_retrieved_document_creation(self, mock_env_vars):
        """Test RetrievedDocument dataclass."""
        from src.agents.corrective_rag_agent import RetrievedDocument, RelevanceLevel

        doc = RetrievedDocument(
            id="doc-001",
            content="Authentication flow documentation...",
            source="docs/auth.md",
            retrieval_score=0.9,
            relevance_score=0.85,
            relevance_level=RelevanceLevel.HIGHLY_RELEVANT,
            metadata={"type": "documentation", "version": "2.0"},
        )

        assert doc.id == "doc-001"
        assert doc.content == "Authentication flow documentation..."
        assert doc.relevance_level == RelevanceLevel.HIGHLY_RELEVANT
        assert doc.relevance_score == 0.85

    def test_crag_result_creation(self, mock_env_vars):
        """Test CRAGResult dataclass."""
        from src.agents.corrective_rag_agent import CRAGResult, RetrievalAction

        result = CRAGResult(
            query="How does authentication work?",
            response="Authentication uses JWT tokens...",
            documents_used=1,
            retrieval_rounds=1,
            actions_taken=[RetrievalAction.CORRECT],
            overall_relevance=0.9,
            confidence=0.88,
            web_search_used=False,
        )

        assert result.confidence == 0.88
        assert result.actions_taken == [RetrievalAction.CORRECT]
        assert result.documents_used == 1


class TestCorrectiveRAGEnums:
    """Tests for Corrective RAG enums."""

    def test_relevance_level_values(self, mock_env_vars):
        """Test RelevanceLevel enum."""
        from src.agents.corrective_rag_agent import RelevanceLevel

        assert RelevanceLevel.HIGHLY_RELEVANT.value == "highly_relevant"
        assert RelevanceLevel.RELEVANT.value == "relevant"
        assert RelevanceLevel.MARGINALLY_RELEVANT.value == "marginally_relevant"
        assert RelevanceLevel.IRRELEVANT.value == "irrelevant"

    def test_retrieval_action_values(self, mock_env_vars):
        """Test RetrievalAction enum."""
        from src.agents.corrective_rag_agent import RetrievalAction

        assert RetrievalAction.CORRECT.value == "correct"
        assert RetrievalAction.INCORRECT.value == "incorrect"
        assert RetrievalAction.AMBIGUOUS.value == "ambiguous"
        assert RetrievalAction.WEB_SEARCH.value == "web_search"


class TestCorrectiveRAGAgent:
    """Tests for CorrectiveRAGAgent class."""

    def test_crag_agent_initialization(self, mock_env_vars):
        """Test CorrectiveRAGAgent initialization."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        assert agent is not None
        assert hasattr(agent, 'query')

    @pytest.mark.asyncio
    async def test_query_with_high_relevance(self, mock_env_vars):
        """Test query when documents are highly relevant."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "response": "The login works with JWT tokens",
                    "documents_used": 1,
                    "retrieval_rounds": 1,
                    "actions_taken": ["correct"],
                    "overall_relevance": 0.9,
                    "confidence": 0.9,
                    "web_search_used": false
                }'''
            )

            result = await agent.query(
                query="How does the login work?",
                context={"project_id": "test_project"},
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
                    "response": "Authentication process involves several steps",
                    "documents_used": 0,
                    "retrieval_rounds": 2,
                    "actions_taken": ["incorrect", "correct"],
                    "confidence": 0.7,
                    "query_reformulations": ["authentication process steps"]
                }'''
            )

            # This tests the reformulation path
            result = await agent.query(
                query="auth?",  # Vague query
                context={"project_id": "test_project"},
            )

            assert result is not None


class TestAdaptiveRAGRouter:
    """Tests for AdaptiveRAGRouter class."""

    def test_adaptive_router_initialization(self, mock_env_vars):
        """Test AdaptiveRAGRouter initialization."""
        from src.agents.corrective_rag_agent import AdaptiveRAGRouter, CorrectiveRAGAgent

        base_agent = CorrectiveRAGAgent()
        router = AdaptiveRAGRouter(base_agent=base_agent)

        assert router is not None
        assert hasattr(router, 'route')

    @pytest.mark.asyncio
    async def test_route_simple_query(self, mock_env_vars):
        """Test routing a simple query."""
        from src.agents.corrective_rag_agent import AdaptiveRAGRouter, AdaptiveRAGStrategy, CorrectiveRAGAgent

        base_agent = CorrectiveRAGAgent()
        router = AdaptiveRAGRouter(base_agent=base_agent)

        # AdaptiveRAGRouter uses self.agent._call_ai, so mock that
        with patch.object(base_agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='{"strategy": "single_step"}'
            )

            strategy = await router.route("What is the API endpoint?")

            assert strategy is not None

    @pytest.mark.asyncio
    async def test_route_complex_query(self, mock_env_vars):
        """Test routing a complex query."""
        from src.agents.corrective_rag_agent import AdaptiveRAGRouter, AdaptiveRAGStrategy, CorrectiveRAGAgent

        base_agent = CorrectiveRAGAgent()
        router = AdaptiveRAGRouter(base_agent=base_agent)

        # AdaptiveRAGRouter uses self.agent._call_ai, so mock that
        with patch.object(base_agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='{"strategy": "iterative"}'
            )

            strategy = await router.route(
                "Explain the authentication flow including OAuth, JWT validation, and session management"
            )

            assert strategy is not None


class TestAdaptiveRAGStrategy:
    """Tests for AdaptiveRAGStrategy enum."""

    def test_strategy_values(self, mock_env_vars):
        """Test AdaptiveRAGStrategy enum values."""
        from src.agents.corrective_rag_agent import AdaptiveRAGStrategy

        assert AdaptiveRAGStrategy.NO_RETRIEVAL.value == "no_retrieval"
        assert AdaptiveRAGStrategy.SINGLE_STEP.value == "single_step"
        assert AdaptiveRAGStrategy.CORRECTIVE.value == "corrective"
        assert AdaptiveRAGStrategy.ITERATIVE.value == "iterative"
        assert AdaptiveRAGStrategy.AGENT.value == "agent"


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
                    "response": "The system uses JWT-based authentication with refresh tokens.",
                    "documents_used": 1,
                    "retrieval_rounds": 1,
                    "actions_taken": ["correct"],
                    "overall_relevance": 0.92,
                    "confidence": 0.92,
                    "web_search_used": false
                }'''
            )

            result = await agent.query(
                query="How does authentication work in this system?",
                context={"project_id": "proj_456"},
            )

            assert result is not None
            mock_ai.assert_called()

    @pytest.mark.asyncio
    async def test_crag_web_search_fallback(self, mock_env_vars):
        """Test CRAG falls back to web search when needed."""
        from src.agents.corrective_rag_agent import CorrectiveRAGAgent

        agent = CorrectiveRAGAgent()

        with patch.object(agent, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "response": "Based on web search...",
                    "documents_used": 0,
                    "retrieval_rounds": 2,
                    "actions_taken": ["incorrect", "web_search"],
                    "confidence": 0.7,
                    "web_search_used": true
                }'''
            )

            result = await agent.query(
                query="Latest Python 3.13 features",  # Query about recent info
                context={"project_id": "proj_456"},
            )

            assert result is not None
