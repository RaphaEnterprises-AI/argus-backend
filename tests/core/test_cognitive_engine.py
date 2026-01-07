"""Tests for the Cognitive Testing Engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestUnderstandingLevel:
    """Tests for UnderstandingLevel enum."""

    def test_understanding_levels(self, mock_env_vars):
        """Test UnderstandingLevel enum values."""
        from src.core.cognitive_engine import UnderstandingLevel
        
        assert UnderstandingLevel.STRUCTURAL.value == "structural"
        assert UnderstandingLevel.BEHAVIORAL.value == "behavioral"
        assert UnderstandingLevel.SEMANTIC.value == "semantic"
        assert UnderstandingLevel.PREDICTIVE.value == "predictive"


class TestApplicationModel:
    """Tests for ApplicationModel dataclass."""

    def test_application_model_creation(self, mock_env_vars):
        """Test creating an ApplicationModel."""
        from src.core.cognitive_engine import ApplicationModel
        
        model = ApplicationModel(
            app_id="test-app",
            name="Test Application",
            purpose="Testing platform",
        )
        
        assert model.app_id == "test-app"
        assert model.purpose == "Testing platform"
        assert model.user_personas == []
        assert model.confidence_score == 0.0

    def test_application_model_with_data(self, mock_env_vars):
        """Test ApplicationModel with full data."""
        from src.core.cognitive_engine import ApplicationModel
        
        model = ApplicationModel(
            app_id="test-app",
            name="Test Application",
            purpose="E-commerce platform",
            user_personas=[{"name": "Shopper", "goals": ["Buy products"]}],
            core_user_journeys=[{"name": "Checkout", "steps": ["Add to cart", "Pay"]}],
            business_rules=[{"rule": "Must be logged in", "severity": "high"}],
            invariants=["Cart total >= 0"],
            confidence_score=0.85,
        )
        
        assert len(model.user_personas) == 1
        assert len(model.core_user_journeys) == 1
        assert model.confidence_score == 0.85


class TestCognitiveInsight:
    """Tests for CognitiveInsight dataclass."""

    def test_cognitive_insight_creation(self, mock_env_vars):
        """Test creating a CognitiveInsight."""
        from src.core.cognitive_engine import CognitiveInsight
        
        insight = CognitiveInsight(
            type="prediction",
            severity="high",
            title="Potential Login Failure",
            description="Login may fail under high load",
            evidence=["Error rate increased 20%"],
            recommended_action="Add rate limiting",
            confidence=0.75,
        )
        
        assert insight.type == "prediction"
        assert insight.severity == "high"
        assert insight.confidence == 0.75
        assert len(insight.evidence) == 1


class TestCognitiveTestingEngine:
    """Tests for CognitiveTestingEngine class."""

    def test_engine_initialization(self, mock_env_vars):
        """Test CognitiveTestingEngine initialization."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine
            from src.core.model_registry import get_model_id

            engine = CognitiveTestingEngine()

            # Engine uses get_model_id for default model
            assert engine.model == get_model_id("claude-sonnet-4-5")
            assert engine.app_models == {}

    def test_engine_custom_model(self, mock_env_vars):
        """Test CognitiveTestingEngine with custom model."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine

            engine = CognitiveTestingEngine(model="claude-opus-4-5-20250514")

            assert engine.model == "claude-opus-4-5-20250514"

    def test_generate_app_id(self, mock_env_vars):
        """Test app ID generation."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine

            engine = CognitiveTestingEngine()
            app_id = engine._generate_app_id("https://example.com")

            assert len(app_id) == 12
            assert app_id.isalnum()

    def test_generate_app_id_consistent(self, mock_env_vars):
        """Test that same URL generates same ID."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine

            engine = CognitiveTestingEngine()
            id1 = engine._generate_app_id("https://example.com")
            id2 = engine._generate_app_id("https://example.com")

            assert id1 == id2

    @pytest.mark.asyncio
    async def test_analyze_structure(self, mock_env_vars):
        """Test structural analysis (placeholder)."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine

            engine = CognitiveTestingEngine()
            result = await engine._analyze_structure("https://example.com")

            assert "pages" in result
            assert "components" in result
            assert "navigation" in result

    @pytest.mark.asyncio
    async def test_analyze_behavior(self, mock_env_vars):
        """Test behavioral analysis (placeholder)."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine

            engine = CognitiveTestingEngine()
            structure = {"pages": [], "components": []}

            result = await engine._analyze_behavior("https://example.com", structure)

            assert "state_machine" in result
            assert "transitions" in result

    @pytest.mark.asyncio
    async def test_learn_from_production(self, mock_env_vars):
        """Test learning from production data (placeholder)."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine

            engine = CognitiveTestingEngine()
            result = await engine._learn_from_production(
                logs=[{"event": "error"}],
                sessions=[{"user": "test"}],
            )

            assert "common_paths" in result
            assert "error_patterns" in result

    @pytest.mark.asyncio
    async def test_build_predictions(self, mock_env_vars):
        """Test building predictions (placeholder)."""
        with patch('src.core.cognitive_engine.AsyncAnthropic'):
            from src.core.cognitive_engine import CognitiveTestingEngine

            engine = CognitiveTestingEngine()
            result = await engine._build_predictions(
                structural={},
                behavioral={},
                semantic={},
            )

            assert "risk_areas" in result
            assert "likely_regressions" in result

    @pytest.mark.asyncio
    async def test_build_semantic_model(self, mock_env_vars, mock_async_anthropic_client):
        """Test building semantic model."""
        with patch('src.core.cognitive_engine.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_async_anthropic_client
            from src.core.cognitive_engine import CognitiveTestingEngine

            # Mock response with JSON
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"purpose": "E-commerce platform", "confidence": 0.8}')]
            mock_async_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

            engine = CognitiveTestingEngine()
            engine.client = mock_async_anthropic_client

            result = await engine._build_semantic_model(
                app_url="https://example.com",
                structural_data={"pages": []},
                behavioral_data={"state_machine": {}},
                source_code=None,
                api_specs=None,
                design_docs=None,
            )

            assert result["purpose"] == "E-commerce platform"
            assert result["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_build_semantic_model_parse_error(self, mock_env_vars, mock_async_anthropic_client):
        """Test semantic model handles parse errors."""
        with patch('src.core.cognitive_engine.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_async_anthropic_client
            from src.core.cognitive_engine import CognitiveTestingEngine

            # Mock response with invalid JSON
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='Invalid JSON response')]
            mock_async_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

            engine = CognitiveTestingEngine()
            engine.client = mock_async_anthropic_client

            result = await engine._build_semantic_model(
                app_url="https://example.com",
                structural_data={},
                behavioral_data={},
                source_code=None,
                api_specs=None,
                design_docs=None,
            )

            # Should return default on parse error
            assert result["purpose"] == "Unknown"
            assert result["confidence"] == 0.3

    @pytest.mark.asyncio
    async def test_learn_application(self, mock_env_vars, mock_async_anthropic_client):
        """Test learning an application."""
        with patch('src.core.cognitive_engine.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_async_anthropic_client
            from src.core.cognitive_engine import CognitiveTestingEngine

            # Mock all the responses
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"purpose": "Test app", "confidence": 0.8, "personas": [], "journeys": [], "rules": [], "invariants": []}')]
            mock_async_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

            engine = CognitiveTestingEngine()
            engine.client = mock_async_anthropic_client

            model = await engine.learn_application(
                app_url="https://example.com",
            )

            assert model is not None
            assert model.app_id is not None
            assert engine.app_models[model.app_id] == model

    @pytest.mark.asyncio
    async def test_generate_autonomous_tests(self, mock_env_vars, mock_async_anthropic_client):
        """Test autonomous test generation."""
        with patch('src.core.cognitive_engine.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_async_anthropic_client
            from src.core.cognitive_engine import CognitiveTestingEngine, ApplicationModel

            app_model = ApplicationModel(
                app_id="test-app",
                name="Test App",
                purpose="E-commerce",
                core_user_journeys=[{"name": "Checkout", "steps": ["Pay"]}],
                business_rules=[{"rule": "Must be logged in"}],
                invariants=["Cart >= 0"],
                risk_areas=[{"area": "Payment"}],
            )

            # Mock response with test array
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='[{"name": "Login Test", "priority": "high", "steps": []}]')]
            mock_async_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

            engine = CognitiveTestingEngine()
            engine.client = mock_async_anthropic_client

            tests = []
            async for test in engine.generate_autonomous_tests(app_model):
                tests.append(test)

            assert len(tests) == 1
            assert tests[0]["name"] == "Login Test"

    @pytest.mark.asyncio
    async def test_predict_failures(self, mock_env_vars, mock_async_anthropic_client):
        """Test failure prediction."""
        with patch('src.core.cognitive_engine.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_async_anthropic_client
            from src.core.cognitive_engine import CognitiveTestingEngine, ApplicationModel

            app_model = ApplicationModel(
                app_id="test-app",
                name="Test App",
                purpose="E-commerce",
                risk_areas=[{"area": "Payment", "risk": "high"}],
                error_patterns=[{"pattern": "timeout"}],
            )

            # Mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='[{"type": "prediction", "severity": "high", "title": "Payment failure risk", "description": "High risk", "evidence": [], "confidence": 0.8}]')]
            mock_async_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

            engine = CognitiveTestingEngine()
            engine.client = mock_async_anthropic_client

            insights = await engine.predict_failures(app_model)

            assert len(insights) == 1
            assert insights[0].type == "prediction"
            assert insights[0].severity == "high"

    @pytest.mark.asyncio
    async def test_explain_failure(self, mock_env_vars, mock_async_anthropic_client):
        """Test failure explanation."""
        with patch('src.core.cognitive_engine.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_async_anthropic_client
            from src.core.cognitive_engine import CognitiveTestingEngine, ApplicationModel

            app_model = ApplicationModel(
                app_id="test-app",
                name="Test App",
                purpose="E-commerce",
            )

            failure_context = {
                "test_name": "Login Test",
                "error": "Element not found",
                "severity": "high",
            }

            # Mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='The login button was not found because the page layout changed.')]
            mock_async_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

            engine = CognitiveTestingEngine()
            engine.client = mock_async_anthropic_client

            insight = await engine.explain_failure(app_model, failure_context)

            assert insight.type == "explanation"
            assert "Login Test" in insight.title

    @pytest.mark.asyncio
    async def test_suggest_test_improvements(self, mock_env_vars, mock_async_anthropic_client):
        """Test test improvement suggestions."""
        with patch('src.core.cognitive_engine.AsyncAnthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_async_anthropic_client
            from src.core.cognitive_engine import CognitiveTestingEngine, ApplicationModel

            app_model = ApplicationModel(
                app_id="test-app",
                name="Test App",
                purpose="E-commerce",
                core_user_journeys=[{"name": "Checkout", "steps": []}],
                business_rules=[{"rule": "Must be logged in"}],
                risk_areas=[],
            )

            current_tests = [
                {"name": "Login Test", "category": "journey"},
            ]

            # Mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='[{"type": "suggestion", "severity": "medium", "title": "Add checkout test", "description": "Missing test", "confidence": 0.7}]')]
            mock_async_anthropic_client.messages.create = AsyncMock(return_value=mock_response)

            engine = CognitiveTestingEngine()
            engine.client = mock_async_anthropic_client

            insights = await engine.suggest_test_improvements(app_model, current_tests)

            assert len(insights) == 1
            assert insights[0].type == "suggestion"
