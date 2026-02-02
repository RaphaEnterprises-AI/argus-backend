"""Tests for the HallucinationDetectorAgent (RAP-330).

Tests the SelfCheckGPT-based hallucination detection implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.hallucination_detector import (
    HallucinationDetectorAgent,
    HallucinationResult,
    HallucinationSeverity,
    HallucinationType,
    SentenceScore,
    GroundingCheckResult,
    HallucinationCheckMiddleware,
    create_hallucination_detector,
    create_hallucination_middleware,
)


class TestHallucinationEnums:
    """Test enum definitions."""

    def test_hallucination_severity_values(self):
        """Test HallucinationSeverity has expected values."""
        assert HallucinationSeverity.NONE.value == "none"
        assert HallucinationSeverity.LOW.value == "low"
        assert HallucinationSeverity.MEDIUM.value == "medium"
        assert HallucinationSeverity.HIGH.value == "high"
        assert HallucinationSeverity.CRITICAL.value == "critical"

    def test_hallucination_type_values(self):
        """Test HallucinationType has expected values."""
        assert HallucinationType.FACTUAL_ERROR.value == "factual_error"
        assert HallucinationType.UNSUPPORTED_CLAIM.value == "unsupported_claim"
        assert HallucinationType.INCONSISTENT.value == "inconsistent"
        assert HallucinationType.FABRICATED.value == "fabricated"
        assert HallucinationType.EXAGGERATED.value == "exaggerated"
        assert HallucinationType.OUTDATED.value == "outdated"


class TestSentenceScore:
    """Test SentenceScore dataclass."""

    def test_sentence_score_creation(self):
        """Test creating a SentenceScore."""
        score = SentenceScore(
            sentence="The API returns JSON.",
            index=0,
            consistency_score=0.9,
            support_count=3,
            total_samples=3,
        )

        assert score.sentence == "The API returns JSON."
        assert score.index == 0
        assert score.consistency_score == 0.9
        assert score.support_count == 3
        assert score.total_samples == 3
        assert score.is_hallucinated is False

    def test_sentence_score_hallucinated(self):
        """Test is_hallucinated property."""
        score = SentenceScore(
            sentence="Made up fact.",
            index=0,
            consistency_score=0.3,
            support_count=0,
            total_samples=3,
            hallucination_type=HallucinationType.FABRICATED,
        )

        assert score.is_hallucinated is True
        assert score.hallucination_type == HallucinationType.FABRICATED

    def test_sentence_score_to_dict(self):
        """Test serialization to dict."""
        score = SentenceScore(
            sentence="Test sentence.",
            index=1,
            consistency_score=0.7,
            support_count=2,
            total_samples=3,
            hallucination_type=HallucinationType.UNSUPPORTED_CLAIM,
            evidence=["Sample 1 says different"],
        )

        d = score.to_dict()
        assert d["sentence"] == "Test sentence."
        assert d["index"] == 1
        assert d["consistency_score"] == 0.7
        assert d["support_count"] == 2
        assert d["total_samples"] == 3
        assert d["is_hallucinated"] is False  # 0.7 >= 0.5
        assert d["hallucination_type"] == "unsupported_claim"
        assert "Sample 1 says different" in d["evidence"]


class TestHallucinationResult:
    """Test HallucinationResult dataclass."""

    def test_result_creation(self):
        """Test creating a HallucinationResult."""
        result = HallucinationResult(
            id="test_123",
            original_response="The API returns JSON with user data.",
            overall_score=0.85,
            hallucination_rate=0.1,
            severity=HallucinationSeverity.LOW,
            sentence_scores=[],
            hallucinated_sentences=[],
            samples_generated=3,
            grounding_used=True,
        )

        assert result.id == "test_123"
        assert result.overall_score == 0.85
        assert result.hallucination_rate == 0.1
        assert result.severity == HallucinationSeverity.LOW
        assert result.is_reliable is True  # LOW severity is reliable

    def test_result_not_reliable(self):
        """Test is_reliable for high severity."""
        result = HallucinationResult(
            id="test_456",
            original_response="Made up content.",
            overall_score=0.3,
            hallucination_rate=0.6,
            severity=HallucinationSeverity.HIGH,
            sentence_scores=[],
            hallucinated_sentences=["Made up content."],
            samples_generated=3,
            grounding_used=False,
        )

        assert result.is_reliable is False  # HIGH severity is not reliable

    def test_result_to_dict(self):
        """Test serialization to dict."""
        result = HallucinationResult(
            id="test_789",
            original_response="Test response.",
            overall_score=0.95,
            hallucination_rate=0.0,
            severity=HallucinationSeverity.NONE,
            sentence_scores=[],
            hallucinated_sentences=[],
            samples_generated=3,
            grounding_used=True,
            total_cost=0.05,
            total_tokens=1000,
        )

        d = result.to_dict()
        assert d["id"] == "test_789"
        assert d["overall_score"] == 0.95
        assert d["severity"] == "none"
        assert d["is_reliable"] is True
        assert d["total_cost"] == 0.05
        assert d["total_tokens"] == 1000


class TestGroundingCheckResult:
    """Test GroundingCheckResult dataclass."""

    def test_grounding_result_creation(self):
        """Test creating a GroundingCheckResult."""
        result = GroundingCheckResult(
            is_grounded=True,
            grounding_score=0.85,
            supported_claims=["The API uses OAuth 2.0"],
            unsupported_claims=[],
            contradicted_claims=[],
            context_coverage=0.7,
        )

        assert result.is_grounded is True
        assert result.grounding_score == 0.85
        assert len(result.supported_claims) == 1

    def test_grounding_result_to_dict(self):
        """Test serialization to dict."""
        result = GroundingCheckResult(
            is_grounded=False,
            grounding_score=0.3,
            supported_claims=["claim 1"],
            unsupported_claims=["claim 2"],
            contradicted_claims=["claim 3"],
            context_coverage=0.4,
        )

        d = result.to_dict()
        assert d["is_grounded"] is False
        assert d["grounding_score"] == 0.3
        assert "claim 3" in d["contradicted_claims"]


class TestHallucinationDetectorAgent:
    """Test HallucinationDetectorAgent."""

    def test_agent_initialization(self):
        """Test agent initialization with default values."""
        agent = HallucinationDetectorAgent()

        assert agent.num_samples == 3
        assert agent.consistency_threshold == 0.5
        assert agent.enable_grounding is True
        assert agent.sample_temperature == 0.7

    def test_agent_initialization_custom(self):
        """Test agent initialization with custom values."""
        agent = HallucinationDetectorAgent(
            num_samples=5,
            consistency_threshold=0.7,
            enable_grounding=False,
            sample_temperature=0.9,
            max_cost_per_check=0.10,
        )

        assert agent.num_samples == 5
        assert agent.consistency_threshold == 0.7
        assert agent.enable_grounding is False
        assert agent.sample_temperature == 0.9
        assert agent.max_cost_per_check == 0.10

    def test_agent_capabilities(self):
        """Test agent declares expected capabilities."""
        assert "hallucination_detection" in HallucinationDetectorAgent.CAPABILITIES
        assert "consistency_checking" in HallucinationDetectorAgent.CAPABILITIES
        assert "factual_verification" in HallucinationDetectorAgent.CAPABILITIES

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        agent = HallucinationDetectorAgent()

        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = agent._split_into_sentences(text)

        assert len(sentences) >= 2  # Should split on sentence boundaries

    def test_determine_severity_none(self):
        """Test severity determination for no hallucination."""
        agent = HallucinationDetectorAgent()

        severity = agent._determine_severity(
            hallucination_rate=0.0,
            overall_score=0.95,
        )

        assert severity == HallucinationSeverity.NONE

    def test_determine_severity_low(self):
        """Test severity determination for low hallucination."""
        agent = HallucinationDetectorAgent()

        severity = agent._determine_severity(
            hallucination_rate=0.05,
            overall_score=0.8,
        )

        assert severity == HallucinationSeverity.LOW

    def test_determine_severity_medium(self):
        """Test severity determination for medium hallucination."""
        agent = HallucinationDetectorAgent()

        severity = agent._determine_severity(
            hallucination_rate=0.2,
            overall_score=0.6,
        )

        assert severity == HallucinationSeverity.MEDIUM

    def test_determine_severity_high(self):
        """Test severity determination for high hallucination."""
        agent = HallucinationDetectorAgent()

        severity = agent._determine_severity(
            hallucination_rate=0.4,
            overall_score=0.4,
        )

        assert severity == HallucinationSeverity.HIGH

    def test_determine_severity_critical(self):
        """Test severity determination for critical hallucination."""
        agent = HallucinationDetectorAgent()

        severity = agent._determine_severity(
            hallucination_rate=0.7,
            overall_score=0.2,
        )

        assert severity == HallucinationSeverity.CRITICAL


class TestHallucinationCheckMiddleware:
    """Test HallucinationCheckMiddleware."""

    def test_middleware_initialization(self):
        """Test middleware initialization with defaults."""
        middleware = HallucinationCheckMiddleware()

        assert middleware.threshold == 0.6
        assert middleware.auto_retry is False
        assert middleware.max_retries == 2
        assert middleware.detector is not None

    def test_middleware_initialization_custom(self):
        """Test middleware initialization with custom values."""
        middleware = HallucinationCheckMiddleware(
            threshold=0.8,
            auto_retry=True,
            max_retries=3,
        )

        assert middleware.threshold == 0.8
        assert middleware.auto_retry is True
        assert middleware.max_retries == 3


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_hallucination_detector(self):
        """Test create_hallucination_detector factory."""
        detector = create_hallucination_detector(
            num_samples=5,
            consistency_threshold=0.7,
        )

        assert isinstance(detector, HallucinationDetectorAgent)
        assert detector.num_samples == 5
        assert detector.consistency_threshold == 0.7

    def test_create_hallucination_middleware(self):
        """Test create_hallucination_middleware factory."""
        middleware = create_hallucination_middleware(
            threshold=0.75,
            auto_retry=True,
            max_retries=5,
        )

        assert isinstance(middleware, HallucinationCheckMiddleware)
        assert middleware.threshold == 0.75
        assert middleware.auto_retry is True
        assert middleware.max_retries == 5


class TestIntegrationWithBaseAgent:
    """Test integration with BaseAgent."""

    def test_base_agent_has_hallucination_method(self):
        """Test BaseAgent has the hallucination check method."""
        from src.agents.base import BaseAgent

        assert hasattr(BaseAgent, "_call_ai_with_hallucination_check")

    @pytest.mark.asyncio
    async def test_detect_with_mocked_ai(self):
        """Test detection with mocked AI calls."""
        agent = HallucinationDetectorAgent()

        # Mock the _call_ai method
        mock_response = MagicMock()
        mock_response.content = "Sample response"
        mock_response.cost = 0.01
        mock_response.total_tokens = 100

        mock_consistency_response = MagicMock()
        mock_consistency_response.content = '''{
            "sentence_scores": [
                {
                    "index": 0,
                    "consistency_score": 0.9,
                    "support_count": 3,
                    "hallucination_type": null,
                    "evidence": []
                }
            ]
        }'''

        with patch.object(agent, "_call_ai", new_callable=AsyncMock) as mock_call:
            # First calls generate samples, last call checks consistency
            mock_call.side_effect = [
                mock_response,  # Sample 1
                mock_response,  # Sample 2
                mock_response,  # Sample 3
                mock_consistency_response,  # Consistency check
            ]

            result = await agent.detect(
                response="The API returns JSON data.",
                query="What does the API return?",
            )

            assert isinstance(result, HallucinationResult)
            assert result.samples_generated == 3
