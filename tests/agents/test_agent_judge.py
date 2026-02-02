"""Tests for the Agent-as-Judge module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAgentJudgeDataClasses:
    """Tests for Agent-as-Judge data classes."""

    def test_evaluation_issue_creation(self, mock_env_vars):
        """Test EvaluationIssue dataclass."""
        from src.agents.agent_judge import EvaluationIssue, IssueSeverity, EvaluationCategory

        issue = EvaluationIssue(
            id="issue-001",
            severity=IssueSeverity.CRITICAL,
            category=EvaluationCategory.CORRECTNESS,
            description="Output missing required field",
            location="response.data",
            suggested_fix="Include 'status' field in response",
        )

        assert issue.id == "issue-001"
        assert issue.severity == IssueSeverity.CRITICAL
        assert "status" in issue.suggested_fix

    def test_agent_evaluation_creation(self, mock_env_vars):
        """Test AgentEvaluation dataclass."""
        from src.agents.agent_judge import AgentEvaluation, EvaluationIssue, IssueSeverity, EvaluationCategory, VerificationMethod

        evaluation = AgentEvaluation(
            id="eval-001",
            agent_type="CodeAnalyzerAgent",
            overall_score=0.85,
            criteria_scores={"accuracy": 0.9, "completeness": 0.8},
            issues=[
                EvaluationIssue(
                    id="issue-002",
                    severity=IssueSeverity.MINOR,
                    category=EvaluationCategory.COMPLETENESS,
                    description="Missing docstring analysis",
                )
            ],
            suggestions=["Add docstring extraction"],
            verification_method=VerificationMethod.LLM_ANALYSIS,
        )

        assert evaluation.overall_score == 0.85
        assert len(evaluation.issues) == 1
        assert len(evaluation.suggestions) == 1
        assert evaluation.id == "eval-001"
        assert evaluation.verification_method == VerificationMethod.LLM_ANALYSIS

    def test_test_validation_creation(self, mock_env_vars):
        """Test TestValidation dataclass."""
        from src.agents.agent_judge import TestValidation

        validation = TestValidation(
            is_valid=True,
            syntactic_validity=True,
            semantic_validity=True,
            edge_case_coverage=0.8,
            flakiness_risk=0.1,
            issues=[],
            suggested_improvements=["Add edge case for empty input"],
        )

        assert validation.is_valid is True
        assert validation.syntactic_validity is True
        assert len(validation.suggested_improvements) == 1

    def test_healing_validation_creation(self, mock_env_vars):
        """Test HealingValidation dataclass."""
        from src.agents.agent_judge import HealingValidation

        validation = HealingValidation(
            is_valid=True,
            addresses_root_cause=True,
            preserves_test_intent=True,
            has_side_effects=False,
            confidence=0.92,
            alternative_suggestions=["Use aria-label selector"],
            reasoning="Selector update addresses the DOM change",
        )

        assert validation.is_valid is True
        assert validation.confidence == 0.92
        assert validation.has_side_effects is False


class TestAgentJudgeEnums:
    """Tests for Agent-as-Judge enums."""

    def test_issue_severity_values(self, mock_env_vars):
        """Test IssueSeverity enum."""
        from src.agents.agent_judge import IssueSeverity

        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.MAJOR.value == "major"
        assert IssueSeverity.MINOR.value == "minor"
        assert IssueSeverity.INFO.value == "info"

    def test_evaluation_category_values(self, mock_env_vars):
        """Test EvaluationCategory enum."""
        from src.agents.agent_judge import EvaluationCategory

        assert EvaluationCategory.CORRECTNESS.value == "correctness"
        assert EvaluationCategory.COMPLETENESS.value == "completeness"
        assert EvaluationCategory.EFFICIENCY.value == "efficiency"
        assert EvaluationCategory.SAFETY.value == "safety"

    def test_verification_method_values(self, mock_env_vars):
        """Test VerificationMethod enum."""
        from src.agents.agent_judge import VerificationMethod

        assert VerificationMethod.LLM_ANALYSIS.value == "llm_analysis"
        assert VerificationMethod.EXECUTION.value == "execution"
        assert VerificationMethod.GROUND_TRUTH.value == "ground_truth"
        assert VerificationMethod.HYBRID.value == "hybrid"


class TestJudgeCapability:
    """Tests for JudgeCapability constants."""

    def test_judge_capabilities_defined(self, mock_env_vars):
        """Test JudgeCapability class has expected constants."""
        from src.agents.agent_judge import JudgeCapability

        assert hasattr(JudgeCapability, 'OUTPUT_EVALUATION')
        assert hasattr(JudgeCapability, 'TEST_VALIDATION')
        assert hasattr(JudgeCapability, 'CODE_VERIFICATION')
        assert hasattr(JudgeCapability, 'MULTI_AGENT_DEBATE')


class TestAgentAsJudge:
    """Tests for AgentAsJudge class."""

    def test_agent_judge_initialization(self, mock_env_vars):
        """Test AgentAsJudge initialization."""
        from src.agents.agent_judge import AgentAsJudge

        judge = AgentAsJudge()

        assert judge is not None
        assert hasattr(judge, 'evaluate_agent_output')
        assert hasattr(judge, 'validate_generated_test')
        assert hasattr(judge, 'verify_healing_suggestion')

    @pytest.mark.asyncio
    async def test_evaluate_agent_output(self, mock_env_vars):
        """Test evaluate_agent_output method."""
        from src.agents.agent_judge import AgentAsJudge

        judge = AgentAsJudge()

        with patch.object(judge, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "overall_score": 0.85,
                    "criteria_scores": {"accuracy": 0.9},
                    "issues": [],
                    "suggestions": [],
                    "reasoning": "Good quality output"
                }'''
            )

            result = await judge.evaluate_agent_output(
                agent_type="CodeAnalyzerAgent",
                input_context={"task": "Analyze authentication module"},
                output={"functions": ["login", "logout"], "complexity": "medium"},
            )

            assert result is not None
            mock_ai.assert_called()

    @pytest.mark.asyncio
    async def test_validate_generated_test(self, mock_env_vars):
        """Test validate_generated_test method."""
        from src.agents.agent_judge import AgentAsJudge

        judge = AgentAsJudge()

        with patch.object(judge, '_call_ai', new_callable=AsyncMock) as mock_ai:
            mock_ai.return_value = MagicMock(
                content='''{
                    "is_valid": true,
                    "syntactic_validity": true,
                    "semantic_validity": true,
                    "edge_case_coverage": 0.8,
                    "flakiness_risk": 0.1,
                    "issues": [],
                    "suggested_improvements": []
                }'''
            )

            test_spec = {
                "name": "test_login_success",
                "steps": ["Navigate to login", "Enter credentials", "Click submit"],
                "assertions": ["User is logged in", "Dashboard visible"],
            }

            result = await judge.validate_generated_test(
                test_spec=test_spec,
                target_app_context={"app": "auth_service"},
            )

            assert result is not None
            mock_ai.assert_called()


class TestMetaJudge:
    """Tests for MetaJudge class."""

    def test_meta_judge_initialization(self, mock_env_vars):
        """Test MetaJudge initialization."""
        from src.agents.agent_judge import MetaJudge

        meta_judge = MetaJudge()

        assert meta_judge is not None
        assert hasattr(meta_judge, 'evaluate_with_debate')

    @pytest.mark.asyncio
    async def test_evaluate_with_debate(self, mock_env_vars):
        """Test evaluate_with_debate method."""
        from src.agents.agent_judge import MetaJudge, AgentAsJudge, AgentEvaluation, VerificationMethod

        # Create the original evaluation that will be debated
        original_evaluation = AgentEvaluation(
            id="eval-debate-001",
            agent_type="TestAgent",
            overall_score=0.88,
            criteria_scores={"correctness": 0.9, "completeness": 0.85},
            issues=[],
            suggestions=[],
            verification_method=VerificationMethod.LLM_ANALYSIS,
        )

        meta_judge = MetaJudge()

        # Mock the internal debate methods that use AI
        with patch.object(meta_judge, '_critic_challenge', new_callable=AsyncMock) as mock_critic, \
             patch.object(meta_judge, '_commander_resolve', new_callable=AsyncMock) as mock_commander:

            mock_critic.return_value = ["Minor concern about edge cases"]
            mock_commander.return_value = ("Resolution: evaluation is sound", 0.02)

            result = await meta_judge.evaluate_with_debate(
                original_evaluation=original_evaluation,
                debate_rounds=2,
            )

            assert result is not None
            assert result.final_score is not None
            assert len(result.debate_rounds) == 2


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_agent_judge(self, mock_env_vars):
        """Test create_agent_judge factory."""
        from src.agents.agent_judge import create_agent_judge

        judge = create_agent_judge()

        assert judge is not None

    def test_create_meta_judge(self, mock_env_vars):
        """Test create_meta_judge factory."""
        from src.agents.agent_judge import create_meta_judge

        meta_judge = create_meta_judge()

        assert meta_judge is not None


class TestEvaluateAgentOutputSimple:
    """Tests for evaluate_agent_output_simple helper."""

    @pytest.mark.asyncio
    async def test_simple_evaluation(self, mock_env_vars):
        """Test simple evaluation helper function."""
        from src.agents.agent_judge import evaluate_agent_output_simple

        with patch('src.agents.agent_judge.AgentAsJudge') as MockJudge:
            mock_instance = MockJudge.return_value
            mock_instance.evaluate_agent_output = AsyncMock(return_value=MagicMock(
                overall_score=0.9,
                issues=[],
            ))

            result = await evaluate_agent_output_simple(
                agent_type="TestAgent",
                input_context={"task": "Test task"},
                output={"data": "test"},
            )

            assert result is not None
