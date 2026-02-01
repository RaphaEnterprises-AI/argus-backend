"""Tests for the Agent-as-Judge module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAgentJudgeDataClasses:
    """Tests for Agent-as-Judge data classes."""

    def test_evaluation_issue_creation(self, mock_env_vars):
        """Test EvaluationIssue dataclass."""
        from src.agents.agent_judge import EvaluationIssue, IssueSeverity, EvaluationCategory

        issue = EvaluationIssue(
            category=EvaluationCategory.CORRECTNESS,
            severity=IssueSeverity.HIGH,
            description="Output missing required field",
            location="response.data",
            suggestion="Include 'status' field in response",
        )

        assert issue.category == EvaluationCategory.CORRECTNESS
        assert issue.severity == IssueSeverity.HIGH
        assert "status" in issue.suggestion

    def test_agent_evaluation_creation(self, mock_env_vars):
        """Test AgentEvaluation dataclass."""
        from src.agents.agent_judge import AgentEvaluation, EvaluationIssue, IssueSeverity, EvaluationCategory

        evaluation = AgentEvaluation(
            agent_type="CodeAnalyzerAgent",
            task_description="Analyze Python file",
            output_quality_score=0.85,
            issues=[
                EvaluationIssue(
                    category=EvaluationCategory.COMPLETENESS,
                    severity=IssueSeverity.LOW,
                    description="Missing docstring analysis",
                )
            ],
            strengths=["Thorough function detection", "Accurate type inference"],
            improvement_suggestions=["Add docstring extraction"],
            overall_assessment="Good quality with minor gaps",
        )

        assert evaluation.output_quality_score == 0.85
        assert len(evaluation.issues) == 1
        assert len(evaluation.strengths) == 2

    def test_test_validation_creation(self, mock_env_vars):
        """Test TestValidation dataclass."""
        from src.agents.agent_judge import TestValidation, VerificationMethod

        validation = TestValidation(
            test_id="TEST-001",
            is_valid=True,
            correctness_score=0.9,
            completeness_score=0.85,
            robustness_score=0.8,
            verification_methods=[
                VerificationMethod.SYNTAX_CHECK,
                VerificationMethod.SCHEMA_VALIDATION,
            ],
            issues=[],
            recommendations=["Add edge case for empty input"],
        )

        assert validation.is_valid is True
        assert validation.correctness_score == 0.9
        assert len(validation.verification_methods) == 2

    def test_healing_validation_creation(self, mock_env_vars):
        """Test HealingValidation dataclass."""
        from src.agents.agent_judge import HealingValidation

        validation = HealingValidation(
            original_failure="Selector not found: #login-btn",
            proposed_fix="Updated selector to [data-testid='login']",
            is_valid_fix=True,
            confidence=0.92,
            risk_assessment="Low risk - data-testid is stable",
            side_effects=[],
            alternative_fixes=["Use aria-label selector"],
        )

        assert validation.is_valid_fix is True
        assert validation.confidence == 0.92
        assert len(validation.side_effects) == 0


class TestAgentJudgeEnums:
    """Tests for Agent-as-Judge enums."""

    def test_issue_severity_values(self, mock_env_vars):
        """Test IssueSeverity enum."""
        from src.agents.agent_judge import IssueSeverity

        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.HIGH.value == "high"
        assert IssueSeverity.MEDIUM.value == "medium"
        assert IssueSeverity.LOW.value == "low"
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

        assert VerificationMethod.SYNTAX_CHECK.value == "syntax_check"
        assert VerificationMethod.SCHEMA_VALIDATION.value == "schema_validation"
        assert VerificationMethod.EXECUTION_TEST.value == "execution_test"


class TestJudgeCapability:
    """Tests for JudgeCapability constants."""

    def test_judge_capabilities_defined(self, mock_env_vars):
        """Test JudgeCapability class has expected constants."""
        from src.agents.agent_judge import JudgeCapability

        assert hasattr(JudgeCapability, 'EVALUATE_OUTPUT')
        assert hasattr(JudgeCapability, 'VALIDATE_TEST')
        assert hasattr(JudgeCapability, 'VERIFY_HEALING')
        assert hasattr(JudgeCapability, 'META_EVALUATE')


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
                    "output_quality_score": 0.85,
                    "issues": [],
                    "strengths": ["Accurate analysis"],
                    "improvement_suggestions": [],
                    "overall_assessment": "Good quality output"
                }'''
            )

            result = await judge.evaluate_agent_output(
                agent_type="CodeAnalyzerAgent",
                task_description="Analyze authentication module",
                agent_output={"functions": ["login", "logout"], "complexity": "medium"},
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
                    "correctness_score": 0.9,
                    "completeness_score": 0.85,
                    "robustness_score": 0.8,
                    "issues": [],
                    "recommendations": []
                }'''
            )

            test_case = {
                "name": "test_login_success",
                "steps": ["Navigate to login", "Enter credentials", "Click submit"],
                "assertions": ["User is logged in", "Dashboard visible"],
            }

            result = await judge.validate_generated_test(
                test_case=test_case,
                context={"app": "auth_service"},
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
        from src.agents.agent_judge import MetaJudge

        meta_judge = MetaJudge()

        with patch.object(meta_judge, '_run_debate', new_callable=AsyncMock) as mock_debate:
            mock_debate.return_value = MagicMock(
                final_score=0.88,
                consensus_reached=True,
                debate_rounds=[],
            )

            result = await meta_judge.evaluate_with_debate(
                content="Test output to evaluate",
                criteria=["correctness", "completeness"],
            )

            assert result is not None


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
                output_quality_score=0.9,
                issues=[],
            ))

            result = await evaluate_agent_output_simple(
                agent_type="TestAgent",
                output={"data": "test"},
            )

            assert result is not None
