"""Tests for A/B Test Execution Harness (RAP-340).

Tests the statistical functions, experiment lifecycle, and Langfuse integration.
"""

import math
import pytest
import statistics
from unittest.mock import AsyncMock, MagicMock, patch

from src.optimization.ab_test_runner import (
    ABTestRunner,
    Experiment,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    StatisticalTest,
    calculate_sample_size,
    mann_whitney_u_test,
    run_ab_test,
    welch_t_test,
)


# =============================================================================
# Statistical Function Tests
# =============================================================================


class TestWelchTTest:
    """Tests for Welch's t-test implementation."""

    def test_identical_samples_no_difference(self):
        """Identical samples should have p-value close to 1."""
        scores_a = [0.5, 0.5, 0.5, 0.5, 0.5]
        scores_b = [0.5, 0.5, 0.5, 0.5, 0.5]

        t_stat, p_value = welch_t_test(scores_a, scores_b)

        assert t_stat == 0.0
        assert p_value >= 0.99  # No difference

    def test_clearly_different_samples(self):
        """Clearly different samples should have low p-value."""
        scores_a = [0.2, 0.25, 0.22, 0.23, 0.21, 0.24, 0.22, 0.23, 0.21, 0.24]
        scores_b = [0.8, 0.85, 0.82, 0.83, 0.81, 0.84, 0.82, 0.83, 0.81, 0.84]

        t_stat, p_value = welch_t_test(scores_a, scores_b)

        assert p_value < 0.01  # Highly significant
        assert t_stat < 0  # A < B

    def test_insufficient_samples(self):
        """Single sample should return p=1.0."""
        scores_a = [0.5]
        scores_b = [0.7]

        t_stat, p_value = welch_t_test(scores_a, scores_b)

        assert p_value == 1.0

    def test_unequal_variances(self):
        """Test with unequal variances (Welch's strength)."""
        scores_a = [0.5, 0.51, 0.49, 0.5, 0.5]  # Low variance
        scores_b = [0.3, 0.7, 0.4, 0.6, 0.5]  # High variance

        t_stat, p_value = welch_t_test(scores_a, scores_b)

        # Should still compute without error
        assert 0.0 <= p_value <= 1.0


class TestMannWhitneyU:
    """Tests for Mann-Whitney U test implementation."""

    def test_clearly_different_samples(self):
        """Non-overlapping samples should be significant."""
        scores_a = [1, 2, 3, 4, 5]
        scores_b = [6, 7, 8, 9, 10]

        u_stat, p_value = mann_whitney_u_test(scores_a, scores_b)

        assert p_value < 0.05

    def test_overlapping_samples(self):
        """Highly overlapping samples should not be significant."""
        scores_a = [1, 3, 5, 7, 9]
        scores_b = [2, 4, 6, 8, 10]

        u_stat, p_value = mann_whitney_u_test(scores_a, scores_b)

        assert p_value > 0.05  # Not significant

    def test_handles_ties(self):
        """Should handle tied values correctly."""
        scores_a = [1, 1, 2, 2, 3]
        scores_b = [2, 2, 3, 3, 4]

        u_stat, p_value = mann_whitney_u_test(scores_a, scores_b)

        # Should compute without error
        assert 0.0 <= p_value <= 1.0


class TestSampleSizeCalculation:
    """Tests for power analysis sample size calculation."""

    def test_standard_parameters(self):
        """Standard parameters should give reasonable sample size."""
        n = calculate_sample_size(
            effect_size=0.2,  # 20% improvement
            alpha=0.05,
            power=0.8,
            baseline_std=0.3,
        )

        # Should be a reasonable number for this effect size
        assert 10 <= n <= 100

    def test_small_effect_needs_more_samples(self):
        """Smaller effect sizes need more samples."""
        n_large = calculate_sample_size(effect_size=0.2)
        n_small = calculate_sample_size(effect_size=0.05)

        assert n_small > n_large

    def test_higher_power_needs_more_samples(self):
        """Higher power requirements need more samples."""
        n_low = calculate_sample_size(effect_size=0.1, power=0.7)
        n_high = calculate_sample_size(effect_size=0.1, power=0.95)

        assert n_high > n_low

    def test_minimum_sample_size(self):
        """Should enforce minimum sample size."""
        n = calculate_sample_size(
            effect_size=1.0,  # Very large effect
            alpha=0.05,
            power=0.5,
        )

        assert n >= 10  # Minimum enforced


# =============================================================================
# Experiment Configuration Tests
# =============================================================================


class TestExperimentConfig:
    """Tests for ExperimentConfig validation."""

    def test_valid_config(self):
        """Valid configuration should create successfully."""
        config = ExperimentConfig(
            agent_type="CodeAnalyzerAgent",
            variant_a="default",
            variant_b="optimized",
            minimum_effect_size=0.05,
        )

        assert config.agent_type == "CodeAnalyzerAgent"
        assert config.traffic_split == 0.5

    def test_invalid_traffic_split(self):
        """Invalid traffic split should raise error."""
        with pytest.raises(ValueError, match="traffic_split"):
            ExperimentConfig(
                agent_type="TestAgent",
                traffic_split=1.0,  # Invalid
            )

    def test_invalid_significance_level(self):
        """Invalid significance level should raise error."""
        with pytest.raises(ValueError, match="significance_level"):
            ExperimentConfig(
                agent_type="TestAgent",
                significance_level=0.0,  # Invalid
            )


# =============================================================================
# A/B Test Runner Tests
# =============================================================================


class TestABTestRunner:
    """Tests for ABTestRunner class."""

    @pytest.fixture
    def runner(self):
        """Create a test runner."""
        return ABTestRunner(use_langfuse=False, storage_backend="memory")

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return ExperimentConfig(
            agent_type="TestAgent",
            variant_a="control",
            variant_b="treatment",
            minimum_effect_size=0.1,
        )

    @pytest.mark.asyncio
    async def test_create_experiment(self, runner, config):
        """Should create experiment with valid config."""
        experiment = await runner.create_experiment(
            name="test_experiment",
            config=config,
        )

        assert experiment.id.startswith("exp_")
        assert experiment.name == "test_experiment"
        assert experiment.status == ExperimentStatus.DRAFT
        assert experiment.required_sample_size > 0

    @pytest.mark.asyncio
    async def test_start_experiment(self, runner, config):
        """Should transition experiment to running state."""
        experiment = await runner.create_experiment(
            name="test_experiment",
            config=config,
        )

        started = await runner.start_experiment(experiment.id)

        assert started.status == ExperimentStatus.RUNNING
        assert started.started_at is not None

    @pytest.mark.asyncio
    async def test_start_already_running_experiment(self, runner, config):
        """Should raise error if experiment already running."""
        experiment = await runner.create_experiment(
            name="test_experiment",
            config=config,
        )
        await runner.start_experiment(experiment.id)

        with pytest.raises(ValueError, match="DRAFT"):
            await runner.start_experiment(experiment.id)

    @pytest.mark.asyncio
    async def test_deterministic_variant_assignment(self, runner, config):
        """Same user should always get same variant."""
        experiment = await runner.create_experiment(name="test", config=config)

        # Assign same user multiple times
        variants = [
            runner.assign_variant(experiment.id, "user123") for _ in range(10)
        ]

        # Should always get same variant
        assert len(set(variants)) == 1

    @pytest.mark.asyncio
    async def test_traffic_split(self, runner):
        """Traffic split should approximately match configuration."""
        config = ExperimentConfig(
            agent_type="TestAgent",
            variant_a="control",
            variant_b="treatment",
            traffic_split=0.3,  # 30% to treatment
        )
        experiment = await runner.create_experiment(name="test", config=config)

        # Assign many users
        variants = [
            runner.assign_variant(experiment.id, f"user_{i}") for i in range(1000)
        ]

        treatment_count = variants.count("treatment")
        treatment_ratio = treatment_count / 1000

        # Should be approximately 30% (with some variance)
        assert 0.2 < treatment_ratio < 0.4

    @pytest.mark.asyncio
    async def test_record_observation(self, runner, config):
        """Should record observation and track variant."""
        experiment = await runner.create_experiment(
            name="test_experiment",
            config=config,
        )
        await runner.start_experiment(experiment.id)

        observation = await runner.record_observation(
            experiment_id=experiment.id,
            user_id="user123",
            metric_value=0.85,
            input_data={"code": "test"},
            output_data={"result": "pass"},
        )

        assert observation.metric_value == 0.85
        assert observation.variant in ["control", "treatment"]
        assert len(experiment.observations) == 1

    @pytest.mark.asyncio
    async def test_analyze_experiment(self, runner):
        """Should analyze experiment and compute statistics."""
        # Use a config with early stopping disabled
        config = ExperimentConfig(
            agent_type="TestAgent",
            variant_a="control",
            variant_b="treatment",
            minimum_effect_size=0.1,
            enable_early_stopping=False,  # Disable to allow all observations
        )
        experiment = await runner.create_experiment(
            name="test_experiment",
            config=config,
        )
        await runner.start_experiment(experiment.id)

        # Record observations for both variants
        for i in range(20):
            # Control variant (lower scores)
            experiment.assignments[f"user_a_{i}"] = MagicMock(
                variant="control", user_id=f"user_a_{i}"
            )
            await runner.record_observation(
                experiment_id=experiment.id,
                user_id=f"user_a_{i}",
                metric_value=0.5 + (i % 5) * 0.02,
                input_data={"idx": i},
            )

            # Treatment variant (higher scores)
            experiment.assignments[f"user_b_{i}"] = MagicMock(
                variant="treatment", user_id=f"user_b_{i}"
            )
            await runner.record_observation(
                experiment_id=experiment.id,
                user_id=f"user_b_{i}",
                metric_value=0.7 + (i % 5) * 0.02,
                input_data={"idx": i},
            )

        result = runner.analyze_experiment(experiment.id)

        assert result.variant_a_samples == 20
        assert result.variant_b_samples == 20
        assert result.variant_b_mean > result.variant_a_mean
        assert result.p_value < 0.05  # Should be significant
        assert result.is_significant
        assert result.winner == "treatment"

    @pytest.mark.asyncio
    async def test_list_experiments(self, runner, config):
        """Should list experiments with filtering."""
        await runner.create_experiment(name="exp1", config=config)
        await runner.create_experiment(name="exp2", config=config)

        all_experiments = runner.list_experiments()
        assert len(all_experiments) == 2

        draft_only = runner.list_experiments(status=ExperimentStatus.DRAFT)
        assert len(draft_only) == 2

    @pytest.mark.asyncio
    async def test_get_experiment(self, runner, config):
        """Should retrieve experiment by ID."""
        created = await runner.create_experiment(name="test", config=config)
        retrieved = await runner.get_experiment(created.id)

        assert retrieved.id == created.id
        assert retrieved.name == "test"

    @pytest.mark.asyncio
    async def test_get_nonexistent_experiment(self, runner):
        """Should raise error for nonexistent experiment."""
        with pytest.raises(ValueError, match="not found"):
            await runner.get_experiment("exp_nonexistent")


# =============================================================================
# Integration Tests
# =============================================================================


class TestABTestRunnerIntegration:
    """Integration tests for full experiment lifecycle."""

    @pytest.mark.asyncio
    async def test_full_experiment_lifecycle(self):
        """Test complete experiment from creation to result."""
        runner = ABTestRunner(use_langfuse=False)

        # Create experiment with early stopping disabled for consistent testing
        config = ExperimentConfig(
            agent_type="TestAgent",
            variant_a="control",
            variant_b="treatment",
            minimum_effect_size=0.1,
            enable_early_stopping=False,  # Disable for consistent test results
        )
        experiment = await runner.create_experiment(
            name="full_lifecycle_test",
            config=config,
            description="Test experiment",
            tags=["test"],
        )

        assert experiment.status == ExperimentStatus.DRAFT

        # Start experiment
        await runner.start_experiment(experiment.id)
        assert experiment.status == ExperimentStatus.RUNNING

        # Simulate observations
        for i in range(30):
            # Force assignment to specific variants
            experiment.assignments[f"ctrl_{i}"] = MagicMock(
                variant="control", user_id=f"ctrl_{i}"
            )
            experiment.assignments[f"treat_{i}"] = MagicMock(
                variant="treatment", user_id=f"treat_{i}"
            )

            await runner.record_observation(
                experiment_id=experiment.id,
                user_id=f"ctrl_{i}",
                metric_value=0.45 + (i % 10) * 0.01,
                input_data={"test": i},
            )
            await runner.record_observation(
                experiment_id=experiment.id,
                user_id=f"treat_{i}",
                metric_value=0.65 + (i % 10) * 0.01,
                input_data={"test": i},
            )

        # Analyze
        result = runner.analyze_experiment(experiment.id)

        assert result.is_significant
        assert result.winner == "treatment"
        assert result.relative_improvement > 0
        assert result.effect_size > 0

    @pytest.mark.asyncio
    async def test_experiment_with_no_significant_difference(self):
        """Test experiment where variants are equivalent."""
        runner = ABTestRunner(use_langfuse=False)

        config = ExperimentConfig(
            agent_type="TestAgent",
            minimum_effect_size=0.1,
            significance_level=0.05,
        )
        experiment = await runner.create_experiment(
            name="no_difference_test",
            config=config,
        )
        await runner.start_experiment(experiment.id)

        # Record similar observations for both variants
        import random

        random.seed(42)
        for i in range(30):
            experiment.assignments[f"ctrl_{i}"] = MagicMock(
                variant="default", user_id=f"ctrl_{i}"
            )
            experiment.assignments[f"treat_{i}"] = MagicMock(
                variant="treatment", user_id=f"treat_{i}"
            )

            base_score = 0.5 + random.uniform(-0.1, 0.1)
            await runner.record_observation(
                experiment_id=experiment.id,
                user_id=f"ctrl_{i}",
                metric_value=base_score,
                input_data={"test": i},
            )
            await runner.record_observation(
                experiment_id=experiment.id,
                user_id=f"treat_{i}",
                metric_value=base_score + random.uniform(-0.05, 0.05),
                input_data={"test": i},
            )

        result = runner.analyze_experiment(experiment.id)

        # With similar distributions, should not be significant
        assert result.p_value > 0.05 or abs(result.effect_size) < 0.2


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestRunABTestConvenience:
    """Tests for the run_ab_test convenience function."""

    @pytest.mark.asyncio
    @patch("src.optimization.ab_test_runner.ABTestRunner.run_experiment")
    @patch("src.optimization.ab_test_runner.ABTestRunner.create_experiment")
    async def test_run_ab_test_calls_runner(
        self, mock_create, mock_run
    ):
        """Convenience function should use runner correctly."""
        mock_experiment = MagicMock()
        mock_experiment.id = "exp_test"
        mock_create.return_value = mock_experiment

        mock_result = ExperimentResult(
            experiment_id="exp_test",
            variant_a="default",
            variant_b="treatment",
            variant_a_mean=0.5,
            variant_b_mean=0.6,
            variant_a_std=0.1,
            variant_b_std=0.1,
            variant_a_samples=10,
            variant_b_samples=10,
            effect_size=0.5,
            relative_improvement=0.2,
            test_statistic=2.5,
            p_value=0.01,
            confidence_interval=(0.05, 0.15),
            is_significant=True,
            winner="treatment",
            recommendation="Adopt treatment",
            statistical_test="welch_t",
        )
        mock_run.return_value = mock_result

        test_cases = [{"input": {"code": "test"}, "expected": {"result": "pass"}}]

        result = await run_ab_test(
            agent_type="TestAgent",
            variant_a="default",
            variant_b="treatment",
            test_cases=test_cases,
        )

        mock_create.assert_called_once()
        mock_run.assert_called_once()
