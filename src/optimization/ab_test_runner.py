"""A/B Test Execution Harness for Prompt Experiments.

RAP-340: Production-grade A/B testing infrastructure for comparing prompt variants
with proper statistical rigor and Langfuse experiment tracking integration.

Features:
- Proper statistical significance testing (Welch's t-test, Mann-Whitney U)
- Traffic randomization with consistent user assignment
- Sample size calculation via power analysis
- Langfuse experiment tracking integration
- Sequential testing for early stopping
- Concurrent experiment execution
- Persistent experiment state

Usage:
    ```python
    from src.optimization.ab_test_runner import ABTestRunner, ExperimentConfig

    runner = ABTestRunner()

    # Create experiment
    experiment = await runner.create_experiment(
        name="code_analyzer_v2",
        config=ExperimentConfig(
            agent_type="CodeAnalyzerAgent",
            variant_a="default",
            variant_b="optimized_v2",
            metric_name="accuracy",
            minimum_effect_size=0.05,  # 5% improvement
            significance_level=0.05,
            power=0.8,
        ),
    )

    # Run experiment with test cases
    result = await runner.run_experiment(
        experiment_id=experiment.id,
        test_cases=test_data,
    )

    # Check if significant
    if result.is_significant:
        print(f"Winner: {result.winner} (p={result.p_value:.4f})")
    ```

References:
- Langfuse Experiments: https://langfuse.com/docs/experimentation
- Sequential Testing: https://arxiv.org/abs/1906.09712
"""

import asyncio
import hashlib
import json
import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Statistical Functions
# =============================================================================


def welch_t_test(scores_a: list[float], scores_b: list[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances.

    More robust than Student's t-test when sample sizes or variances differ.

    Args:
        scores_a: Scores from variant A
        scores_b: Scores from variant B

    Returns:
        Tuple of (t_statistic, p_value)
    """
    n_a = len(scores_a)
    n_b = len(scores_b)

    if n_a < 2 or n_b < 2:
        return 0.0, 1.0

    mean_a = statistics.mean(scores_a)
    mean_b = statistics.mean(scores_b)
    var_a = statistics.variance(scores_a)
    var_b = statistics.variance(scores_b)

    # Welch's t-statistic
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1

    # Approximate p-value using t-distribution CDF approximation
    # Using a simple approximation for the cumulative t-distribution
    x = df / (df + t_stat**2)
    p_value = _incomplete_beta(df / 2, 0.5, x)

    # Two-tailed test
    p_value = min(1.0, 2 * min(p_value, 1 - p_value))

    return t_stat, p_value


def mann_whitney_u_test(
    scores_a: list[float], scores_b: list[float]
) -> tuple[float, float]:
    """Mann-Whitney U test (non-parametric alternative).

    Does not assume normal distribution. Good for ordinal data or
    when sample sizes are small.

    Args:
        scores_a: Scores from variant A
        scores_b: Scores from variant B

    Returns:
        Tuple of (U_statistic, p_value)
    """
    n_a = len(scores_a)
    n_b = len(scores_b)

    if n_a < 1 or n_b < 1:
        return 0.0, 1.0

    # Combine and rank
    combined = [(s, "a") for s in scores_a] + [(s, "b") for s in scores_b]
    combined.sort(key=lambda x: x[0])

    # Assign ranks (handling ties)
    ranks: dict[int, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        # Average rank for ties
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of ranks for group A
    r_a = sum(ranks[i] for i, (_, group) in enumerate(combined) if group == "a")

    # U statistic
    u_a = r_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)

    # Normal approximation for large samples
    mu = n_a * n_b / 2
    sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)

    if sigma == 0:
        return u, 1.0

    z = (u - mu) / sigma
    p_value = 2 * _standard_normal_cdf(-abs(z))

    return u, p_value


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Approximate incomplete beta function for p-value calculation."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use continued fraction approximation
    # This is a simplified version
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1 - x)
    )

    if x < (a + 1) / (a + b + 2):
        return bt * _beta_cf(a, b, x) / a
    else:
        return 1.0 - bt * _beta_cf(b, a, 1 - x) / b


def _beta_cf(a: float, b: float, x: float) -> float:
    """Continued fraction for incomplete beta."""
    max_iter = 100
    eps = 1e-10

    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap

    if abs(d) < eps:
        d = eps
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < eps:
            d = eps
        c = 1.0 + aa / c
        if abs(c) < eps:
            c = eps
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def _standard_normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    # Error function approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-(x * x))

    return 0.5 * (1.0 + sign * y)


def calculate_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    baseline_std: float = 0.2,
) -> int:
    """Calculate required sample size per variant.

    Uses standard power analysis formula for two-sample t-test.

    Args:
        effect_size: Minimum detectable effect (difference in means)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)
        baseline_std: Expected standard deviation of the metric

    Returns:
        Required sample size per variant
    """
    # Z-scores for alpha and power
    z_alpha = _z_score(1 - alpha / 2)  # Two-tailed
    z_power = _z_score(power)

    # Sample size formula
    n = 2 * ((z_alpha + z_power) * baseline_std / effect_size) ** 2

    return max(10, int(math.ceil(n)))


def _z_score(p: float) -> float:
    """Inverse standard normal (approximate)."""
    if p <= 0:
        return -10
    if p >= 1:
        return 10
    if abs(p - 0.5) < 1e-10:
        return 0.0

    # Rational approximation (Abramowitz and Stegun)
    if p < 0.5:
        t = math.sqrt(-2 * math.log(p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))
    else:
        # For p > 0.5, use symmetry without recursion
        t = math.sqrt(-2 * math.log(1 - p))
        c0 = 2.515517
        c1 = 0.802853
        c2 = 0.010328
        d1 = 1.432788
        d2 = 0.189269
        d3 = 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


# =============================================================================
# Data Classes
# =============================================================================


class ExperimentStatus(str, Enum):
    """Status of an A/B experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"
    FAILED = "failed"


class StatisticalTest(str, Enum):
    """Statistical test to use for significance."""

    WELCH_T = "welch_t"  # Parametric, assumes normality
    MANN_WHITNEY_U = "mann_whitney_u"  # Non-parametric


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment.

    Attributes:
        agent_type: Agent class name (e.g., "CodeAnalyzerAgent")
        variant_a: Name of control variant
        variant_b: Name of treatment variant
        metric_name: Name of the primary metric to optimize
        minimum_effect_size: Minimum improvement to detect (e.g., 0.05 for 5%)
        significance_level: Alpha level for statistical significance (default 0.05)
        power: Statistical power for sample size calculation (default 0.8)
        max_samples: Maximum samples before stopping
        enable_early_stopping: Whether to use sequential testing
        early_stopping_threshold: Threshold for early stopping (default 0.001)
        statistical_test: Which test to use for significance
        traffic_split: Fraction of traffic to variant B (default 0.5)
    """

    agent_type: str
    variant_a: str = "default"
    variant_b: str = "treatment"
    metric_name: str = "accuracy"
    minimum_effect_size: float = 0.05
    significance_level: float = 0.05
    power: float = 0.8
    max_samples: int | None = None
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.001
    statistical_test: StatisticalTest = StatisticalTest.WELCH_T
    traffic_split: float = 0.5

    def __post_init__(self):
        """Validate configuration."""
        if self.traffic_split <= 0 or self.traffic_split >= 1:
            raise ValueError("traffic_split must be between 0 and 1 (exclusive)")
        if self.significance_level <= 0 or self.significance_level >= 1:
            raise ValueError("significance_level must be between 0 and 1")
        if self.power <= 0 or self.power >= 1:
            raise ValueError("power must be between 0 and 1")


@dataclass
class ExperimentAssignment:
    """Assignment of a user/session to a variant."""

    user_id: str
    variant: str
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assignment_hash: str = ""

    def __post_init__(self):
        if not self.assignment_hash:
            self.assignment_hash = hashlib.sha256(
                f"{self.user_id}:{self.variant}".encode()
            ).hexdigest()[:16]


@dataclass
class ExperimentObservation:
    """Single observation in an experiment."""

    observation_id: str
    user_id: str
    variant: str
    metric_value: float
    input_data: dict
    output_data: dict | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Result of an A/B experiment.

    Contains statistical analysis and recommendation.
    """

    experiment_id: str
    variant_a: str
    variant_b: str
    variant_a_mean: float
    variant_b_mean: float
    variant_a_std: float
    variant_b_std: float
    variant_a_samples: int
    variant_b_samples: int
    effect_size: float
    relative_improvement: float
    test_statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    winner: str | None
    recommendation: str
    statistical_test: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "variant_a": self.variant_a,
            "variant_b": self.variant_b,
            "variant_a_mean": self.variant_a_mean,
            "variant_b_mean": self.variant_b_mean,
            "variant_a_std": self.variant_a_std,
            "variant_b_std": self.variant_b_std,
            "variant_a_samples": self.variant_a_samples,
            "variant_b_samples": self.variant_b_samples,
            "effect_size": self.effect_size,
            "relative_improvement": self.relative_improvement,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "confidence_interval": list(self.confidence_interval),
            "is_significant": self.is_significant,
            "winner": self.winner,
            "recommendation": self.recommendation,
            "statistical_test": self.statistical_test,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


@dataclass
class Experiment:
    """An A/B experiment instance."""

    id: str
    name: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    required_sample_size: int = 0
    observations: list[ExperimentObservation] = field(default_factory=list)
    assignments: dict[str, ExperimentAssignment] = field(default_factory=dict)
    results: list[ExperimentResult] = field(default_factory=list)
    langfuse_dataset_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.required_sample_size == 0:
            self.required_sample_size = calculate_sample_size(
                effect_size=self.config.minimum_effect_size,
                alpha=self.config.significance_level,
                power=self.config.power,
            )
            if self.config.max_samples:
                self.required_sample_size = min(
                    self.required_sample_size, self.config.max_samples
                )

    @property
    def total_observations(self) -> int:
        """Total number of observations."""
        return len(self.observations)

    @property
    def variant_a_observations(self) -> list[ExperimentObservation]:
        """Observations for variant A."""
        return [o for o in self.observations if o.variant == self.config.variant_a]

    @property
    def variant_b_observations(self) -> list[ExperimentObservation]:
        """Observations for variant B."""
        return [o for o in self.observations if o.variant == self.config.variant_b]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "config": {
                "agent_type": self.config.agent_type,
                "variant_a": self.config.variant_a,
                "variant_b": self.config.variant_b,
                "metric_name": self.config.metric_name,
                "minimum_effect_size": self.config.minimum_effect_size,
                "significance_level": self.config.significance_level,
                "power": self.config.power,
                "max_samples": self.config.max_samples,
                "enable_early_stopping": self.config.enable_early_stopping,
                "statistical_test": self.config.statistical_test.value,
                "traffic_split": self.config.traffic_split,
            },
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "required_sample_size": self.required_sample_size,
            "total_observations": self.total_observations,
            "langfuse_dataset_id": self.langfuse_dataset_id,
            "metadata": self.metadata,
        }


# =============================================================================
# A/B Test Runner
# =============================================================================


class ABTestRunner:
    """Production-grade A/B test execution harness.

    Manages the full lifecycle of prompt A/B experiments:
    1. Experiment creation with power analysis
    2. User assignment to variants (consistent hashing)
    3. Observation collection
    4. Statistical analysis with proper tests
    5. Early stopping for efficiency
    6. Langfuse integration for tracking

    Example:
        ```python
        runner = ABTestRunner()

        # Create experiment
        experiment = await runner.create_experiment(
            name="code_analyzer_v2",
            config=ExperimentConfig(
                agent_type="CodeAnalyzerAgent",
                variant_a="default",
                variant_b="optimized_v2",
                minimum_effect_size=0.05,
            ),
        )

        # Run with test cases
        result = await runner.run_experiment(
            experiment_id=experiment.id,
            test_cases=test_data,
            metric_fn=accuracy_metric,
        )
        ```
    """

    def __init__(
        self,
        use_langfuse: bool = True,
        storage_backend: str = "memory",  # "memory" or "cognee"
    ):
        """Initialize the A/B test runner.

        Args:
            use_langfuse: Whether to track experiments in Langfuse
            storage_backend: Where to persist experiment state
        """
        self.use_langfuse = use_langfuse
        self.storage_backend = storage_backend
        self._experiments: dict[str, Experiment] = {}
        self._prompt_manager = None
        self.log = logger.bind(component="ABTestRunner")

    @property
    def prompt_manager(self):
        """Lazy-load prompt manager."""
        if self._prompt_manager is None:
            from .prompt_optimizer import AgentPromptManager

            self._prompt_manager = AgentPromptManager()
        return self._prompt_manager

    async def create_experiment(
        self,
        name: str,
        config: ExperimentConfig,
        description: str = "",
        tags: list[str] | None = None,
    ) -> Experiment:
        """Create a new A/B experiment.

        Args:
            name: Human-readable experiment name
            config: Experiment configuration
            description: Optional description
            tags: Optional tags for filtering

        Returns:
            Created Experiment instance
        """
        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"

        experiment = Experiment(
            id=experiment_id,
            name=name,
            config=config,
            metadata={
                "description": description,
                "tags": tags or [],
            },
        )

        # Create Langfuse dataset if enabled
        if self.use_langfuse:
            experiment.langfuse_dataset_id = await self._create_langfuse_dataset(
                experiment, description, tags
            )

        self._experiments[experiment_id] = experiment

        # Persist to storage
        await self._save_experiment(experiment)

        self.log.info(
            "Experiment created",
            experiment_id=experiment_id,
            name=name,
            required_sample_size=experiment.required_sample_size,
            variant_a=config.variant_a,
            variant_b=config.variant_b,
        )

        return experiment

    async def start_experiment(self, experiment_id: str) -> Experiment:
        """Start an experiment (change status to running).

        Args:
            experiment_id: The experiment ID

        Returns:
            Updated experiment
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(
                f"Can only start experiments in DRAFT status, "
                f"current status: {experiment.status}"
            )

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now(timezone.utc)

        await self._save_experiment(experiment)

        self.log.info("Experiment started", experiment_id=experiment_id)

        return experiment

    def assign_variant(
        self,
        experiment_id: str,
        user_id: str,
    ) -> str:
        """Assign a user to a variant (deterministic).

        Uses consistent hashing so the same user always gets
        the same variant within an experiment.

        Args:
            experiment_id: The experiment ID
            user_id: User identifier

        Returns:
            Assigned variant name
        """
        experiment = self._get_experiment(experiment_id)

        # Check if already assigned
        if user_id in experiment.assignments:
            return experiment.assignments[user_id].variant

        # Deterministic assignment using hash
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        fraction = (hash_value % 10000) / 10000

        if fraction < experiment.config.traffic_split:
            variant = experiment.config.variant_b  # Treatment
        else:
            variant = experiment.config.variant_a  # Control

        # Record assignment
        assignment = ExperimentAssignment(user_id=user_id, variant=variant)
        experiment.assignments[user_id] = assignment

        return variant

    async def record_observation(
        self,
        experiment_id: str,
        user_id: str,
        metric_value: float,
        input_data: dict,
        output_data: dict | None = None,
        latency_ms: float = 0.0,
        cost_usd: float = 0.0,
        metadata: dict | None = None,
    ) -> ExperimentObservation:
        """Record an observation for an experiment.

        Args:
            experiment_id: The experiment ID
            user_id: User who generated this observation
            metric_value: The primary metric value
            input_data: Input that was tested
            output_data: Output from the prompt
            latency_ms: Response latency
            cost_usd: Cost of the call
            metadata: Additional metadata

        Returns:
            Recorded observation
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status not in (ExperimentStatus.RUNNING, ExperimentStatus.DRAFT):
            raise ValueError(
                f"Cannot record observations for experiment in status: {experiment.status}"
            )

        # Get or assign variant
        variant = self.assign_variant(experiment_id, user_id)

        observation = ExperimentObservation(
            observation_id=f"obs_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            variant=variant,
            metric_value=metric_value,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )

        experiment.observations.append(observation)

        # Track in Langfuse
        if self.use_langfuse and experiment.langfuse_dataset_id:
            await self._record_langfuse_observation(experiment, observation)

        # Check for early stopping
        if experiment.config.enable_early_stopping:
            should_stop, result = await self._check_early_stopping(experiment)
            if should_stop:
                experiment.status = ExperimentStatus.STOPPED_EARLY
                experiment.completed_at = datetime.now(timezone.utc)
                experiment.results.append(result)
                self.log.info(
                    "Experiment stopped early",
                    experiment_id=experiment_id,
                    p_value=result.p_value,
                    winner=result.winner,
                )

        return observation

    async def run_experiment(
        self,
        experiment_id: str,
        test_cases: list[dict],
        metric_fn: Callable[[dict, dict], float] | None = None,
        concurrent_limit: int = 5,
    ) -> ExperimentResult:
        """Run a complete experiment with test cases.

        Executes both variants on all test cases and computes results.

        Args:
            experiment_id: The experiment ID
            test_cases: List of test cases with "input" and "expected" keys
            metric_fn: Function to compute metric (actual, expected) -> float
            concurrent_limit: Max concurrent executions

        Returns:
            Final experiment result
        """
        experiment = self._get_experiment(experiment_id)

        if experiment.status == ExperimentStatus.DRAFT:
            await self.start_experiment(experiment_id)

        if metric_fn is None:
            metric_fn = self._default_metric

        # Load prompts
        prompt_a = await self.prompt_manager.get_prompt(
            experiment.config.agent_type, experiment.config.variant_a
        )
        prompt_b = await self.prompt_manager.get_prompt(
            experiment.config.agent_type, experiment.config.variant_b
        )

        if not prompt_a or not prompt_b:
            raise ValueError(
                f"Could not load prompts for variants: "
                f"{experiment.config.variant_a}, {experiment.config.variant_b}"
            )

        self.log.info(
            "Running experiment",
            experiment_id=experiment_id,
            test_cases=len(test_cases),
            concurrent_limit=concurrent_limit,
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def run_single_case(
            case_idx: int,
            case: dict,
            prompt: str,
            variant: str,
        ) -> tuple[float, dict]:
            """Run a single test case."""
            async with semaphore:
                input_data = case.get("input", case)
                expected = case.get("expected", case.get("output", {}))

                start_time = time.time()
                try:
                    output = await self.prompt_manager._run_prompt(prompt, input_data)
                    score = metric_fn(output, expected)
                    latency_ms = (time.time() - start_time) * 1000
                    return score, {
                        "output": output,
                        "latency_ms": latency_ms,
                        "success": True,
                    }
                except Exception as e:
                    latency_ms = (time.time() - start_time) * 1000
                    self.log.warning(
                        "Test case failed",
                        case_idx=case_idx,
                        variant=variant,
                        error=str(e),
                    )
                    return 0.0, {
                        "output": None,
                        "latency_ms": latency_ms,
                        "success": False,
                        "error": str(e),
                    }

        # Run all test cases for both variants
        tasks_a = [
            run_single_case(i, case, prompt_a, experiment.config.variant_a)
            for i, case in enumerate(test_cases)
        ]
        tasks_b = [
            run_single_case(i, case, prompt_b, experiment.config.variant_b)
            for i, case in enumerate(test_cases)
        ]

        results_a = await asyncio.gather(*tasks_a)
        results_b = await asyncio.gather(*tasks_b)

        # Record observations
        for i, (case, (score, details)) in enumerate(zip(test_cases, results_a)):
            user_id = f"case_{i}_a"
            # Manually assign to variant A
            experiment.assignments[user_id] = ExperimentAssignment(
                user_id=user_id, variant=experiment.config.variant_a
            )
            await self.record_observation(
                experiment_id=experiment_id,
                user_id=user_id,
                metric_value=score,
                input_data=case.get("input", case),
                output_data=details.get("output"),
                latency_ms=details.get("latency_ms", 0),
                metadata={"case_idx": i},
            )

        for i, (case, (score, details)) in enumerate(zip(test_cases, results_b)):
            user_id = f"case_{i}_b"
            # Manually assign to variant B
            experiment.assignments[user_id] = ExperimentAssignment(
                user_id=user_id, variant=experiment.config.variant_b
            )
            await self.record_observation(
                experiment_id=experiment_id,
                user_id=user_id,
                metric_value=score,
                input_data=case.get("input", case),
                output_data=details.get("output"),
                latency_ms=details.get("latency_ms", 0),
                metadata={"case_idx": i},
            )

        # Compute final result
        result = self.analyze_experiment(experiment_id)

        # Mark as completed
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        experiment.results.append(result)

        await self._save_experiment(experiment)

        # Score in Langfuse
        if self.use_langfuse:
            await self._score_langfuse_experiment(experiment, result)

        self.log.info(
            "Experiment completed",
            experiment_id=experiment_id,
            winner=result.winner,
            p_value=result.p_value,
            is_significant=result.is_significant,
            effect_size=result.effect_size,
        )

        return result

    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results.

        Computes statistical significance and effect size.

        Args:
            experiment_id: The experiment ID

        Returns:
            Analysis result
        """
        experiment = self._get_experiment(experiment_id)

        scores_a = [o.metric_value for o in experiment.variant_a_observations]
        scores_b = [o.metric_value for o in experiment.variant_b_observations]

        if len(scores_a) < 2 or len(scores_b) < 2:
            return ExperimentResult(
                experiment_id=experiment_id,
                variant_a=experiment.config.variant_a,
                variant_b=experiment.config.variant_b,
                variant_a_mean=statistics.mean(scores_a) if scores_a else 0.0,
                variant_b_mean=statistics.mean(scores_b) if scores_b else 0.0,
                variant_a_std=0.0,
                variant_b_std=0.0,
                variant_a_samples=len(scores_a),
                variant_b_samples=len(scores_b),
                effect_size=0.0,
                relative_improvement=0.0,
                test_statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                winner=None,
                recommendation="Insufficient data for analysis",
                statistical_test=experiment.config.statistical_test.value,
            )

        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)
        std_a = statistics.stdev(scores_a)
        std_b = statistics.stdev(scores_b)

        # Effect size (Cohen's d)
        pooled_std = math.sqrt(((len(scores_a) - 1) * std_a**2 + (len(scores_b) - 1) * std_b**2) / (len(scores_a) + len(scores_b) - 2))
        effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0.0

        # Relative improvement
        relative_improvement = (mean_b - mean_a) / mean_a if mean_a != 0 else 0.0

        # Statistical test
        if experiment.config.statistical_test == StatisticalTest.WELCH_T:
            test_stat, p_value = welch_t_test(scores_a, scores_b)
        else:
            test_stat, p_value = mann_whitney_u_test(scores_a, scores_b)

        # Confidence interval for difference (approximation)
        se = math.sqrt(std_a**2 / len(scores_a) + std_b**2 / len(scores_b))
        z = 1.96  # 95% CI
        diff = mean_b - mean_a
        ci_lower = diff - z * se
        ci_upper = diff + z * se

        # Determine significance and winner
        is_significant = p_value < experiment.config.significance_level

        if is_significant:
            if mean_b > mean_a:
                winner = experiment.config.variant_b
                recommendation = (
                    f"Adopt {experiment.config.variant_b}: "
                    f"{relative_improvement * 100:.1f}% improvement (p={p_value:.4f})"
                )
            else:
                winner = experiment.config.variant_a
                recommendation = (
                    f"Keep {experiment.config.variant_a}: "
                    f"Treatment showed {-relative_improvement * 100:.1f}% regression (p={p_value:.4f})"
                )
        else:
            winner = None
            recommendation = (
                f"No significant difference detected (p={p_value:.4f}). "
                f"Consider collecting more data or the variants are equivalent."
            )

        return ExperimentResult(
            experiment_id=experiment_id,
            variant_a=experiment.config.variant_a,
            variant_b=experiment.config.variant_b,
            variant_a_mean=mean_a,
            variant_b_mean=mean_b,
            variant_a_std=std_a,
            variant_b_std=std_b,
            variant_a_samples=len(scores_a),
            variant_b_samples=len(scores_b),
            effect_size=effect_size,
            relative_improvement=relative_improvement,
            test_statistic=test_stat,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            winner=winner,
            recommendation=recommendation,
            statistical_test=experiment.config.statistical_test.value,
            details={
                "scores_a": scores_a,
                "scores_b": scores_b,
            },
        )

    async def get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID.

        Args:
            experiment_id: The experiment ID

        Returns:
            The experiment
        """
        return self._get_experiment(experiment_id)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
        agent_type: str | None = None,
    ) -> list[Experiment]:
        """List all experiments with optional filtering.

        Args:
            status: Filter by status
            agent_type: Filter by agent type

        Returns:
            List of experiments
        """
        experiments = list(self._experiments.values())

        if status:
            experiments = [e for e in experiments if e.status == status]
        if agent_type:
            experiments = [e for e in experiments if e.config.agent_type == agent_type]

        return sorted(experiments, key=lambda e: e.created_at, reverse=True)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _get_experiment(self, experiment_id: str) -> Experiment:
        """Get experiment by ID or raise error."""
        if experiment_id not in self._experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")
        return self._experiments[experiment_id]

    async def _save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to storage backend."""
        if self.storage_backend == "cognee":
            try:
                from ..knowledge.cognee_client import get_cognee_client

                cognee = get_cognee_client()
                await cognee.put(
                    key=f"experiment_{experiment.id}",
                    value=experiment.to_dict(),
                    namespace="ab_experiments",
                    embeddings=False,
                )
            except Exception as e:
                self.log.warning(
                    "Failed to save experiment to Cognee",
                    experiment_id=experiment.id,
                    error=str(e),
                )

    async def _check_early_stopping(
        self, experiment: Experiment
    ) -> tuple[bool, ExperimentResult | None]:
        """Check if experiment should stop early using sequential testing.

        Uses the O'Brien-Fleming spending function for group sequential testing.
        """
        # Need minimum samples for early stopping
        min_samples = max(10, experiment.required_sample_size // 4)
        if (
            len(experiment.variant_a_observations) < min_samples
            or len(experiment.variant_b_observations) < min_samples
        ):
            return False, None

        result = self.analyze_experiment(experiment.id)

        # O'Brien-Fleming alpha spending
        info_fraction = min(
            experiment.total_observations / (2 * experiment.required_sample_size), 1.0
        )
        # Adjusted significance threshold
        adjusted_alpha = (
            experiment.config.early_stopping_threshold
            * (2 * _standard_normal_cdf(-_z_score(1 - experiment.config.significance_level / 2) / math.sqrt(info_fraction)))
        )

        if result.p_value < adjusted_alpha:
            return True, result

        return False, None

    async def _create_langfuse_dataset(
        self,
        experiment: Experiment,
        description: str,
        tags: list[str] | None,
    ) -> str | None:
        """Create a Langfuse dataset for the experiment."""
        try:
            import os

            if os.environ.get("LANGFUSE_ENABLED", "true").lower() == "false":
                return None

            from langfuse import Langfuse

            langfuse = Langfuse()

            dataset = langfuse.create_dataset(
                name=f"ab_experiment_{experiment.id}",
                description=description or f"A/B test: {experiment.name}",
                metadata={
                    "experiment_id": experiment.id,
                    "variant_a": experiment.config.variant_a,
                    "variant_b": experiment.config.variant_b,
                    "agent_type": experiment.config.agent_type,
                    "metric": experiment.config.metric_name,
                },
            )

            self.log.debug(
                "Created Langfuse dataset",
                dataset_name=dataset.name,
                experiment_id=experiment.id,
            )

            return dataset.name

        except Exception as e:
            self.log.warning(
                "Failed to create Langfuse dataset",
                error=str(e),
            )
            return None

    async def _record_langfuse_observation(
        self,
        experiment: Experiment,
        observation: ExperimentObservation,
    ) -> None:
        """Record observation in Langfuse dataset."""
        try:
            import os

            if os.environ.get("LANGFUSE_ENABLED", "true").lower() == "false":
                return

            from langfuse import Langfuse

            langfuse = Langfuse()

            langfuse.create_dataset_item(
                dataset_name=experiment.langfuse_dataset_id,
                input=observation.input_data,
                expected_output=observation.output_data,
                metadata={
                    "observation_id": observation.observation_id,
                    "variant": observation.variant,
                    "metric_value": observation.metric_value,
                    "latency_ms": observation.latency_ms,
                    "cost_usd": observation.cost_usd,
                },
            )

        except Exception as e:
            self.log.warning(
                "Failed to record Langfuse observation",
                error=str(e),
            )

    async def _score_langfuse_experiment(
        self,
        experiment: Experiment,
        result: ExperimentResult,
    ) -> None:
        """Score the experiment in Langfuse."""
        try:
            from ..orchestrator.langfuse_integration import score_trace

            # Score overall experiment result
            score_trace(
                trace_id=experiment.id,
                name=f"ab_experiment_{experiment.config.metric_name}",
                value=1.0 if result.is_significant else 0.0,
                comment=result.recommendation,
                data_type="NUMERIC",
            )

            # Score effect size
            score_trace(
                trace_id=experiment.id,
                name="effect_size",
                value=result.effect_size,
                comment=f"Cohen's d: {result.effect_size:.3f}",
                data_type="NUMERIC",
            )

        except Exception as e:
            self.log.warning(
                "Failed to score Langfuse experiment",
                error=str(e),
            )

    def _default_metric(self, actual: dict, expected: dict) -> float:
        """Default metric: key overlap and value similarity."""
        if not actual or not expected:
            return 0.0

        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        key_overlap = (
            len(actual_keys & expected_keys) / len(expected_keys) if expected_keys else 0
        )

        value_scores = []
        for key in actual_keys & expected_keys:
            actual_val = str(actual[key])
            expected_val = str(expected[key])

            if actual_val == expected_val:
                value_scores.append(1.0)
            elif expected_val in actual_val or actual_val in expected_val:
                value_scores.append(0.7)
            else:
                actual_words = set(actual_val.lower().split())
                expected_words = set(expected_val.lower().split())
                if actual_words or expected_words:
                    jaccard = len(actual_words & expected_words) / len(
                        actual_words | expected_words
                    )
                    value_scores.append(jaccard)
                else:
                    value_scores.append(0.0)

        value_similarity = statistics.mean(value_scores) if value_scores else 0.0

        return 0.3 * key_overlap + 0.7 * value_similarity


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_ab_test(
    agent_type: str,
    variant_a: str,
    variant_b: str,
    test_cases: list[dict],
    metric_fn: Callable[[dict, dict], float] | None = None,
    minimum_effect_size: float = 0.05,
    significance_level: float = 0.05,
) -> ExperimentResult:
    """Convenience function to run a quick A/B test.

    Args:
        agent_type: Agent class name
        variant_a: Control variant name
        variant_b: Treatment variant name
        test_cases: Test cases with input/expected
        metric_fn: Metric function
        minimum_effect_size: Minimum effect to detect
        significance_level: Alpha level

    Returns:
        ExperimentResult with analysis

    Example:
        ```python
        result = await run_ab_test(
            agent_type="CodeAnalyzerAgent",
            variant_a="default",
            variant_b="optimized_v2",
            test_cases=test_data,
        )
        print(f"Winner: {result.winner}, p={result.p_value:.4f}")
        ```
    """
    runner = ABTestRunner(use_langfuse=True)

    experiment = await runner.create_experiment(
        name=f"quick_test_{agent_type}_{variant_a}_vs_{variant_b}",
        config=ExperimentConfig(
            agent_type=agent_type,
            variant_a=variant_a,
            variant_b=variant_b,
            minimum_effect_size=minimum_effect_size,
            significance_level=significance_level,
        ),
    )

    return await runner.run_experiment(
        experiment_id=experiment.id,
        test_cases=test_cases,
        metric_fn=metric_fn,
    )
