"""DSPy-style Prompt Optimization Module.

This module provides:
- PromptSignature: Declarative prompt definitions (input/output behavior)
- PromptOptimizer: Automatic prompt optimization with multiple strategies
- AgentPromptManager: Version control and A/B testing for agent prompts
- AgentBenchmarkRunner: Benchmark runner for tracking agent performance
- ABTestRunner: Production-grade A/B test execution harness (RAP-340)

Optimizers available:
- COPRO: Coordinate Ascent Prompt Optimization (hill-climbing)
- MIPRO: Mixed Integer Programming for Prompt Optimization (Bayesian)
- SIMBA: Stochastic Mini-Batch Analysis (self-reflective)
- GEPA: Reflective Prompt Evolution (evolutionary)

A/B Testing (RAP-340):
- ABTestRunner: Full experiment lifecycle management
- ExperimentConfig: Configuration for A/B experiments
- ExperimentResult: Statistical analysis results
- run_ab_test: Convenience function for quick experiments
"""

from .ab_test_runner import (
    ABTestRunner,
    Experiment,
    ExperimentConfig,
    ExperimentObservation,
    ExperimentResult,
    ExperimentStatus,
    StatisticalTest,
    calculate_sample_size,
    run_ab_test,
    welch_t_test,
    mann_whitney_u_test,
)
from .prompt_optimizer import (
    ABTestResult,
    AgentBenchmarkRunner,
    AgentPromptManager,
    BaseOptimizer,
    BenchmarkResult,
    COPROOptimizer,
    GEPAOptimizer,
    MIPROOptimizer,
    OptimizationResult,
    OptimizedPrompt,
    PromptOptimizer,
    PromptSignature,
    SignatureField,
    SIMBAOptimizer,
)

__all__ = [
    # Prompt Signatures
    "SignatureField",
    "PromptSignature",
    "OptimizedPrompt",
    "OptimizationResult",
    "ABTestResult",
    "BenchmarkResult",
    # Optimizers
    "BaseOptimizer",
    "COPROOptimizer",
    "MIPROOptimizer",
    "SIMBAOptimizer",
    "GEPAOptimizer",
    "PromptOptimizer",
    # Managers
    "AgentPromptManager",
    "AgentBenchmarkRunner",
    # A/B Testing (RAP-340)
    "ABTestRunner",
    "Experiment",
    "ExperimentConfig",
    "ExperimentObservation",
    "ExperimentResult",
    "ExperimentStatus",
    "StatisticalTest",
    "calculate_sample_size",
    "run_ab_test",
    "welch_t_test",
    "mann_whitney_u_test",
]
