"""DSPy-style Prompt Optimization Module.

This module provides:
- PromptSignature: Declarative prompt definitions (input/output behavior)
- PromptOptimizer: Automatic prompt optimization with multiple strategies
- AgentPromptManager: Version control and A/B testing for agent prompts
- AgentBenchmarkRunner: Benchmark runner for tracking agent performance

Optimizers available:
- COPRO: Coordinate Ascent Prompt Optimization (hill-climbing)
- MIPRO: Mixed Integer Programming for Prompt Optimization (Bayesian)
- SIMBA: Stochastic Mini-Batch Analysis (self-reflective)
- GEPA: Reflective Prompt Evolution (evolutionary)
"""

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
    "SignatureField",
    "PromptSignature",
    "OptimizedPrompt",
    "OptimizationResult",
    "ABTestResult",
    "BenchmarkResult",
    "BaseOptimizer",
    "COPROOptimizer",
    "MIPROOptimizer",
    "SIMBAOptimizer",
    "GEPAOptimizer",
    "PromptOptimizer",
    "AgentPromptManager",
    "AgentBenchmarkRunner",
]
