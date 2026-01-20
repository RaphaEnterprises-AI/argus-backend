"""
Agent Evaluation Framework for E2E Testing Agent.

This module provides comprehensive evaluation of the multi-agent system's
performance, intelligence, and AI capabilities using industry-standard
benchmarks and real-world test scenarios.

Evaluation Categories:
1. Agent Quality Metrics - Accuracy, reasoning, decision making
2. System Performance - Latency, throughput, cost efficiency
3. Integration Tests - End-to-end workflow validation
4. Competitive Benchmarks - Comparison with commercial tools

Based on research from:
- WebArena/BrowserArena benchmarks
- AgentBench multi-domain evaluation
- BFCL (Berkeley Function-Calling Leaderboard)
- LangChain State of AI Agents 2025/2026

References:
- https://o-mega.ai/articles/the-best-ai-agent-evals-and-benchmarks-full-2025-guide
- https://www.langchain.com/state-of-agent-engineering
- https://www.evidentlyai.com/blog/ai-agent-benchmarks
"""

from .metrics import EvaluationMetrics, AgentScore
from .runner import EvaluationRunner

__all__ = ["EvaluationMetrics", "AgentScore", "EvaluationRunner"]
