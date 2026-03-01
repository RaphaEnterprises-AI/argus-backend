"""Lightweight intent verifier — checks whether agent results satisfy the original request.

Runs a single cheap LLM call after swarm execution to detect:
- Unaddressed goals from the original intent
- Partial completions (e.g., only 2 of 5 requested areas covered)
- Misaligned outputs (agents ran but didn't answer what was asked)

Cost: ~$0.0001 per verification (flash-tier model, <500 tokens out).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class IntentVerdict:
    """Result of intent verification."""

    completion_score: float  # 0.0 - 1.0
    verdict: str  # "complete", "partial", "incomplete"
    gaps: list[str] = field(default_factory=list)
    summary: str = ""


def extract_intent(config) -> str:
    """Build a human-readable intent string from SwarmConfig."""
    from .swarm_orchestrator import SwarmMode

    mode = config.mode
    parts = []

    if mode == SwarmMode.FULL_CRAWL:
        parts.append(f"Comprehensive security, accessibility, performance, and UI analysis of {config.target_url}")
    elif mode == SwarmMode.TARGETED_BLITZ:
        flow = config.target_flow or "the main flow"
        parts.append(f"Targeted test of {flow} on {config.target_url}")
    elif mode == SwarmMode.PR_ANALYSIS:
        n = len(config.changed_files) if config.changed_files else 0
        pr = f" (PR #{config.pr_number})" if config.pr_number else ""
        parts.append(f"Analyze {n} changed files{pr} for test impact, code quality, and security")
    elif mode == SwarmMode.DISCOVERY_SWARM:
        parts.append(f"Discover all testable surfaces of {config.target_url} and test each one")
    else:
        parts.append(f"{mode.value} analysis")

    if config.target_url:
        parts.append(f"Target: {config.target_url}")

    return ". ".join(parts)


def _summarize_results(worker_results: list) -> str:
    """Build a compact summary of what workers actually produced."""
    lines = []
    for r in worker_results:
        status = "OK" if r.success else "FAILED"
        findings = len(r.findings) if r.findings else 0
        lines.append(
            f"- {r.agent_type}: {status}, {findings} findings, "
            f"confidence={r.confidence:.0%}"
            + (f", error={r.error}" if r.error else "")
        )
    return "\n".join(lines) if lines else "(no results)"


async def verify_intent(
    config,
    worker_results: list,
    total_cost: float = 0.0,
) -> IntentVerdict:
    """Verify whether worker results satisfy the original intent.

    Makes one cheap LLM call (flash-tier). Falls back to heuristic
    scoring if the LLM call fails, so this never blocks the pipeline.
    """
    intent = extract_intent(config)
    results_summary = _summarize_results(worker_results)

    # Fast heuristic first — if all workers succeeded with decent confidence,
    # skip the LLM call entirely (saves cost on happy path)
    successful = [r for r in worker_results if r.success]
    if len(successful) == len(worker_results) and all(
        r.confidence >= 0.6 for r in successful
    ):
        return IntentVerdict(
            completion_score=round(
                sum(r.confidence for r in successful) / max(len(successful), 1), 2
            ),
            verdict="complete",
            gaps=[],
            summary="All agents completed successfully with good confidence",
        )

    # If there are failures or low confidence, use LLM to evaluate
    prompt = f"""You are an intent-completion verifier. Compare what was REQUESTED vs what was DELIVERED and identify gaps.

REQUESTED INTENT:
{intent}

AGENT RESULTS:
{results_summary}

Respond with ONLY valid JSON (no markdown):
{{
  "completion_score": <0.0 to 1.0>,
  "verdict": "<complete|partial|incomplete>",
  "gaps": ["list of unaddressed goals or failed areas"],
  "summary": "one-line explanation"
}}

Rules:
- Score 1.0 = all goals met, 0.5 = major gaps, 0.0 = nothing useful
- "complete" if score >= 0.8, "partial" if >= 0.4, "incomplete" otherwise
- List specific gaps, not vague observations
- An agent that failed or returned an error result is a gap"""

    try:
        from src.core.model_router import ModelRouter, TaskType

        router = ModelRouter()
        response = await router.complete(
            task_type=TaskType.ELEMENT_CLASSIFICATION,  # Cheapest tier
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
            json_mode=True,
        )

        content = response.get("content", "")
        # Strip markdown fences if model wraps in ```json
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(content)

        score = float(parsed.get("completion_score", 0.5))
        verdict_str = parsed.get("verdict", "partial")
        if verdict_str not in ("complete", "partial", "incomplete"):
            verdict_str = "complete" if score >= 0.8 else "partial" if score >= 0.4 else "incomplete"

        return IntentVerdict(
            completion_score=round(score, 2),
            verdict=verdict_str,
            gaps=parsed.get("gaps", []),
            summary=parsed.get("summary", ""),
        )

    except Exception as e:
        logger.warning("Intent verification LLM call failed, using heuristic", error=str(e))
        return _heuristic_verdict(worker_results)


def _heuristic_verdict(worker_results: list) -> IntentVerdict:
    """Fallback heuristic when LLM verification is unavailable."""
    if not worker_results:
        return IntentVerdict(
            completion_score=0.0,
            verdict="incomplete",
            gaps=["No agents produced results"],
            summary="No results to evaluate",
        )

    successful = [r for r in worker_results if r.success]
    failed = [r for r in worker_results if not r.success]
    total = len(worker_results)

    score = len(successful) / total if total > 0 else 0.0

    # Penalize low-confidence successes
    if successful:
        avg_conf = sum(r.confidence for r in successful) / len(successful)
        score = score * 0.6 + avg_conf * 0.4

    gaps = [f"{r.agent_type} failed: {r.error or 'unknown error'}" for r in failed]

    # Flag error/empty results (confidence == 0, no findings)
    for r in successful:
        if r.confidence == 0.0 and not r.findings:
            gaps.append(f"{r.agent_type} returned empty results")

    verdict = "complete" if score >= 0.8 else "partial" if score >= 0.4 else "incomplete"

    return IntentVerdict(
        completion_score=round(score, 2),
        verdict=verdict,
        gaps=gaps,
        summary=f"{len(successful)}/{total} agents succeeded (heuristic evaluation)",
    )
