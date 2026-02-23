"""Benchmark Aggregator Service.

Computes Pass@k and CLEAR framework aggregates from raw agent_evaluations,
then stores results in the agent_benchmarks table.

Designed to run daily via the schedule daemon (4 AM UTC).
Can also be invoked on-demand via the benchmarks API.

Metrics computed:
- Pass@1, Pass@3, Pass@8 (consecutive success probability)
- CLEAR averages (efficacy, latency, cost, assurance)
- Percentile latencies (P50, P95, P99)
- Cost-Normalized Accuracy (CNA = efficacy / cost * 100)
- Self-correction rate, human escalation rate
"""

import statistics
from datetime import UTC, date, datetime, timedelta

import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger(__name__)


def compute_pass_at_k(results: list[bool], k: int) -> float | None:
    """Compute Pass@k: probability of k consecutive successes.

    Uses a sliding window over the result sequence. A window of size k
    counts as a success if ALL k results are True.

    Args:
        results: Ordered list of pass/fail booleans.
        k: Window size (e.g., 1, 3, 8).

    Returns:
        Fraction of windows that are all-True, or None if < k results.
    """
    if len(results) < k:
        return None
    total_windows = len(results) - k + 1
    successes = sum(1 for i in range(total_windows) if all(results[i:i + k]))
    return successes / total_windows


def compute_percentile(values: list[int | float], p: int) -> int | None:
    """Compute the p-th percentile of a list of numeric values."""
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    idx = min(idx, len(sorted_vals) - 1)
    return int(sorted_vals[idx])


async def aggregate_benchmarks(
    period_days: int = 1,
    organization_id: str | None = None,
) -> list[dict]:
    """Aggregate agent evaluations into benchmark records.

    Queries agent_evaluations for the given period and computes
    CLEAR metrics per agent_type.

    Args:
        period_days: Number of days to aggregate (1=daily, 7=weekly).
        organization_id: Scope to a single org, or None for global.

    Returns:
        List of benchmark records that were upserted.
    """
    supabase = get_supabase_client()

    period_end = date.today()
    period_start = period_end - timedelta(days=period_days)
    start_iso = datetime(period_start.year, period_start.month, period_start.day, tzinfo=UTC).isoformat().replace("+00:00", "Z")

    # Build query filters
    filters = f"created_at=gte.{start_iso}&order=created_at.asc"
    if organization_id:
        filters += f"&organization_id=eq.{organization_id}"

    # Fetch evaluations
    response = await supabase.request(
        f"/agent_evaluations?{filters}&select=agent_type,task_type,task_completed,efficacy_score,latency_ms,cost_usd,assurance_score,input_tokens,output_tokens,self_corrections,human_escalated,error_type"
    )

    rows = response.get("data", [])
    if not rows:
        logger.info("No evaluations found for aggregation", period_start=str(period_start))
        return []

    # Group by agent_type
    by_agent: dict[str, list[dict]] = {}
    for row in rows:
        agent = row.get("agent_type", "unknown")
        by_agent.setdefault(agent, []).append(row)

    benchmarks = []
    for agent_type, evals in by_agent.items():
        benchmark = _compute_agent_benchmark(
            agent_type=agent_type,
            evals=evals,
            period_start=period_start,
            period_end=period_end,
            organization_id=organization_id,
        )
        benchmarks.append(benchmark)

        # Upsert into agent_benchmarks
        try:
            # Delete existing record for this period/agent/org
            delete_filters = (
                f"agent_type=eq.{agent_type}"
                f"&period_start=eq.{period_start.isoformat()}"
            )
            if organization_id:
                delete_filters += f"&organization_id=eq.{organization_id}"
            else:
                delete_filters += "&organization_id=is.null"

            await supabase.request(
                f"/agent_benchmarks?{delete_filters}",
                method="DELETE",
            )

            # Insert new record
            await supabase.insert("agent_benchmarks", benchmark)

            logger.info(
                "Upserted agent benchmark",
                agent_type=agent_type,
                total_executions=benchmark["total_executions"],
                pass_at_1=benchmark.get("pass_at_1"),
                pass_at_8=benchmark.get("pass_at_8"),
            )
        except Exception as e:
            logger.error(
                "Failed to upsert benchmark",
                agent_type=agent_type,
                error=str(e),
            )

    return benchmarks


def _compute_agent_benchmark(
    agent_type: str,
    evals: list[dict],
    period_start: date,
    period_end: date,
    organization_id: str | None,
) -> dict:
    """Compute all benchmark metrics for a single agent type."""
    total = len(evals)

    # Extract ordered pass/fail sequence
    results = [bool(e.get("task_completed")) for e in evals]

    # Pass@k
    pass_at_1 = compute_pass_at_k(results, 1)
    pass_at_3 = compute_pass_at_k(results, 3)
    pass_at_8 = compute_pass_at_k(results, 8)

    # CLEAR averages
    efficacy_scores = [e["efficacy_score"] for e in evals if e.get("efficacy_score") is not None]
    latencies = [e["latency_ms"] for e in evals if e.get("latency_ms") is not None]
    costs = [e["cost_usd"] for e in evals if e.get("cost_usd") is not None]
    assurance_scores = [e["assurance_score"] for e in evals if e.get("assurance_score") is not None]

    avg_efficacy = statistics.mean(efficacy_scores) if efficacy_scores else None
    avg_latency = statistics.mean(latencies) if latencies else None
    avg_cost = statistics.mean(costs) if costs else None
    avg_assurance = statistics.mean(assurance_scores) if assurance_scores else None

    # Percentile latencies
    p50 = compute_percentile(latencies, 50)
    p95 = compute_percentile(latencies, 95)
    p99 = compute_percentile(latencies, 99)

    # Cost-Normalized Accuracy
    cna = None
    if avg_efficacy is not None and avg_cost and avg_cost > 0:
        cna = avg_efficacy / avg_cost * 100

    # Self-correction rate
    corrections = [e.get("self_corrections", 0) for e in evals]
    correction_rate = sum(1 for c in corrections if c > 0) / total if total > 0 else 0

    # Human escalation rate
    escalations = [e.get("human_escalated", False) for e in evals]
    escalation_rate = sum(1 for h in escalations if h) / total if total > 0 else 0

    return {
        "organization_id": organization_id,
        "agent_type": agent_type,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "total_executions": total,
        "pass_at_1": pass_at_1,
        "pass_at_3": pass_at_3,
        "pass_at_8": pass_at_8,
        "avg_efficacy": avg_efficacy,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "avg_cost_usd": avg_cost,
        "cost_normalized_accuracy": cna,
        "avg_assurance": avg_assurance,
        "self_correction_rate": correction_rate,
        "human_escalation_rate": escalation_rate,
        "hallucination_rate": None,  # Requires hallucination detector integration
        "false_positive_rate": None,  # Requires ground-truth labels
        "mttr_seconds": None,  # SRE-specific, filled by SRE aggregation
        "mttd_seconds": None,
        "automated_resolution_rate": None,
    }
