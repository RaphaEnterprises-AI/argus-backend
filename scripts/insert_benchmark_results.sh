#!/bin/bash
# Insert QA Intelligence Agent benchmark results from 2026-02-26 test runs.
# Uses the /api/v1/evaluations/record-batch endpoint.

API_URL="https://argus-brain-production.up.railway.app"
API_KEY="argus_sk_9a1819e0e9faa47011c28ff75bbeb442ffa285779233ea3e81aed7e992367198"
ORG_ID="229904aa-3084-4b25-8bf8-c07ae749888c"

echo "Inserting QA agent benchmark results..."

curl -s -X POST "${API_URL}/api/v1/evaluations/record-batch" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  -H "X-Organization-Id: ${ORG_ID}" \
  -d '{
  "global_benchmark": true,
  "evaluations": [
    {
      "agent_type": "TestGenAgent",
      "task_type": "requirements_analysis",
      "task_completed": true,
      "efficacy_score": 0.85,
      "latency_ms": 47100,
      "cost_usd": 0.02,
      "assurance_score": 0.70,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "authentication",
        "requirements_count": 5,
        "avg_ambiguity": 0.30,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "requirements_analysis",
      "task_completed": true,
      "efficacy_score": 0.80,
      "latency_ms": 56900,
      "cost_usd": 0.03,
      "assurance_score": 0.59,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "e-commerce",
        "requirements_count": 7,
        "avg_ambiguity": 0.41,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "requirements_analysis",
      "task_completed": true,
      "efficacy_score": 0.78,
      "latency_ms": 62300,
      "cost_usd": 0.03,
      "assurance_score": 0.55,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "api_design",
        "requirements_count": 6,
        "avg_ambiguity": 0.45,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "requirements_analysis",
      "task_completed": true,
      "efficacy_score": 0.80,
      "latency_ms": 59000,
      "cost_usd": 0.03,
      "assurance_score": 0.58,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "healthcare",
        "requirements_count": 6,
        "avg_ambiguity": 0.42,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "requirements_analysis",
      "task_completed": true,
      "efficacy_score": 0.75,
      "latency_ms": 41800,
      "cost_usd": 0.02,
      "assurance_score": 0.50,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "collaboration",
        "requirements_count": 5,
        "avg_ambiguity": 0.50,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "test_generation",
      "task_completed": true,
      "efficacy_score": 0.76,
      "latency_ms": 89000,
      "cost_usd": 0.039,
      "assurance_score": 0.76,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "authentication",
        "framework": "playwright",
        "language": "typescript",
        "requirements_parsed": 4,
        "scenarios_generated": 5,
        "tests_generated": 5,
        "quality_score": 0.76,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "test_generation",
      "task_completed": true,
      "efficacy_score": 0.76,
      "latency_ms": 124000,
      "cost_usd": 0.102,
      "assurance_score": 0.76,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "e-commerce",
        "framework": "cypress",
        "language": "javascript",
        "requirements_parsed": 5,
        "scenarios_generated": 5,
        "tests_generated": 5,
        "quality_score": 0.76,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "test_generation",
      "task_completed": true,
      "efficacy_score": 0.78,
      "latency_ms": 118000,
      "cost_usd": 0.106,
      "assurance_score": 0.78,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "api_design",
        "framework": "pytest",
        "language": "python",
        "requirements_parsed": 5,
        "scenarios_generated": 5,
        "tests_generated": 5,
        "quality_score": 0.78,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "test_generation",
      "task_completed": true,
      "efficacy_score": 0.77,
      "latency_ms": 105000,
      "cost_usd": 0.088,
      "assurance_score": 0.77,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "multi-page-app",
        "framework": "playwright",
        "language": "typescript",
        "requirements_parsed": 6,
        "scenarios_generated": 6,
        "tests_generated": 6,
        "quality_score": 0.77,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "test_generation",
      "task_completed": true,
      "efficacy_score": 0.60,
      "latency_ms": 78000,
      "cost_usd": 0.041,
      "assurance_score": 0.60,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "login_flow",
        "framework": "playwright",
        "language": "typescript",
        "requirements_parsed": 3,
        "scenarios_generated": 3,
        "tests_generated": 3,
        "quality_score": 0.60,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestGenAgent",
      "task_type": "test_generation",
      "task_completed": true,
      "efficacy_score": 0.65,
      "latency_ms": 72000,
      "cost_usd": 0.041,
      "assurance_score": 0.65,
      "trigger_source": "benchmark",
      "metadata": {
        "domain": "cart_flow",
        "framework": "playwright",
        "language": "typescript",
        "requirements_parsed": 1,
        "scenarios_generated": 2,
        "tests_generated": 2,
        "quality_score": 0.65,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestHealerCICDAgent",
      "task_type": "healing_scan",
      "task_completed": true,
      "efficacy_score": 0.80,
      "latency_ms": 45000,
      "cost_usd": 0.035,
      "assurance_score": 0.75,
      "trigger_source": "benchmark",
      "metadata": {
        "scan_type": "proactive",
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestHealerCICDAgent",
      "task_type": "healing_scan",
      "task_completed": true,
      "efficacy_score": 0.85,
      "latency_ms": 52000,
      "cost_usd": 0.042,
      "assurance_score": 0.82,
      "trigger_source": "benchmark",
      "metadata": {
        "scan_type": "ci",
        "failure_type": "selector_not_found",
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "TestHealerCICDAgent",
      "task_type": "healing_scan",
      "task_completed": true,
      "efficacy_score": 0.82,
      "latency_ms": 48000,
      "cost_usd": 0.038,
      "assurance_score": 0.78,
      "trigger_source": "benchmark",
      "metadata": {
        "scan_type": "ci",
        "failure_type": "assertion_failed",
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "QAEngineerAgent",
      "task_type": "quality_audit",
      "task_completed": true,
      "efficacy_score": 0.75,
      "latency_ms": 95000,
      "cost_usd": 0.065,
      "assurance_score": 0.70,
      "trigger_source": "benchmark",
      "metadata": {
        "analysis_type": "full",
        "quality_score": 33.75,
        "recommendations_count": 8,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    },
    {
      "agent_type": "QAEngineerAgent",
      "task_type": "quality_audit",
      "task_completed": true,
      "efficacy_score": 0.73,
      "latency_ms": 82000,
      "cost_usd": 0.055,
      "assurance_score": 0.68,
      "trigger_source": "benchmark",
      "metadata": {
        "analysis_type": "pr",
        "quality_score": 33.75,
        "recommendations_count": 8,
        "benchmark_suite": "qa_agents_v1",
        "benchmark_date": "2026-02-26"
      }
    }
  ]
}' | python3 -m json.tool

echo ""
echo "Done. Now triggering aggregation (global sentinel org)..."

# Aggregate with sentinel org so benchmarks are globally visible
curl -s -X POST "${API_URL}/api/v1/benchmarks/aggregate?period_days=30" \
  -H "X-API-Key: ${API_KEY}" \
  -H "X-Organization-Id: ${ORG_ID}" | python3 -m json.tool

echo ""
echo "Benchmark results inserted and aggregated."
