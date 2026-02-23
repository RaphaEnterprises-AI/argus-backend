"""Tests for benchmark scoring functions — validates dataset loading and scoring logic.

These tests do NOT call real agents or LLMs. They verify that:
1. All datasets load correctly
2. Scoring functions handle ground truth properly
3. Edge cases (None, empty data) produce sensible results
"""

import json
from unittest.mock import MagicMock

import pytest

from src.services.benchmark_runner import (
    SUITES,
    ScenarioResult,
    SuiteResult,
    _load_dataset,
    score_code_analysis,
    score_hallucination_detection,
    score_self_healing,
    score_sre_analysis,
)


# =============================================================================
# Dataset Loading Tests
# =============================================================================


class TestDatasetLoading:
    def test_all_datasets_load(self):
        """All JSON datasets should load without errors."""
        datasets = {
            "self_healing.json": 8,
            "code_analysis.json": 4,
            "hallucination_detection.json": 8,
            "sre_incidents.json": 6,
        }
        for filename, expected_count in datasets.items():
            data = _load_dataset(filename)
            assert isinstance(data, list), f"{filename} should be a list"
            assert len(data) == expected_count, f"{filename} should have {expected_count} scenarios"

    def test_self_healing_dataset_structure(self):
        """Self-healing scenarios should have required fields."""
        data = _load_dataset("self_healing.json")
        for scenario in data:
            assert "id" in scenario
            assert "failure_data" in scenario
            assert "ground_truth" in scenario
            fd = scenario["failure_data"]
            assert "error_type" in fd
            assert "failed_selector" in fd
            gt = scenario["ground_truth"]
            assert "must_contain_in_suggestion" in gt

    def test_code_analysis_dataset_structure(self):
        """Code analysis scenarios should have required fields."""
        data = _load_dataset("code_analysis.json")
        for scenario in data:
            assert "id" in scenario
            assert "file_contents" in scenario
            assert "ground_truth" in scenario
            gt = scenario["ground_truth"]
            assert "min_surfaces" in gt
            assert "must_find_elements" in gt

    def test_hallucination_dataset_structure(self):
        """Hallucination scenarios should have required fields."""
        data = _load_dataset("hallucination_detection.json")
        for scenario in data:
            assert "id" in scenario
            assert "query" in scenario
            assert "context" in scenario
            assert "response" in scenario
            gt = scenario["ground_truth"]
            assert "is_hallucination" in gt

    def test_sre_dataset_structure(self):
        """SRE incident scenarios should have required fields."""
        data = _load_dataset("sre_incidents.json")
        for scenario in data:
            assert "id" in scenario
            assert "alert_data" in scenario
            assert "ground_truth" in scenario
            gt = scenario["ground_truth"]
            assert "severity" in gt
            assert "root_cause_must_mention" in gt

    def test_missing_dataset_raises(self):
        """Loading a non-existent dataset should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_dataset("nonexistent.json")


# =============================================================================
# Scoring Function Tests
# =============================================================================


class TestScoringFunctions:
    def test_self_healing_none_result(self):
        ok, score, details = score_self_healing(None, {"must_contain_in_suggestion": ["button"]})
        assert ok is False
        assert score == 0.0

    def test_self_healing_perfect_match(self):
        """When agent output contains all expected terms, score should be 1.0."""
        result = MagicMock()
        result.data = {
            "healed_selector": "#submitButton",
            "strategy_used": "code_aware",
            "suggestions": [{"selector": "#submitButton"}],
        }
        gt = {"must_contain_in_suggestion": ["submitButton", "#submitButton"]}
        ok, score, details = score_self_healing(result, gt)
        assert ok is True
        assert score == 1.0

    def test_self_healing_partial_match(self):
        """When only some terms match, score should be proportional."""
        result = MagicMock()
        result.data = {
            "healed_selector": ".nav-item-link",
            "strategy_used": "dom_analysis",
            "suggestions": [],
        }
        gt = {"must_contain_in_suggestion": ["nav-item-link", "data-testid"]}
        ok, score, details = score_self_healing(result, gt)
        assert score == 0.5  # 1 of 2 terms found
        assert ok is True  # 0.5 >= 0.5 threshold

    def test_code_analysis_none_result(self):
        ok, score, details = score_code_analysis(None, {"must_find_elements": ["form"]})
        assert ok is False
        assert score == 0.0

    def test_code_analysis_surfaces_found(self):
        result = MagicMock()
        result.data = {
            "testable_surfaces": [
                {"name": "checkout form"},
                {"name": "email input"},
                {"name": "/orders endpoint"},
            ],
        }
        gt = {
            "min_surfaces": 3,
            "must_find_elements": ["checkout form", "email input"],
            "must_find_api_endpoints": ["/orders"],
        }
        ok, score, details = score_code_analysis(result, gt)
        assert ok is True
        assert score > 0.5

    def test_hallucination_correct_detection(self):
        """When hallucination is correctly detected, score should be high.

        The HallucinationDetectorAgent returns severity/hallucination_rate/is_reliable,
        NOT is_hallucination/confidence.
        """
        result = MagicMock()
        result.data = {"severity": "critical", "hallucination_rate": 0.9, "is_reliable": False}
        gt = {"is_hallucination": True}
        ok, score, details = score_hallucination_detection(result, gt)
        assert ok is True
        assert score >= 0.5

    def test_hallucination_correct_non_detection(self):
        """When non-hallucination is correctly identified, score should be high."""
        result = MagicMock()
        result.data = {"severity": "none", "hallucination_rate": 0.1, "is_reliable": True}
        gt = {"is_hallucination": False}
        ok, score, details = score_hallucination_detection(result, gt)
        assert ok is True
        assert score >= 0.5

    def test_hallucination_wrong_detection(self):
        """False positive (flagging correct content as hallucination) should score 0."""
        result = MagicMock()
        result.data = {"severity": "critical", "hallucination_rate": 0.9, "is_reliable": False}
        gt = {"is_hallucination": False}
        ok, score, details = score_hallucination_detection(result, gt)
        assert ok is False
        assert score == 0.0

    def test_sre_analysis_none_result(self):
        ok, score, details = score_sre_analysis(None, {"root_cause_must_mention": ["disk"]})
        assert ok is False
        assert score == 0.0

    def test_sre_analysis_correct_diagnosis(self):
        """SREAgent returns CorrelatedIncident dataclass in 'incident' key.

        The scoring function extracts text from analysis/suggested_actions/incident fields.
        """
        # Simulate the CorrelatedIncident dataclass
        incident = MagicMock()
        incident.severity = "sev2"
        incident.root_cause_hypothesis = "connection pool exhausted due to max_connections limit"
        incident.title = "Database connection pool exhaustion"
        incident.affected_services = ["api-gateway", "database"]
        incident.suggested_actions = [
            {"action": "increase pool size", "priority": "high"},
            {"action": "check for connection leaks", "priority": "medium"},
        ]

        result = MagicMock()
        result.data = {
            "incident": incident,
            "analysis": {
                "root_cause": "connection pool exhausted due to max_connections limit",
                "confidence": 0.85,
                "affected_services": ["api-gateway", "database"],
            },
            "suggested_actions": [
                {"action": "increase pool size", "priority": "high"},
                {"action": "check for connection leaks", "priority": "medium"},
            ],
        }
        gt = {
            "severity": "sev2",
            "root_cause_must_mention": ["connection pool", "max_connections"],
            "suggested_actions_must_include": ["increase pool size", "connection leak"],
        }
        ok, score, details = score_sre_analysis(result, gt)
        assert ok is True
        assert score > 0.5
        assert details["severity_correct"] is True


# =============================================================================
# Suite Result Tests
# =============================================================================


class TestSuiteResult:
    def test_pass_sequence(self):
        suite = SuiteResult(suite_name="test", agent_type="Test", total_scenarios=1, total_runs=3)
        suite.results = [
            ScenarioResult("s1_run0", "Test", True, 0.8, 100, 0.01),
            ScenarioResult("s1_run1", "Test", False, 0.3, 150, 0.01),
            ScenarioResult("s1_run2", "Test", True, 0.9, 120, 0.01),
        ]
        assert suite.pass_sequence == [True, False, True]
        assert suite.avg_score == pytest.approx(0.6667, abs=0.001)
        assert suite.avg_latency_ms == pytest.approx(123.33, abs=0.1)
        assert suite.total_cost == pytest.approx(0.03, abs=0.001)

    def test_suite_registry(self):
        """All 4 suites should be registered."""
        assert set(SUITES.keys()) == {
            "self_healing",
            "code_analysis",
            "hallucination_detection",
            "sre_incidents",
        }
