"""
Test Impact Analysis

CRITICAL FOR CI/CD SPEED: Don't run ALL tests on every change.
Run only the tests that are AFFECTED by the code changes.

This provides:
1. 10-100x faster CI/CD pipelines
2. Intelligent test selection based on code changes
3. Risk-based test prioritization
4. Coverage gap detection
5. Change-aware test generation

This is what Facebook/Google/Microsoft do internally.
Now we bring it to everyone.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime

from anthropic import Anthropic

from src.config import get_settings


@dataclass
class CodeChange:
    """A code change (commit, PR, etc)."""
    id: str  # commit hash or PR number
    files: list[dict]  # [{path, additions, deletions, patch}]
    message: str
    author: str
    timestamp: datetime
    branch: str


@dataclass
class TestMapping:
    """Mapping between code and tests."""
    code_path: str
    test_ids: list[str]
    confidence: float
    last_updated: datetime


@dataclass
class ImpactAnalysis:
    """Result of test impact analysis."""
    change_id: str
    affected_tests: list[str]
    unaffected_tests: list[str]
    new_tests_suggested: list[dict]
    risk_score: float  # 0-1, higher = riskier change
    estimated_time_saved: float  # seconds
    coverage_gaps: list[str]
    recommendations: list[str]


@dataclass
class ChangeRisk:
    """Risk assessment of a code change."""
    file_path: str
    risk_level: str  # "critical", "high", "medium", "low"
    reasons: list[str]
    suggested_tests: list[str]


class TestImpactAnalyzer:
    """
    Analyzes code changes to determine test impact.

    Key capabilities:
    1. Static analysis of code dependencies
    2. Historical correlation of changes to test failures
    3. AI-powered impact prediction
    4. Coverage-aware test selection
    """

    def __init__(self):
        self.settings = get_settings()
        api_key = self.settings.anthropic_api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        self.client = Anthropic(api_key=api_key)
        self.mappings: dict[str, TestMapping] = {}
        self.failure_history: dict[str, list[dict]] = {}  # code_path -> [{change_id, test_id, failed}]

    async def analyze_impact(
        self,
        change: CodeChange,
        all_tests: list[dict],
        coverage_data: dict | None = None
    ) -> ImpactAnalysis:
        """
        Analyze the impact of a code change on tests.

        This is the core function that makes CI/CD 10-100x faster.
        """
        affected_tests = set()
        unaffected_tests = set()

        # Phase 1: Direct mapping (fastest)
        for file_change in change.files:
            file_path = file_change["path"]

            # Check existing mappings
            if file_path in self.mappings:
                mapping = self.mappings[file_path]
                affected_tests.update(mapping.test_ids)

            # Check coverage data
            if coverage_data:
                covered_tests = coverage_data.get(file_path, [])
                affected_tests.update(covered_tests)

        # Phase 2: Dependency analysis
        dependencies = await self._analyze_dependencies(change)
        for dep_path in dependencies:
            if dep_path in self.mappings:
                affected_tests.update(self.mappings[dep_path].test_ids)

        # Phase 3: Historical correlation
        historical_tests = self._get_historically_affected_tests(change)
        affected_tests.update(historical_tests)

        # Phase 4: AI-powered prediction for uncovered code
        uncovered_files = [
            f for f in change.files
            if f["path"] not in self.mappings and f["path"] not in (coverage_data or {})
        ]
        if uncovered_files:
            ai_predicted = await self._ai_predict_affected_tests(
                uncovered_files, all_tests
            )
            affected_tests.update(ai_predicted)

        # Determine unaffected tests
        all_test_ids = {t["id"] for t in all_tests}
        unaffected_tests = all_test_ids - affected_tests

        # Calculate risk score
        risk_score = await self._calculate_risk_score(change)

        # Identify coverage gaps
        coverage_gaps = self._identify_coverage_gaps(change, affected_tests, all_tests)

        # Suggest new tests
        new_tests = await self._suggest_new_tests(change, coverage_gaps)

        # Estimate time saved
        affected_time = sum(t.get("avg_duration", 30) for t in all_tests if t["id"] in affected_tests)
        total_time = sum(t.get("avg_duration", 30) for t in all_tests)
        time_saved = total_time - affected_time

        # Generate recommendations
        recommendations = self._generate_recommendations(
            change, risk_score, coverage_gaps, affected_tests
        )

        return ImpactAnalysis(
            change_id=change.id,
            affected_tests=list(affected_tests),
            unaffected_tests=list(unaffected_tests),
            new_tests_suggested=new_tests,
            risk_score=risk_score,
            estimated_time_saved=time_saved,
            coverage_gaps=coverage_gaps,
            recommendations=recommendations
        )

    async def _analyze_dependencies(self, change: CodeChange) -> list[str]:
        """Analyze code dependencies to find indirectly affected files."""
        dependencies = []

        for file_change in change.files:
            file_change["path"]

            # Extract imports (simplified - would use AST in production)
            patch = file_change.get("patch", "")

            # Python imports
            python_imports = re.findall(r'^[+-]\s*(?:from|import)\s+([\w.]+)', patch, re.MULTILINE)
            for imp in python_imports:
                module_path = imp.replace(".", "/") + ".py"
                dependencies.append(module_path)

            # JavaScript imports
            js_imports = re.findall(r'^[+-]\s*import\s+.*from\s+[\'"](.+?)[\'"]', patch, re.MULTILINE)
            for imp in js_imports:
                if not imp.startswith('.'):
                    continue  # Skip node_modules
                dependencies.append(imp.lstrip('./'))

        return dependencies

    def _get_historically_affected_tests(self, change: CodeChange) -> set[str]:
        """Get tests that historically fail when these files change."""
        affected = set()

        for file_change in change.files:
            file_path = file_change["path"]
            history = self.failure_history.get(file_path, [])

            # Find tests that failed more than 30% of the time when this file changed
            test_failures = {}
            test_runs = {}

            for entry in history:
                test_id = entry["test_id"]
                test_runs[test_id] = test_runs.get(test_id, 0) + 1
                if entry["failed"]:
                    test_failures[test_id] = test_failures.get(test_id, 0) + 1

            for test_id, failures in test_failures.items():
                runs = test_runs[test_id]
                if failures / runs > 0.3:  # 30% failure rate
                    affected.add(test_id)

        return affected

    async def _ai_predict_affected_tests(
        self,
        uncovered_files: list[dict],
        all_tests: list[dict]
    ) -> set[str]:
        """Use AI to predict which tests might be affected by uncovered changes."""
        affected = set()

        # Group tests by category for easier matching
        test_summaries = [
            {"id": t["id"], "name": t["name"], "category": t.get("category", "unknown")}
            for t in all_tests[:100]  # Limit for context
        ]

        for file_change in uncovered_files:
            prompt = f"""Predict which tests might be affected by this code change.

FILE CHANGED: {file_change['path']}

PATCH:
```
{file_change.get('patch', '')[:2000]}
```

AVAILABLE TESTS:
{json.dumps(test_summaries, indent=2)}

Based on the code change, which tests are likely to be affected?
Consider:
1. What functionality is being changed?
2. Which tests cover that functionality?
3. What are the side effects of this change?

Return JSON array of test IDs: ["test-1", "test-2", ...]"""

            from src.core.model_registry import get_model_id
            response = self.client.messages.create(
                model=get_model_id("claude-sonnet-4-5"),
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            try:
                text = response.content[0].text
                json_start = text.find("[")
                json_end = text.rfind("]") + 1
                if json_start >= 0:
                    predicted = json.loads(text[json_start:json_end])
                    affected.update(predicted)
            except:
                pass

        return affected

    async def _calculate_risk_score(self, change: CodeChange) -> float:
        """Calculate risk score for a change."""
        risk_factors = []

        for file_change in change.files:
            file_path = file_change["path"]
            additions = file_change.get("additions", 0)
            deletions = file_change.get("deletions", 0)

            # Large changes are riskier
            if additions + deletions > 500:
                risk_factors.append(0.3)
            elif additions + deletions > 200:
                risk_factors.append(0.2)
            elif additions + deletions > 50:
                risk_factors.append(0.1)

            # Certain file patterns are riskier
            risky_patterns = [
                (r"auth", 0.3),
                (r"payment", 0.4),
                (r"security", 0.3),
                (r"database|db|migration", 0.3),
                (r"config", 0.2),
                (r"api", 0.2),
            ]
            for pattern, risk in risky_patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    risk_factors.append(risk)

            # Deletions are riskier than additions
            if deletions > additions:
                risk_factors.append(0.1)

        if not risk_factors:
            return 0.1

        # Combine risk factors (not just sum, as that could exceed 1)
        combined_risk = 1 - (1 - sum(risk_factors) / len(risk_factors))
        return min(combined_risk, 1.0)

    def _identify_coverage_gaps(
        self,
        change: CodeChange,
        affected_tests: set[str],
        all_tests: list[dict]
    ) -> list[str]:
        """Identify parts of the change not covered by tests."""
        gaps = []

        for file_change in change.files:
            file_path = file_change["path"]

            # Check if any test covers this file
            if file_path not in self.mappings:
                gaps.append(f"No tests map to: {file_path}")
                continue

            # Check coverage depth
            mapping = self.mappings[file_path]
            if len(mapping.test_ids) < 2:
                gaps.append(f"Low test coverage for: {file_path} (only {len(mapping.test_ids)} tests)")

        return gaps

    async def _suggest_new_tests(
        self,
        change: CodeChange,
        coverage_gaps: list[str]
    ) -> list[dict]:
        """Suggest new tests to fill coverage gaps."""
        if not coverage_gaps:
            return []

        suggestions = []

        prompt = f"""Suggest new tests for these coverage gaps.

CODE CHANGES:
{json.dumps([{"path": f["path"], "patch": f.get("patch", "")[:500]} for f in change.files], indent=2)}

COVERAGE GAPS:
{json.dumps(coverage_gaps, indent=2)}

Suggest specific tests that would improve coverage:
[
    {{
        "name": "Test name",
        "description": "What it tests",
        "priority": "critical|high|medium|low",
        "type": "unit|integration|e2e",
        "coverage_target": "what gap it fills"
    }}
]"""

        from src.core.model_registry import get_model_id
        response = self.client.messages.create(
            model=get_model_id("claude-sonnet-4-5"),
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            text = response.content[0].text
            json_start = text.find("[")
            json_end = text.rfind("]") + 1
            if json_start >= 0:
                suggestions = json.loads(text[json_start:json_end])
        except:
            pass

        return suggestions

    def _generate_recommendations(
        self,
        change: CodeChange,
        risk_score: float,
        coverage_gaps: list[str],
        affected_tests: set[str]
    ) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if risk_score > 0.7:
            recommendations.append(
                "HIGH RISK CHANGE: Consider additional review and testing"
            )

        if risk_score > 0.5 and len(affected_tests) < 5:
            recommendations.append(
                "Risky change with low test coverage - add more tests"
            )

        if coverage_gaps:
            recommendations.append(
                f"Found {len(coverage_gaps)} coverage gaps - consider adding tests"
            )

        if len(affected_tests) == 0:
            recommendations.append(
                "No tests affected - this could mean lack of coverage"
            )

        if len(affected_tests) > 50:
            recommendations.append(
                "Large blast radius - consider breaking into smaller changes"
            )

        return recommendations

    def update_mapping(
        self,
        code_path: str,
        test_id: str,
        confidence: float = 1.0
    ):
        """Update the code-to-test mapping."""
        if code_path not in self.mappings:
            self.mappings[code_path] = TestMapping(
                code_path=code_path,
                test_ids=[],
                confidence=confidence,
                last_updated=datetime.utcnow()
            )

        if test_id not in self.mappings[code_path].test_ids:
            self.mappings[code_path].test_ids.append(test_id)
            self.mappings[code_path].last_updated = datetime.utcnow()

    def record_test_result(
        self,
        change_id: str,
        code_path: str,
        test_id: str,
        failed: bool
    ):
        """Record a test result for historical analysis."""
        if code_path not in self.failure_history:
            self.failure_history[code_path] = []

        self.failure_history[code_path].append({
            "change_id": change_id,
            "test_id": test_id,
            "failed": failed,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Keep last 100 entries per file
        self.failure_history[code_path] = self.failure_history[code_path][-100:]

        # Also update mapping
        self.update_mapping(code_path, test_id)


class SmartTestSelector:
    """
    Selects which tests to run based on multiple factors.

    Goes beyond simple impact analysis to consider:
    - Historical failure patterns
    - Test execution time
    - Risk of the change
    - Available CI resources
    - Time budget
    """

    def __init__(self, impact_analyzer: TestImpactAnalyzer):
        self.impact_analyzer = impact_analyzer

    async def select_tests(
        self,
        change: CodeChange,
        all_tests: list[dict],
        time_budget_seconds: int | None = None,
        risk_tolerance: str = "medium"  # "low", "medium", "high"
    ) -> dict:
        """
        Select optimal tests to run given constraints.

        Returns:
        {
            "must_run": [...],      # Critical tests that must run
            "should_run": [...],    # Recommended tests
            "can_skip": [...],      # Safe to skip
            "deferred": [...],      # Run later if time permits
            "estimated_time": float,
            "coverage_estimate": float
        }
        """
        # Get impact analysis
        analysis = await self.impact_analyzer.analyze_impact(
            change, all_tests
        )

        # Categorize tests
        must_run = []
        should_run = []
        can_skip = []
        deferred = []

        # Critical tests always run
        for test_id in analysis.affected_tests:
            test = next((t for t in all_tests if t["id"] == test_id), None)
            if not test:
                continue

            priority = test.get("priority", "medium")
            is_critical = priority == "critical" or test.get("category") in ["smoke", "sanity"]

            if is_critical:
                must_run.append(test_id)
            elif priority == "high":
                should_run.append(test_id)
            else:
                if risk_tolerance == "low":
                    should_run.append(test_id)
                else:
                    deferred.append(test_id)

        # Unaffected tests can be skipped
        can_skip = list(analysis.unaffected_tests)

        # Apply time budget if specified
        if time_budget_seconds:
            must_run, should_run, deferred = self._apply_time_budget(
                must_run, should_run, deferred,
                all_tests, time_budget_seconds
            )

        # Calculate estimates
        estimated_time = sum(
            next((t.get("avg_duration", 30) for t in all_tests if t["id"] == tid), 30)
            for tid in must_run + should_run
        )

        coverage_estimate = len(must_run + should_run) / len(all_tests) if all_tests else 0

        return {
            "must_run": must_run,
            "should_run": should_run,
            "can_skip": can_skip,
            "deferred": deferred,
            "estimated_time": estimated_time,
            "coverage_estimate": coverage_estimate,
            "risk_score": analysis.risk_score,
            "recommendations": analysis.recommendations
        }

    def _apply_time_budget(
        self,
        must_run: list[str],
        should_run: list[str],
        deferred: list[str],
        all_tests: list[dict],
        time_budget: int
    ) -> tuple[list[str], list[str], list[str]]:
        """Apply time budget constraints to test selection."""
        test_times = {
            t["id"]: t.get("avg_duration", 30)
            for t in all_tests
        }

        # Must run tests are non-negotiable
        must_run_time = sum(test_times.get(tid, 30) for tid in must_run)

        if must_run_time > time_budget:
            # Can't fit even critical tests - run them anyway but warn
            return must_run, [], should_run + deferred

        remaining_time = time_budget - must_run_time

        # Add should_run tests that fit
        final_should_run = []
        for tid in should_run:
            test_time = test_times.get(tid, 30)
            if remaining_time >= test_time:
                final_should_run.append(tid)
                remaining_time -= test_time
            else:
                deferred.append(tid)

        return must_run, final_should_run, deferred
