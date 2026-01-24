"""
AI-Powered Root Cause Analysis Engine

This is a KEY DIFFERENTIATOR from competitors. Instead of just showing failures,
we use Claude to deeply analyze WHY a test failed and categorize the root cause.

Categories:
- UI_CHANGE: Visual/structural change in the application
- NETWORK_ERROR: API/network-related failure
- TIMING_ISSUE: Race condition or flaky timing
- DATA_MISMATCH: Test data or state issue
- REAL_BUG: Actual application defect
- ENVIRONMENT: Infrastructure/environment issue
- TEST_DEFECT: Bug in the test itself
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from anthropic import Anthropic

from src.agents.prompts import get_enhanced_prompt
from src.config import get_settings


class FailureCategory(str, Enum):
    UI_CHANGE = "ui_change"
    NETWORK_ERROR = "network_error"
    TIMING_ISSUE = "timing_issue"
    DATA_MISMATCH = "data_mismatch"
    REAL_BUG = "real_bug"
    ENVIRONMENT = "environment"
    TEST_DEFECT = "test_defect"
    UNKNOWN = "unknown"


@dataclass
class RootCauseResult:
    """Result of root cause analysis."""
    category: FailureCategory
    confidence: float  # 0.0 to 1.0
    summary: str
    detailed_analysis: str
    suggested_fix: str
    related_failures: list[str] = field(default_factory=list)
    code_location: str | None = None
    is_flaky: bool = False
    flaky_confidence: float = 0.0
    historical_occurrences: int = 0
    auto_healable: bool = False
    healing_suggestion: dict | None = None


@dataclass
class FailureContext:
    """Context for analyzing a test failure."""
    test_id: str
    test_name: str
    error_message: str
    stack_trace: str | None = None
    screenshot_base64: str | None = None
    html_snapshot: str | None = None
    network_logs: list[dict] | None = None
    console_logs: list[str] | None = None
    step_history: list[dict] | None = None
    expected_vs_actual: dict | None = None
    previous_runs: list[dict] | None = None
    code_diff: str | None = None  # Recent code changes
    environment: dict | None = None


class RootCauseAnalyzer:
    """
    AI-powered root cause analysis for test failures.

    This goes beyond simple error messages to understand the TRUE cause
    of failures, reducing debugging time from hours to seconds.
    """

    def __init__(self):
        self.settings = get_settings()
        api_key = self.settings.anthropic_api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        self.client = Anthropic(api_key=api_key)
        self.failure_history: dict[str, list[dict]] = {}  # For pattern detection

    async def analyze(self, context: FailureContext) -> RootCauseResult:
        """
        Perform comprehensive root cause analysis on a test failure.

        This is where Claude shines - understanding complex failure patterns
        that rule-based systems miss.
        """
        # Build rich context for Claude
        messages = self._build_analysis_prompt(context)

        # Use Claude for deep analysis
        response = self.client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=self._get_system_prompt(),
            messages=messages
        )

        # Parse Claude's analysis
        result = self._parse_analysis(response.content[0].text, context)

        # Enhance with historical pattern detection
        result = self._enhance_with_history(result, context)

        # Check if auto-healable
        result = self._check_auto_healability(result, context)

        # Store for future pattern detection
        self._record_failure(context, result)

        return result

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for root cause analysis."""
        enhanced = get_enhanced_prompt("root_cause_analyzer")
        if enhanced:
            return enhanced

        return """You are an expert test failure analyst with deep knowledge of:
- Web application architecture and common failure patterns
- Browser behavior and timing issues
- Network protocols and API failures
- UI testing best practices and anti-patterns
- Database and state management issues

Your job is to analyze test failures and determine the TRUE root cause, not just the surface-level error.

You MUST respond with a JSON object containing:
{
    "category": "ui_change|network_error|timing_issue|data_mismatch|real_bug|environment|test_defect|unknown",
    "confidence": 0.0-1.0,
    "summary": "One-line summary of the root cause",
    "detailed_analysis": "Detailed explanation of what happened and why",
    "suggested_fix": "Specific actionable fix recommendation",
    "code_location": "File and line number if identifiable, or null",
    "is_flaky": true/false,
    "flaky_confidence": 0.0-1.0
}

Key analysis patterns to consider:
1. If element not found after navigation, likely TIMING_ISSUE
2. If element not found but page loaded, likely UI_CHANGE
3. If API returns 500, check if it's REAL_BUG or ENVIRONMENT
4. If assertion fails on dynamic content, likely DATA_MISMATCH
5. If same test passes/fails intermittently, likely TIMING_ISSUE (flaky)
6. If failure only in CI, likely ENVIRONMENT
7. If failure matches recent code change, likely REAL_BUG or UI_CHANGE"""

    def _build_analysis_prompt(self, context: FailureContext) -> list[dict]:
        """Build a rich prompt with all available context."""
        content_parts = []

        # Core failure info
        content_parts.append({
            "type": "text",
            "text": f"""## Test Failure Analysis Request

**Test:** {context.test_name} (ID: {context.test_id})

**Error Message:**
```
{context.error_message}
```
"""
        })

        # Stack trace if available
        if context.stack_trace:
            content_parts.append({
                "type": "text",
                "text": f"""
**Stack Trace:**
```
{context.stack_trace[:2000]}  # Truncate if too long
```
"""
            })

        # Screenshot for visual analysis
        if context.screenshot_base64:
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": context.screenshot_base64
                }
            })
            content_parts.append({
                "type": "text",
                "text": "**Screenshot at failure point (above)**"
            })

        # Network logs
        if context.network_logs:
            failed_requests = [
                log for log in context.network_logs
                if log.get("status", 200) >= 400
            ]
            if failed_requests:
                content_parts.append({
                    "type": "text",
                    "text": f"""
**Failed Network Requests:**
```json
{json.dumps(failed_requests[:10], indent=2)}
```
"""
                })

        # Console errors
        if context.console_logs:
            errors = [log for log in context.console_logs if "error" in log.lower()]
            if errors:
                content_parts.append({
                    "type": "text",
                    "text": f"""
**Console Errors:**
```
{chr(10).join(errors[:20])}
```
"""
                })

        # Step history
        if context.step_history:
            content_parts.append({
                "type": "text",
                "text": f"""
**Test Steps Before Failure:**
```json
{json.dumps(context.step_history[-5:], indent=2)}
```
"""
            })

        # Previous runs for flaky detection
        if context.previous_runs:
            pass_count = sum(1 for r in context.previous_runs if r.get("passed"))
            fail_count = len(context.previous_runs) - pass_count
            content_parts.append({
                "type": "text",
                "text": f"""
**Historical Pattern:** {pass_count} passed, {fail_count} failed in last {len(context.previous_runs)} runs
"""
            })

        # Recent code changes
        if context.code_diff:
            content_parts.append({
                "type": "text",
                "text": f"""
**Recent Code Changes:**
```diff
{context.code_diff[:1500]}
```
"""
            })

        content_parts.append({
            "type": "text",
            "text": "\nAnalyze this failure and provide your assessment in JSON format."
        })

        return [{"role": "user", "content": content_parts}]

    def _parse_analysis(self, response_text: str, context: FailureContext) -> RootCauseResult:
        """Parse Claude's analysis response."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            text = response_text

            # Remove markdown code block wrappers if present
            if "```json" in text:
                text = text.split("```json", 1)[1]
                if "```" in text:
                    text = text.split("```", 1)[0]
            elif "```" in text:
                # Generic code block
                parts = text.split("```")
                if len(parts) >= 3:
                    text = parts[1]  # Content between first pair of ```

            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                analysis = json.loads(text[json_start:json_end])
            else:
                raise ValueError("No JSON found in response")

            # Handle both flat format and nested root_cause format
            root_cause = analysis.get("root_cause", {})
            resolution = analysis.get("resolution", {})

            # Try nested format first, then flat format
            category_str = (
                root_cause.get("category") or
                analysis.get("category") or
                "unknown"
            )
            # Normalize category names (e.g., "config_error" -> "environment")
            category_map = {
                "config_error": "environment",
                "infrastructure": "environment",
                "configuration_error": "environment",
                "assertion_failure": "data_mismatch",
                "element_not_found": "ui_change",
                "selector_issue": "ui_change",
            }
            category_str = category_map.get(category_str, category_str)

            # Validate category is in enum
            try:
                category = FailureCategory(category_str)
            except ValueError:
                category = FailureCategory.UNKNOWN

            confidence = float(
                root_cause.get("confidence") or
                analysis.get("confidence") or
                0.5
            )
            summary = (
                root_cause.get("description") or
                analysis.get("summary") or
                "Unable to determine root cause"
            )
            detailed_analysis = (
                analysis.get("detailed_analysis") or
                json.dumps(analysis.get("causal_chain", []), indent=2) or
                ""
            )
            suggested_fix = (
                resolution.get("immediate_fix") or
                analysis.get("suggested_fix") or
                ""
            )

            return RootCauseResult(
                category=category,
                confidence=confidence,
                summary=summary[:500] if summary else "Unable to determine root cause",  # Truncate long summaries
                detailed_analysis=detailed_analysis,
                suggested_fix=suggested_fix,
                code_location=analysis.get("code_location"),
                is_flaky=analysis.get("is_flaky", False),
                flaky_confidence=float(analysis.get("flaky_confidence", 0.0))
            )
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback if parsing fails
            return RootCauseResult(
                category=FailureCategory.UNKNOWN,
                confidence=0.3,
                summary=f"Analysis parsing failed: {str(e)[:100]}",
                detailed_analysis=response_text,
                suggested_fix="Manual investigation required"
            )

    def _enhance_with_history(
        self,
        result: RootCauseResult,
        context: FailureContext
    ) -> RootCauseResult:
        """Enhance analysis with historical pattern detection."""
        # Create a fingerprint for this failure type
        fingerprint = self._create_failure_fingerprint(context)

        if fingerprint in self.failure_history:
            history = self.failure_history[fingerprint]
            result.historical_occurrences = len(history)

            # If same failure happened multiple times, increase flaky confidence
            if len(history) >= 3:
                pass_rate = sum(1 for h in history if h.get("recovered")) / len(history)
                if pass_rate > 0.3:  # Sometimes passes
                    result.is_flaky = True
                    result.flaky_confidence = max(result.flaky_confidence, pass_rate)

            # Find related failures
            result.related_failures = [h["test_id"] for h in history[-5:]]

        return result

    def _check_auto_healability(
        self,
        result: RootCauseResult,
        context: FailureContext
    ) -> RootCauseResult:
        """Check if this failure can be automatically healed."""
        healable_categories = {
            FailureCategory.UI_CHANGE,
            FailureCategory.TIMING_ISSUE,
        }

        if result.category in healable_categories and result.confidence > 0.7:
            result.auto_healable = True

            if result.category == FailureCategory.UI_CHANGE:
                result.healing_suggestion = {
                    "type": "update_locator",
                    "description": "UI element changed, locator needs update",
                    "action": "self_heal_locator"
                }
            elif result.category == FailureCategory.TIMING_ISSUE:
                result.healing_suggestion = {
                    "type": "add_wait",
                    "description": "Add explicit wait for element stability",
                    "action": "add_smart_wait"
                }

        return result

    def _create_failure_fingerprint(self, context: FailureContext) -> str:
        """Create a unique fingerprint for this failure pattern."""
        key_parts = [
            context.test_name,
            context.error_message[:100] if context.error_message else "",
        ]
        return hashlib.md5("|".join(key_parts).encode(), usedforsecurity=False).hexdigest()

    def _record_failure(self, context: FailureContext, result: RootCauseResult):
        """Record failure for future pattern detection."""
        fingerprint = self._create_failure_fingerprint(context)

        if fingerprint not in self.failure_history:
            self.failure_history[fingerprint] = []

        self.failure_history[fingerprint].append({
            "test_id": context.test_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "category": result.category.value,
            "recovered": False  # Updated if test passes on retry
        })

        # Keep only last 50 occurrences
        self.failure_history[fingerprint] = self.failure_history[fingerprint][-50:]

    async def analyze_batch(
        self,
        failures: list[FailureContext]
    ) -> tuple[list[RootCauseResult], dict]:
        """
        Analyze multiple failures together to find patterns.

        Returns individual results plus aggregate insights.
        """
        results = []
        for failure in failures:
            result = await self.analyze(failure)
            results.append(result)

        # Aggregate insights
        category_counts = {}
        for r in results:
            category_counts[r.category.value] = category_counts.get(r.category.value, 0) + 1

        flaky_tests = [r for r in results if r.is_flaky]
        auto_healable = [r for r in results if r.auto_healable]

        insights = {
            "total_failures": len(failures),
            "category_distribution": category_counts,
            "flaky_test_count": len(flaky_tests),
            "auto_healable_count": len(auto_healable),
            "top_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "recommendation": self._generate_batch_recommendation(results)
        }

        return results, insights

    def _generate_batch_recommendation(self, results: list[RootCauseResult]) -> str:
        """Generate overall recommendation based on batch analysis."""
        categories = [r.category for r in results]

        if categories.count(FailureCategory.TIMING_ISSUE) > len(results) * 0.5:
            return "High flakiness detected. Consider adding smart waits or reviewing async operations."
        elif categories.count(FailureCategory.UI_CHANGE) > len(results) * 0.3:
            return "Multiple UI changes detected. Review recent frontend changes and update test locators."
        elif categories.count(FailureCategory.NETWORK_ERROR) > len(results) * 0.3:
            return "API instability detected. Check backend services and consider adding retry logic."
        elif categories.count(FailureCategory.REAL_BUG) > 0:
            return f"Found {categories.count(FailureCategory.REAL_BUG)} likely real bugs. Prioritize investigation."
        else:
            return "Mixed failure causes. Review individual analyses for specific fixes."
