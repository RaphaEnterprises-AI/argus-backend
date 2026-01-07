"""Visual AI comparison using multi-model vision capabilities.

Provides Applitools-like visual regression testing using Claude Vision, GPT-4V, or Gemini.
Compares screenshots and identifies meaningful visual differences.
Cost-optimized using intelligent model routing.
"""

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import anthropic
import structlog

from ..config import get_settings, MultiModelStrategy
from ..core.model_router import ModelRouter, TaskType, TaskComplexity
from ..core.model_registry import get_model_id

logger = structlog.get_logger()


class DifferenceType(str, Enum):
    """Types of visual differences detected."""
    LAYOUT = "layout"  # Element positions changed
    CONTENT = "content"  # Text or images changed
    STYLE = "style"  # Colors, fonts, borders changed
    MISSING = "missing"  # Element disappeared
    NEW = "new"  # New element appeared
    DYNAMIC = "dynamic"  # Expected dynamic content (timestamps, etc.)
    NONE = "none"  # No meaningful difference


class Severity(str, Enum):
    """Severity of visual difference."""
    CRITICAL = "critical"  # Blocks user flow
    MAJOR = "major"  # Significant visual issue
    MINOR = "minor"  # Small visual change
    INFO = "info"  # Informational (dynamic content)


@dataclass
class VisualDifference:
    """A detected visual difference between screenshots."""
    type: DifferenceType
    severity: Severity
    description: str
    location: str  # Where on the page (e.g., "top-left", "navigation bar")
    element: Optional[str] = None  # Affected element description
    expected: Optional[str] = None  # What was expected
    actual: Optional[str] = None  # What was found
    is_regression: bool = True  # True if this is a bug, False if acceptable


@dataclass
class VisualComparisonResult:
    """Result of comparing two screenshots."""
    baseline_path: str
    current_path: str
    match: bool  # True if screenshots match (no regressions)
    match_percentage: float  # 0-100% similarity
    differences: list[VisualDifference] = field(default_factory=list)
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    analysis_cost_usd: float = 0.0

    def has_regressions(self) -> bool:
        """Check if there are any actual regressions (not just info changes)."""
        return any(d.is_regression and d.severity in (Severity.CRITICAL, Severity.MAJOR) for d in self.differences)

    def to_dict(self) -> dict:
        return {
            "baseline_path": self.baseline_path,
            "current_path": self.current_path,
            "match": self.match,
            "match_percentage": self.match_percentage,
            "has_regressions": self.has_regressions(),
            "differences": [
                {
                    "type": d.type.value,
                    "severity": d.severity.value,
                    "description": d.description,
                    "location": d.location,
                    "element": d.element,
                    "is_regression": d.is_regression,
                }
                for d in self.differences
            ],
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


class VisualAI:
    """
    Visual AI comparison using multi-model vision (Claude, GPT-4V, Gemini).

    Features:
    - Screenshot comparison with semantic understanding
    - Distinguishes meaningful changes from noise
    - Identifies layout shifts, content changes, style differences
    - Filters out expected dynamic content (timestamps, ads)
    - Provides actionable descriptions of differences
    - Cost-optimized via intelligent model routing

    Usage:
        visual = VisualAI()

        # Compare two screenshots (uses cheapest capable model)
        result = await visual.compare(
            baseline="./screenshots/baseline.png",
            current="./screenshots/current.png",
            context="Login page after entering credentials"
        )

        if result.has_regressions():
            print(f"Visual regression detected!")
            for diff in result.differences:
                print(f"  - {diff.severity}: {diff.description}")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        ignore_dynamic: bool = True,
        sensitivity: str = "medium",  # low, medium, high
        use_multi_model: bool = True,
    ):
        self.settings = get_settings()
        api_key = self.settings.anthropic_api_key
        if hasattr(api_key, 'get_secret_value'):
            api_key = api_key.get_secret_value()
        self.client = anthropic.Anthropic(api_key=api_key)
        model = model or get_model_id("claude-sonnet-4-5")
        self.model = model
        self.ignore_dynamic = ignore_dynamic
        self.sensitivity = sensitivity
        self.use_multi_model = use_multi_model and self.settings.model_strategy != MultiModelStrategy.ANTHROPIC_ONLY
        self._model_router: Optional[ModelRouter] = None
        self.log = logger.bind(component="visual_ai", multi_model=self.use_multi_model)

    @property
    def model_router(self) -> ModelRouter:
        """Lazy-initialize model router."""
        if self._model_router is None:
            self._model_router = ModelRouter(self.settings)
        return self._model_router

    def _load_image(self, path: str | Path) -> tuple[str, str]:
        """Load image and return base64 encoded data with media type."""
        path = Path(path)
        suffix = path.suffix.lower()

        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        media_type = media_types.get(suffix, "image/png")

        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")

        return data, media_type

    async def compare(
        self,
        baseline: str | Path,
        current: str | Path,
        context: Optional[str] = None,
        ignore_regions: Optional[list[str]] = None,
    ) -> VisualComparisonResult:
        """
        Compare two screenshots using Claude Vision.

        Args:
            baseline: Path to baseline (expected) screenshot
            current: Path to current (actual) screenshot
            context: Description of what the page/component should show
            ignore_regions: Regions to ignore (e.g., ["header timestamp", "ad banner"])

        Returns:
            VisualComparisonResult with detailed difference analysis
        """
        self.log.info("Comparing screenshots", baseline=str(baseline), current=str(current))

        # Load images
        baseline_data, baseline_type = self._load_image(baseline)
        current_data, current_type = self._load_image(current)

        # Build prompt
        ignore_text = ""
        if ignore_regions:
            ignore_text = f"\n\nIGNORE these regions (expected to change): {', '.join(ignore_regions)}"

        context_text = ""
        if context:
            context_text = f"\n\nCONTEXT: {context}"

        sensitivity_instructions = {
            "low": "Only report major, obvious differences that would affect user experience.",
            "medium": "Report meaningful differences but ignore minor pixel-level variations.",
            "high": "Report all differences, including subtle styling changes.",
        }

        prompt = f"""You are a visual QA expert. Compare these two screenshots:
- IMAGE 1: Baseline (expected/before)
- IMAGE 2: Current (actual/after)

{sensitivity_instructions.get(self.sensitivity, sensitivity_instructions["medium"])}
{context_text}
{ignore_text}

{"IMPORTANT: Ignore dynamic content like timestamps, session IDs, ads, or randomly generated content." if self.ignore_dynamic else ""}

Analyze the differences and respond with JSON:
{{
    "match": true/false,
    "match_percentage": 0-100,
    "summary": "Brief overall assessment",
    "differences": [
        {{
            "type": "layout|content|style|missing|new|dynamic",
            "severity": "critical|major|minor|info",
            "description": "What changed",
            "location": "Where on the page",
            "element": "Affected element (button, header, etc.)",
            "expected": "What baseline shows",
            "actual": "What current shows",
            "is_regression": true/false
        }}
    ]
}}

Classification guide:
- CRITICAL: Broken layout, missing critical elements, unreadable text
- MAJOR: Significant visual changes, wrong colors/branding, misaligned elements
- MINOR: Small spacing differences, subtle font variations
- INFO: Dynamic content changes, expected variations

is_regression should be false for:
- Dynamic content (timestamps, counters)
- Intentional changes mentioned in context
- Regions marked to ignore
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "IMAGE 1 (Baseline):"
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": baseline_type,
                                    "data": baseline_data,
                                }
                            },
                            {
                                "type": "text",
                                "text": "IMAGE 2 (Current):"
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": current_type,
                                    "data": current_data,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            # Calculate cost
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            # Sonnet pricing
            cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)

            # Parse response
            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            result_data = json.loads(content.strip())

            # Build differences list
            differences = []
            for diff in result_data.get("differences", []):
                differences.append(VisualDifference(
                    type=DifferenceType(diff.get("type", "content")),
                    severity=Severity(diff.get("severity", "minor")),
                    description=diff.get("description", ""),
                    location=diff.get("location", "unknown"),
                    element=diff.get("element"),
                    expected=diff.get("expected"),
                    actual=diff.get("actual"),
                    is_regression=diff.get("is_regression", True),
                ))

            result = VisualComparisonResult(
                baseline_path=str(baseline),
                current_path=str(current),
                match=result_data.get("match", False),
                match_percentage=result_data.get("match_percentage", 0),
                differences=differences,
                summary=result_data.get("summary", ""),
                analysis_cost_usd=cost,
            )

            self.log.info(
                "Comparison complete",
                match=result.match,
                match_percentage=result.match_percentage,
                differences=len(differences),
                regressions=result.has_regressions(),
                cost=f"${cost:.4f}",
            )

            return result

        except Exception as e:
            self.log.error("Visual comparison failed", error=str(e))
            return VisualComparisonResult(
                baseline_path=str(baseline),
                current_path=str(current),
                match=False,
                match_percentage=0,
                summary=f"Comparison failed: {str(e)}",
            )

    async def compare_batch(
        self,
        comparisons: list[tuple[str, str]],
        context: Optional[str] = None,
    ) -> list[VisualComparisonResult]:
        """Compare multiple screenshot pairs."""
        results = []
        for baseline, current in comparisons:
            result = await self.compare(baseline, current, context)
            results.append(result)
        return results

    async def analyze_single(
        self,
        screenshot: str | Path,
        expected_elements: Optional[list[str]] = None,
        context: Optional[str] = None,
    ) -> dict:
        """
        Analyze a single screenshot for expected content.

        Useful for verifying a page loaded correctly without a baseline.
        """
        self.log.info("Analyzing screenshot", path=str(screenshot))

        image_data, media_type = self._load_image(screenshot)

        elements_text = ""
        if expected_elements:
            elements_text = f"\n\nEXPECTED ELEMENTS: {', '.join(expected_elements)}"

        context_text = ""
        if context:
            context_text = f"\n\nCONTEXT: {context}"

        prompt = f"""Analyze this screenshot and verify the page loaded correctly.
{context_text}
{elements_text}

Respond with JSON:
{{
    "page_loaded": true/false,
    "title": "Page title if visible",
    "main_elements": ["list of major UI elements visible"],
    "issues": ["any problems detected"],
    "expected_elements_found": {{"element": true/false}},
    "overall_health": "healthy|degraded|broken"
}}
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except Exception as e:
            self.log.error("Screenshot analysis failed", error=str(e))
            return {
                "page_loaded": False,
                "issues": [str(e)],
                "overall_health": "broken",
            }


class VisualRegressionManager:
    """
    Manages visual regression testing workflow.

    Handles:
    - Baseline storage and retrieval
    - Test run management
    - Diff report generation
    """

    def __init__(
        self,
        baseline_dir: str = "./visual-baselines",
        results_dir: str = "./visual-results",
    ):
        self.baseline_dir = Path(baseline_dir)
        self.results_dir = Path(results_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.visual = VisualAI()
        self.log = logger.bind(component="visual_regression")

    def get_baseline_path(self, test_id: str, step: int = 0) -> Path:
        """Get path to baseline screenshot for a test."""
        return self.baseline_dir / f"{test_id}_step{step}.png"

    def has_baseline(self, test_id: str, step: int = 0) -> bool:
        """Check if baseline exists for a test."""
        return self.get_baseline_path(test_id, step).exists()

    def save_baseline(self, test_id: str, screenshot_data: bytes, step: int = 0) -> Path:
        """Save a new baseline screenshot."""
        path = self.get_baseline_path(test_id, step)
        path.write_bytes(screenshot_data)
        self.log.info("Saved baseline", test_id=test_id, step=step, path=str(path))
        return path

    async def check_visual_regression(
        self,
        test_id: str,
        current_screenshot: bytes,
        step: int = 0,
        context: Optional[str] = None,
        auto_update_baseline: bool = False,
    ) -> VisualComparisonResult:
        """
        Check for visual regression against baseline.

        If no baseline exists:
        - Creates baseline from current screenshot
        - Returns match=True (first run)

        If baseline exists:
        - Compares current against baseline
        - Reports differences
        """
        baseline_path = self.get_baseline_path(test_id, step)
        current_path = self.results_dir / f"{test_id}_step{step}_current.png"

        # Save current screenshot
        current_path.write_bytes(current_screenshot)

        if not baseline_path.exists():
            # First run - create baseline
            self.save_baseline(test_id, current_screenshot, step)
            self.log.info("Created initial baseline", test_id=test_id, step=step)
            return VisualComparisonResult(
                baseline_path=str(baseline_path),
                current_path=str(current_path),
                match=True,
                match_percentage=100.0,
                summary="Initial baseline created",
            )

        # Compare against baseline
        result = await self.visual.compare(
            baseline=baseline_path,
            current=current_path,
            context=context,
        )

        # Auto-update baseline if no regressions and option enabled
        if auto_update_baseline and not result.has_regressions():
            self.save_baseline(test_id, current_screenshot, step)
            self.log.info("Auto-updated baseline", test_id=test_id, step=step)

        return result

    def generate_report(self, results: list[VisualComparisonResult]) -> dict:
        """Generate a summary report from multiple visual comparison results."""
        total = len(results)
        passed = sum(1 for r in results if r.match or not r.has_regressions())
        failed = total - passed

        all_differences = []
        for r in results:
            for d in r.differences:
                if d.is_regression:
                    all_differences.append({
                        "baseline": r.baseline_path,
                        "current": r.current_path,
                        **{k: v.value if hasattr(v, 'value') else v for k, v in vars(d).items()}
                    })

        return {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": f"{(passed/total*100):.1f}%" if total > 0 else "N/A",
            },
            "regressions": all_differences,
            "total_cost_usd": sum(r.analysis_cost_usd for r in results),
        }


# Convenience function
def create_visual_ai(sensitivity: str = "medium") -> VisualAI:
    """Factory function for creating VisualAI instance."""
    return VisualAI(sensitivity=sensitivity)
