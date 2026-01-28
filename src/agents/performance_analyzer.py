"""Performance Analyzer Agent - Analyzes application performance metrics.

Uses Lighthouse-style analysis to evaluate:
- Core Web Vitals (LCP, FID, CLS)
- Time to First Byte (TTFB)
- First Contentful Paint (FCP)
- Speed Index
- Total Blocking Time
- JavaScript execution time
- Resource optimization opportunities
"""

from dataclasses import dataclass, field
from enum import Enum

from ..core.model_router import TaskType
from .base import AgentCapability, AgentResult, BaseAgent
from .prompts import get_enhanced_prompt


class PerformanceGrade(str, Enum):
    """Performance grade levels."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    NEEDS_WORK = "needs_work"  # 50-74
    POOR = "poor"           # 0-49


@dataclass
class CoreWebVitals:
    """Core Web Vitals metrics."""
    lcp_ms: float  # Largest Contentful Paint
    fid_ms: float  # First Input Delay
    cls: float     # Cumulative Layout Shift
    inp_ms: float  # Interaction to Next Paint

    def get_lcp_grade(self) -> PerformanceGrade:
        if self.lcp_ms <= 2500:
            return PerformanceGrade.EXCELLENT
        elif self.lcp_ms <= 4000:
            return PerformanceGrade.NEEDS_WORK
        return PerformanceGrade.POOR

    def get_fid_grade(self) -> PerformanceGrade:
        if self.fid_ms <= 100:
            return PerformanceGrade.EXCELLENT
        elif self.fid_ms <= 300:
            return PerformanceGrade.NEEDS_WORK
        return PerformanceGrade.POOR

    def get_cls_grade(self) -> PerformanceGrade:
        if self.cls <= 0.1:
            return PerformanceGrade.EXCELLENT
        elif self.cls <= 0.25:
            return PerformanceGrade.NEEDS_WORK
        return PerformanceGrade.POOR


@dataclass
class PerformanceMetrics:
    """Full performance metrics."""
    # Core Web Vitals
    core_vitals: CoreWebVitals

    # Timing metrics
    ttfb_ms: float = 0.0
    fcp_ms: float = 0.0
    speed_index: float = 0.0
    tti_ms: float = 0.0  # Time to Interactive
    tbt_ms: float = 0.0  # Total Blocking Time

    # Resource metrics
    total_requests: int = 0
    total_transfer_size_kb: float = 0.0
    js_execution_time_ms: float = 0.0
    dom_content_loaded_ms: float = 0.0
    load_time_ms: float = 0.0

    # Scores (0-100)
    performance_score: int = 0
    accessibility_score: int = 0
    best_practices_score: int = 0
    seo_score: int = 0


@dataclass
class PerformanceIssue:
    """A specific performance issue found."""
    category: str  # "render-blocking", "large-resources", "unused-code", etc.
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    savings_ms: float = 0.0
    savings_kb: float = 0.0
    affected_resources: list[str] = field(default_factory=list)
    fix_suggestion: str = ""


@dataclass
class PerformanceAnalysisResult:
    """Result of performance analysis."""
    url: str
    metrics: PerformanceMetrics
    issues: list[PerformanceIssue]
    recommendations: list[str]
    overall_grade: PerformanceGrade
    summary: str


class PerformanceAnalyzerAgent(BaseAgent):
    """
    Agent that analyzes application performance using Lighthouse-style metrics.

    Capabilities:
    - Core Web Vitals measurement and grading
    - Resource loading analysis
    - Render-blocking detection
    - JavaScript execution profiling
    - Performance recommendations

    Uses Claude for intelligent analysis of metrics and generating
    actionable recommendations based on the specific application context.
    """

    DEFAULT_TASK_TYPE = TaskType.TEXT_EXTRACTION

    # RAP-231: Agent capabilities for A2A discovery
    CAPABILITIES = [
        AgentCapability.PERFORMANCE_ANALYSIS,
    ]

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for performance analysis."""
        enhanced = get_enhanced_prompt("performance_analyzer")
        if enhanced:
            return enhanced

        return """# Role
You are a world-class web performance engineer with deep expertise in:
- Google's Core Web Vitals and Lighthouse scoring
- Browser rendering pipelines and critical rendering path
- JavaScript execution optimization
- Network waterfall analysis
- Resource prioritization strategies

# Analysis Framework
When analyzing performance data:
1. Identify Critical Issues First (render-blocking, large bundles, unoptimized images)
2. Calculate Impact (time savings, user impact, mobile vs desktop)
3. Provide Actionable Fixes (specific code changes, before/after examples)

# Output Requirements
- JSON with typed fields
- Severity ratings: critical, high, medium, low
- Time savings estimates in milliseconds
- Confidence scores for each recommendation"""

    async def execute(
        self,
        url: str,
        device: str = "mobile",
        include_trace: bool = False,
    ) -> AgentResult[PerformanceAnalysisResult]:
        """
        Analyze performance for a URL.

        Args:
            url: URL to analyze
            device: Device type ("mobile" or "desktop")
            include_trace: Whether to include detailed trace data

        Returns:
            PerformanceAnalysisResult with metrics and recommendations
        """
        try:
            # Step 1: Collect performance metrics via browser
            metrics = await self._collect_metrics(url, device)

            # Step 2: Analyze metrics with AI
            analysis = await self._analyze_metrics(url, metrics, device)

            return AgentResult(
                success=True,
                data=analysis,
                input_tokens=self.usage.total_input_tokens,
                output_tokens=self.usage.total_output_tokens,
                cost=self.usage.total_cost,
            )

        except Exception as e:
            self.log.exception("Performance analysis failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    async def _collect_metrics(self, url: str, device: str) -> dict:
        """Collect raw performance metrics from browser."""
        # This would integrate with the Worker's browser for real metrics
        # For now, return structure that would come from Performance API
        return {
            "navigation_timing": {},
            "resource_timing": [],
            "layout_shifts": [],
            "long_tasks": [],
            "largest_contentful_paint": None,
            "first_input_delay": None,
        }

    async def _analyze_metrics(
        self,
        url: str,
        raw_metrics: dict,
        device: str,
    ) -> PerformanceAnalysisResult:
        """Analyze metrics using AI for intelligent insights."""
        prompt = f"""Analyze these performance metrics for {url} on {device}:

RAW METRICS:
{raw_metrics}

Provide:
1. Core Web Vitals assessment with grades
2. Top 5 performance issues ranked by impact
3. Specific, actionable recommendations
4. Overall performance grade

RESPOND IN JSON:
{{
  "core_vitals": {{
    "lcp_ms": <number>,
    "lcp_grade": "excellent|good|needs_work|poor",
    "fid_ms": <number>,
    "fid_grade": "excellent|good|needs_work|poor",
    "cls": <number>,
    "cls_grade": "excellent|good|needs_work|poor"
  }},
  "issues": [
    {{
      "category": "<category>",
      "severity": "critical|high|medium|low",
      "title": "<title>",
      "description": "<description>",
      "savings_ms": <number>,
      "fix_suggestion": "<specific fix>"
    }}
  ],
  "recommendations": ["<actionable recommendation>"],
  "overall_grade": "excellent|good|needs_work|poor",
  "summary": "<2-3 sentence summary>"
}}"""

        response = await self._call_model(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.TEXT_EXTRACTION,
            max_tokens=2000,
        )

        data = self._parse_json_response(response["content"], {})

        # Convert to typed result
        core_vitals = CoreWebVitals(
            lcp_ms=data.get("core_vitals", {}).get("lcp_ms", 0),
            fid_ms=data.get("core_vitals", {}).get("fid_ms", 0),
            cls=data.get("core_vitals", {}).get("cls", 0),
            inp_ms=data.get("core_vitals", {}).get("inp_ms", 0),
        )

        issues = [
            PerformanceIssue(
                category=i.get("category", "unknown"),
                severity=i.get("severity", "medium"),
                title=i.get("title", ""),
                description=i.get("description", ""),
                savings_ms=i.get("savings_ms", 0),
                fix_suggestion=i.get("fix_suggestion", ""),
            )
            for i in data.get("issues", [])
        ]

        return PerformanceAnalysisResult(
            url=url,
            metrics=PerformanceMetrics(core_vitals=core_vitals),
            issues=issues,
            recommendations=data.get("recommendations", []),
            overall_grade=PerformanceGrade(data.get("overall_grade", "needs_work")),
            summary=data.get("summary", ""),
        )

    async def compare_before_after(
        self,
        before_metrics: PerformanceMetrics,
        after_metrics: PerformanceMetrics,
    ) -> dict:
        """Compare performance before and after a change."""
        prompt = f"""Compare these performance metrics before and after a change:

BEFORE:
- LCP: {before_metrics.core_vitals.lcp_ms}ms
- FID: {before_metrics.core_vitals.fid_ms}ms
- CLS: {before_metrics.core_vitals.cls}
- Total Load: {before_metrics.load_time_ms}ms

AFTER:
- LCP: {after_metrics.core_vitals.lcp_ms}ms
- FID: {after_metrics.core_vitals.fid_ms}ms
- CLS: {after_metrics.core_vitals.cls}
- Total Load: {after_metrics.load_time_ms}ms

Provide a summary of:
1. What improved
2. What regressed
3. Net impact assessment
4. Recommendation (ship/don't ship/investigate)

RESPOND IN JSON."""

        response = await self._call_model(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.TEXT_EXTRACTION,
            max_tokens=1000,
        )

        return self._parse_json_response(response["content"], {})
