"""
Quality Auditor - Accessibility & Performance Testing

This is a KEY DIFFERENTIATOR. We go beyond functional testing to include:
1. WCAG 2.1 Accessibility Compliance (axe-core integration)
2. Core Web Vitals performance metrics
3. Lighthouse-style scoring
4. Best practices and SEO checks
5. Automated remediation suggestions
"""

import json
import re
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from anthropic import Anthropic

from src.config import get_settings


class AccessibilityLevel(str, Enum):
    A = "A"         # Minimum accessibility
    AA = "AA"       # Standard (most common requirement)
    AAA = "AAA"     # Highest accessibility


class ViolationSeverity(str, Enum):
    CRITICAL = "critical"   # Blocks users completely
    SERIOUS = "serious"     # Significantly impacts users
    MODERATE = "moderate"   # Some impact on users
    MINOR = "minor"         # Minor inconvenience


@dataclass
class AccessibilityViolation:
    """A single accessibility violation."""
    rule_id: str
    description: str
    severity: ViolationSeverity
    wcag_criteria: list[str]
    affected_elements: list[str]
    fix_suggestion: str
    help_url: str


@dataclass
class PerformanceMetric:
    """A performance metric measurement."""
    name: str
    value: float
    unit: str
    score: float  # 0-100
    threshold_good: float
    threshold_needs_improvement: float
    status: str  # "good", "needs_improvement", "poor"


@dataclass
class AccessibilityReport:
    """Complete accessibility audit report."""
    url: str
    timestamp: str
    level_tested: AccessibilityLevel
    total_violations: int
    violations_by_severity: dict[str, int]
    violations: list[AccessibilityViolation]
    passes: int
    score: float  # 0-100
    compliant: bool


@dataclass
class PerformanceReport:
    """Complete performance audit report."""
    url: str
    timestamp: str
    overall_score: float
    metrics: list[PerformanceMetric]
    opportunities: list[dict]
    diagnostics: list[dict]


@dataclass
class QualityReport:
    """Combined quality audit report."""
    accessibility: AccessibilityReport
    performance: PerformanceReport
    best_practices_score: float
    seo_score: float
    overall_grade: str  # A, B, C, D, F


class AccessibilityAuditor:
    """
    Automated accessibility testing following WCAG 2.1 guidelines.

    Uses a combination of:
    - Automated rule checking (like axe-core)
    - AI-powered visual analysis for contrast and readability
    - Semantic structure analysis
    """

    # Common accessibility rules to check
    RULES = {
        "color-contrast": {
            "description": "Text must have sufficient color contrast",
            "wcag": ["1.4.3", "1.4.6"],
            "severity": ViolationSeverity.SERIOUS
        },
        "image-alt": {
            "description": "Images must have alternative text",
            "wcag": ["1.1.1"],
            "severity": ViolationSeverity.CRITICAL
        },
        "button-name": {
            "description": "Buttons must have discernible text",
            "wcag": ["4.1.2"],
            "severity": ViolationSeverity.CRITICAL
        },
        "link-name": {
            "description": "Links must have discernible text",
            "wcag": ["2.4.4", "4.1.2"],
            "severity": ViolationSeverity.SERIOUS
        },
        "label": {
            "description": "Form elements must have labels",
            "wcag": ["1.3.1", "4.1.2"],
            "severity": ViolationSeverity.CRITICAL
        },
        "heading-order": {
            "description": "Heading levels should increase by one",
            "wcag": ["1.3.1"],
            "severity": ViolationSeverity.MODERATE
        },
        "duplicate-id": {
            "description": "IDs must be unique",
            "wcag": ["4.1.1"],
            "severity": ViolationSeverity.SERIOUS
        },
        "html-lang": {
            "description": "HTML element must have a lang attribute",
            "wcag": ["3.1.1"],
            "severity": ViolationSeverity.SERIOUS
        },
        "landmark-one-main": {
            "description": "Page must have one main landmark",
            "wcag": ["1.3.1"],
            "severity": ViolationSeverity.MODERATE
        },
        "focus-visible": {
            "description": "Focus indicator must be visible",
            "wcag": ["2.4.7"],
            "severity": ViolationSeverity.SERIOUS
        },
        "aria-valid-attr": {
            "description": "ARIA attributes must be valid",
            "wcag": ["4.1.2"],
            "severity": ViolationSeverity.CRITICAL
        },
        "keyboard-accessible": {
            "description": "All interactive elements must be keyboard accessible",
            "wcag": ["2.1.1"],
            "severity": ViolationSeverity.CRITICAL
        },
    }

    def __init__(self):
        self.settings = get_settings()
        self.client = Anthropic(api_key=self.settings.anthropic_api_key)

    async def audit(
        self,
        page_html: str,
        page_url: str,
        screenshot: Optional[str] = None,  # base64
        level: AccessibilityLevel = AccessibilityLevel.AA
    ) -> AccessibilityReport:
        """Perform accessibility audit on a page."""
        from datetime import datetime

        violations = []
        passes = 0

        # Run automated checks
        html_violations = self._check_html_rules(page_html)
        violations.extend(html_violations)

        # Use AI for visual analysis if screenshot provided
        if screenshot:
            visual_violations = await self._ai_visual_analysis(screenshot)
            violations.extend(visual_violations)

        # Calculate score
        total_checks = len(self.RULES)
        passes = total_checks - len(set(v.rule_id for v in violations))

        severity_counts = {s.value: 0 for s in ViolationSeverity}
        for v in violations:
            severity_counts[v.severity.value] += 1

        # Score calculation (weighted by severity)
        severity_weights = {
            ViolationSeverity.CRITICAL.value: 25,
            ViolationSeverity.SERIOUS.value: 15,
            ViolationSeverity.MODERATE.value: 5,
            ViolationSeverity.MINOR.value: 2
        }

        deductions = sum(
            count * severity_weights.get(sev, 5)
            for sev, count in severity_counts.items()
        )
        score = max(0, 100 - deductions)

        # Check compliance
        compliant = severity_counts[ViolationSeverity.CRITICAL.value] == 0

        return AccessibilityReport(
            url=page_url,
            timestamp=datetime.utcnow().isoformat(),
            level_tested=level,
            total_violations=len(violations),
            violations_by_severity=severity_counts,
            violations=violations,
            passes=passes,
            score=score,
            compliant=compliant
        )

    def _check_html_rules(self, html: str) -> list[AccessibilityViolation]:
        """Check HTML against accessibility rules."""
        violations = []

        # Check for images without alt
        img_pattern = r'<img[^>]*>'
        for match in re.finditer(img_pattern, html, re.IGNORECASE):
            img_tag = match.group()
            if 'alt=' not in img_tag.lower():
                violations.append(AccessibilityViolation(
                    rule_id="image-alt",
                    description=self.RULES["image-alt"]["description"],
                    severity=self.RULES["image-alt"]["severity"],
                    wcag_criteria=self.RULES["image-alt"]["wcag"],
                    affected_elements=[img_tag[:100]],
                    fix_suggestion="Add alt attribute describing the image content",
                    help_url="https://dequeuniversity.com/rules/axe/4.4/image-alt"
                ))

        # Check for buttons without text
        button_pattern = r'<button[^>]*>(\s*)</button>'
        for match in re.finditer(button_pattern, html, re.IGNORECASE):
            if not match.group(1).strip():
                violations.append(AccessibilityViolation(
                    rule_id="button-name",
                    description=self.RULES["button-name"]["description"],
                    severity=self.RULES["button-name"]["severity"],
                    wcag_criteria=self.RULES["button-name"]["wcag"],
                    affected_elements=[match.group()[:100]],
                    fix_suggestion="Add text content or aria-label to button",
                    help_url="https://dequeuniversity.com/rules/axe/4.4/button-name"
                ))

        # Check for missing lang attribute
        if '<html' in html.lower() and 'lang=' not in html.lower()[:500]:
            violations.append(AccessibilityViolation(
                rule_id="html-lang",
                description=self.RULES["html-lang"]["description"],
                severity=self.RULES["html-lang"]["severity"],
                wcag_criteria=self.RULES["html-lang"]["wcag"],
                affected_elements=["<html>"],
                fix_suggestion="Add lang attribute to HTML element (e.g., lang='en')",
                help_url="https://dequeuniversity.com/rules/axe/4.4/html-has-lang"
            ))

        # Check for duplicate IDs
        id_pattern = r'id=["\']([^"\']+)["\']'
        ids = re.findall(id_pattern, html)
        seen_ids = set()
        duplicate_ids = set()
        for id_val in ids:
            if id_val in seen_ids:
                duplicate_ids.add(id_val)
            seen_ids.add(id_val)

        for dup_id in duplicate_ids:
            violations.append(AccessibilityViolation(
                rule_id="duplicate-id",
                description=self.RULES["duplicate-id"]["description"],
                severity=self.RULES["duplicate-id"]["severity"],
                wcag_criteria=self.RULES["duplicate-id"]["wcag"],
                affected_elements=[f'id="{dup_id}"'],
                fix_suggestion=f"Make ID '{dup_id}' unique across the page",
                help_url="https://dequeuniversity.com/rules/axe/4.4/duplicate-id"
            ))

        # Check for form inputs without labels
        input_pattern = r'<input[^>]*type=["\'](?:text|email|password|tel|number)["\'][^>]*>'
        for match in re.finditer(input_pattern, html, re.IGNORECASE):
            input_tag = match.group()
            input_id = re.search(r'id=["\']([^"\']+)["\']', input_tag)
            if input_id:
                label_pattern = f'for=["\']?{re.escape(input_id.group(1))}["\']?'
                if not re.search(label_pattern, html, re.IGNORECASE):
                    # Check for aria-label
                    if 'aria-label' not in input_tag.lower():
                        violations.append(AccessibilityViolation(
                            rule_id="label",
                            description=self.RULES["label"]["description"],
                            severity=self.RULES["label"]["severity"],
                            wcag_criteria=self.RULES["label"]["wcag"],
                            affected_elements=[input_tag[:100]],
                            fix_suggestion="Add a <label> element with matching 'for' attribute",
                            help_url="https://dequeuniversity.com/rules/axe/4.4/label"
                        ))

        return violations

    async def _ai_visual_analysis(
        self,
        screenshot: str
    ) -> list[AccessibilityViolation]:
        """Use AI to analyze visual accessibility issues."""
        violations = []

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": screenshot
                    }
                },
                {
                    "type": "text",
                    "text": """Analyze this screenshot for accessibility issues. Look for:
1. Low contrast text (text that's hard to read against its background)
2. Small text that may be hard to read
3. Interactive elements that may be too small to click (< 44x44 pixels)
4. Missing visual focus indicators
5. Poor visual hierarchy
6. Problematic color-only information

Return a JSON array of issues found:
[
  {
    "issue_type": "color-contrast|small-target|focus-indicator|color-only",
    "description": "What the issue is",
    "location": "Where on the screen",
    "severity": "critical|serious|moderate|minor",
    "fix": "How to fix it"
  }
]

Return empty array [] if no issues found."""
                }
            ]
        }]

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=1024,
                messages=messages
            )

            text = response.content[0].text
            json_match = re.search(r'\[[\s\S]*\]', text)
            if json_match:
                issues = json.loads(json_match.group())
                for issue in issues:
                    severity_map = {
                        "critical": ViolationSeverity.CRITICAL,
                        "serious": ViolationSeverity.SERIOUS,
                        "moderate": ViolationSeverity.MODERATE,
                        "minor": ViolationSeverity.MINOR
                    }
                    violations.append(AccessibilityViolation(
                        rule_id=issue.get("issue_type", "visual-issue"),
                        description=issue.get("description", ""),
                        severity=severity_map.get(
                            issue.get("severity", "moderate"),
                            ViolationSeverity.MODERATE
                        ),
                        wcag_criteria=["1.4.3"] if "contrast" in issue.get("issue_type", "") else ["2.5.5"],
                        affected_elements=[issue.get("location", "Unknown")],
                        fix_suggestion=issue.get("fix", ""),
                        help_url="https://www.w3.org/WAI/WCAG21/quickref/"
                    ))
        except Exception:
            pass  # Visual analysis is optional enhancement

        return violations


class PerformanceAuditor:
    """
    Performance testing following Core Web Vitals standards.

    Measures:
    - LCP (Largest Contentful Paint)
    - FID (First Input Delay) / INP (Interaction to Next Paint)
    - CLS (Cumulative Layout Shift)
    - TTFB (Time to First Byte)
    - FCP (First Contentful Paint)
    """

    # Core Web Vitals thresholds (in ms or score)
    THRESHOLDS = {
        "LCP": {"good": 2500, "poor": 4000, "unit": "ms"},
        "FID": {"good": 100, "poor": 300, "unit": "ms"},
        "INP": {"good": 200, "poor": 500, "unit": "ms"},
        "CLS": {"good": 0.1, "poor": 0.25, "unit": "score"},
        "TTFB": {"good": 800, "poor": 1800, "unit": "ms"},
        "FCP": {"good": 1800, "poor": 3000, "unit": "ms"},
        "TTI": {"good": 3800, "poor": 7300, "unit": "ms"},
        "Speed_Index": {"good": 3400, "poor": 5800, "unit": "ms"},
        "Total_Blocking_Time": {"good": 200, "poor": 600, "unit": "ms"},
    }

    def __init__(self):
        self.settings = get_settings()

    async def audit(
        self,
        page_url: str,
        metrics: dict[str, float]
    ) -> PerformanceReport:
        """Generate performance report from collected metrics."""
        from datetime import datetime

        metric_results = []
        total_score = 0
        metric_count = 0

        for name, value in metrics.items():
            if name in self.THRESHOLDS:
                threshold = self.THRESHOLDS[name]
                score, status = self._calculate_score(value, threshold)

                metric_results.append(PerformanceMetric(
                    name=name,
                    value=value,
                    unit=threshold["unit"],
                    score=score,
                    threshold_good=threshold["good"],
                    threshold_needs_improvement=threshold["poor"],
                    status=status
                ))

                total_score += score
                metric_count += 1

        overall_score = total_score / metric_count if metric_count > 0 else 0

        # Generate opportunities
        opportunities = self._generate_opportunities(metric_results)

        # Generate diagnostics
        diagnostics = self._generate_diagnostics(metrics)

        return PerformanceReport(
            url=page_url,
            timestamp=datetime.utcnow().isoformat(),
            overall_score=round(overall_score, 1),
            metrics=metric_results,
            opportunities=opportunities,
            diagnostics=diagnostics
        )

    def _calculate_score(
        self,
        value: float,
        threshold: dict
    ) -> tuple[float, str]:
        """Calculate performance score for a metric."""
        good = threshold["good"]
        poor = threshold["poor"]

        if value <= good:
            score = 90 + (10 * (good - value) / good)
            status = "good"
        elif value <= poor:
            score = 50 + (40 * (poor - value) / (poor - good))
            status = "needs_improvement"
        else:
            score = max(0, 50 * (1 - (value - poor) / poor))
            status = "poor"

        return min(100, max(0, score)), status

    def _generate_opportunities(
        self,
        metrics: list[PerformanceMetric]
    ) -> list[dict]:
        """Generate optimization opportunities."""
        opportunities = []

        for metric in metrics:
            if metric.status != "good":
                if metric.name == "LCP":
                    opportunities.append({
                        "title": "Improve Largest Contentful Paint",
                        "impact": "high" if metric.status == "poor" else "medium",
                        "suggestions": [
                            "Optimize and compress images",
                            "Use a CDN for static assets",
                            "Preload critical resources",
                            "Remove render-blocking resources"
                        ]
                    })
                elif metric.name == "CLS":
                    opportunities.append({
                        "title": "Reduce Cumulative Layout Shift",
                        "impact": "high" if metric.status == "poor" else "medium",
                        "suggestions": [
                            "Set explicit dimensions on images and videos",
                            "Reserve space for dynamic content",
                            "Avoid inserting content above existing content",
                            "Use CSS transform for animations"
                        ]
                    })
                elif metric.name in ["FID", "INP", "Total_Blocking_Time"]:
                    opportunities.append({
                        "title": "Improve Responsiveness",
                        "impact": "high" if metric.status == "poor" else "medium",
                        "suggestions": [
                            "Break up long JavaScript tasks",
                            "Use web workers for heavy computation",
                            "Optimize third-party scripts",
                            "Reduce JavaScript execution time"
                        ]
                    })
                elif metric.name == "TTFB":
                    opportunities.append({
                        "title": "Reduce Server Response Time",
                        "impact": "high" if metric.status == "poor" else "medium",
                        "suggestions": [
                            "Use a CDN",
                            "Optimize database queries",
                            "Implement server-side caching",
                            "Upgrade server hardware"
                        ]
                    })

        return opportunities

    def _generate_diagnostics(self, metrics: dict) -> list[dict]:
        """Generate diagnostic information."""
        diagnostics = []

        if metrics.get("Total_Blocking_Time", 0) > 300:
            diagnostics.append({
                "title": "Main thread blocked",
                "description": f"Main thread was blocked for {metrics.get('Total_Blocking_Time', 0)}ms",
                "severity": "warning"
            })

        if metrics.get("resource_count", 0) > 100:
            diagnostics.append({
                "title": "Too many resources",
                "description": f"Page loaded {metrics.get('resource_count', 0)} resources",
                "severity": "warning"
            })

        if metrics.get("total_bytes", 0) > 3000000:  # 3MB
            diagnostics.append({
                "title": "Large page weight",
                "description": f"Total page size is {metrics.get('total_bytes', 0) / 1000000:.1f}MB",
                "severity": "warning"
            })

        return diagnostics


class QualityAuditor:
    """Combined quality auditor for accessibility, performance, and best practices."""

    def __init__(self):
        self.accessibility_auditor = AccessibilityAuditor()
        self.performance_auditor = PerformanceAuditor()

    async def full_audit(
        self,
        page_url: str,
        page_html: str,
        screenshot: Optional[str] = None,
        performance_metrics: Optional[dict] = None
    ) -> QualityReport:
        """Perform comprehensive quality audit."""
        # Accessibility audit
        accessibility = await self.accessibility_auditor.audit(
            page_html=page_html,
            page_url=page_url,
            screenshot=screenshot
        )

        # Performance audit
        performance = await self.performance_auditor.audit(
            page_url=page_url,
            metrics=performance_metrics or {}
        )

        # Best practices score (simplified)
        best_practices_score = self._calculate_best_practices(page_html)

        # SEO score (simplified)
        seo_score = self._calculate_seo(page_html, page_url)

        # Overall grade
        avg_score = (
            accessibility.score * 0.3 +
            performance.overall_score * 0.3 +
            best_practices_score * 0.2 +
            seo_score * 0.2
        )

        if avg_score >= 90:
            grade = "A"
        elif avg_score >= 80:
            grade = "B"
        elif avg_score >= 70:
            grade = "C"
        elif avg_score >= 60:
            grade = "D"
        else:
            grade = "F"

        return QualityReport(
            accessibility=accessibility,
            performance=performance,
            best_practices_score=best_practices_score,
            seo_score=seo_score,
            overall_grade=grade
        )

    def _calculate_best_practices(self, html: str) -> float:
        """Calculate best practices score."""
        score = 100
        html_lower = html.lower()

        # Check for HTTPS
        if 'http://' in html_lower and 'http://localhost' not in html_lower:
            score -= 10

        # Check for deprecated elements
        deprecated = ['<font', '<center', '<marquee', '<blink']
        for dep in deprecated:
            if dep in html_lower:
                score -= 5

        # Check for inline styles (not always bad, but can indicate issues)
        inline_styles = html_lower.count('style="')
        if inline_styles > 20:
            score -= 10

        # Check for console errors in scripts (simplified)
        if 'console.error' in html_lower:
            score -= 5

        return max(0, score)

    def _calculate_seo(self, html: str, url: str) -> float:
        """Calculate SEO score."""
        score = 100
        html_lower = html.lower()

        # Check for title
        if '<title>' not in html_lower or '</title>' not in html_lower:
            score -= 20

        # Check for meta description
        if 'name="description"' not in html_lower:
            score -= 15

        # Check for heading structure
        if '<h1' not in html_lower:
            score -= 10

        # Check for meta viewport
        if 'name="viewport"' not in html_lower:
            score -= 10

        # Check for canonical
        if 'rel="canonical"' not in html_lower:
            score -= 5

        # Check for robots meta
        if 'noindex' in html_lower:
            score -= 20

        return max(0, score)
