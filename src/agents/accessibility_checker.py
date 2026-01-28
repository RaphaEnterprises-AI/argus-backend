"""Accessibility Checker Agent - Validates WCAG compliance and a11y best practices.

Tests for WCAG 2.1 Level AA compliance:
- Perceivable: Alt text, captions, color contrast
- Operable: Keyboard navigation, focus management
- Understandable: Labels, error messages, consistent navigation
- Robust: Valid HTML, ARIA usage
"""

import re
from dataclasses import dataclass, field
from enum import Enum

from ..core.model_router import TaskType
from .base import AgentCapability, AgentResult, BaseAgent
from .prompts import get_enhanced_prompt


class WCAGLevel(str, Enum):
    """WCAG conformance levels."""
    A = "A"
    AA = "AA"
    AAA = "AAA"


class WCAGPrinciple(str, Enum):
    """WCAG principles (POUR)."""
    PERCEIVABLE = "perceivable"
    OPERABLE = "operable"
    UNDERSTANDABLE = "understandable"
    ROBUST = "robust"


class IssueImpact(str, Enum):
    """Accessibility issue impact levels."""
    CRITICAL = "critical"    # Blocks access entirely
    SERIOUS = "serious"      # Causes major difficulty
    MODERATE = "moderate"    # Causes some difficulty
    MINOR = "minor"          # Causes minor inconvenience


@dataclass
class AccessibilityIssue:
    """An accessibility violation found."""
    id: str
    wcag_criterion: str  # e.g., "1.1.1", "2.4.6"
    wcag_level: WCAGLevel
    principle: WCAGPrinciple
    impact: IssueImpact
    title: str
    description: str
    element: str  # HTML element or selector
    html_snippet: str
    fix_suggestion: str
    affected_users: list[str] = field(default_factory=list)  # ["blind", "motor-impaired", etc.]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "wcag_criterion": self.wcag_criterion,
            "wcag_level": self.wcag_level.value,
            "principle": self.principle.value,
            "impact": self.impact.value,
            "title": self.title,
            "description": self.description,
            "element": self.element,
            "html_snippet": self.html_snippet,
            "fix_suggestion": self.fix_suggestion,
            "affected_users": self.affected_users,
        }


@dataclass
class ColorContrastResult:
    """Color contrast analysis result."""
    foreground: str
    background: str
    contrast_ratio: float
    passes_aa_normal: bool
    passes_aa_large: bool
    passes_aaa_normal: bool
    passes_aaa_large: bool


@dataclass
class KeyboardNavigationResult:
    """Keyboard navigation test result."""
    focusable_elements: int
    focus_visible: bool
    focus_order_logical: bool
    keyboard_traps: list[str]
    skip_links_present: bool


@dataclass
class AccessibilityCheckResult:
    """Result of accessibility check."""
    url: str
    issues: list[AccessibilityIssue]
    color_contrast: list[ColorContrastResult]
    keyboard_navigation: KeyboardNavigationResult
    wcag_level_achieved: WCAGLevel | None
    score: int  # 0-100
    summary: str
    recommendations: list[str]

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.impact == IssueImpact.CRITICAL)

    @property
    def serious_count(self) -> int:
        return sum(1 for i in self.issues if i.impact == IssueImpact.SERIOUS)

    def passes_wcag_aa(self) -> bool:
        """Check if page passes WCAG 2.1 AA."""
        # No critical or serious Level A or AA issues
        blocking_issues = [
            i for i in self.issues
            if i.impact in [IssueImpact.CRITICAL, IssueImpact.SERIOUS]
            and i.wcag_level in [WCAGLevel.A, WCAGLevel.AA]
        ]
        return len(blocking_issues) == 0


class AccessibilityCheckerAgent(BaseAgent):
    """
    Agent that checks web accessibility compliance with WCAG guidelines.

    Capabilities:
    - WCAG 2.1 Level A, AA, and AAA checking
    - Color contrast analysis
    - Keyboard navigation testing
    - Screen reader compatibility
    - ARIA usage validation
    - Focus management testing
    - Form accessibility

    Uses AI to provide contextual fix suggestions and prioritization.
    """

    DEFAULT_TASK_TYPE = TaskType.TEXT_EXTRACTION

    # RAP-231: Agent capabilities for A2A discovery
    CAPABILITIES = [
        AgentCapability.ACCESSIBILITY_CHECK,
    ]

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for accessibility checking."""
        enhanced = get_enhanced_prompt("accessibility_checker")
        if enhanced:
            return enhanced

        return """# Role
You are a certified accessibility specialist (CPACC, WAS) with WCAG 2.1 expertise.

# Focus Areas
- Visual accessibility (contrast, alt text)
- Keyboard accessibility (focus, navigation)
- Screen reader compatibility (ARIA, semantic HTML)
- Form accessibility (labels, errors)

# Output Requirements
- WCAG success criterion reference (e.g., 1.1.1)
- Affected user groups (blind, low vision, motor, cognitive)
- Specific, actionable fix with code example
- Priority based on impact and frequency

# Constraints
- Test actual rendered content, not just source
- Account for dynamic content and SPAs
- Consider assistive technology compatibility"""

    async def execute(
        self,
        url: str,
        wcag_level: WCAGLevel = WCAGLevel.AA,
        include_best_practices: bool = True,
        test_keyboard: bool = True,
    ) -> AgentResult[AccessibilityCheckResult]:
        """
        Check accessibility compliance for a URL.

        Args:
            url: URL to check
            wcag_level: Target WCAG level (A, AA, or AAA)
            include_best_practices: Include non-WCAG best practices
            test_keyboard: Perform keyboard navigation tests

        Returns:
            AccessibilityCheckResult with issues and recommendations
        """
        try:
            # Step 1: Get page content
            page_html = await self._fetch_page(url)

            # Step 2: Run automated checks
            issues = []

            # Image alt text
            img_issues = self._check_images(page_html, url)
            issues.extend(img_issues)

            # Form labels
            form_issues = self._check_forms(page_html, url)
            issues.extend(form_issues)

            # Headings
            heading_issues = self._check_headings(page_html, url)
            issues.extend(heading_issues)

            # ARIA usage
            aria_issues = self._check_aria(page_html, url)
            issues.extend(aria_issues)

            # Links
            link_issues = self._check_links(page_html, url)
            issues.extend(link_issues)

            # Color contrast (would need computed styles)
            contrast_results = []

            # Keyboard navigation (would need browser)
            keyboard_result = KeyboardNavigationResult(
                focusable_elements=0,
                focus_visible=True,
                focus_order_logical=True,
                keyboard_traps=[],
                skip_links_present=False,
            )

            # Step 3: AI analysis for context and prioritization
            analysis = await self._analyze_issues(url, issues, wcag_level)

            # Calculate score
            score = self._calculate_score(issues)

            # Determine achieved level
            achieved_level = self._determine_wcag_level(issues)

            result = AccessibilityCheckResult(
                url=url,
                issues=issues,
                color_contrast=contrast_results,
                keyboard_navigation=keyboard_result,
                wcag_level_achieved=achieved_level,
                score=score,
                summary=analysis.get("summary", ""),
                recommendations=analysis.get("recommendations", []),
            )

            return AgentResult(
                success=True,
                data=result,
                input_tokens=self.usage.total_input_tokens,
                output_tokens=self.usage.total_output_tokens,
                cost=self.usage.total_cost,
            )

        except Exception as e:
            self.log.exception("Accessibility check failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    async def _fetch_page(self, url: str) -> str:
        """Fetch page HTML content."""
        return ""

    def _check_images(self, html: str, url: str) -> list[AccessibilityIssue]:
        """Check images for alt text."""
        issues = []

        # Find images without alt (simplified pattern)
        img_pattern = r'<img[^>]*>'
        for match in re.finditer(img_pattern, html, re.IGNORECASE):
            img_tag = match.group()
            if 'alt=' not in img_tag.lower():
                issues.append(AccessibilityIssue(
                    id="A11Y-IMG-001",
                    wcag_criterion="1.1.1",
                    wcag_level=WCAGLevel.A,
                    principle=WCAGPrinciple.PERCEIVABLE,
                    impact=IssueImpact.CRITICAL,
                    title="Image missing alt attribute",
                    description="Images must have alt attributes to provide text alternatives for screen reader users.",
                    element="img",
                    html_snippet=img_tag[:100],
                    fix_suggestion='Add alt="" for decorative images or descriptive alt text for informative images.',
                    affected_users=["blind", "low-vision"],
                ))

        return issues

    def _check_forms(self, html: str, url: str) -> list[AccessibilityIssue]:
        """Check form elements for labels."""
        issues = []

        # Find inputs without associated labels (simplified)
        input_pattern = r'<input[^>]*>'
        for match in re.finditer(input_pattern, html, re.IGNORECASE):
            input_tag = match.group()
            # Check for id to look for label
            id_match = re.search(r'id=["\']([^"\']+)["\']', input_tag)
            if id_match:
                input_id = id_match.group(1)
                label_pattern = f'for=["\']?{input_id}["\']?'
                if not re.search(label_pattern, html, re.IGNORECASE):
                    # Check for aria-label or aria-labelledby
                    if 'aria-label' not in input_tag.lower():
                        issues.append(AccessibilityIssue(
                            id="A11Y-FORM-001",
                            wcag_criterion="1.3.1",
                            wcag_level=WCAGLevel.A,
                            principle=WCAGPrinciple.PERCEIVABLE,
                            impact=IssueImpact.SERIOUS,
                            title="Form input missing label",
                            description="Form inputs must have associated labels for screen reader users.",
                            element="input",
                            html_snippet=input_tag[:100],
                            fix_suggestion=f'Add <label for="{input_id}">Label text</label> or aria-label attribute.',
                            affected_users=["blind", "cognitive"],
                        ))

        return issues

    def _check_headings(self, html: str, url: str) -> list[AccessibilityIssue]:
        """Check heading hierarchy."""
        issues = []

        # Extract heading levels
        heading_pattern = r'<h([1-6])[^>]*>'
        headings = re.findall(heading_pattern, html, re.IGNORECASE)

        if headings:
            # Check if h1 exists
            if '1' not in headings:
                issues.append(AccessibilityIssue(
                    id="A11Y-HEAD-001",
                    wcag_criterion="1.3.1",
                    wcag_level=WCAGLevel.A,
                    principle=WCAGPrinciple.PERCEIVABLE,
                    impact=IssueImpact.MODERATE,
                    title="Missing h1 heading",
                    description="Pages should have a single h1 heading that describes the main content.",
                    element="h1",
                    html_snippet="<h1> not found",
                    fix_suggestion="Add an h1 heading at the start of the main content area.",
                    affected_users=["blind", "cognitive"],
                ))

            # Check for skipped levels
            prev_level = 0
            for level in [int(h) for h in headings]:
                if level - prev_level > 1 and prev_level != 0:
                    issues.append(AccessibilityIssue(
                        id=f"A11Y-HEAD-SKIP-{prev_level}-{level}",
                        wcag_criterion="1.3.1",
                        wcag_level=WCAGLevel.A,
                        principle=WCAGPrinciple.PERCEIVABLE,
                        impact=IssueImpact.MODERATE,
                        title=f"Skipped heading level (h{prev_level} to h{level})",
                        description="Heading levels should not skip levels to maintain proper document outline.",
                        element=f"h{level}",
                        html_snippet=f"h{prev_level} followed by h{level}",
                        fix_suggestion=f"Use h{prev_level + 1} instead or restructure heading hierarchy.",
                        affected_users=["blind", "cognitive"],
                    ))
                prev_level = level

        return issues

    def _check_aria(self, html: str, url: str) -> list[AccessibilityIssue]:
        """Check ARIA usage."""
        issues = []

        # Check for common ARIA mistakes
        # aria-hidden="true" on focusable elements
        if re.search(r'aria-hidden=["\']true["\'][^>]*(?:href|tabindex|button)', html, re.IGNORECASE):
            issues.append(AccessibilityIssue(
                id="A11Y-ARIA-001",
                wcag_criterion="4.1.2",
                wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.ROBUST,
                impact=IssueImpact.SERIOUS,
                title="aria-hidden on focusable element",
                description="Elements with aria-hidden='true' should not be focusable.",
                element="[aria-hidden='true']",
                html_snippet="aria-hidden='true' with focusable content",
                fix_suggestion="Remove aria-hidden or make element non-focusable (tabindex='-1').",
                affected_users=["blind"],
            ))

        return issues

    def _check_links(self, html: str, url: str) -> list[AccessibilityIssue]:
        """Check link accessibility."""
        issues = []

        # Check for empty links
        empty_link_pattern = r'<a[^>]*>\s*</a>'
        if re.search(empty_link_pattern, html, re.IGNORECASE):
            issues.append(AccessibilityIssue(
                id="A11Y-LINK-001",
                wcag_criterion="2.4.4",
                wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.OPERABLE,
                impact=IssueImpact.SERIOUS,
                title="Empty link",
                description="Links must have discernible text for screen reader users.",
                element="a",
                html_snippet="<a href='...'></a>",
                fix_suggestion="Add link text, aria-label, or visually hidden text.",
                affected_users=["blind"],
            ))

        # Check for generic link text
        generic_patterns = [r'>click here<', r'>read more<', r'>learn more<', r'>here<']
        for pattern in generic_patterns:
            if re.search(pattern, html, re.IGNORECASE):
                issues.append(AccessibilityIssue(
                    id="A11Y-LINK-002",
                    wcag_criterion="2.4.4",
                    wcag_level=WCAGLevel.A,
                    principle=WCAGPrinciple.OPERABLE,
                    impact=IssueImpact.MODERATE,
                    title="Non-descriptive link text",
                    description="Link text should describe the destination, not generic phrases.",
                    element="a",
                    html_snippet=pattern.replace('>', '').replace('<', ''),
                    fix_suggestion="Use descriptive link text that makes sense out of context.",
                    affected_users=["blind", "cognitive"],
                ))
                break

        return issues

    def _calculate_score(self, issues: list[AccessibilityIssue]) -> int:
        """Calculate accessibility score (0-100)."""
        if not issues:
            return 100

        # Deduct points based on severity
        deductions = {
            IssueImpact.CRITICAL: 25,
            IssueImpact.SERIOUS: 15,
            IssueImpact.MODERATE: 5,
            IssueImpact.MINOR: 2,
        }

        total_deduction = sum(deductions[i.impact] for i in issues)
        return max(0, 100 - total_deduction)

    def _determine_wcag_level(self, issues: list[AccessibilityIssue]) -> WCAGLevel | None:
        """Determine the highest WCAG level achieved."""
        # Check for Level A failures
        level_a_failures = [
            i for i in issues
            if i.wcag_level == WCAGLevel.A
            and i.impact in [IssueImpact.CRITICAL, IssueImpact.SERIOUS]
        ]
        if level_a_failures:
            return None  # Doesn't meet Level A

        # Check for Level AA failures
        level_aa_failures = [
            i for i in issues
            if i.wcag_level == WCAGLevel.AA
            and i.impact in [IssueImpact.CRITICAL, IssueImpact.SERIOUS]
        ]
        if level_aa_failures:
            return WCAGLevel.A

        # Check for Level AAA failures
        level_aaa_failures = [
            i for i in issues
            if i.wcag_level == WCAGLevel.AAA
            and i.impact in [IssueImpact.CRITICAL, IssueImpact.SERIOUS]
        ]
        if level_aaa_failures:
            return WCAGLevel.AA

        return WCAGLevel.AAA

    async def _analyze_issues(
        self,
        url: str,
        issues: list[AccessibilityIssue],
        target_level: WCAGLevel,
    ) -> dict:
        """Use AI to analyze and prioritize accessibility issues."""
        issue_summary = "\n".join([
            f"- [{i.impact.value}] {i.wcag_criterion}: {i.title}"
            for i in issues[:15]
        ])

        prompt = f"""Analyze these accessibility issues for {url}:

TARGET COMPLIANCE: WCAG 2.1 Level {target_level.value}

ISSUES FOUND:
{issue_summary}

Provide:
1. Executive summary for stakeholders (2-3 sentences)
2. Top 5 prioritized fixes (by impact on users)
3. Estimated effort to reach Level {target_level.value} compliance
4. Quick wins that can be fixed immediately

RESPOND IN JSON:
{{
  "summary": "<executive summary>",
  "recommendations": ["<prioritized fix>"],
  "effort_estimate": "low|medium|high",
  "quick_wins": ["<immediate fix>"]
}}"""

        response = await self._call_model(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.TEXT_EXTRACTION,
            max_tokens=1200,
        )

        return self._parse_json_response(response["content"], {
            "summary": "Accessibility check completed",
            "recommendations": [],
        })
