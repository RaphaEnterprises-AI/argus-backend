"""Tests for Accessibility Checker Agent - comprehensive WCAG compliance testing.

Following industry best practices:
- Parametrization for edge cases
- Proper mocking for external dependencies
- Async testing with pytest-asyncio
- Clear test naming with docstrings
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.accessibility_checker import (
    WCAGLevel,
    WCAGPrinciple,
    IssueImpact,
    AccessibilityIssue,
    ColorContrastResult,
    KeyboardNavigationResult,
    AccessibilityCheckResult,
    AccessibilityCheckerAgent,
)


class TestWCAGLevelEnum:
    """Tests for WCAGLevel enum."""

    @pytest.mark.parametrize("level,expected_value", [
        (WCAGLevel.A, "A"),
        (WCAGLevel.AA, "AA"),
        (WCAGLevel.AAA, "AAA"),
    ])
    def test_wcag_level_values(self, mock_env_vars, level, expected_value):
        """Test WCAGLevel enum values."""
        assert level.value == expected_value


class TestWCAGPrincipleEnum:
    """Tests for WCAGPrinciple enum (POUR principles)."""

    @pytest.mark.parametrize("principle,expected_value", [
        (WCAGPrinciple.PERCEIVABLE, "perceivable"),
        (WCAGPrinciple.OPERABLE, "operable"),
        (WCAGPrinciple.UNDERSTANDABLE, "understandable"),
        (WCAGPrinciple.ROBUST, "robust"),
    ])
    def test_wcag_principle_values(self, mock_env_vars, principle, expected_value):
        """Test WCAGPrinciple enum values (POUR)."""
        assert principle.value == expected_value


class TestIssueImpactEnum:
    """Tests for IssueImpact enum."""

    @pytest.mark.parametrize("impact,expected_value", [
        (IssueImpact.CRITICAL, "critical"),
        (IssueImpact.SERIOUS, "serious"),
        (IssueImpact.MODERATE, "moderate"),
        (IssueImpact.MINOR, "minor"),
    ])
    def test_issue_impact_values(self, mock_env_vars, impact, expected_value):
        """Test IssueImpact enum values."""
        assert impact.value == expected_value


class TestAccessibilityIssue:
    """Tests for AccessibilityIssue dataclass."""

    def test_accessibility_issue_creation(self, mock_env_vars):
        """Test AccessibilityIssue creation with all fields."""
        issue = AccessibilityIssue(
            id="A11Y-IMG-001",
            wcag_criterion="1.1.1",
            wcag_level=WCAGLevel.A,
            principle=WCAGPrinciple.PERCEIVABLE,
            impact=IssueImpact.CRITICAL,
            title="Image missing alt attribute",
            description="Images must have alt text",
            element="img",
            html_snippet='<img src="photo.jpg">',
            fix_suggestion='Add alt="" or descriptive text',
            affected_users=["blind", "low-vision"],
        )

        assert issue.id == "A11Y-IMG-001"
        assert issue.wcag_criterion == "1.1.1"
        assert issue.wcag_level == WCAGLevel.A
        assert issue.impact == IssueImpact.CRITICAL
        assert "blind" in issue.affected_users

    def test_accessibility_issue_to_dict(self, mock_env_vars):
        """Test AccessibilityIssue serialization."""
        issue = AccessibilityIssue(
            id="A11Y-FORM-001",
            wcag_criterion="1.3.1",
            wcag_level=WCAGLevel.A,
            principle=WCAGPrinciple.PERCEIVABLE,
            impact=IssueImpact.SERIOUS,
            title="Form input missing label",
            description="Form inputs must have labels",
            element="input",
            html_snippet='<input type="text">',
            fix_suggestion="Add a <label> element",
            affected_users=["blind", "cognitive"],
        )

        data = issue.to_dict()

        assert data["id"] == "A11Y-FORM-001"
        assert data["wcag_criterion"] == "1.3.1"
        assert data["wcag_level"] == "A"
        assert data["principle"] == "perceivable"
        assert data["impact"] == "serious"


class TestColorContrastResult:
    """Tests for ColorContrastResult dataclass."""

    def test_color_contrast_result(self, mock_env_vars):
        """Test ColorContrastResult creation."""
        result = ColorContrastResult(
            foreground="#000000",
            background="#FFFFFF",
            contrast_ratio=21.0,
            passes_aa_normal=True,
            passes_aa_large=True,
            passes_aaa_normal=True,
            passes_aaa_large=True,
        )

        assert result.contrast_ratio == 21.0
        assert result.passes_aa_normal is True
        assert result.passes_aaa_normal is True

    @pytest.mark.parametrize("ratio,passes_aa_normal,passes_aa_large", [
        (4.5, True, True),   # Meets AA for normal text
        (4.0, False, True),  # Only meets AA for large text
        (3.0, False, True),  # Large text threshold is 3:1
        (2.5, False, False), # Fails both
    ])
    def test_color_contrast_thresholds(self, mock_env_vars, ratio, passes_aa_normal, passes_aa_large):
        """Test color contrast threshold scenarios."""
        result = ColorContrastResult(
            foreground="#333",
            background="#fff",
            contrast_ratio=ratio,
            passes_aa_normal=passes_aa_normal,
            passes_aa_large=passes_aa_large,
            passes_aaa_normal=False,
            passes_aaa_large=False,
        )
        assert result.passes_aa_normal == passes_aa_normal
        assert result.passes_aa_large == passes_aa_large


class TestKeyboardNavigationResult:
    """Tests for KeyboardNavigationResult dataclass."""

    def test_keyboard_navigation_result(self, mock_env_vars):
        """Test KeyboardNavigationResult creation."""
        result = KeyboardNavigationResult(
            focusable_elements=15,
            focus_visible=True,
            focus_order_logical=True,
            keyboard_traps=[],
            skip_links_present=True,
        )

        assert result.focusable_elements == 15
        assert result.focus_visible is True
        assert len(result.keyboard_traps) == 0

    def test_keyboard_navigation_with_traps(self, mock_env_vars):
        """Test KeyboardNavigationResult with keyboard traps."""
        result = KeyboardNavigationResult(
            focusable_elements=10,
            focus_visible=False,
            focus_order_logical=False,
            keyboard_traps=["modal dialog", "date picker"],
            skip_links_present=False,
        )

        assert len(result.keyboard_traps) == 2
        assert result.focus_visible is False


class TestAccessibilityCheckResult:
    """Tests for AccessibilityCheckResult dataclass."""

    def test_accessibility_check_result(self, mock_env_vars):
        """Test AccessibilityCheckResult creation."""
        result = AccessibilityCheckResult(
            url="https://example.com",
            issues=[],
            color_contrast=[],
            keyboard_navigation=KeyboardNavigationResult(
                focusable_elements=10,
                focus_visible=True,
                focus_order_logical=True,
                keyboard_traps=[],
                skip_links_present=True,
            ),
            wcag_level_achieved=WCAGLevel.AA,
            score=95,
            summary="Page passes WCAG AA",
            recommendations=[],
        )

        assert result.url == "https://example.com"
        assert result.score == 95
        assert result.wcag_level_achieved == WCAGLevel.AA

    def test_critical_count(self, mock_env_vars):
        """Test critical issue count property."""
        issues = [
            AccessibilityIssue(
                id="1", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.CRITICAL,
                title="Critical 1", description="", element="", html_snippet="", fix_suggestion="",
            ),
            AccessibilityIssue(
                id="2", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.CRITICAL,
                title="Critical 2", description="", element="", html_snippet="", fix_suggestion="",
            ),
            AccessibilityIssue(
                id="3", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.SERIOUS,
                title="Serious 1", description="", element="", html_snippet="", fix_suggestion="",
            ),
        ]

        result = AccessibilityCheckResult(
            url="https://example.com",
            issues=issues,
            color_contrast=[],
            keyboard_navigation=KeyboardNavigationResult(0, True, True, [], False),
            wcag_level_achieved=None,
            score=50,
            summary="Issues found",
            recommendations=[],
        )

        assert result.critical_count == 2
        assert result.serious_count == 1

    def test_passes_wcag_aa_with_no_blocking_issues(self, mock_env_vars):
        """Test WCAG AA compliance check with no blocking issues."""
        result = AccessibilityCheckResult(
            url="https://example.com",
            issues=[
                AccessibilityIssue(
                    id="1", wcag_criterion="2.4.6", wcag_level=WCAGLevel.AAA,
                    principle=WCAGPrinciple.OPERABLE, impact=IssueImpact.MODERATE,
                    title="AAA issue", description="", element="", html_snippet="", fix_suggestion="",
                ),
            ],
            color_contrast=[],
            keyboard_navigation=KeyboardNavigationResult(10, True, True, [], True),
            wcag_level_achieved=WCAGLevel.AA,
            score=90,
            summary="Passes AA",
            recommendations=[],
        )

        assert result.passes_wcag_aa() is True

    def test_passes_wcag_aa_with_blocking_issues(self, mock_env_vars):
        """Test WCAG AA compliance check with blocking issues."""
        result = AccessibilityCheckResult(
            url="https://example.com",
            issues=[
                AccessibilityIssue(
                    id="1", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
                    principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.CRITICAL,
                    title="Critical A issue", description="", element="", html_snippet="", fix_suggestion="",
                ),
            ],
            color_contrast=[],
            keyboard_navigation=KeyboardNavigationResult(10, True, True, [], True),
            wcag_level_achieved=None,
            score=30,
            summary="Fails AA",
            recommendations=[],
        )

        assert result.passes_wcag_aa() is False


class TestAccessibilityCheckerAgent:
    """Tests for AccessibilityCheckerAgent class."""

    def test_agent_initialization(self, mock_env_vars):
        """Test agent initialization."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            assert agent is not None

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            prompt = agent._get_system_prompt()

            assert "WCAG" in prompt
            assert "accessibility" in prompt.lower()

    def test_check_images_finds_missing_alt(self, mock_env_vars):
        """Test image alt text checking."""
        html = '''
        <html>
        <body>
            <img src="photo1.jpg">
            <img src="photo2.jpg" alt="A nice photo">
            <img src="icon.png">
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_images(html, "https://example.com")

            # Should find 2 images missing alt
            assert len(issues) == 2
            assert all(i.wcag_criterion == "1.1.1" for i in issues)
            assert all(i.impact == IssueImpact.CRITICAL for i in issues)

    def test_check_images_no_issues(self, mock_env_vars):
        """Test image checking when all have alt text."""
        html = '''
        <html>
        <body>
            <img src="photo1.jpg" alt="Description">
            <img src="photo2.jpg" alt="">
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_images(html, "https://example.com")

            assert len(issues) == 0

    def test_check_forms_finds_missing_labels(self, mock_env_vars):
        """Test form input label checking."""
        html = '''
        <html>
        <body>
            <form>
                <input type="text" id="username">
                <label for="email">Email</label>
                <input type="email" id="email">
                <input type="password" id="password">
            </form>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_forms(html, "https://example.com")

            # username and password should be flagged (email has label)
            assert len(issues) == 2

    def test_check_forms_with_aria_label(self, mock_env_vars):
        """Test form checking with aria-label."""
        html = '''
        <html>
        <body>
            <form>
                <input type="text" id="search" aria-label="Search">
            </form>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_forms(html, "https://example.com")

            # aria-label should be acceptable
            assert len(issues) == 0

    def test_check_headings_missing_h1(self, mock_env_vars):
        """Test heading hierarchy - missing h1."""
        html = '''
        <html>
        <body>
            <h2>Section 1</h2>
            <h3>Subsection</h3>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_headings(html, "https://example.com")

            assert any("h1" in i.title.lower() for i in issues)

    def test_check_headings_skipped_level(self, mock_env_vars):
        """Test heading hierarchy - skipped levels."""
        html = '''
        <html>
        <body>
            <h1>Main Title</h1>
            <h3>Skipped to h3</h3>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_headings(html, "https://example.com")

            assert any("skip" in i.title.lower() for i in issues)

    def test_check_headings_proper_hierarchy(self, mock_env_vars):
        """Test proper heading hierarchy."""
        html = '''
        <html>
        <body>
            <h1>Main Title</h1>
            <h2>Section 1</h2>
            <h3>Subsection 1.1</h3>
            <h2>Section 2</h2>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_headings(html, "https://example.com")

            # Proper hierarchy should have no issues (except maybe missing h1 if strict)
            skip_issues = [i for i in issues if "skip" in i.title.lower()]
            assert len(skip_issues) == 0

    def test_check_aria_hidden_on_focusable(self, mock_env_vars):
        """Test ARIA - aria-hidden on focusable element."""
        # The regex checks for aria-hidden="true" followed by href/tabindex/button in the attribute string
        html = '''
        <html>
        <body>
            <div aria-hidden="true" tabindex="0">Hidden but focusable</div>
            <span aria-hidden="true" data-action="button">Another focusable</span>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_aria(html, "https://example.com")

            # The regex pattern matches aria-hidden followed by focusable indicators
            assert len(issues) > 0
            assert any(i.wcag_criterion == "4.1.2" for i in issues)

    def test_check_links_empty_link(self, mock_env_vars):
        """Test link checking - empty links."""
        html = '''
        <html>
        <body>
            <a href="/page"></a>
            <a href="/other">Valid Link</a>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_links(html, "https://example.com")

            assert any("empty" in i.title.lower() for i in issues)

    def test_check_links_generic_text(self, mock_env_vars):
        """Test link checking - generic link text."""
        html = '''
        <html>
        <body>
            <a href="/page">click here</a>
            <a href="/other">Read More</a>
        </body>
        </html>
        '''

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            issues = agent._check_links(html, "https://example.com")

            assert any("descriptive" in i.title.lower() or "generic" in i.description.lower() for i in issues)

    def test_calculate_score_no_issues(self, mock_env_vars):
        """Test score calculation with no issues."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            score = agent._calculate_score([])
            assert score == 100

    @pytest.mark.parametrize("impact,expected_deduction", [
        (IssueImpact.CRITICAL, 25),
        (IssueImpact.SERIOUS, 15),
        (IssueImpact.MODERATE, 5),
        (IssueImpact.MINOR, 2),
    ])
    def test_calculate_score_single_issue(self, mock_env_vars, impact, expected_deduction):
        """Test score calculation with single issue of each severity."""
        issue = AccessibilityIssue(
            id="1", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
            principle=WCAGPrinciple.PERCEIVABLE, impact=impact,
            title="Test", description="", element="", html_snippet="", fix_suggestion="",
        )

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            score = agent._calculate_score([issue])
            assert score == 100 - expected_deduction

    def test_calculate_score_multiple_issues(self, mock_env_vars):
        """Test score calculation with multiple issues."""
        issues = [
            AccessibilityIssue(
                id="1", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.CRITICAL,
                title="Critical", description="", element="", html_snippet="", fix_suggestion="",
            ),
            AccessibilityIssue(
                id="2", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.SERIOUS,
                title="Serious", description="", element="", html_snippet="", fix_suggestion="",
            ),
        ]

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            score = agent._calculate_score(issues)
            assert score == 100 - 25 - 15  # 60

    def test_determine_wcag_level_no_issues(self, mock_env_vars):
        """Test WCAG level determination with no issues."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            level = agent._determine_wcag_level([])
            assert level == WCAGLevel.AAA

    def test_determine_wcag_level_with_a_failures(self, mock_env_vars):
        """Test WCAG level determination with Level A failures."""
        issues = [
            AccessibilityIssue(
                id="1", wcag_criterion="1.1.1", wcag_level=WCAGLevel.A,
                principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.CRITICAL,
                title="A issue", description="", element="", html_snippet="", fix_suggestion="",
            ),
        ]

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            level = agent._determine_wcag_level(issues)
            assert level is None  # Doesn't meet Level A

    def test_determine_wcag_level_meets_a_not_aa(self, mock_env_vars):
        """Test WCAG level determination - meets A but not AA."""
        issues = [
            AccessibilityIssue(
                id="1", wcag_criterion="1.4.3", wcag_level=WCAGLevel.AA,
                principle=WCAGPrinciple.PERCEIVABLE, impact=IssueImpact.SERIOUS,
                title="AA issue", description="", element="", html_snippet="", fix_suggestion="",
            ),
        ]

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            level = agent._determine_wcag_level(issues)
            assert level == WCAGLevel.A

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars):
        """Test successful accessibility check execution."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = AccessibilityCheckerAgent()
            agent._fetch_page = AsyncMock(return_value="<html><body><h1>Test</h1></body></html>")

            mock_ai_response = {
                "content": '{"summary": "Page is accessible", "recommendations": ["Add skip links"]}'
            }

            with patch.object(agent, '_call_model', new_callable=AsyncMock) as mock_model:
                mock_model.return_value = mock_ai_response

                result = await agent.execute(
                    url="https://example.com",
                    wcag_level=WCAGLevel.AA,
                )

                assert result.success is True
                assert result.data is not None

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_env_vars):
        """Test accessibility check execution failure."""
        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            agent = AccessibilityCheckerAgent()
            agent._fetch_page = AsyncMock(side_effect=Exception("Network error"))

            result = await agent.execute(url="https://example.com")

            assert result.success is False
            assert result.error is not None
