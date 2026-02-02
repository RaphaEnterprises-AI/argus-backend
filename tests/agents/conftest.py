"""Shared fixtures for agent tests.

This module provides reusable fixtures for testing all agents in the
e2e-testing-agent project. Fixtures follow the patterns established
in tests/conftest.py.

Fixtures included:
- mock_model_router: Mocked ModelRouter for testing AI routing
- sample_agent_config: Standard agent configuration
- mock_browser_pool_client: Mocked BrowserPoolClient for browser automation
- sample_test_spec_ui: UI test specification
- sample_file_changes: File changes for MR analysis
- sample_vulnerabilities: Security vulnerabilities
- sample_accessibility_issues: Accessibility issues
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Model Router Fixtures
# =============================================================================


@pytest.fixture
def mock_model_router():
    """Create a mocked ModelRouter for testing.

    This fixture provides a mock ModelRouter that can be used to test
    agents without making actual API calls. It simulates the complete()
    method with customizable responses.

    Example:
        async def test_agent_with_router(mock_model_router):
            mock_model_router.complete.return_value = {
                "content": "Custom response",
                "model": "test-model",
                ...
            }
            agent = MyAgent()
            agent._model_router = mock_model_router
            result = await agent.execute(...)
    """
    mock_router = MagicMock()
    mock_router.complete = AsyncMock(return_value={
        "content": '{"result": "success"}',
        "model": "claude-sonnet-4-5-20250514",
        "model_name": "sonnet",
        "provider": "anthropic",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost": 0.001,
        "latency_ms": 500,
    })
    mock_router.get_model_for_task = MagicMock(return_value={
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250514",
        "tier": "standard",
    })
    mock_router.estimate_cost = MagicMock(return_value=0.001)
    mock_router.get_budget_status = AsyncMock(return_value={
        "remaining": 100.0,
        "used": 0.0,
        "limit": 100.0,
    })
    return mock_router


@pytest.fixture
def mock_model_router_with_json_response():
    """Create a mocked ModelRouter with customizable JSON responses.

    Returns a factory function that creates mock routers with specific responses.

    Example:
        def test_something(mock_model_router_with_json_response):
            router = mock_model_router_with_json_response({"status": "ok"})
            agent._model_router = router
    """
    def _create_router(json_response: dict):
        import json
        mock_router = MagicMock()
        mock_router.complete = AsyncMock(return_value={
            "content": json.dumps(json_response),
            "model": "claude-sonnet-4-5-20250514",
            "model_name": "sonnet",
            "provider": "anthropic",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.001,
            "latency_ms": 500,
        })
        return mock_router
    return _create_router


# =============================================================================
# Agent Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_agent_config():
    """Create a standard agent configuration for testing.

    Returns an AgentConfig with sensible defaults for testing.
    Uses ModelName.SONNET as the default model.
    """
    from src.config import AgentConfig, ModelName

    return AgentConfig(
        name="test_agent",
        model=ModelName.SONNET,
        max_tokens=4096,
        temperature=0.0,
        timeout_seconds=60,
        max_retries=3,
        retry_delay=0.1,  # Fast retries for tests
    )


@pytest.fixture
def sample_agent_config_haiku():
    """Create an agent configuration using the Haiku model.

    Useful for testing cost-optimized scenarios.
    """
    from src.config import AgentConfig, ModelName

    return AgentConfig(
        name="test_agent_haiku",
        model=ModelName.HAIKU,
        max_tokens=2048,
        temperature=0.0,
        timeout_seconds=30,
        max_retries=2,
        retry_delay=0.1,
    )


@pytest.fixture
def sample_agent_config_opus():
    """Create an agent configuration using the Opus model.

    Useful for testing complex reasoning scenarios.
    """
    from src.config import AgentConfig, ModelName

    return AgentConfig(
        name="test_agent_opus",
        model=ModelName.OPUS,
        max_tokens=8192,
        temperature=0.0,
        timeout_seconds=180,
        max_retries=2,
        retry_delay=0.5,
    )


# =============================================================================
# Browser Pool Client Fixtures
# =============================================================================


@pytest.fixture
def mock_browser_pool_client():
    """Create a mocked BrowserPoolClient for browser automation testing.

    This fixture provides a comprehensive mock of the BrowserPoolClient
    with all common methods mocked for testing UI automation agents.

    Mocked methods:
    - observe(): Returns mock element observations
    - act(): Executes browser actions
    - extract(): Extracts data from pages
    - screenshot(): Takes screenshots
    - navigate(): Navigates to URLs
    - close(): Closes browser sessions
    - health_check(): Returns pool health status

    Example:
        async def test_ui_agent(mock_browser_pool_client):
            agent = UITesterAgent(browser_client=mock_browser_pool_client)
            result = await agent.execute(test_spec)
    """
    mock_client = AsyncMock()

    # Mock observe method - returns element discovery results
    mock_client.observe = AsyncMock(return_value={
        "elements": [
            {
                "selector": "#login-btn",
                "type": "button",
                "text": "Login",
                "visible": True,
                "clickable": True,
            },
            {
                "selector": "#email",
                "type": "input",
                "text": "",
                "visible": True,
                "placeholder": "Enter email",
            },
            {
                "selector": "#password",
                "type": "input",
                "text": "",
                "visible": True,
                "placeholder": "Enter password",
            },
        ],
        "url": "https://example.com/login",
        "title": "Login Page",
    })

    # Mock act method - executes browser actions
    mock_client.act = AsyncMock(return_value={
        "success": True,
        "action": "click",
        "selector": "#login-btn",
        "duration_ms": 150,
    })

    # Mock extract method - extracts data from page
    mock_client.extract = AsyncMock(return_value={
        "data": {"title": "Dashboard", "user": "test@example.com"},
        "success": True,
    })

    # Mock screenshot method - returns base64 screenshot
    mock_client.screenshot = AsyncMock(return_value=b"fake_screenshot_data")

    # Mock navigate method
    mock_client.navigate = AsyncMock(return_value={
        "success": True,
        "url": "https://example.com/dashboard",
        "status_code": 200,
    })

    # Mock close method
    mock_client.close = AsyncMock()

    # Mock health_check method
    mock_client.health_check = AsyncMock(return_value={
        "healthy": True,
        "available_browsers": 3,
        "active_sessions": 1,
    })

    # Mock run_test method for full test execution
    mock_client.run_test = AsyncMock(return_value={
        "success": True,
        "steps_passed": 4,
        "steps_failed": 0,
        "duration_ms": 2500,
        "screenshots": [],
    })

    # Support async context manager
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client


@pytest.fixture
def mock_browser_pool_client_failure():
    """Create a BrowserPoolClient mock that simulates failures.

    Useful for testing error handling and recovery scenarios.
    """
    mock_client = AsyncMock()

    mock_client.observe = AsyncMock(side_effect=Exception("Browser pool unavailable"))
    mock_client.act = AsyncMock(side_effect=Exception("Element not found"))
    mock_client.navigate = AsyncMock(side_effect=Exception("Navigation timeout"))
    mock_client.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
    mock_client.health_check = AsyncMock(return_value={
        "healthy": False,
        "available_browsers": 0,
        "error": "Pool exhausted",
    })

    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client


# =============================================================================
# Test Specification Fixtures
# =============================================================================


@pytest.fixture
def sample_test_spec_ui():
    """Create a comprehensive UI test specification.

    This fixture provides a realistic UI test specification for testing
    UI automation agents. It includes multiple steps and assertions.
    """
    return {
        "id": "ui-test-001",
        "name": "User Login Flow Test",
        "type": "ui",
        "priority": "high",
        "description": "Test the complete user login flow including error handling",
        "tags": ["auth", "login", "smoke"],
        "steps": [
            {
                "action": "goto",
                "target": "/login",
                "value": None,
                "timeout": 10000,
                "description": "Navigate to login page",
            },
            {
                "action": "wait_for_selector",
                "target": "#login-form",
                "value": None,
                "timeout": 5000,
                "description": "Wait for login form to load",
            },
            {
                "action": "fill",
                "target": "#email",
                "value": "test@example.com",
                "timeout": 5000,
                "description": "Enter email address",
            },
            {
                "action": "fill",
                "target": "#password",
                "value": "SecurePassword123!",
                "timeout": 5000,
                "description": "Enter password",
            },
            {
                "action": "click",
                "target": "#submit-btn",
                "value": None,
                "timeout": 5000,
                "description": "Click login button",
            },
            {
                "action": "wait_for_navigation",
                "target": None,
                "value": None,
                "timeout": 10000,
                "description": "Wait for navigation to complete",
            },
        ],
        "assertions": [
            {
                "type": "url_contains",
                "target": None,
                "expected": "/dashboard",
                "description": "Should redirect to dashboard",
            },
            {
                "type": "element_visible",
                "target": ".welcome-message",
                "expected": None,
                "description": "Welcome message should be visible",
            },
            {
                "type": "text_contains",
                "target": ".user-name",
                "expected": "test@example.com",
                "description": "Username should be displayed",
            },
        ],
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "created_by": "test_fixture",
            "estimated_duration_ms": 15000,
        },
    }


@pytest.fixture
def sample_test_spec_api():
    """Create an API test specification.

    This fixture provides an API test specification for testing
    API testing agents.
    """
    return {
        "id": "api-test-001",
        "name": "User API CRUD Test",
        "type": "api",
        "priority": "high",
        "description": "Test user API endpoints",
        "steps": [
            {
                "action": "post",
                "target": "/api/v1/users",
                "headers": {"Content-Type": "application/json"},
                "body": {"email": "newuser@example.com", "name": "New User"},
            },
            {
                "action": "get",
                "target": "/api/v1/users/{user_id}",
                "headers": {"Content-Type": "application/json"},
            },
        ],
        "assertions": [
            {"type": "status_code", "expected": 201},
            {"type": "response_json_path", "path": "$.email", "expected": "newuser@example.com"},
        ],
    }


# =============================================================================
# File Change Fixtures (for MR Analysis)
# =============================================================================


@pytest.fixture
def sample_file_changes():
    """Create sample file changes for MR/PR analysis.

    This fixture provides realistic file change data as would be
    received from GitHub or GitLab APIs.
    """
    return [
        {
            "path": "src/auth/login.py",
            "status": "modified",
            "additions": 25,
            "deletions": 10,
            "patch": """@@ -15,10 +15,25 @@ def login(username: str, password: str):
-    if not validate_password(password):
-        raise AuthError("Invalid password")
+    # Enhanced password validation with rate limiting
+    if not validate_password_with_checks(password, username):
+        log_failed_attempt(username)
+        raise AuthError("Invalid credentials")
+
+    # Add MFA check
+    if user.mfa_enabled:
+        return require_mfa(user)
     return generate_token(user)""",
            "previous_path": None,
        },
        {
            "path": "src/auth/mfa.py",
            "status": "added",
            "additions": 85,
            "deletions": 0,
            "patch": """@@ -0,0 +1,85 @@
+\"\"\"Multi-factor authentication module.\"\"\"
+
+import pyotp
+from .models import User
+
+def generate_mfa_secret(user: User) -> str:
+    \"\"\"Generate MFA secret for user.\"\"\"
+    return pyotp.random_base32()
+
+def verify_mfa_token(user: User, token: str) -> bool:
+    \"\"\"Verify MFA token.\"\"\"
+    totp = pyotp.TOTP(user.mfa_secret)
+    return totp.verify(token)""",
            "previous_path": None,
        },
        {
            "path": "tests/test_login.py",
            "status": "modified",
            "additions": 45,
            "deletions": 5,
            "patch": """@@ -20,5 +20,50 @@ def test_login_success():
+def test_login_with_mfa():
+    \"\"\"Test login with MFA enabled.\"\"\"
+    user = create_test_user(mfa_enabled=True)
+    result = login(user.email, "password")
+    assert result.requires_mfa is True""",
            "previous_path": None,
        },
        {
            "path": "config/settings.yaml",
            "status": "modified",
            "additions": 3,
            "deletions": 0,
            "patch": """@@ -10,0 +11,3 @@
+mfa:
+  enabled: true
+  issuer: "MyApp\"""",
            "previous_path": None,
        },
        {
            "path": "docs/auth.md",
            "status": "modified",
            "additions": 20,
            "deletions": 2,
            "patch": """@@ -1,2 +1,20 @@
-# Authentication
+# Authentication Guide
+
+## Multi-Factor Authentication
+
+MFA can be enabled per user...""",
            "previous_path": None,
        },
    ]


@pytest.fixture
def sample_file_changes_simple():
    """Create simple file changes for basic testing."""
    return [
        {
            "path": "src/utils/helpers.py",
            "status": "modified",
            "additions": 5,
            "deletions": 2,
            "patch": "@@ -1,2 +1,5 @@\n+# New helper function\n+def format_date(d): return d.isoformat()",
            "previous_path": None,
        },
    ]


@pytest.fixture
def sample_file_changes_refactor():
    """Create file changes representing a refactoring operation."""
    return [
        {
            "path": "src/services/user_service.py",
            "status": "renamed",
            "additions": 10,
            "deletions": 10,
            "patch": "@@ -1,10 +1,10 @@\n class UserService:\n     pass",
            "previous_path": "src/services/user.py",
        },
        {
            "path": "src/services/__init__.py",
            "status": "modified",
            "additions": 1,
            "deletions": 1,
            "patch": "@@ -1 +1 @@\n-from .user import UserService\n+from .user_service import UserService",
            "previous_path": None,
        },
    ]


# =============================================================================
# Security Vulnerability Fixtures
# =============================================================================


@pytest.fixture
def sample_vulnerabilities():
    """Create sample security vulnerabilities for testing.

    This fixture provides realistic vulnerability data following
    OWASP Top 10 categories and CVSS scoring.
    """
    from src.agents.security_scanner import (
        Vulnerability,
        VulnerabilityCategory,
        VulnerabilitySeverity,
    )

    return [
        Vulnerability(
            id="VULN-001",
            category=VulnerabilityCategory.XSS,
            severity=VulnerabilitySeverity.HIGH,
            title="Reflected XSS in search parameter",
            description="User input in the 'q' parameter is reflected without sanitization",
            location="https://example.com/search?q=<script>",
            evidence="<script>alert(1)</script> was reflected in response",
            cvss_score=7.5,
            cwe_id="CWE-79",
            remediation="Sanitize user input and use Content-Security-Policy headers",
            references=["https://owasp.org/www-community/attacks/xss/"],
            false_positive_likelihood=0.1,
        ),
        Vulnerability(
            id="VULN-002",
            category=VulnerabilityCategory.INJECTION,
            severity=VulnerabilitySeverity.CRITICAL,
            title="SQL Injection in user lookup",
            description="SQL injection vulnerability in user search endpoint",
            location="https://example.com/api/users?id=1",
            evidence="Error: You have an error in your SQL syntax",
            cvss_score=9.8,
            cwe_id="CWE-89",
            remediation="Use parameterized queries or ORM",
            references=["https://owasp.org/www-community/attacks/SQL_Injection"],
            false_positive_likelihood=0.05,
        ),
        Vulnerability(
            id="VULN-003",
            category=VulnerabilityCategory.SENSITIVE_DATA,
            severity=VulnerabilitySeverity.MEDIUM,
            title="Missing HTTPS redirect",
            description="Application does not enforce HTTPS",
            location="http://example.com/",
            evidence="HTTP requests are not redirected to HTTPS",
            cvss_score=5.3,
            cwe_id="CWE-319",
            remediation="Configure Strict-Transport-Security header and HTTPS redirect",
            references=[],
            false_positive_likelihood=0.0,
        ),
        Vulnerability(
            id="VULN-004",
            category=VulnerabilityCategory.SECURITY_MISCONFIG,
            severity=VulnerabilitySeverity.LOW,
            title="Server version disclosure",
            description="Server header reveals version information",
            location="https://example.com/",
            evidence="Server: nginx/1.18.0",
            cvss_score=2.1,
            cwe_id="CWE-200",
            remediation="Remove or obfuscate server version headers",
            references=[],
            false_positive_likelihood=0.0,
        ),
        Vulnerability(
            id="VULN-005",
            category=VulnerabilityCategory.BROKEN_AUTH,
            severity=VulnerabilitySeverity.HIGH,
            title="Weak password policy",
            description="Application accepts weak passwords",
            location="https://example.com/register",
            evidence="Password '123456' was accepted",
            cvss_score=7.0,
            cwe_id="CWE-521",
            remediation="Implement strong password requirements",
            references=[],
            false_positive_likelihood=0.0,
        ),
    ]


@pytest.fixture
def sample_vulnerability_critical():
    """Create a single critical vulnerability for focused testing."""
    from src.agents.security_scanner import (
        Vulnerability,
        VulnerabilityCategory,
        VulnerabilitySeverity,
    )

    return Vulnerability(
        id="VULN-CRITICAL-001",
        category=VulnerabilityCategory.INSECURE_DESERIAL,
        severity=VulnerabilitySeverity.CRITICAL,
        title="Remote Code Execution via unsafe deserialization",
        description="Untrusted data is deserialized without validation",
        location="https://example.com/api/import",
        evidence="Arbitrary code execution confirmed via unsafe deserialization",
        cvss_score=10.0,
        cwe_id="CWE-502",
        remediation="Never deserialize untrusted data; use safe formats like JSON",
        references=["https://owasp.org/www-project-web-security-testing-guide/"],
        false_positive_likelihood=0.0,
    )


@pytest.fixture
def sample_security_headers_missing():
    """Create security headers object with common missing headers."""
    from src.agents.security_scanner import SecurityHeaders

    return SecurityHeaders(
        content_security_policy=None,
        x_frame_options=None,
        x_content_type_options="nosniff",
        strict_transport_security=None,
        x_xss_protection="1; mode=block",
        referrer_policy=None,
        permissions_policy=None,
    )


@pytest.fixture
def sample_security_headers_complete():
    """Create security headers object with all headers present."""
    from src.agents.security_scanner import SecurityHeaders

    return SecurityHeaders(
        content_security_policy="default-src 'self'; script-src 'self'",
        x_frame_options="DENY",
        x_content_type_options="nosniff",
        strict_transport_security="max-age=31536000; includeSubDomains",
        x_xss_protection="1; mode=block",
        referrer_policy="strict-origin-when-cross-origin",
        permissions_policy="geolocation=(), microphone=()",
    )


# =============================================================================
# Accessibility Issue Fixtures
# =============================================================================


@pytest.fixture
def sample_accessibility_issues():
    """Create sample accessibility issues for testing.

    This fixture provides realistic WCAG-based accessibility issues
    covering various principles and impact levels.
    """
    from src.agents.accessibility_checker import (
        AccessibilityIssue,
        IssueImpact,
        WCAGLevel,
        WCAGPrinciple,
    )

    return [
        AccessibilityIssue(
            id="A11Y-IMG-001",
            wcag_criterion="1.1.1",
            wcag_level=WCAGLevel.A,
            principle=WCAGPrinciple.PERCEIVABLE,
            impact=IssueImpact.CRITICAL,
            title="Image missing alt attribute",
            description="Images must have alternative text for screen reader users",
            element="img",
            html_snippet='<img src="hero-banner.jpg">',
            fix_suggestion='Add alt="Description of the image" or alt="" for decorative images',
            affected_users=["blind", "low-vision"],
        ),
        AccessibilityIssue(
            id="A11Y-FORM-001",
            wcag_criterion="1.3.1",
            wcag_level=WCAGLevel.A,
            principle=WCAGPrinciple.PERCEIVABLE,
            impact=IssueImpact.SERIOUS,
            title="Form input missing label",
            description="Form inputs must have associated labels",
            element="input",
            html_snippet='<input type="text" id="username" placeholder="Username">',
            fix_suggestion='Add <label for="username">Username</label> before the input',
            affected_users=["blind", "cognitive"],
        ),
        AccessibilityIssue(
            id="A11Y-COLOR-001",
            wcag_criterion="1.4.3",
            wcag_level=WCAGLevel.AA,
            principle=WCAGPrinciple.PERCEIVABLE,
            impact=IssueImpact.SERIOUS,
            title="Insufficient color contrast",
            description="Text has insufficient contrast ratio (2.5:1, needs 4.5:1)",
            element="p",
            html_snippet='<p style="color: #999;">Light gray text</p>',
            fix_suggestion="Use a darker color (#666 or darker) to meet 4.5:1 ratio",
            affected_users=["low-vision", "color-blind"],
        ),
        AccessibilityIssue(
            id="A11Y-FOCUS-001",
            wcag_criterion="2.4.7",
            wcag_level=WCAGLevel.AA,
            principle=WCAGPrinciple.OPERABLE,
            impact=IssueImpact.MODERATE,
            title="Focus indicator not visible",
            description="Interactive elements must have visible focus indicators",
            element="button",
            html_snippet='<button style="outline: none;">Click me</button>',
            fix_suggestion="Remove outline:none or provide custom visible focus styles",
            affected_users=["keyboard-users", "cognitive"],
        ),
        AccessibilityIssue(
            id="A11Y-HEADING-001",
            wcag_criterion="1.3.1",
            wcag_level=WCAGLevel.A,
            principle=WCAGPrinciple.PERCEIVABLE,
            impact=IssueImpact.MODERATE,
            title="Skipped heading level",
            description="Heading levels should not be skipped (h1 to h3)",
            element="h3",
            html_snippet="<h1>Title</h1><h3>Subsection</h3>",
            fix_suggestion="Use h2 instead of h3 to maintain proper heading hierarchy",
            affected_users=["blind", "cognitive"],
        ),
        AccessibilityIssue(
            id="A11Y-LINK-001",
            wcag_criterion="2.4.4",
            wcag_level=WCAGLevel.A,
            principle=WCAGPrinciple.OPERABLE,
            impact=IssueImpact.MINOR,
            title="Non-descriptive link text",
            description="Link text should describe the destination",
            element="a",
            html_snippet='<a href="/products">Click here</a>',
            fix_suggestion='Use descriptive text like "View our products"',
            affected_users=["blind", "cognitive"],
        ),
    ]


@pytest.fixture
def sample_accessibility_issue_critical():
    """Create a single critical accessibility issue."""
    from src.agents.accessibility_checker import (
        AccessibilityIssue,
        IssueImpact,
        WCAGLevel,
        WCAGPrinciple,
    )

    return AccessibilityIssue(
        id="A11Y-CRITICAL-001",
        wcag_criterion="2.1.1",
        wcag_level=WCAGLevel.A,
        principle=WCAGPrinciple.OPERABLE,
        impact=IssueImpact.CRITICAL,
        title="Keyboard trap in modal dialog",
        description="Users cannot escape the modal using keyboard navigation",
        element="div",
        html_snippet='<div class="modal" tabindex="0">...<button onclick="close()">X</button></div>',
        fix_suggestion="Implement proper focus management and allow Escape key to close modal",
        affected_users=["keyboard-users", "blind", "motor-impaired"],
    )


@pytest.fixture
def sample_color_contrast_results():
    """Create sample color contrast analysis results."""
    from src.agents.accessibility_checker import ColorContrastResult

    return [
        ColorContrastResult(
            foreground="#000000",
            background="#FFFFFF",
            contrast_ratio=21.0,
            passes_aa_normal=True,
            passes_aa_large=True,
            passes_aaa_normal=True,
            passes_aaa_large=True,
        ),
        ColorContrastResult(
            foreground="#767676",
            background="#FFFFFF",
            contrast_ratio=4.54,
            passes_aa_normal=True,
            passes_aa_large=True,
            passes_aaa_normal=False,
            passes_aaa_large=True,
        ),
        ColorContrastResult(
            foreground="#999999",
            background="#FFFFFF",
            contrast_ratio=2.85,
            passes_aa_normal=False,
            passes_aa_large=False,
            passes_aaa_normal=False,
            passes_aaa_large=False,
        ),
    ]


@pytest.fixture
def sample_keyboard_navigation_result():
    """Create sample keyboard navigation analysis result."""
    from src.agents.accessibility_checker import KeyboardNavigationResult

    return KeyboardNavigationResult(
        focusable_elements=15,
        focus_visible=True,
        focus_order_logical=True,
        keyboard_traps=[],
        skip_links_present=True,
    )


@pytest.fixture
def sample_keyboard_navigation_result_with_issues():
    """Create keyboard navigation result with accessibility issues."""
    from src.agents.accessibility_checker import KeyboardNavigationResult

    return KeyboardNavigationResult(
        focusable_elements=10,
        focus_visible=False,
        focus_order_logical=False,
        keyboard_traps=["modal-dialog", "date-picker"],
        skip_links_present=False,
    )


# =============================================================================
# Performance Analysis Fixtures
# =============================================================================


@pytest.fixture
def sample_performance_metrics():
    """Create sample performance metrics for testing."""
    return {
        "url": "https://example.com/",
        "load_time_ms": 2500,
        "first_contentful_paint_ms": 800,
        "largest_contentful_paint_ms": 1500,
        "time_to_interactive_ms": 2000,
        "total_blocking_time_ms": 300,
        "cumulative_layout_shift": 0.05,
        "speed_index": 1200,
        "dom_content_loaded_ms": 1000,
        "requests_count": 45,
        "total_transfer_size_kb": 1200,
        "javascript_size_kb": 450,
        "css_size_kb": 120,
        "image_size_kb": 580,
        "font_size_kb": 50,
    }


@pytest.fixture
def sample_performance_issues():
    """Create sample performance issues for testing."""
    return [
        {
            "id": "PERF-001",
            "severity": "high",
            "title": "Large JavaScript bundle",
            "description": "Main bundle is 450KB (should be under 200KB)",
            "impact": "Increases time to interactive",
            "recommendation": "Code split the bundle and lazy load non-critical code",
        },
        {
            "id": "PERF-002",
            "severity": "medium",
            "title": "Unoptimized images",
            "description": "Several images are not using modern formats",
            "impact": "Increases page load time",
            "recommendation": "Convert images to WebP or AVIF format",
        },
        {
            "id": "PERF-003",
            "severity": "low",
            "title": "Missing resource hints",
            "description": "No preconnect hints for third-party domains",
            "impact": "Delayed resource loading",
            "recommendation": "Add <link rel='preconnect'> for critical domains",
        },
    ]


# =============================================================================
# Test Result Fixtures
# =============================================================================


@pytest.fixture
def sample_test_failure():
    """Create a sample test failure for self-healing testing."""
    return {
        "test_id": "ui-test-001",
        "test_name": "Login Flow Test",
        "status": "failed",
        "failure_type": "element_not_found",
        "error_message": "Element '#submit-btn' not found after 5000ms timeout",
        "step_index": 4,
        "step_action": "click",
        "step_target": "#submit-btn",
        "screenshot": b"fake_failure_screenshot",
        "dom_snapshot": "<html><body><button id='login-button'>Sign In</button></body></html>",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_test_failure_timing():
    """Create a timing-related test failure."""
    return {
        "test_id": "ui-test-002",
        "test_name": "Dashboard Load Test",
        "status": "failed",
        "failure_type": "timeout",
        "error_message": "Timeout waiting for element '.dashboard-content' after 10000ms",
        "step_index": 2,
        "step_action": "wait_for_selector",
        "step_target": ".dashboard-content",
        "screenshot": None,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_test_failure_assertion():
    """Create an assertion failure."""
    return {
        "test_id": "ui-test-003",
        "test_name": "User Profile Test",
        "status": "failed",
        "failure_type": "assertion_failed",
        "error_message": "Expected URL to contain '/profile', but got '/account/settings'",
        "assertion_type": "url_contains",
        "expected": "/profile",
        "actual": "/account/settings",
        "screenshot": b"fake_assertion_screenshot",
        "timestamp": datetime.now(UTC).isoformat(),
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def mock_cognee_client():
    """Create a mocked Cognee client for knowledge graph operations."""
    mock_client = AsyncMock()
    mock_client.put = AsyncMock()
    mock_client.get = AsyncMock(return_value=None)
    mock_client.semantic_search = AsyncMock(return_value=[])
    mock_client.delete = AsyncMock()
    return mock_client


@pytest.fixture
def mock_event_producer():
    """Create a mocked Kafka event producer."""
    mock_producer = AsyncMock()
    mock_producer.send = AsyncMock()
    mock_producer.flush = AsyncMock()
    mock_producer.close = AsyncMock()
    return mock_producer


@pytest.fixture
def sample_html_page():
    """Create a sample HTML page for accessibility/security testing."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Page</title>
</head>
<body>
    <header>
        <nav>
            <a href="/">Home</a>
            <a href="/about">About</a>
            <a href="/contact">Contact</a>
        </nav>
    </header>
    <main>
        <h1>Welcome to Test Page</h1>
        <img src="hero.jpg">
        <form id="login-form">
            <input type="text" id="username" placeholder="Username">
            <input type="password" id="password" placeholder="Password">
            <button type="submit" id="submit-btn">Login</button>
        </form>
    </main>
    <footer>
        <p>&copy; 2025 Test Company</p>
    </footer>
</body>
</html>
"""


@pytest.fixture
def sample_html_page_accessible():
    """Create a fully accessible HTML page."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Accessible Test Page</title>
</head>
<body>
    <a href="#main" class="skip-link">Skip to main content</a>
    <header>
        <nav aria-label="Main navigation">
            <ul>
                <li><a href="/">Home</a></li>
                <li><a href="/about">About Us</a></li>
                <li><a href="/contact">Contact Us</a></li>
            </ul>
        </nav>
    </header>
    <main id="main">
        <h1>Welcome to Our Accessible Test Page</h1>
        <img src="hero.jpg" alt="A beautiful landscape showing mountains at sunset">
        <form id="login-form" aria-labelledby="form-title">
            <h2 id="form-title">Login Form</h2>
            <div>
                <label for="username">Username</label>
                <input type="text" id="username" required>
            </div>
            <div>
                <label for="password">Password</label>
                <input type="password" id="password" required>
            </div>
            <button type="submit">Login</button>
        </form>
    </main>
    <footer>
        <p>&copy; 2025 Test Company</p>
    </footer>
</body>
</html>
"""
