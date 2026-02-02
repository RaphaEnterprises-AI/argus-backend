"""Tests for Security Scanner Agent - comprehensive OWASP Top 10 vulnerability testing.

Following industry best practices:
- Parametrization for edge cases
- Proper mocking for external dependencies
- Async testing with pytest-asyncio
- Clear test naming with docstrings
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.security_scanner import (
    VulnerabilitySeverity,
    VulnerabilityCategory,
    Vulnerability,
    SecurityHeaders,
    SecurityScanResult,
    SecurityScannerAgent,
)


class TestVulnerabilitySeverityEnum:
    """Tests for VulnerabilitySeverity enum (CVSS-based)."""

    @pytest.mark.parametrize("severity,expected_value", [
        (VulnerabilitySeverity.CRITICAL, "critical"),
        (VulnerabilitySeverity.HIGH, "high"),
        (VulnerabilitySeverity.MEDIUM, "medium"),
        (VulnerabilitySeverity.LOW, "low"),
        (VulnerabilitySeverity.INFO, "info"),
    ])
    def test_vulnerability_severity_values(self, mock_env_vars, severity, expected_value):
        """Test VulnerabilitySeverity enum values."""
        assert severity.value == expected_value

    def test_vulnerability_severity_is_string_enum(self, mock_env_vars):
        """Test that VulnerabilitySeverity is a string enum."""
        assert isinstance(VulnerabilitySeverity.CRITICAL, str)
        assert isinstance(VulnerabilitySeverity.HIGH, str)


class TestVulnerabilityCategoryEnum:
    """Tests for VulnerabilityCategory enum (OWASP Top 10)."""

    @pytest.mark.parametrize("category,expected_value", [
        (VulnerabilityCategory.INJECTION, "A03:2021-Injection"),
        (VulnerabilityCategory.BROKEN_AUTH, "A07:2021-Identification and Authentication Failures"),
        (VulnerabilityCategory.SENSITIVE_DATA, "A02:2021-Cryptographic Failures"),
        (VulnerabilityCategory.XXE, "A05:2021-Security Misconfiguration"),
        (VulnerabilityCategory.BROKEN_ACCESS, "A01:2021-Broken Access Control"),
        (VulnerabilityCategory.SECURITY_MISCONFIG, "A05:2021-Security Misconfiguration"),
        (VulnerabilityCategory.XSS, "A03:2021-Injection"),
        (VulnerabilityCategory.INSECURE_DESERIAL, "A08:2021-Software and Data Integrity Failures"),
        (VulnerabilityCategory.VULNERABLE_COMPONENTS, "A06:2021-Vulnerable and Outdated Components"),
        (VulnerabilityCategory.LOGGING, "A09:2021-Security Logging and Monitoring Failures"),
        (VulnerabilityCategory.SSRF, "A10:2021-Server-Side Request Forgery"),
    ])
    def test_vulnerability_category_values(self, mock_env_vars, category, expected_value):
        """Test VulnerabilityCategory enum values match OWASP 2021."""
        assert category.value == expected_value

    def test_vulnerability_category_is_string_enum(self, mock_env_vars):
        """Test that VulnerabilityCategory is a string enum."""
        assert isinstance(VulnerabilityCategory.INJECTION, str)
        assert isinstance(VulnerabilityCategory.XSS, str)


class TestVulnerability:
    """Tests for Vulnerability dataclass."""

    def test_vulnerability_creation(self, mock_env_vars):
        """Test Vulnerability creation with all fields."""
        vuln = Vulnerability(
            id="SEC-XSS-001",
            category=VulnerabilityCategory.XSS,
            severity=VulnerabilitySeverity.HIGH,
            title="Cross-Site Scripting in Search Field",
            description="User input is reflected without sanitization",
            location="https://example.com/search",
            evidence="<script>alert(1)</script> reflected in response",
            cvss_score=7.5,
            cwe_id="CWE-79",
            remediation="Sanitize all user input before rendering",
            references=["https://owasp.org/www-community/xss-filter-evasion-cheatsheet"],
            false_positive_likelihood=0.1,
        )

        assert vuln.id == "SEC-XSS-001"
        assert vuln.category == VulnerabilityCategory.XSS
        assert vuln.severity == VulnerabilitySeverity.HIGH
        assert vuln.cvss_score == 7.5
        assert vuln.cwe_id == "CWE-79"
        assert len(vuln.references) == 1
        assert vuln.false_positive_likelihood == 0.1

    def test_vulnerability_creation_with_defaults(self, mock_env_vars):
        """Test Vulnerability creation with default values."""
        vuln = Vulnerability(
            id="SEC-HDR-001",
            category=VulnerabilityCategory.SECURITY_MISCONFIG,
            severity=VulnerabilitySeverity.LOW,
            title="Missing X-Frame-Options",
            description="X-Frame-Options header not set",
            location="https://example.com",
            evidence="Header not present in response",
        )

        assert vuln.cvss_score == 0.0
        assert vuln.cwe_id is None
        assert vuln.remediation == ""
        assert vuln.references == []
        assert vuln.false_positive_likelihood == 0.0

    def test_vulnerability_to_dict(self, mock_env_vars):
        """Test Vulnerability serialization to dict."""
        vuln = Vulnerability(
            id="SEC-SQL-001",
            category=VulnerabilityCategory.INJECTION,
            severity=VulnerabilitySeverity.CRITICAL,
            title="SQL Injection",
            description="Unsanitized input in SQL query",
            location="https://example.com/api/users",
            evidence="Error: SQL syntax error near 'OR 1=1'",
            cvss_score=9.8,
            cwe_id="CWE-89",
            remediation="Use parameterized queries",
            references=["https://owasp.org/www-community/attacks/SQL_Injection"],
        )

        data = vuln.to_dict()

        assert data["id"] == "SEC-SQL-001"
        assert data["category"] == "A03:2021-Injection"
        assert data["severity"] == "critical"
        assert data["title"] == "SQL Injection"
        assert data["cvss_score"] == 9.8
        assert data["cwe_id"] == "CWE-89"
        assert len(data["references"]) == 1

    def test_vulnerability_to_dict_preserves_enum_values(self, mock_env_vars):
        """Test that to_dict converts enums to their string values."""
        vuln = Vulnerability(
            id="TEST-001",
            category=VulnerabilityCategory.BROKEN_ACCESS,
            severity=VulnerabilitySeverity.HIGH,
            title="Test",
            description="Test",
            location="https://example.com",
            evidence="Test",
        )

        data = vuln.to_dict()

        # Enums should be converted to string values
        assert isinstance(data["category"], str)
        assert isinstance(data["severity"], str)
        assert data["category"] == "A01:2021-Broken Access Control"
        assert data["severity"] == "high"


class TestSecurityHeaders:
    """Tests for SecurityHeaders dataclass."""

    def test_security_headers_creation(self, mock_env_vars):
        """Test SecurityHeaders creation with all fields."""
        headers = SecurityHeaders(
            content_security_policy="default-src 'self'",
            x_frame_options="DENY",
            x_content_type_options="nosniff",
            strict_transport_security="max-age=31536000; includeSubDomains",
            x_xss_protection="1; mode=block",
            referrer_policy="strict-origin-when-cross-origin",
            permissions_policy="geolocation=(), microphone=()",
        )

        assert headers.content_security_policy == "default-src 'self'"
        assert headers.x_frame_options == "DENY"
        assert headers.strict_transport_security is not None

    def test_security_headers_defaults(self, mock_env_vars):
        """Test SecurityHeaders default values are None."""
        headers = SecurityHeaders()

        assert headers.content_security_policy is None
        assert headers.x_frame_options is None
        assert headers.x_content_type_options is None
        assert headers.strict_transport_security is None
        assert headers.x_xss_protection is None
        assert headers.referrer_policy is None
        assert headers.permissions_policy is None

    def test_get_missing_headers_all_missing(self, mock_env_vars):
        """Test get_missing_headers when all critical headers are missing."""
        headers = SecurityHeaders()

        missing = headers.get_missing_headers()

        assert "Content-Security-Policy" in missing
        assert "X-Frame-Options" in missing
        assert "Strict-Transport-Security" in missing
        assert "X-Content-Type-Options" in missing
        assert len(missing) == 4

    def test_get_missing_headers_none_missing(self, mock_env_vars):
        """Test get_missing_headers when all critical headers are present."""
        headers = SecurityHeaders(
            content_security_policy="default-src 'self'",
            x_frame_options="DENY",
            x_content_type_options="nosniff",
            strict_transport_security="max-age=31536000",
        )

        missing = headers.get_missing_headers()

        assert len(missing) == 0

    def test_get_missing_headers_partial(self, mock_env_vars):
        """Test get_missing_headers with some headers present."""
        headers = SecurityHeaders(
            content_security_policy="default-src 'self'",
            x_frame_options="DENY",
            # Missing: strict_transport_security and x_content_type_options
        )

        missing = headers.get_missing_headers()

        assert "Strict-Transport-Security" in missing
        assert "X-Content-Type-Options" in missing
        assert "Content-Security-Policy" not in missing
        assert "X-Frame-Options" not in missing
        assert len(missing) == 2

    @pytest.mark.parametrize("csp,x_frame,hsts,x_cto,expected_count", [
        (None, None, None, None, 4),
        ("policy", None, None, None, 3),
        ("policy", "DENY", None, None, 2),
        ("policy", "DENY", "max-age=100", None, 1),
        ("policy", "DENY", "max-age=100", "nosniff", 0),
    ])
    def test_get_missing_headers_combinations(self, mock_env_vars, csp, x_frame, hsts, x_cto, expected_count):
        """Test various combinations of present/missing headers."""
        headers = SecurityHeaders(
            content_security_policy=csp,
            x_frame_options=x_frame,
            strict_transport_security=hsts,
            x_content_type_options=x_cto,
        )

        missing = headers.get_missing_headers()
        assert len(missing) == expected_count


class TestSecurityScanResult:
    """Tests for SecurityScanResult dataclass."""

    def test_security_scan_result_creation(self, mock_env_vars):
        """Test SecurityScanResult creation."""
        headers = SecurityHeaders()
        vulnerabilities = []

        result = SecurityScanResult(
            url="https://example.com",
            vulnerabilities=vulnerabilities,
            headers=headers,
            risk_score=25.0,
            summary="No critical vulnerabilities found",
            scan_duration_ms=1500,
            recommendations=["Enable HTTPS", "Add CSP header"],
        )

        assert result.url == "https://example.com"
        assert result.risk_score == 25.0
        assert result.scan_duration_ms == 1500
        assert len(result.recommendations) == 2

    def test_critical_count_no_vulnerabilities(self, mock_env_vars):
        """Test critical_count property with no vulnerabilities."""
        result = SecurityScanResult(
            url="https://example.com",
            vulnerabilities=[],
            headers=SecurityHeaders(),
            risk_score=0.0,
            summary="Clean",
            scan_duration_ms=1000,
            recommendations=[],
        )

        assert result.critical_count == 0

    def test_critical_count_with_critical_vulnerabilities(self, mock_env_vars):
        """Test critical_count property with critical vulnerabilities."""
        vulnerabilities = [
            Vulnerability(
                id="1", category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.CRITICAL,
                title="SQL Injection", description="", location="", evidence="",
            ),
            Vulnerability(
                id="2", category=VulnerabilityCategory.XSS,
                severity=VulnerabilitySeverity.CRITICAL,
                title="XSS", description="", location="", evidence="",
            ),
            Vulnerability(
                id="3", category=VulnerabilityCategory.SECURITY_MISCONFIG,
                severity=VulnerabilitySeverity.HIGH,
                title="Missing Header", description="", location="", evidence="",
            ),
            Vulnerability(
                id="4", category=VulnerabilityCategory.BROKEN_AUTH,
                severity=VulnerabilitySeverity.CRITICAL,
                title="Auth Bypass", description="", location="", evidence="",
            ),
        ]

        result = SecurityScanResult(
            url="https://example.com",
            vulnerabilities=vulnerabilities,
            headers=SecurityHeaders(),
            risk_score=90.0,
            summary="Critical issues found",
            scan_duration_ms=2000,
            recommendations=[],
        )

        assert result.critical_count == 3

    def test_high_count_no_vulnerabilities(self, mock_env_vars):
        """Test high_count property with no vulnerabilities."""
        result = SecurityScanResult(
            url="https://example.com",
            vulnerabilities=[],
            headers=SecurityHeaders(),
            risk_score=0.0,
            summary="Clean",
            scan_duration_ms=1000,
            recommendations=[],
        )

        assert result.high_count == 0

    def test_high_count_with_high_vulnerabilities(self, mock_env_vars):
        """Test high_count property with high severity vulnerabilities."""
        vulnerabilities = [
            Vulnerability(
                id="1", category=VulnerabilityCategory.XSS,
                severity=VulnerabilitySeverity.HIGH,
                title="XSS", description="", location="", evidence="",
            ),
            Vulnerability(
                id="2", category=VulnerabilityCategory.INJECTION,
                severity=VulnerabilitySeverity.CRITICAL,
                title="SQL Injection", description="", location="", evidence="",
            ),
            Vulnerability(
                id="3", category=VulnerabilityCategory.BROKEN_ACCESS,
                severity=VulnerabilitySeverity.HIGH,
                title="IDOR", description="", location="", evidence="",
            ),
            Vulnerability(
                id="4", category=VulnerabilityCategory.SECURITY_MISCONFIG,
                severity=VulnerabilitySeverity.MEDIUM,
                title="Missing Header", description="", location="", evidence="",
            ),
        ]

        result = SecurityScanResult(
            url="https://example.com",
            vulnerabilities=vulnerabilities,
            headers=SecurityHeaders(),
            risk_score=75.0,
            summary="Issues found",
            scan_duration_ms=2000,
            recommendations=[],
        )

        assert result.high_count == 2

    @pytest.mark.parametrize("severities,expected_critical,expected_high", [
        ([], 0, 0),
        ([VulnerabilitySeverity.LOW], 0, 0),
        ([VulnerabilitySeverity.MEDIUM], 0, 0),
        ([VulnerabilitySeverity.HIGH], 0, 1),
        ([VulnerabilitySeverity.CRITICAL], 1, 0),
        ([VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH], 1, 1),
        ([VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH], 2, 1),
    ])
    def test_severity_counts_combinations(self, mock_env_vars, severities, expected_critical, expected_high):
        """Test critical_count and high_count with various severity combinations."""
        vulnerabilities = [
            Vulnerability(
                id=str(i), category=VulnerabilityCategory.SECURITY_MISCONFIG,
                severity=sev, title="Test", description="", location="", evidence="",
            )
            for i, sev in enumerate(severities)
        ]

        result = SecurityScanResult(
            url="https://example.com",
            vulnerabilities=vulnerabilities,
            headers=SecurityHeaders(),
            risk_score=0.0,
            summary="Test",
            scan_duration_ms=1000,
            recommendations=[],
        )

        assert result.critical_count == expected_critical
        assert result.high_count == expected_high


class TestSecurityScannerAgent:
    """Tests for SecurityScannerAgent class."""

    def test_agent_initialization(self, mock_env_vars):
        """Test agent initialization."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            assert agent is not None

    def test_agent_capabilities(self, mock_env_vars):
        """Test that agent declares SECURITY_SCAN capability."""
        from src.agents.base import AgentCapability

        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            assert AgentCapability.SECURITY_SCAN in agent.CAPABILITIES

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            prompt = agent._get_system_prompt()

            assert "security" in prompt.lower()
            assert "OWASP" in prompt or "CVSS" in prompt or "CWE" in prompt

    def test_check_security_headers_all_missing(self, mock_env_vars):
        """Test _check_security_headers with all headers missing."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            headers = SecurityHeaders()

            vulnerabilities = agent._check_security_headers(headers, "https://example.com")

            assert len(vulnerabilities) == 4
            assert all(v.category == VulnerabilityCategory.SECURITY_MISCONFIG for v in vulnerabilities)

    def test_check_security_headers_none_missing(self, mock_env_vars):
        """Test _check_security_headers with all headers present."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            headers = SecurityHeaders(
                content_security_policy="default-src 'self'",
                x_frame_options="DENY",
                x_content_type_options="nosniff",
                strict_transport_security="max-age=31536000",
            )

            vulnerabilities = agent._check_security_headers(headers, "https://example.com")

            assert len(vulnerabilities) == 0

    def test_check_security_headers_csp_is_medium_severity(self, mock_env_vars):
        """Test that missing CSP header is marked as MEDIUM severity."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            headers = SecurityHeaders(
                x_frame_options="DENY",
                x_content_type_options="nosniff",
                strict_transport_security="max-age=31536000",
                # CSP is missing
            )

            vulnerabilities = agent._check_security_headers(headers, "https://example.com")

            assert len(vulnerabilities) == 1
            csp_vuln = vulnerabilities[0]
            assert "Content-Security-Policy" in csp_vuln.title
            assert csp_vuln.severity == VulnerabilitySeverity.MEDIUM

    def test_check_security_headers_other_headers_are_low_severity(self, mock_env_vars):
        """Test that other missing headers are marked as LOW severity."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            headers = SecurityHeaders(
                content_security_policy="default-src 'self'",
                # Other headers missing
            )

            vulnerabilities = agent._check_security_headers(headers, "https://example.com")

            for vuln in vulnerabilities:
                assert vuln.severity == VulnerabilitySeverity.LOW

    def test_check_security_headers_url_in_location(self, mock_env_vars):
        """Test that URL is used as the location field."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            headers = SecurityHeaders()
            test_url = "https://test.example.com/page"

            vulnerabilities = agent._check_security_headers(headers, test_url)

            assert all(v.location == test_url for v in vulnerabilities)

    @pytest.mark.asyncio
    async def test_test_xss_finds_innerhtml(self, mock_env_vars):
        """Test _test_xss finds innerHTML patterns."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            content = '''
            <script>
                document.getElementById("result").innerHTML = userInput;
            </script>
            '''

            vulnerabilities = await agent._test_xss("https://example.com", content)

            assert len(vulnerabilities) >= 1
            assert any("innerHTML" in v.title for v in vulnerabilities)
            assert all(v.category == VulnerabilityCategory.XSS for v in vulnerabilities)
            assert all(v.severity == VulnerabilitySeverity.HIGH for v in vulnerabilities)

    @pytest.mark.asyncio
    async def test_test_xss_finds_document_write(self, mock_env_vars):
        """Test _test_xss finds document.write patterns."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            content = '''
            <script>
                document.write("<p>" + userInput + "</p>");
            </script>
            '''

            vulnerabilities = await agent._test_xss("https://example.com", content)

            assert len(vulnerabilities) >= 1
            assert any("document.write" in v.title for v in vulnerabilities)

    @pytest.mark.asyncio
    async def test_test_xss_finds_eval_pattern(self, mock_env_vars):
        """Test _test_xss finds code execution patterns."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            # Using a simple pattern that matches what the scanner looks for
            content = '''
            <script>
                window["ev" + "al"](userInput);
            </script>
            '''

            # The actual pattern is: eval\s*\(
            content_with_pattern = "result = eval(userInput);"

            vulnerabilities = await agent._test_xss("https://example.com", content_with_pattern)

            assert len(vulnerabilities) >= 1

    @pytest.mark.asyncio
    async def test_test_xss_finds_jquery_html(self, mock_env_vars):
        """Test _test_xss finds jQuery .html() patterns."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            content = '''
            <script>
                $("#result").html(userInput);
            </script>
            '''

            vulnerabilities = await agent._test_xss("https://example.com", content)

            assert len(vulnerabilities) >= 1
            assert any(".html()" in v.title for v in vulnerabilities)

    @pytest.mark.asyncio
    async def test_test_xss_no_vulnerabilities(self, mock_env_vars):
        """Test _test_xss with safe content."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            content = '''
            <html>
            <body>
                <p>Safe content</p>
                <script>
                    const sanitized = DOMPurify.sanitize(input);
                    element.textContent = sanitized;
                </script>
            </body>
            </html>
            '''

            vulnerabilities = await agent._test_xss("https://example.com", content)

            # Should not find vulnerabilities in safe content
            assert len(vulnerabilities) == 0

    @pytest.mark.asyncio
    async def test_test_xss_case_insensitive(self, mock_env_vars):
        """Test _test_xss pattern matching is case insensitive."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            content = '''
            <script>
                element.INNERHTML = data;
            </script>
            '''

            vulnerabilities = await agent._test_xss("https://example.com", content)

            # Should still find the pattern even with different case
            assert len(vulnerabilities) >= 1

    @pytest.mark.asyncio
    async def test_test_xss_multiple_patterns(self, mock_env_vars):
        """Test _test_xss finds multiple vulnerability patterns."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            # Note: Using actual patterns that the regex will match
            content = '''
            <script>
                document.getElementById("a").innerHTML = input1;
                document.write("<p>" + input2 + "</p>");
                window.eval(input3);
                $("#b").html(input4);
            </script>
            '''

            vulnerabilities = await agent._test_xss("https://example.com", content)

            assert len(vulnerabilities) == 4

    @pytest.mark.asyncio
    async def test_test_xss_cwe_id(self, mock_env_vars):
        """Test _test_xss vulnerabilities have correct CWE ID."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            content = 'document.write(x)'

            vulnerabilities = await agent._test_xss("https://example.com", content)

            assert len(vulnerabilities) >= 1
            assert all(v.cwe_id == "CWE-79" for v in vulnerabilities)

    def test_calculate_risk_score_empty(self, mock_env_vars):
        """Test _calculate_risk_score with no vulnerabilities."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()

            score = agent._calculate_risk_score([])

            assert score == 0.0

    def test_calculate_risk_score_single_critical(self, mock_env_vars):
        """Test _calculate_risk_score with single critical vulnerability."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            vulnerabilities = [
                Vulnerability(
                    id="1", category=VulnerabilityCategory.INJECTION,
                    severity=VulnerabilitySeverity.CRITICAL,
                    title="SQL Injection", description="", location="", evidence="",
                )
            ]

            score = agent._calculate_risk_score(vulnerabilities)

            assert score == 40.0  # Critical = 40 points

    def test_calculate_risk_score_single_high(self, mock_env_vars):
        """Test _calculate_risk_score with single high vulnerability."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            vulnerabilities = [
                Vulnerability(
                    id="1", category=VulnerabilityCategory.XSS,
                    severity=VulnerabilitySeverity.HIGH,
                    title="XSS", description="", location="", evidence="",
                )
            ]

            score = agent._calculate_risk_score(vulnerabilities)

            assert score == 25.0  # High = 25 points

    @pytest.mark.parametrize("severity,expected_weight", [
        (VulnerabilitySeverity.CRITICAL, 40),
        (VulnerabilitySeverity.HIGH, 25),
        (VulnerabilitySeverity.MEDIUM, 10),
        (VulnerabilitySeverity.LOW, 5),
        (VulnerabilitySeverity.INFO, 1),
    ])
    def test_calculate_risk_score_weights(self, mock_env_vars, severity, expected_weight):
        """Test _calculate_risk_score uses correct weights for each severity."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            vulnerabilities = [
                Vulnerability(
                    id="1", category=VulnerabilityCategory.SECURITY_MISCONFIG,
                    severity=severity,
                    title="Test", description="", location="", evidence="",
                )
            ]

            score = agent._calculate_risk_score(vulnerabilities)

            assert score == expected_weight

    def test_calculate_risk_score_cumulative(self, mock_env_vars):
        """Test _calculate_risk_score accumulates vulnerability weights."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            vulnerabilities = [
                Vulnerability(
                    id="1", category=VulnerabilityCategory.INJECTION,
                    severity=VulnerabilitySeverity.CRITICAL,
                    title="Critical", description="", location="", evidence="",
                ),
                Vulnerability(
                    id="2", category=VulnerabilityCategory.XSS,
                    severity=VulnerabilitySeverity.HIGH,
                    title="High", description="", location="", evidence="",
                ),
                Vulnerability(
                    id="3", category=VulnerabilityCategory.SECURITY_MISCONFIG,
                    severity=VulnerabilitySeverity.MEDIUM,
                    title="Medium", description="", location="", evidence="",
                ),
            ]

            score = agent._calculate_risk_score(vulnerabilities)

            # 40 + 25 + 10 = 75
            assert score == 75.0

    def test_calculate_risk_score_capped_at_100(self, mock_env_vars):
        """Test _calculate_risk_score is capped at 100."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            # Create enough critical vulnerabilities to exceed 100
            vulnerabilities = [
                Vulnerability(
                    id=str(i), category=VulnerabilityCategory.INJECTION,
                    severity=VulnerabilitySeverity.CRITICAL,
                    title=f"Critical {i}", description="", location="", evidence="",
                )
                for i in range(5)  # 5 * 40 = 200
            ]

            score = agent._calculate_risk_score(vulnerabilities)

            assert score == 100.0

    def test_calculate_risk_score_mixed_severities(self, mock_env_vars):
        """Test _calculate_risk_score with all severity levels."""
        with patch("src.agents.base.anthropic.Anthropic"):
            agent = SecurityScannerAgent()
            vulnerabilities = [
                Vulnerability(id="1", category=VulnerabilityCategory.INJECTION, severity=VulnerabilitySeverity.CRITICAL, title="", description="", location="", evidence=""),
                Vulnerability(id="2", category=VulnerabilityCategory.XSS, severity=VulnerabilitySeverity.HIGH, title="", description="", location="", evidence=""),
                Vulnerability(id="3", category=VulnerabilityCategory.SECURITY_MISCONFIG, severity=VulnerabilitySeverity.MEDIUM, title="", description="", location="", evidence=""),
                Vulnerability(id="4", category=VulnerabilityCategory.LOGGING, severity=VulnerabilitySeverity.LOW, title="", description="", location="", evidence=""),
                Vulnerability(id="5", category=VulnerabilityCategory.LOGGING, severity=VulnerabilitySeverity.INFO, title="", description="", location="", evidence=""),
            ]

            score = agent._calculate_risk_score(vulnerabilities)

            # 40 + 25 + 10 + 5 + 1 = 81
            assert score == 81.0

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_env_vars):
        """Test successful security scan execution."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "No critical issues", "recommendations": ["Enable HSTS"]}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders())
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "No critical issues",
                "recommendations": ["Enable HSTS"],
            })

            result = await agent.execute(url="https://example.com")

            assert result.success is True
            assert result.data is not None
            assert result.data.url == "https://example.com"

    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_env_vars):
        """Test security scan execution failure."""
        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(side_effect=Exception("Network error"))

            result = await agent.execute(url="https://example.com")

            assert result.success is False
            assert result.error is not None
            assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_execute_quick_scan(self, mock_env_vars):
        """Test security scan with quick scan type."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "Quick scan complete", "recommendations": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders())
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._test_injection = AsyncMock(return_value=[])
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "Quick scan complete",
                "recommendations": [],
            })

            result = await agent.execute(url="https://example.com", scan_type="quick")

            assert result.success is True
            # Quick scan should NOT call injection testing
            agent._test_injection.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_standard_scan(self, mock_env_vars):
        """Test security scan with standard scan type."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "Standard scan", "recommendations": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders())
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._test_injection = AsyncMock(return_value=[])
            agent._test_authentication = AsyncMock(return_value=[])
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "Standard scan",
                "recommendations": [],
            })

            result = await agent.execute(url="https://example.com", scan_type="standard")

            assert result.success is True
            # Standard scan should call injection testing but NOT auth testing
            agent._test_injection.assert_called_once()
            agent._test_authentication.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_deep_scan(self, mock_env_vars):
        """Test security scan with deep scan type."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "Deep scan", "recommendations": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders())
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._test_injection = AsyncMock(return_value=[])
            agent._test_authentication = AsyncMock(return_value=[])
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "Deep scan",
                "recommendations": [],
            })

            result = await agent.execute(url="https://example.com", scan_type="deep")

            assert result.success is True
            # Deep scan should call BOTH injection and auth testing
            agent._test_injection.assert_called_once()
            agent._test_authentication.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_includes_header_vulnerabilities(self, mock_env_vars):
        """Test that execute includes header-based vulnerabilities."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "Issues found", "recommendations": ["Add security headers"]}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            # Return headers with missing CSP
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders(
                x_frame_options="DENY",
                x_content_type_options="nosniff",
                strict_transport_security="max-age=31536000",
            ))
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "Issues found",
                "recommendations": ["Add CSP header"],
            })

            result = await agent.execute(url="https://example.com", scan_type="quick")

            assert result.success is True
            # Should have 1 vulnerability for missing CSP
            assert len(result.data.vulnerabilities) >= 1
            assert any("Content-Security-Policy" in v.title for v in result.data.vulnerabilities)

    @pytest.mark.asyncio
    async def test_execute_includes_xss_vulnerabilities(self, mock_env_vars):
        """Test that execute includes XSS vulnerabilities from page content."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "XSS found", "recommendations": ["Sanitize input"]}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders(
                content_security_policy="default-src 'self'",
                x_frame_options="DENY",
                x_content_type_options="nosniff",
                strict_transport_security="max-age=31536000",
            ))
            agent._fetch_page_content = AsyncMock(return_value='''
                <script>
                    document.getElementById("out").innerHTML = userInput;
                </script>
            ''')
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "XSS found",
                "recommendations": ["Sanitize input"],
            })

            result = await agent.execute(url="https://example.com", scan_type="quick")

            assert result.success is True
            # Should find XSS vulnerability
            xss_vulns = [v for v in result.data.vulnerabilities if v.category == VulnerabilityCategory.XSS]
            assert len(xss_vulns) >= 1

    @pytest.mark.asyncio
    async def test_execute_calculates_risk_score(self, mock_env_vars):
        """Test that execute calculates risk score based on vulnerabilities."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "Test", "recommendations": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            # All headers missing = 4 vulnerabilities (1 medium + 3 low = 10 + 15 = 25)
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders())
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "Test",
                "recommendations": [],
            })

            result = await agent.execute(url="https://example.com", scan_type="quick")

            assert result.success is True
            # Risk score should be 10 (CSP medium) + 5 + 5 + 5 (3 low) = 25
            assert result.data.risk_score == 25.0

    @pytest.mark.asyncio
    async def test_execute_returns_duration(self, mock_env_vars):
        """Test that execute returns scan duration."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "Test", "recommendations": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders())
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "Test",
                "recommendations": [],
            })

            result = await agent.execute(url="https://example.com", scan_type="quick")

            assert result.success is True
            assert result.data.scan_duration_ms >= 0
            assert isinstance(result.data.scan_duration_ms, int)

    @pytest.mark.asyncio
    async def test_execute_returns_token_usage(self, mock_env_vars):
        """Test that execute returns token usage information."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"summary": "Test", "recommendations": []}')]
        mock_response.usage = MagicMock(input_tokens=500, output_tokens=100)

        with patch("src.agents.base.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            agent = SecurityScannerAgent()
            agent._analyze_headers = AsyncMock(return_value=SecurityHeaders())
            agent._fetch_page_content = AsyncMock(return_value="<html></html>")
            agent._analyze_findings = AsyncMock(return_value={
                "summary": "Test",
                "recommendations": [],
            })

            result = await agent.execute(url="https://example.com", scan_type="quick")

            assert result.success is True
            # Token usage should be populated from AI calls
            assert hasattr(result, 'input_tokens')
            assert hasattr(result, 'output_tokens')
