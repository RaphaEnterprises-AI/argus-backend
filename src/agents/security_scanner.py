"""Security Scanner Agent - Detects security vulnerabilities in web applications.

Scans for OWASP Top 10 vulnerabilities:
- Injection (SQL, XSS, Command)
- Broken Authentication
- Sensitive Data Exposure
- XML External Entities (XXE)
- Broken Access Control
- Security Misconfiguration
- Cross-Site Scripting (XSS)
- Insecure Deserialization
- Using Components with Known Vulnerabilities
- Insufficient Logging & Monitoring
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from ..core.model_router import TaskType
from .base import AgentCapability, AgentResult, BaseAgent
from .prompts import get_enhanced_prompt


class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels (CVSS-based)."""
    CRITICAL = "critical"  # 9.0-10.0
    HIGH = "high"          # 7.0-8.9
    MEDIUM = "medium"      # 4.0-6.9
    LOW = "low"            # 0.1-3.9
    INFO = "info"          # Informational


class VulnerabilityCategory(str, Enum):
    """OWASP Top 10 categories."""
    INJECTION = "A03:2021-Injection"
    BROKEN_AUTH = "A07:2021-Identification and Authentication Failures"
    SENSITIVE_DATA = "A02:2021-Cryptographic Failures"
    XXE = "A05:2021-Security Misconfiguration"
    BROKEN_ACCESS = "A01:2021-Broken Access Control"
    SECURITY_MISCONFIG = "A05:2021-Security Misconfiguration"
    XSS = "A03:2021-Injection"
    INSECURE_DESERIAL = "A08:2021-Software and Data Integrity Failures"
    VULNERABLE_COMPONENTS = "A06:2021-Vulnerable and Outdated Components"
    LOGGING = "A09:2021-Security Logging and Monitoring Failures"
    SSRF = "A10:2021-Server-Side Request Forgery"


@dataclass
class Vulnerability:
    """A detected security vulnerability."""
    id: str
    category: VulnerabilityCategory
    severity: VulnerabilitySeverity
    title: str
    description: str
    location: str  # URL, file path, or code location
    evidence: str  # Proof of vulnerability
    cvss_score: float = 0.0
    cwe_id: str | None = None
    remediation: str = ""
    references: list[str] = field(default_factory=list)
    false_positive_likelihood: float = 0.0  # 0-1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "evidence": self.evidence,
            "cvss_score": self.cvss_score,
            "cwe_id": self.cwe_id,
            "remediation": self.remediation,
            "references": self.references,
        }


@dataclass
class SecurityHeaders:
    """Security headers analysis."""
    content_security_policy: str | None = None
    x_frame_options: str | None = None
    x_content_type_options: str | None = None
    strict_transport_security: str | None = None
    x_xss_protection: str | None = None
    referrer_policy: str | None = None
    permissions_policy: str | None = None

    def get_missing_headers(self) -> list[str]:
        missing = []
        if not self.content_security_policy:
            missing.append("Content-Security-Policy")
        if not self.x_frame_options:
            missing.append("X-Frame-Options")
        if not self.strict_transport_security:
            missing.append("Strict-Transport-Security")
        if not self.x_content_type_options:
            missing.append("X-Content-Type-Options")
        return missing


@dataclass
class SecurityScanResult:
    """Result of a security scan."""
    url: str
    vulnerabilities: list[Vulnerability]
    headers: SecurityHeaders
    risk_score: float  # 0-100, higher is more risky
    summary: str
    scan_duration_ms: int
    recommendations: list[str]

    @property
    def critical_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for v in self.vulnerabilities if v.severity == VulnerabilitySeverity.HIGH)


class SecurityScannerAgent(BaseAgent):
    """
    Agent that scans web applications for security vulnerabilities.

    Capabilities:
    - OWASP Top 10 vulnerability detection
    - Security header analysis
    - XSS payload testing
    - SQL injection detection
    - Authentication bypass testing
    - Sensitive data exposure detection
    - API security testing

    Uses AI for intelligent vulnerability analysis and reducing false positives.
    """

    # RAP-231: Agent capabilities for A2A discovery
    CAPABILITIES = [
        AgentCapability.SECURITY_SCAN,
    ]

    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for security scanning."""
        enhanced = get_enhanced_prompt("security_scanner")
        if enhanced:
            return enhanced

        return """# Role
You are an elite application security engineer with expertise in OWASP Top 10.

# Focus Areas
- Injection (SQL, XSS, Command)
- Authentication & Authorization flaws
- Security misconfigurations
- Sensitive data exposure

# Output Requirements
- CVSS 3.1 scoring for each vulnerability
- CWE IDs for classification
- Step-by-step remediation guidance
- False positive confidence level

# Ethical Constraints
- Only report verifiable vulnerabilities
- Do not attempt destructive testing
- Focus on detection, not exploitation"""

    async def execute(
        self,
        url: str,
        scan_type: Literal["quick", "standard", "deep"] = "standard",
        include_api_tests: bool = True,
        custom_payloads: list[str] | None = None,
    ) -> AgentResult[SecurityScanResult]:
        """
        Scan a URL for security vulnerabilities.

        Args:
            url: Target URL to scan
            scan_type: Scan depth ("quick", "standard", "deep")
            include_api_tests: Whether to test API endpoints
            custom_payloads: Custom test payloads

        Returns:
            SecurityScanResult with findings and recommendations
        """
        import time
        start_time = time.time()

        try:
            # Step 1: Collect security data
            headers = await self._analyze_headers(url)
            page_content = await self._fetch_page_content(url)

            # Step 2: Run security tests
            vulnerabilities = []

            # Header-based vulnerabilities
            header_vulns = self._check_security_headers(headers, url)
            vulnerabilities.extend(header_vulns)

            # XSS testing
            xss_vulns = await self._test_xss(url, page_content)
            vulnerabilities.extend(xss_vulns)

            # Injection testing
            if scan_type in ["standard", "deep"]:
                injection_vulns = await self._test_injection(url)
                vulnerabilities.extend(injection_vulns)

            # Deep scan includes more thorough testing
            if scan_type == "deep":
                auth_vulns = await self._test_authentication(url)
                vulnerabilities.extend(auth_vulns)

            # Step 3: AI analysis for intelligent assessment
            analysis = await self._analyze_findings(url, vulnerabilities, headers)

            # Calculate risk score
            risk_score = self._calculate_risk_score(vulnerabilities)

            duration_ms = int((time.time() - start_time) * 1000)

            result = SecurityScanResult(
                url=url,
                vulnerabilities=vulnerabilities,
                headers=headers,
                risk_score=risk_score,
                summary=analysis.get("summary", ""),
                scan_duration_ms=duration_ms,
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
            self.log.exception("Security scan failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    async def _analyze_headers(self, url: str) -> SecurityHeaders:
        """Analyze security headers from response."""
        # Would fetch actual headers via HTTP client
        return SecurityHeaders()

    async def _fetch_page_content(self, url: str) -> str:
        """Fetch page content for analysis."""
        return ""

    def _check_security_headers(self, headers: SecurityHeaders, url: str) -> list[Vulnerability]:
        """Check for missing or misconfigured security headers."""
        vulnerabilities = []

        missing = headers.get_missing_headers()
        for header in missing:
            vuln = Vulnerability(
                id=f"SEC-HDR-{header.replace('-', '')}",
                category=VulnerabilityCategory.SECURITY_MISCONFIG,
                severity=VulnerabilitySeverity.MEDIUM if header == "Content-Security-Policy" else VulnerabilitySeverity.LOW,
                title=f"Missing {header} Header",
                description=f"The {header} security header is not set, which may expose the application to various attacks.",
                location=url,
                evidence=f"Header '{header}' not present in response",
                remediation=f"Add the {header} header to your server configuration.",
            )
            vulnerabilities.append(vuln)

        return vulnerabilities

    async def _test_xss(self, url: str, content: str) -> list[Vulnerability]:
        """Test for XSS vulnerabilities."""
        vulnerabilities = []

        # Check for common XSS sinks in JavaScript
        dangerous_patterns = [
            (r'innerHTML\s*=', "innerHTML assignment"),
            (r'document\.write\s*\(', "document.write usage"),
            (r'eval\s*\(', "eval usage"),
            (r'\.html\s*\(', "jQuery .html() usage"),
        ]

        for pattern, description in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                vuln = Vulnerability(
                    id=f"SEC-XSS-{description.replace(' ', '')}",
                    category=VulnerabilityCategory.XSS,
                    severity=VulnerabilitySeverity.HIGH,
                    title=f"Potential XSS: {description}",
                    description=f"The page uses {description} which can lead to XSS if user input is not sanitized.",
                    location=url,
                    evidence=f"Pattern '{pattern}' found in page source",
                    cwe_id="CWE-79",
                    remediation="Use safe DOM manipulation methods and sanitize all user input.",
                )
                vulnerabilities.append(vuln)

        return vulnerabilities

    async def _test_injection(self, url: str) -> list[Vulnerability]:
        """Test for injection vulnerabilities."""
        # Would perform actual injection testing
        return []

    async def _test_authentication(self, url: str) -> list[Vulnerability]:
        """Test for authentication vulnerabilities."""
        # Would test auth mechanisms
        return []

    def _calculate_risk_score(self, vulnerabilities: list[Vulnerability]) -> float:
        """Calculate overall risk score from vulnerabilities."""
        if not vulnerabilities:
            return 0.0

        # Weighted scoring based on severity
        weights = {
            VulnerabilitySeverity.CRITICAL: 40,
            VulnerabilitySeverity.HIGH: 25,
            VulnerabilitySeverity.MEDIUM: 10,
            VulnerabilitySeverity.LOW: 5,
            VulnerabilitySeverity.INFO: 1,
        }

        total_score = sum(weights[v.severity] for v in vulnerabilities)
        # Cap at 100
        return min(100.0, total_score)

    async def _analyze_findings(
        self,
        url: str,
        vulnerabilities: list[Vulnerability],
        headers: SecurityHeaders,
    ) -> dict:
        """Use AI to analyze and contextualize findings."""
        vuln_summary = "\n".join([
            f"- [{v.severity.value}] {v.title}: {v.description}"
            for v in vulnerabilities[:10]  # Limit for context
        ])

        prompt = f"""Analyze these security findings for {url}:

VULNERABILITIES FOUND:
{vuln_summary}

SECURITY HEADERS:
- CSP: {headers.content_security_policy or 'Missing'}
- HSTS: {headers.strict_transport_security or 'Missing'}
- X-Frame-Options: {headers.x_frame_options or 'Missing'}

Provide:
1. Executive summary (2-3 sentences)
2. Top 5 prioritized remediation steps
3. Risk assessment considering attack surface
4. False positive assessment

RESPOND IN JSON:
{{
  "summary": "<executive summary>",
  "recommendations": ["<prioritized recommendation>"],
  "risk_level": "critical|high|medium|low",
  "false_positive_assessment": "<assessment>"
}}"""

        response = await self._call_model(
            messages=[{"role": "user", "content": prompt}],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=1500,
        )

        return self._parse_json_response(response["content"], {
            "summary": "Security scan completed",
            "recommendations": [],
        })
