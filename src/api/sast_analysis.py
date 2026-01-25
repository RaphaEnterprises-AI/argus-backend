"""SAST (Static Application Security Testing) Analysis Module.

Provides Semgrep-based security analysis for:
- SQL injection via string formatting
- JWT verification disabled
- Hardcoded credentials patterns
- Debug mode detection
- And many more security vulnerabilities

RAP-93: SAST Integration implementation.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.context import get_current_organization_id
from src.api.teams import get_current_user, verify_org_access
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/sast", tags=["SAST Analysis"])


# =============================================================================
# Constants
# =============================================================================

# Severity weights for risk score calculation
SEVERITY_WEIGHTS = {
    "critical": 1.0,
    "high": 0.7,
    "medium": 0.4,
    "low": 0.1,
    "info": 0.0,
}

# Map Semgrep severity to our normalized severity levels
SEMGREP_SEVERITY_MAP = {
    "ERROR": "critical",
    "WARNING": "high",
    "INFO": "medium",
    "INVENTORY": "low",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SecurityFinding:
    """A security finding from SAST analysis."""

    rule_id: str
    severity: Literal["critical", "high", "medium", "low", "info"]
    message: str
    file_path: str
    line_number: int
    code_snippet: str
    fix_suggestion: str | None = None
    cwe: str | None = None  # CWE identifier
    owasp: str | None = None  # OWASP category
    confidence: str = "high"  # high, medium, low
    end_line: int | None = None
    column: int | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "fix_suggestion": self.fix_suggestion,
            "cwe": self.cwe,
            "owasp": self.owasp,
            "confidence": self.confidence,
            "end_line": self.end_line,
            "column": self.column,
        }


@dataclass
class SASTResult:
    """Result of SAST analysis."""

    findings: list[SecurityFinding] = field(default_factory=list)
    security_risk_score: float = 0.0
    files_scanned: int = 0
    scan_duration_seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "findings": [f.to_dict() for f in self.findings],
            "security_risk_score": self.security_risk_score,
            "files_scanned": self.files_scanned,
            "scan_duration_seconds": self.scan_duration_seconds,
            "error": self.error,
        }


# =============================================================================
# Request/Response Models
# =============================================================================


class AnalyzeRequest(BaseModel):
    """Request for SAST analysis."""

    files: list[str] | None = Field(
        None,
        description="List of file paths to analyze. If None, scans entire repo.",
    )
    repo_path: str | None = Field(
        None,
        description="Repository path to scan. Defaults to current directory.",
    )
    custom_rules_path: str | None = Field(
        None,
        description="Path to custom Semgrep rules file.",
    )
    include_auto_rules: bool = Field(
        True,
        description="Include Semgrep auto config rules.",
    )


class ScanPRRequest(BaseModel):
    """Request to scan a PR's changed files."""

    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    pr_number: int = Field(..., description="Pull request number")
    github_token: str | None = Field(
        None,
        description="GitHub token for API access. If not provided, uses project integration.",
    )


class SASTAnalysisResponse(BaseModel):
    """Response for SAST analysis."""

    findings: list[dict]
    security_risk_score: float
    files_scanned: int
    scan_duration_seconds: float
    error: str | None = None
    summary: dict = Field(default_factory=dict)


class RuleInfo(BaseModel):
    """Information about a Semgrep rule."""

    id: str
    name: str
    severity: str
    languages: list[str]
    description: str
    source: str  # "builtin" or "custom"


class RulesListResponse(BaseModel):
    """Response for listing available rules."""

    rules: list[RuleInfo]
    total_count: int
    custom_rules_available: bool


# =============================================================================
# Semgrep Runner
# =============================================================================


def run_semgrep(
    target_paths: list[str],
    custom_rules_path: str | None = None,
    include_auto_rules: bool = True,
    timeout_seconds: int = 300,
) -> tuple[list[dict], str | None]:
    """Run Semgrep on target paths and return findings.

    Args:
        target_paths: List of file or directory paths to scan
        custom_rules_path: Optional path to custom rules file
        include_auto_rules: Whether to include Semgrep's auto config
        timeout_seconds: Maximum time for the scan

    Returns:
        Tuple of (findings list, error message or None)
    """
    if not target_paths:
        return [], "No target paths provided"

    # Build Semgrep command
    cmd = [
        "semgrep",
        "--json",
        "--quiet",
        "--no-git-ignore",  # Scan all files
    ]

    # Add rules
    if include_auto_rules:
        cmd.extend(["--config", "auto"])

    if custom_rules_path:
        cmd.extend(["--config", custom_rules_path])

    # If no rules specified, at least use auto
    if not include_auto_rules and not custom_rules_path:
        cmd.extend(["--config", "auto"])

    # Add target paths
    cmd.extend(target_paths)

    logger.info(
        "Running Semgrep",
        target_count=len(target_paths),
        custom_rules=custom_rules_path is not None,
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        # Semgrep returns exit code 0 for success, 1 for findings, >1 for errors
        if result.returncode > 1:
            logger.error(
                "Semgrep error",
                stderr=result.stderr[:500] if result.stderr else None,
                return_code=result.returncode,
            )
            return [], f"Semgrep error: {result.stderr[:500] if result.stderr else 'Unknown error'}"

        # Parse JSON output
        try:
            output = json.loads(result.stdout) if result.stdout else {}
        except json.JSONDecodeError as e:
            logger.error("Failed to parse Semgrep output", error=str(e))
            return [], f"Failed to parse Semgrep output: {str(e)}"

        findings = output.get("results", [])
        errors = output.get("errors", [])

        if errors:
            logger.warning(
                "Semgrep reported errors",
                error_count=len(errors),
                first_error=errors[0] if errors else None,
            )

        logger.info(
            "Semgrep scan completed",
            findings_count=len(findings),
            errors_count=len(errors),
        )

        return findings, None

    except subprocess.TimeoutExpired:
        logger.error("Semgrep timeout", timeout_seconds=timeout_seconds)
        return [], f"Semgrep scan timed out after {timeout_seconds} seconds"

    except FileNotFoundError:
        logger.error("Semgrep not found")
        return [], "Semgrep not installed. Install with: pip install semgrep"

    except Exception as e:
        logger.exception("Unexpected error running Semgrep", error=str(e))
        return [], f"Unexpected error: {str(e)}"


def parse_semgrep_findings(raw_findings: list[dict]) -> list[SecurityFinding]:
    """Parse raw Semgrep findings into SecurityFinding objects.

    Args:
        raw_findings: List of raw findings from Semgrep JSON output

    Returns:
        List of SecurityFinding objects
    """
    findings = []

    for raw in raw_findings:
        # Extract check information
        check_id = raw.get("check_id", "unknown")
        extra = raw.get("extra", {})

        # Map Semgrep severity to our levels
        semgrep_severity = extra.get("severity", "INFO")
        severity = SEMGREP_SEVERITY_MAP.get(semgrep_severity, "medium")

        # Extract location info
        start = raw.get("start", {})
        end = raw.get("end", {})

        # Extract code snippet
        lines = extra.get("lines", "")
        if not lines:
            # Try to get from extra.snippet
            lines = extra.get("snippet", raw.get("extra", {}).get("lines", ""))

        # Extract fix suggestion
        fix = extra.get("fix", None)
        if not fix:
            # Check for message with fix suggestion
            message = extra.get("message", "")
            if "fix:" in message.lower():
                fix = message.split("fix:", 1)[1].strip() if "fix:" in message.lower() else None

        # Extract CWE and OWASP
        metadata = extra.get("metadata", {})
        cwe = None
        owasp = None

        cwe_list = metadata.get("cwe", [])
        if isinstance(cwe_list, list) and cwe_list:
            cwe = cwe_list[0] if isinstance(cwe_list[0], str) else str(cwe_list[0])
        elif isinstance(cwe_list, str):
            cwe = cwe_list

        owasp_list = metadata.get("owasp", [])
        if isinstance(owasp_list, list) and owasp_list:
            owasp = owasp_list[0] if isinstance(owasp_list[0], str) else str(owasp_list[0])
        elif isinstance(owasp_list, str):
            owasp = owasp_list

        finding = SecurityFinding(
            rule_id=check_id,
            severity=severity,
            message=extra.get("message", "Security issue detected"),
            file_path=raw.get("path", "unknown"),
            line_number=start.get("line", 0),
            code_snippet=lines,
            fix_suggestion=fix,
            cwe=cwe,
            owasp=owasp,
            confidence=metadata.get("confidence", "HIGH").lower(),
            end_line=end.get("line"),
            column=start.get("col"),
        )

        findings.append(finding)

    return findings


# =============================================================================
# Risk Score Calculation
# =============================================================================


def calculate_security_risk_score(findings: list[SecurityFinding]) -> float:
    """Calculate normalized security risk score from findings.

    Uses weighted scoring based on severity:
    - critical: 1.0
    - high: 0.7
    - medium: 0.4
    - low: 0.1
    - info: 0.0

    Args:
        findings: List of security findings

    Returns:
        Normalized risk score between 0.0 and 1.0
    """
    if not findings:
        return 0.0

    # Calculate raw weighted score
    raw_score = sum(
        SEVERITY_WEIGHTS.get(f.severity, 0.0) for f in findings
    )

    # Normalize to 0-1 range
    # Using a sigmoid-like normalization that approaches 1.0 asymptotically
    # This ensures even many low-severity findings won't exceed 1.0
    # and a single critical finding contributes significantly
    max_expected_score = 10.0  # Expected maximum meaningful raw score
    normalized = min(1.0, raw_score / max_expected_score)

    # Apply confidence weighting
    confidence_weights = {"high": 1.0, "medium": 0.7, "low": 0.4}
    confidence_factor = sum(
        confidence_weights.get(f.confidence, 0.7) for f in findings
    ) / max(len(findings), 1)

    # Combine normalized score with confidence
    final_score = normalized * confidence_factor

    return round(min(1.0, final_score), 2)


# =============================================================================
# PR Comment Formatting
# =============================================================================


def format_security_findings_for_pr(findings: list[SecurityFinding]) -> str:
    """Format security findings as markdown for PR comments.

    Args:
        findings: List of security findings

    Returns:
        Markdown formatted string for PR comment section
    """
    if not findings:
        return ""

    lines = [
        "### \U0001f512 Security Analysis",
        "",
        "| Severity | Finding | Location |",
        "|----------|---------|----------|",
    ]

    # Sort by severity (critical first)
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f.severity, 5))

    # Group and limit displayed findings
    severity_emoji = {
        "critical": "\U0001f6a8",  # Rotating light
        "high": "\U0001f534",  # Red circle
        "medium": "\U0001f7e1",  # Yellow circle
        "low": "\U0001f7e2",  # Green circle
        "info": "\U0001f535",  # Blue circle
    }

    displayed_count = 0
    max_display = 10  # Limit to prevent huge comments

    for finding in sorted_findings:
        if displayed_count >= max_display:
            break

        emoji = severity_emoji.get(finding.severity, "\u2139\ufe0f")
        severity_label = finding.severity.capitalize()

        # Truncate long messages
        message = finding.message[:60] + "..." if len(finding.message) > 60 else finding.message
        message = message.replace("|", "\\|")  # Escape pipe for table

        # Format location
        location = f"`{finding.file_path}:{finding.line_number}`"

        lines.append(f"| {emoji} {severity_label} | {message} | {location} |")
        displayed_count += 1

    if len(sorted_findings) > max_display:
        lines.append(f"| ... | *{len(sorted_findings) - max_display} more findings* | - |")

    lines.append("")

    # Add summary
    critical_count = sum(1 for f in findings if f.severity == "critical")
    high_count = sum(1 for f in findings if f.severity == "high")

    if critical_count > 0:
        lines.append(f"\u26a0\ufe0f **{critical_count} critical issue(s)** require immediate attention.")
    elif high_count > 0:
        lines.append(f"\u26a0\ufe0f **{high_count} high severity issue(s)** should be reviewed.")

    return "\n".join(lines)


# =============================================================================
# Main Analysis Functions
# =============================================================================


async def analyze_files(
    files: list[str],
    custom_rules_path: str | None = None,
    include_auto_rules: bool = True,
) -> SASTResult:
    """Analyze a list of files for security vulnerabilities.

    Args:
        files: List of file paths to analyze
        custom_rules_path: Optional path to custom Semgrep rules
        include_auto_rules: Whether to include auto config rules

    Returns:
        SASTResult with findings and risk score
    """
    import time

    start_time = time.time()

    # Filter to existing files
    existing_files = [f for f in files if Path(f).exists()]

    if not existing_files:
        logger.warning("No existing files to scan", provided_count=len(files))
        return SASTResult(
            findings=[],
            security_risk_score=0.0,
            files_scanned=0,
            scan_duration_seconds=0.0,
            error="No existing files to scan",
        )

    # Run Semgrep
    raw_findings, error = run_semgrep(
        target_paths=existing_files,
        custom_rules_path=custom_rules_path,
        include_auto_rules=include_auto_rules,
    )

    if error:
        return SASTResult(
            findings=[],
            security_risk_score=0.0,
            files_scanned=len(existing_files),
            scan_duration_seconds=time.time() - start_time,
            error=error,
        )

    # Parse findings
    findings = parse_semgrep_findings(raw_findings)

    # Calculate risk score
    risk_score = calculate_security_risk_score(findings)

    return SASTResult(
        findings=findings,
        security_risk_score=risk_score,
        files_scanned=len(existing_files),
        scan_duration_seconds=round(time.time() - start_time, 2),
    )


async def analyze_repository(
    repo_path: str,
    custom_rules_path: str | None = None,
    include_auto_rules: bool = True,
) -> SASTResult:
    """Analyze an entire repository for security vulnerabilities.

    Args:
        repo_path: Path to the repository root
        custom_rules_path: Optional path to custom Semgrep rules
        include_auto_rules: Whether to include auto config rules

    Returns:
        SASTResult with findings and risk score
    """
    import time

    start_time = time.time()

    repo = Path(repo_path)
    if not repo.exists():
        return SASTResult(
            findings=[],
            security_risk_score=0.0,
            files_scanned=0,
            scan_duration_seconds=0.0,
            error=f"Repository path does not exist: {repo_path}",
        )

    # Run Semgrep on entire repo
    raw_findings, error = run_semgrep(
        target_paths=[str(repo)],
        custom_rules_path=custom_rules_path,
        include_auto_rules=include_auto_rules,
    )

    if error:
        return SASTResult(
            findings=[],
            security_risk_score=0.0,
            files_scanned=0,
            scan_duration_seconds=time.time() - start_time,
            error=error,
        )

    # Parse findings
    findings = parse_semgrep_findings(raw_findings)

    # Calculate risk score
    risk_score = calculate_security_risk_score(findings)

    # Count unique files
    unique_files = len(set(f.file_path for f in findings))

    return SASTResult(
        findings=findings,
        security_risk_score=risk_score,
        files_scanned=unique_files if unique_files > 0 else 1,
        scan_duration_seconds=round(time.time() - start_time, 2),
    )


# =============================================================================
# Custom Rules Management
# =============================================================================


def get_custom_rules_path() -> Path:
    """Get the path to custom Semgrep rules.

    Returns:
        Path to the semgrep-rules directory
    """
    # Look for rules in the project root
    project_root = Path(__file__).parent.parent.parent
    return project_root / "semgrep-rules" / "argus-security.yaml"


def get_available_rules() -> list[RuleInfo]:
    """Get list of available Semgrep rules.

    Returns:
        List of RuleInfo objects
    """
    rules = []

    # Check for custom rules
    custom_rules_path = get_custom_rules_path()
    if custom_rules_path.exists():
        try:
            import yaml

            with open(custom_rules_path) as f:
                custom_config = yaml.safe_load(f)

            for rule in custom_config.get("rules", []):
                rules.append(
                    RuleInfo(
                        id=rule.get("id", "unknown"),
                        name=rule.get("id", "Unknown Rule"),
                        severity=SEMGREP_SEVERITY_MAP.get(
                            rule.get("severity", "WARNING"), "medium"
                        ),
                        languages=rule.get("languages", []),
                        description=rule.get("message", "No description"),
                        source="custom",
                    )
                )
        except Exception as e:
            logger.warning("Failed to parse custom rules", error=str(e))

    # Add info about built-in rules
    builtin_rules = [
        RuleInfo(
            id="semgrep-auto",
            name="Semgrep Auto Config",
            severity="varies",
            languages=["python", "javascript", "typescript", "java", "go", "ruby", "rust", "c", "cpp"],
            description="Automatic rule selection based on detected languages and frameworks",
            source="builtin",
        ),
    ]
    rules.extend(builtin_rules)

    return rules


# =============================================================================
# API Endpoints
# =============================================================================


@router.post("/analyze", response_model=SASTAnalysisResponse)
async def analyze_endpoint(
    request: Request,
    body: AnalyzeRequest,
    project_id: str = Query(..., description="Project ID"),
):
    """Run SAST analysis on provided files or repository.

    Analyzes the provided files or repository for security vulnerabilities
    using Semgrep. Returns findings with severity levels and a normalized
    security risk score.
    """
    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Verify project access
    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id"
    )
    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    org_id = project_result["data"][0]["organization_id"]
    await verify_org_access(
        org_id,
        user["user_id"],
        user_email=user.get("email"),
        request=request,
    )

    # Get custom rules path
    custom_rules = body.custom_rules_path
    if not custom_rules:
        default_rules = get_custom_rules_path()
        if default_rules.exists():
            custom_rules = str(default_rules)

    # Run analysis
    if body.files:
        result = await analyze_files(
            files=body.files,
            custom_rules_path=custom_rules,
            include_auto_rules=body.include_auto_rules,
        )
    elif body.repo_path:
        result = await analyze_repository(
            repo_path=body.repo_path,
            custom_rules_path=custom_rules,
            include_auto_rules=body.include_auto_rules,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Either 'files' or 'repo_path' must be provided",
        )

    logger.info(
        "SAST analysis completed",
        project_id=project_id,
        findings_count=len(result.findings),
        risk_score=result.security_risk_score,
        user_id=user["user_id"],
    )

    # Build summary
    severity_counts = {}
    for finding in result.findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    return SASTAnalysisResponse(
        findings=[f.to_dict() for f in result.findings],
        security_risk_score=result.security_risk_score,
        files_scanned=result.files_scanned,
        scan_duration_seconds=result.scan_duration_seconds,
        error=result.error,
        summary={
            "total_findings": len(result.findings),
            "by_severity": severity_counts,
            "top_rules": _get_top_rules(result.findings),
        },
    )


@router.get("/rules", response_model=RulesListResponse)
async def list_rules(
    request: Request,
):
    """List available Semgrep rules.

    Returns information about both custom Argus rules and
    built-in Semgrep rules.
    """
    user = await get_current_user(request)

    rules = get_available_rules()
    custom_available = get_custom_rules_path().exists()

    logger.info(
        "Listed SAST rules",
        rules_count=len(rules),
        custom_available=custom_available,
        user_id=user["user_id"],
    )

    return RulesListResponse(
        rules=rules,
        total_count=len(rules),
        custom_rules_available=custom_available,
    )


@router.post("/scan-pr", response_model=SASTAnalysisResponse)
async def scan_pr(
    request: Request,
    body: ScanPRRequest,
    project_id: str = Query(..., description="Project ID"),
):
    """Scan a PR's changed files for security vulnerabilities.

    Fetches the list of changed files from the PR and runs SAST
    analysis on them.
    """
    import httpx

    user = await get_current_user(request)
    supabase = get_supabase_client()

    # Verify project access
    project_result = await supabase.request(
        f"/projects?id=eq.{project_id}&select=organization_id"
    )
    if not project_result.get("data"):
        raise HTTPException(status_code=404, detail="Project not found")

    org_id = project_result["data"][0]["organization_id"]
    await verify_org_access(
        org_id,
        user["user_id"],
        user_email=user.get("email"),
        request=request,
    )

    # Get GitHub token
    github_token = body.github_token
    if not github_token:
        # Try to get from project integration
        integration_result = await supabase.request(
            f"/integrations?project_id=eq.{project_id}&type=eq.github&status=eq.connected&select=credentials"
        )
        if integration_result.get("data"):
            from src.services.key_encryption import decrypt_api_key

            encrypted_creds = integration_result["data"][0].get("credentials", {})
            if encrypted_creds.get("token"):
                try:
                    github_token = decrypt_api_key(encrypted_creds["token"])
                except Exception:
                    pass

    if not github_token:
        raise HTTPException(
            status_code=400,
            detail="GitHub token required. Provide in request or configure integration.",
        )

    # Fetch PR files from GitHub
    url = f"https://api.github.com/repos/{body.owner}/{body.repo}/pulls/{body.pr_number}/files"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers)
            if not response.is_success:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to fetch PR files: {response.text[:200]}",
                )

            pr_files = response.json()
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to connect to GitHub: {str(e)}",
            )

    # Extract file paths
    changed_files = [f.get("filename") for f in pr_files if f.get("filename")]

    if not changed_files:
        return SASTAnalysisResponse(
            findings=[],
            security_risk_score=0.0,
            files_scanned=0,
            scan_duration_seconds=0.0,
            summary={"message": "No changed files in PR"},
        )

    # For PR scanning, we need to have local files
    # This assumes the repo is cloned locally or we need to fetch content
    # For now, we'll log a warning if files don't exist locally
    local_files = [f for f in changed_files if Path(f).exists()]

    if not local_files:
        logger.warning(
            "Changed files not found locally",
            pr_number=body.pr_number,
            file_count=len(changed_files),
        )
        return SASTAnalysisResponse(
            findings=[],
            security_risk_score=0.0,
            files_scanned=0,
            scan_duration_seconds=0.0,
            error="Changed files not available locally. Clone the repository first.",
            summary={"changed_files": changed_files},
        )

    # Run analysis on local files
    custom_rules = str(get_custom_rules_path()) if get_custom_rules_path().exists() else None
    result = await analyze_files(
        files=local_files,
        custom_rules_path=custom_rules,
        include_auto_rules=True,
    )

    logger.info(
        "PR SAST scan completed",
        project_id=project_id,
        pr_number=body.pr_number,
        findings_count=len(result.findings),
        risk_score=result.security_risk_score,
        user_id=user["user_id"],
    )

    # Build summary
    severity_counts = {}
    for finding in result.findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    return SASTAnalysisResponse(
        findings=[f.to_dict() for f in result.findings],
        security_risk_score=result.security_risk_score,
        files_scanned=result.files_scanned,
        scan_duration_seconds=result.scan_duration_seconds,
        error=result.error,
        summary={
            "total_findings": len(result.findings),
            "by_severity": severity_counts,
            "pr_number": body.pr_number,
            "files_analyzed": len(local_files),
            "top_rules": _get_top_rules(result.findings),
        },
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _get_top_rules(findings: list[SecurityFinding], limit: int = 5) -> list[dict]:
    """Get the most frequently triggered rules.

    Args:
        findings: List of security findings
        limit: Maximum number of rules to return

    Returns:
        List of dicts with rule_id and count
    """
    rule_counts: dict[str, int] = {}
    for finding in findings:
        rule_counts[finding.rule_id] = rule_counts.get(finding.rule_id, 0) + 1

    sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"rule_id": r[0], "count": r[1]} for r in sorted_rules[:limit]]


# =============================================================================
# Integration with Commit Analysis
# =============================================================================


async def analyze_commit_files(
    files: list[str],
) -> tuple[list[dict], float]:
    """Analyze files from a commit for security vulnerabilities.

    This function is designed to be called from the commit analysis pipeline
    in github_webhooks.py.

    Args:
        files: List of changed file paths

    Returns:
        Tuple of (security_vulnerabilities list, security_risk_score)
    """
    if not files:
        return [], 0.0

    # Get custom rules if available
    custom_rules = str(get_custom_rules_path()) if get_custom_rules_path().exists() else None

    result = await analyze_files(
        files=files,
        custom_rules_path=custom_rules,
        include_auto_rules=True,
    )

    # Convert findings to dict format expected by commit analysis
    vulnerabilities = []
    for finding in result.findings:
        vulnerabilities.append({
            "severity": finding.severity,
            "message": finding.message,
            "file": finding.file_path,
            "line": finding.line_number,
            "rule_id": finding.rule_id,
            "cwe": finding.cwe,
            "fix_suggestion": finding.fix_suggestion,
        })

    return vulnerabilities, result.security_risk_score
