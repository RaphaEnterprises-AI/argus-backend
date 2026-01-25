"""Tests for SAST Analysis API endpoints.

RAP-93: SAST Integration tests.

NOTE: These tests mock the subprocess calls to Semgrep. The actual Semgrep
binary may not be installed in the test environment.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSecurityFinding:
    """Tests for SecurityFinding dataclass."""

    def test_security_finding_creation(self, mock_env_vars):
        """Test creating a SecurityFinding."""
        from src.api.sast_analysis import SecurityFinding

        finding = SecurityFinding(
            rule_id="argus-sql-injection-fstring",
            severity="critical",
            message="SQL injection vulnerability detected",
            file_path="src/api/users.py",
            line_number=45,
            code_snippet='query = f"SELECT * FROM users WHERE id = {user_id}"',
            fix_suggestion="Use parameterized queries",
            cwe="CWE-89",
            owasp="A03:2021",
            confidence="high",
            end_line=45,
            column=10,
        )

        assert finding.rule_id == "argus-sql-injection-fstring"
        assert finding.severity == "critical"
        assert finding.line_number == 45
        assert finding.cwe == "CWE-89"

    def test_security_finding_to_dict(self, mock_env_vars):
        """Test SecurityFinding.to_dict() method."""
        from src.api.sast_analysis import SecurityFinding

        finding = SecurityFinding(
            rule_id="test-rule",
            severity="high",
            message="Test message",
            file_path="test.py",
            line_number=10,
            code_snippet="test code",
        )

        result = finding.to_dict()

        assert isinstance(result, dict)
        assert result["rule_id"] == "test-rule"
        assert result["severity"] == "high"
        assert result["message"] == "Test message"
        assert result["file_path"] == "test.py"
        assert result["line_number"] == 10


class TestSASTResult:
    """Tests for SASTResult dataclass."""

    def test_sast_result_creation(self, mock_env_vars):
        """Test creating a SASTResult."""
        from src.api.sast_analysis import SASTResult, SecurityFinding

        finding = SecurityFinding(
            rule_id="test",
            severity="medium",
            message="Test",
            file_path="test.py",
            line_number=1,
            code_snippet="test",
        )

        result = SASTResult(
            findings=[finding],
            security_risk_score=0.4,
            files_scanned=5,
            scan_duration_seconds=1.5,
        )

        assert len(result.findings) == 1
        assert result.security_risk_score == 0.4
        assert result.files_scanned == 5
        assert result.error is None

    def test_sast_result_to_dict(self, mock_env_vars):
        """Test SASTResult.to_dict() method."""
        from src.api.sast_analysis import SASTResult

        result = SASTResult(
            findings=[],
            security_risk_score=0.0,
            files_scanned=10,
            scan_duration_seconds=2.5,
            error=None,
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data["findings"] == []
        assert data["security_risk_score"] == 0.0
        assert data["files_scanned"] == 10


class TestSeverityWeights:
    """Tests for severity weights and constants."""

    def test_severity_weights_values(self, mock_env_vars):
        """Test severity weight values."""
        from src.api.sast_analysis import SEVERITY_WEIGHTS

        assert SEVERITY_WEIGHTS["critical"] == 1.0
        assert SEVERITY_WEIGHTS["high"] == 0.7
        assert SEVERITY_WEIGHTS["medium"] == 0.4
        assert SEVERITY_WEIGHTS["low"] == 0.1
        assert SEVERITY_WEIGHTS["info"] == 0.0

    def test_semgrep_severity_map(self, mock_env_vars):
        """Test Semgrep severity mapping."""
        from src.api.sast_analysis import SEMGREP_SEVERITY_MAP

        assert SEMGREP_SEVERITY_MAP["ERROR"] == "critical"
        assert SEMGREP_SEVERITY_MAP["WARNING"] == "high"
        assert SEMGREP_SEVERITY_MAP["INFO"] == "medium"


class TestRiskScoreCalculation:
    """Tests for calculate_security_risk_score function."""

    def test_empty_findings_returns_zero(self, mock_env_vars):
        """Test risk score is 0 for empty findings."""
        from src.api.sast_analysis import calculate_security_risk_score

        score = calculate_security_risk_score([])
        assert score == 0.0

    def test_single_critical_finding(self, mock_env_vars):
        """Test risk score for single critical finding."""
        from src.api.sast_analysis import SecurityFinding, calculate_security_risk_score

        findings = [
            SecurityFinding(
                rule_id="test",
                severity="critical",
                message="Critical issue",
                file_path="test.py",
                line_number=1,
                code_snippet="test",
                confidence="high",
            )
        ]

        score = calculate_security_risk_score(findings)
        assert score > 0.0
        assert score <= 1.0

    def test_multiple_findings_weighted(self, mock_env_vars):
        """Test risk score with multiple findings."""
        from src.api.sast_analysis import SecurityFinding, calculate_security_risk_score

        findings = [
            SecurityFinding(
                rule_id="test1",
                severity="critical",
                message="Critical",
                file_path="test.py",
                line_number=1,
                code_snippet="test",
                confidence="high",
            ),
            SecurityFinding(
                rule_id="test2",
                severity="high",
                message="High",
                file_path="test.py",
                line_number=2,
                code_snippet="test",
                confidence="high",
            ),
            SecurityFinding(
                rule_id="test3",
                severity="low",
                message="Low",
                file_path="test.py",
                line_number=3,
                code_snippet="test",
                confidence="medium",
            ),
        ]

        score = calculate_security_risk_score(findings)
        assert score > 0.0
        assert score <= 1.0

    def test_info_findings_contribute_nothing(self, mock_env_vars):
        """Test info severity findings contribute 0 to score."""
        from src.api.sast_analysis import SecurityFinding, calculate_security_risk_score

        findings = [
            SecurityFinding(
                rule_id="test",
                severity="info",
                message="Info",
                file_path="test.py",
                line_number=1,
                code_snippet="test",
                confidence="high",
            )
        ]

        score = calculate_security_risk_score(findings)
        assert score == 0.0


class TestParseSemgrepFindings:
    """Tests for parse_semgrep_findings function."""

    def test_parse_empty_findings(self, mock_env_vars):
        """Test parsing empty findings list."""
        from src.api.sast_analysis import parse_semgrep_findings

        result = parse_semgrep_findings([])
        assert result == []

    def test_parse_basic_finding(self, mock_env_vars):
        """Test parsing a basic Semgrep finding."""
        from src.api.sast_analysis import parse_semgrep_findings

        raw_findings = [
            {
                "check_id": "python.lang.security.audit.dangerous-code-detected",
                "path": "src/api/handler.py",
                "start": {"line": 42, "col": 5},
                "end": {"line": 42, "col": 25},
                "extra": {
                    "severity": "ERROR",
                    "message": "Detected dangerous code usage",
                    "lines": "    result = dangerous_call(user_input)",
                    "metadata": {
                        "cwe": ["CWE-95"],
                        "owasp": ["A03:2021"],
                        "confidence": "HIGH",
                    },
                },
            }
        ]

        findings = parse_semgrep_findings(raw_findings)

        assert len(findings) == 1
        assert findings[0].rule_id == "python.lang.security.audit.dangerous-code-detected"
        assert findings[0].severity == "critical"  # ERROR maps to critical
        assert findings[0].file_path == "src/api/handler.py"
        assert findings[0].line_number == 42
        assert findings[0].cwe == "CWE-95"

    def test_parse_finding_with_fix(self, mock_env_vars):
        """Test parsing finding with fix suggestion."""
        from src.api.sast_analysis import parse_semgrep_findings

        raw_findings = [
            {
                "check_id": "test-rule",
                "path": "test.py",
                "start": {"line": 1},
                "end": {"line": 1},
                "extra": {
                    "severity": "WARNING",
                    "message": "Issue detected",
                    "fix": "Use safe_method() instead",
                    "lines": "dangerous_call()",
                },
            }
        ]

        findings = parse_semgrep_findings(raw_findings)

        assert len(findings) == 1
        assert findings[0].fix_suggestion == "Use safe_method() instead"

    def test_parse_multiple_findings(self, mock_env_vars):
        """Test parsing multiple findings."""
        from src.api.sast_analysis import parse_semgrep_findings

        raw_findings = [
            {
                "check_id": "rule-1",
                "path": "file1.py",
                "start": {"line": 10},
                "end": {"line": 10},
                "extra": {"severity": "ERROR", "message": "Issue 1"},
            },
            {
                "check_id": "rule-2",
                "path": "file2.py",
                "start": {"line": 20},
                "end": {"line": 22},
                "extra": {"severity": "INFO", "message": "Issue 2"},
            },
        ]

        findings = parse_semgrep_findings(raw_findings)

        assert len(findings) == 2
        assert findings[0].file_path == "file1.py"
        assert findings[1].file_path == "file2.py"


class TestRunSemgrep:
    """Tests for run_semgrep function."""

    def test_run_semgrep_no_targets(self, mock_env_vars):
        """Test run_semgrep with no target paths."""
        from src.api.sast_analysis import run_semgrep

        findings, error = run_semgrep([])

        assert findings == []
        assert "No target paths provided" in error

    def test_run_semgrep_success(self, mock_env_vars):
        """Test successful Semgrep run."""
        from src.api.sast_analysis import run_semgrep

        mock_output = {
            "results": [
                {
                    "check_id": "test-rule",
                    "path": "test.py",
                    "start": {"line": 1},
                    "end": {"line": 1},
                    "extra": {"severity": "WARNING", "message": "Test"},
                }
            ],
            "errors": [],
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_output),
                stderr="",
            )

            findings, error = run_semgrep(["/path/to/file.py"])

            assert error is None
            assert len(findings) == 1
            mock_run.assert_called_once()

    def test_run_semgrep_with_findings(self, mock_env_vars):
        """Test Semgrep run that finds issues (exit code 1)."""
        from src.api.sast_analysis import run_semgrep

        mock_output = {
            "results": [
                {
                    "check_id": "security-issue",
                    "path": "vulnerable.py",
                    "start": {"line": 5},
                    "end": {"line": 5},
                    "extra": {"severity": "ERROR", "message": "Security issue"},
                }
            ],
            "errors": [],
        }

        with patch("subprocess.run") as mock_run:
            # Exit code 1 means findings were found
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout=json.dumps(mock_output),
                stderr="",
            )

            findings, error = run_semgrep(["/path/to/vulnerable.py"])

            assert error is None
            assert len(findings) == 1

    def test_run_semgrep_error(self, mock_env_vars):
        """Test Semgrep run with error."""
        from src.api.sast_analysis import run_semgrep

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=2,
                stdout="",
                stderr="Fatal error: invalid config",
            )

            findings, error = run_semgrep(["/path/to/file.py"])

            assert findings == []
            assert error is not None
            assert "Semgrep error" in error

    def test_run_semgrep_timeout(self, mock_env_vars):
        """Test Semgrep run timeout."""
        from subprocess import TimeoutExpired

        from src.api.sast_analysis import run_semgrep

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = TimeoutExpired("semgrep", 300)

            findings, error = run_semgrep(["/path/to/file.py"])

            assert findings == []
            assert "timed out" in error.lower()

    def test_run_semgrep_not_installed(self, mock_env_vars):
        """Test Semgrep not installed."""
        from src.api.sast_analysis import run_semgrep

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("semgrep not found")

            findings, error = run_semgrep(["/path/to/file.py"])

            assert findings == []
            assert "not installed" in error.lower()

    def test_run_semgrep_with_custom_rules(self, mock_env_vars):
        """Test Semgrep with custom rules path."""
        from src.api.sast_analysis import run_semgrep

        mock_output = {"results": [], "errors": []}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_output),
                stderr="",
            )

            findings, error = run_semgrep(
                ["/path/to/file.py"],
                custom_rules_path="/custom/rules.yaml",
            )

            # Verify custom rules were included in command
            call_args = mock_run.call_args[0][0]
            assert "--config" in call_args
            assert "/custom/rules.yaml" in call_args


class TestAnalyzeFiles:
    """Tests for analyze_files function."""

    @pytest.mark.asyncio
    async def test_analyze_files_empty_list(self, mock_env_vars):
        """Test analyzing empty file list."""
        from src.api.sast_analysis import analyze_files

        result = await analyze_files([])

        assert result.findings == []
        assert result.security_risk_score == 0.0
        assert "No existing files" in result.error

    @pytest.mark.asyncio
    async def test_analyze_files_nonexistent(self, mock_env_vars):
        """Test analyzing non-existent files."""
        from src.api.sast_analysis import analyze_files

        result = await analyze_files(["/nonexistent/file.py"])

        assert result.findings == []
        assert result.files_scanned == 0

    @pytest.mark.asyncio
    async def test_analyze_files_success(self, mock_env_vars, tmp_path):
        """Test successful file analysis."""
        from src.api.sast_analysis import analyze_files

        # Create a temporary file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        mock_output = {
            "results": [
                {
                    "check_id": "test-rule",
                    "path": str(test_file),
                    "start": {"line": 1},
                    "end": {"line": 1},
                    "extra": {"severity": "INFO", "message": "Test"},
                }
            ],
            "errors": [],
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_output),
                stderr="",
            )

            result = await analyze_files([str(test_file)])

            assert result.files_scanned == 1
            assert result.error is None


class TestAnalyzeRepository:
    """Tests for analyze_repository function."""

    @pytest.mark.asyncio
    async def test_analyze_repo_nonexistent(self, mock_env_vars):
        """Test analyzing non-existent repository."""
        from src.api.sast_analysis import analyze_repository

        result = await analyze_repository("/nonexistent/repo")

        assert result.findings == []
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_analyze_repo_success(self, mock_env_vars, tmp_path):
        """Test successful repository analysis."""
        from src.api.sast_analysis import analyze_repository

        mock_output = {"results": [], "errors": []}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_output),
                stderr="",
            )

            result = await analyze_repository(str(tmp_path))

            assert result.error is None
            mock_run.assert_called_once()


class TestFormatSecurityFindingsForPR:
    """Tests for format_security_findings_for_pr function."""

    def test_format_empty_findings(self, mock_env_vars):
        """Test formatting empty findings list."""
        from src.api.sast_analysis import format_security_findings_for_pr

        result = format_security_findings_for_pr([])
        assert result == ""

    def test_format_single_finding(self, mock_env_vars):
        """Test formatting single finding."""
        from src.api.sast_analysis import SecurityFinding, format_security_findings_for_pr

        findings = [
            SecurityFinding(
                rule_id="sql-injection",
                severity="critical",
                message="SQL injection vulnerability",
                file_path="users.py",
                line_number=45,
                code_snippet="query = ...",
            )
        ]

        result = format_security_findings_for_pr(findings)

        assert "Security Analysis" in result
        assert "users.py:45" in result
        assert "Critical" in result

    def test_format_multiple_findings_sorted_by_severity(self, mock_env_vars):
        """Test findings are sorted by severity."""
        from src.api.sast_analysis import SecurityFinding, format_security_findings_for_pr

        findings = [
            SecurityFinding(
                rule_id="low-issue",
                severity="low",
                message="Low severity",
                file_path="file1.py",
                line_number=1,
                code_snippet="code",
            ),
            SecurityFinding(
                rule_id="critical-issue",
                severity="critical",
                message="Critical severity",
                file_path="file2.py",
                line_number=2,
                code_snippet="code",
            ),
        ]

        result = format_security_findings_for_pr(findings)

        # Critical should appear before low
        critical_pos = result.find("Critical")
        low_pos = result.find("Low")
        assert critical_pos < low_pos

    def test_format_limits_displayed_findings(self, mock_env_vars):
        """Test that displayed findings are limited."""
        from src.api.sast_analysis import SecurityFinding, format_security_findings_for_pr

        # Create more than max_display (10) findings
        findings = [
            SecurityFinding(
                rule_id=f"rule-{i}",
                severity="medium",
                message=f"Issue {i}",
                file_path=f"file{i}.py",
                line_number=i,
                code_snippet="code",
            )
            for i in range(15)
        ]

        result = format_security_findings_for_pr(findings)

        # Should mention remaining findings
        assert "more findings" in result


class TestGetAvailableRules:
    """Tests for get_available_rules function."""

    def test_get_rules_includes_builtin(self, mock_env_vars):
        """Test that built-in rules are included."""
        from src.api.sast_analysis import get_available_rules

        rules = get_available_rules()

        # Should have at least the builtin semgrep-auto
        assert len(rules) >= 1
        builtin_ids = [r.id for r in rules if r.source == "builtin"]
        assert "semgrep-auto" in builtin_ids


class TestAnalyzeCommitFiles:
    """Tests for analyze_commit_files integration function."""

    @pytest.mark.asyncio
    async def test_analyze_commit_files_empty(self, mock_env_vars):
        """Test analyzing empty file list."""
        from src.api.sast_analysis import analyze_commit_files

        vulnerabilities, score = await analyze_commit_files([])

        assert vulnerabilities == []
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_analyze_commit_files_success(self, mock_env_vars, tmp_path):
        """Test successful commit file analysis."""
        from src.api.sast_analysis import analyze_commit_files

        # Create a temporary file
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        mock_output = {
            "results": [
                {
                    "check_id": "test-rule",
                    "path": str(test_file),
                    "start": {"line": 1},
                    "end": {"line": 1},
                    "extra": {
                        "severity": "ERROR",
                        "message": "Test issue",
                        "metadata": {"cwe": ["CWE-123"]},
                    },
                }
            ],
            "errors": [],
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps(mock_output),
                stderr="",
            )

            vulnerabilities, score = await analyze_commit_files([str(test_file)])

            assert len(vulnerabilities) == 1
            assert vulnerabilities[0]["severity"] == "critical"
            assert vulnerabilities[0]["rule_id"] == "test-rule"
            assert score > 0.0


class TestTopRulesHelper:
    """Tests for _get_top_rules helper function."""

    def test_get_top_rules_empty(self, mock_env_vars):
        """Test with empty findings."""
        from src.api.sast_analysis import _get_top_rules

        result = _get_top_rules([])
        assert result == []

    def test_get_top_rules_counts(self, mock_env_vars):
        """Test counting rules correctly."""
        from src.api.sast_analysis import SecurityFinding, _get_top_rules

        findings = [
            SecurityFinding(
                rule_id="rule-a",
                severity="high",
                message="Msg",
                file_path="f.py",
                line_number=1,
                code_snippet="c",
            ),
            SecurityFinding(
                rule_id="rule-a",
                severity="high",
                message="Msg",
                file_path="f2.py",
                line_number=2,
                code_snippet="c",
            ),
            SecurityFinding(
                rule_id="rule-b",
                severity="medium",
                message="Msg",
                file_path="f3.py",
                line_number=3,
                code_snippet="c",
            ),
        ]

        result = _get_top_rules(findings)

        assert len(result) == 2
        # rule-a should be first with count 2
        assert result[0]["rule_id"] == "rule-a"
        assert result[0]["count"] == 2
        assert result[1]["rule_id"] == "rule-b"
        assert result[1]["count"] == 1

    def test_get_top_rules_limited(self, mock_env_vars):
        """Test that results are limited."""
        from src.api.sast_analysis import SecurityFinding, _get_top_rules

        findings = [
            SecurityFinding(
                rule_id=f"rule-{i}",
                severity="medium",
                message="Msg",
                file_path="f.py",
                line_number=i,
                code_snippet="c",
            )
            for i in range(10)
        ]

        result = _get_top_rules(findings, limit=3)

        assert len(result) == 3


class TestAPIEndpoints:
    """Tests for SAST API endpoints."""

    @pytest.mark.asyncio
    async def test_analyze_endpoint_missing_params(self, mock_env_vars):
        """Test analyze endpoint with missing parameters."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.sast_analysis.get_current_user") as mock_user:
            mock_user.return_value = {"user_id": "test-user"}

            with patch("src.api.sast_analysis.get_supabase_client") as mock_supabase:
                mock_client = MagicMock()
                mock_client.request = AsyncMock(
                    return_value={"data": [{"organization_id": "org-123"}]}
                )
                mock_supabase.return_value = mock_client

                with patch("src.api.sast_analysis.verify_org_access", AsyncMock()):
                    client = TestClient(app)

                    response = client.post(
                        "/api/v1/sast/analyze?project_id=test-project",
                        json={},  # No files or repo_path
                    )

                    assert response.status_code == 400
                    assert "must be provided" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_rules_endpoint(self, mock_env_vars):
        """Test rules listing endpoint."""
        from fastapi.testclient import TestClient

        from src.api.server import app

        with patch("src.api.sast_analysis.get_current_user") as mock_user:
            mock_user.return_value = {"user_id": "test-user"}

            client = TestClient(app)

            response = client.get("/api/v1/sast/rules")

            assert response.status_code == 200
            data = response.json()
            assert "rules" in data
            assert "total_count" in data
            assert isinstance(data["rules"], list)
