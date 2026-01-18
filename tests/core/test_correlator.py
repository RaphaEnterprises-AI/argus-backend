"""Tests for the error correlator module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os


class TestCorrelationType:
    """Tests for CorrelationType enum."""

    def test_correlation_types_exist(self, mock_env_vars):
        """Test that all correlation types are defined."""
        from src.core.correlator import CorrelationType

        assert CorrelationType.DIRECT == "direct"
        assert CorrelationType.FUNCTION == "function"
        assert CorrelationType.FILE == "file"
        assert CorrelationType.MODULE == "module"
        assert CorrelationType.SEMANTIC == "semantic"
        assert CorrelationType.PATTERN == "pattern"
        assert CorrelationType.TEMPORAL == "temporal"


class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum."""

    def test_confidence_levels_exist(self, mock_env_vars):
        """Test that all confidence levels are defined."""
        from src.core.correlator import ConfidenceLevel

        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
        assert ConfidenceLevel.SPECULATIVE == "speculative"


class TestCodeLocation:
    """Tests for CodeLocation dataclass."""

    def test_code_location_creation(self, mock_env_vars):
        """Test creating a CodeLocation instance."""
        from src.core.correlator import CodeLocation

        location = CodeLocation(
            file_path="src/utils/helpers.py",
            function_name="process_data",
            line_number=42,
            line_end=50,
            code_snippet="def process_data():\n    pass",
            last_modified=datetime.utcnow(),
            last_author="developer@example.com",
            commit_sha="abc123def456",
        )

        assert location.file_path == "src/utils/helpers.py"
        assert location.function_name == "process_data"
        assert location.line_number == 42

    def test_code_location_to_dict(self, mock_env_vars):
        """Test CodeLocation to_dict method."""
        from src.core.correlator import CodeLocation

        location = CodeLocation(
            file_path="test.py",
            function_name="test_func",
            line_number=10,
            last_modified=datetime(2024, 1, 15, 12, 0, 0),
            last_author="dev@test.com",
            commit_sha="abc123",
        )

        result = location.to_dict()

        assert result["file_path"] == "test.py"
        assert result["function_name"] == "test_func"
        assert result["line_number"] == 10
        assert result["last_author"] == "dev@test.com"
        assert result["commit_sha"] == "abc123"
        assert "2024-01-15" in result["last_modified"]

    def test_code_location_to_dict_no_dates(self, mock_env_vars):
        """Test CodeLocation to_dict with no dates."""
        from src.core.correlator import CodeLocation

        location = CodeLocation(file_path="test.py")

        result = location.to_dict()

        assert result["last_modified"] is None


class TestCorrelation:
    """Tests for Correlation dataclass."""

    def test_correlation_creation(self, mock_env_vars):
        """Test creating a Correlation instance."""
        from src.core.correlator import Correlation, CorrelationType, ConfidenceLevel, CodeLocation

        location = CodeLocation(file_path="src/auth.py", line_number=25)

        correlation = Correlation(
            id="corr-001",
            event_id="event-001",
            correlation_type=CorrelationType.DIRECT,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.95,
            location=location,
            reason="Direct stack trace match",
            evidence=["Stack frame #1", "Line 25"],
        )

        assert correlation.id == "corr-001"
        assert correlation.confidence == ConfidenceLevel.HIGH
        assert correlation.confidence_score == 0.95

    def test_correlation_to_dict(self, mock_env_vars):
        """Test Correlation to_dict method."""
        from src.core.correlator import Correlation, CorrelationType, ConfidenceLevel, CodeLocation

        location = CodeLocation(file_path="test.py", line_number=10)
        related = [CodeLocation(file_path="helper.py", line_number=5)]

        correlation = Correlation(
            id="corr-002",
            event_id="event-002",
            correlation_type=CorrelationType.FUNCTION,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.75,
            location=location,
            related_locations=related,
            reason="Function match",
            evidence=["Evidence 1"],
            semantic_analysis="Root cause is null check",
            suggested_fix="Add null validation",
        )

        result = correlation.to_dict()

        assert result["id"] == "corr-002"
        assert result["correlation_type"] == "function"
        assert result["confidence"] == "medium"
        assert len(result["related_locations"]) == 1
        assert result["semantic_analysis"] == "Root cause is null check"


class TestErrorPattern:
    """Tests for ErrorPattern dataclass."""

    def test_error_pattern_creation(self, mock_env_vars):
        """Test creating an ErrorPattern instance."""
        from src.core.correlator import ErrorPattern
        from src.core.normalizer import Severity

        pattern = ErrorPattern(
            id="pattern-001",
            name="NullPointerException in User Service",
            description="NPE when user profile is null",
            error_type="NullPointerException",
            affected_files=["src/user/service.py", "src/user/handler.py"],
            affected_components=["UserService", "UserHandler"],
            event_ids=["e1", "e2", "e3"],
            occurrence_count=3,
            first_seen=datetime(2024, 1, 1),
            last_seen=datetime(2024, 1, 15),
            root_cause="Missing null check in profile lookup",
            recommended_fix="Add null validation before accessing profile",
            severity=Severity.ERROR,
        )

        assert pattern.name == "NullPointerException in User Service"
        assert pattern.occurrence_count == 3
        assert len(pattern.event_ids) == 3

    def test_error_pattern_to_dict(self, mock_env_vars):
        """Test ErrorPattern to_dict method."""
        from src.core.correlator import ErrorPattern
        from src.core.normalizer import Severity

        pattern = ErrorPattern(
            id="pattern-002",
            name="Timeout Error",
            description="Database timeout",
            error_type="TimeoutError",
            occurrence_count=5,
            first_seen=datetime(2024, 1, 1),
            last_seen=datetime(2024, 1, 10),
            severity=Severity.WARNING,
        )

        result = pattern.to_dict()

        assert result["id"] == "pattern-002"
        assert result["name"] == "Timeout Error"
        assert result["severity"] == "warning"
        assert "2024-01-01" in result["first_seen"]


class TestErrorCorrelator:
    """Tests for ErrorCorrelator class."""

    def test_correlator_initialization_with_llm(self, mock_env_vars):
        """Test ErrorCorrelator initialization with LLM enabled."""
        from src.core.correlator import ErrorCorrelator

        with patch("anthropic.AsyncAnthropic"):
            correlator = ErrorCorrelator(codebase_path="/test/path", use_llm=True)

            assert correlator.codebase_path == Path("/test/path")
            assert correlator.use_llm is True
            assert correlator.client is not None

    def test_correlator_initialization_without_llm(self, mock_env_vars):
        """Test ErrorCorrelator initialization without LLM."""
        from src.core.correlator import ErrorCorrelator

        correlator = ErrorCorrelator(use_llm=False)

        assert correlator.use_llm is False
        assert correlator.client is None

    def test_is_library_path_node_modules(self, mock_env_vars):
        """Test library path detection for node_modules."""
        from src.core.correlator import ErrorCorrelator

        correlator = ErrorCorrelator(use_llm=False)

        assert correlator._is_library_path("node_modules/react/index.js") is True
        assert correlator._is_library_path("/app/node_modules/lodash/fp.js") is True

    def test_is_library_path_python_venv(self, mock_env_vars):
        """Test library path detection for Python venv."""
        from src.core.correlator import ErrorCorrelator

        correlator = ErrorCorrelator(use_llm=False)

        assert correlator._is_library_path(".venv/lib/python3.9/site-packages/flask/app.py") is True
        assert correlator._is_library_path("venv/lib/requests/api.py") is True
        assert correlator._is_library_path("site-packages/pandas/core.py") is True

    def test_is_library_path_build_dirs(self, mock_env_vars):
        """Test library path detection for build directories."""
        from src.core.correlator import ErrorCorrelator

        correlator = ErrorCorrelator(use_llm=False)

        assert correlator._is_library_path("dist/bundle.js") is True
        assert correlator._is_library_path("build/main.js") is True
        assert correlator._is_library_path(".next/server/pages.js") is True

    def test_is_library_path_app_code(self, mock_env_vars):
        """Test library path detection for app code."""
        from src.core.correlator import ErrorCorrelator

        correlator = ErrorCorrelator(use_llm=False)

        assert correlator._is_library_path("src/utils/helpers.py") is False
        assert correlator._is_library_path("app/components/Button.tsx") is False
        assert correlator._is_library_path("lib/auth.py") is False

    def test_deduplicate_correlations(self, mock_env_vars):
        """Test correlation deduplication."""
        from src.core.correlator import ErrorCorrelator, Correlation, CorrelationType, ConfidenceLevel, CodeLocation

        correlator = ErrorCorrelator(use_llm=False)

        correlations = [
            Correlation(
                id="1",
                event_id="e1",
                correlation_type=CorrelationType.DIRECT,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.95,
                location=CodeLocation(file_path="test.py", line_number=10),
            ),
            Correlation(
                id="2",
                event_id="e1",
                correlation_type=CorrelationType.FUNCTION,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.70,
                location=CodeLocation(file_path="test.py", line_number=10),  # Same location
            ),
            Correlation(
                id="3",
                event_id="e1",
                correlation_type=CorrelationType.FILE,
                confidence=ConfidenceLevel.LOW,
                confidence_score=0.50,
                location=CodeLocation(file_path="other.py", line_number=20),  # Different
            ),
        ]

        result = correlator._deduplicate_correlations(correlations)

        assert len(result) == 2  # Only 2 unique locations
        # Higher confidence should be kept
        test_py_corr = next(c for c in result if c.location.file_path == "test.py")
        assert test_py_corr.confidence_score == 0.95

    @pytest.mark.asyncio
    async def test_correlate_from_stack(self, mock_env_vars):
        """Test correlation from stack trace."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, StackFrame, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        event = NormalizedEvent(
            id="event-001",
            source=EventSource.SENTRY,
            external_id="ext-001",
            event_type=EventType.ERROR,
            title="TypeError in handler",
            severity=Severity.ERROR,
            fingerprint="fp-001",
            created_at=datetime.utcnow(),
            stack_frames=[
                StackFrame(
                    filename="src/api/handler.py",
                    function="process_request",
                    lineno=42,
                    in_app=True,
                ),
                StackFrame(
                    filename="src/utils/validator.py",
                    function="validate",
                    lineno=15,
                    in_app=True,
                ),
                StackFrame(
                    filename="node_modules/express/router.js",
                    function="handle",
                    lineno=100,
                    in_app=False,
                ),
            ],
        )

        correlations = await correlator._correlate_from_stack(event)

        # Should have 2 correlations (skips node_modules)
        assert len(correlations) == 2

        # First frame should have highest confidence
        first_corr = correlations[0]
        assert first_corr.confidence_score == 0.95
        assert first_corr.location.file_path == "src/api/handler.py"

    @pytest.mark.asyncio
    async def test_correlate_from_stack_skips_non_app(self, mock_env_vars):
        """Test that stack correlation skips non-app frames."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, StackFrame, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        event = NormalizedEvent(
            id="event-002",
            source=EventSource.SENTRY,
            external_id="ext-002",
            event_type=EventType.ERROR,
            title="Error",
            severity=Severity.ERROR,
            fingerprint="fp-002",
            created_at=datetime.utcnow(),
            stack_frames=[
                StackFrame(
                    filename="src/app.py",
                    function="main",
                    lineno=10,
                    in_app=False,  # Marked as not in_app
                ),
            ],
        )

        correlations = await correlator._correlate_from_stack(event)

        assert len(correlations) == 0

    @pytest.mark.asyncio
    async def test_correlate_from_file(self, mock_env_vars):
        """Test correlation from explicit file path."""
        from src.core.correlator import ErrorCorrelator, CorrelationType
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        event = NormalizedEvent(
            id="event-003",
            source=EventSource.SENTRY,
            external_id="ext-003",
            event_type=EventType.ERROR,
            title="Parse Error",
            severity=Severity.ERROR,
            fingerprint="fp-003",
            created_at=datetime.utcnow(),
            file_path="src/parser/json_parser.py",
            line_number=55,
            function_name="parse_object",
        )

        correlation = await correlator._correlate_from_file(event)

        assert correlation is not None
        assert correlation.correlation_type == CorrelationType.DIRECT
        assert correlation.confidence_score == 0.98
        assert correlation.location.file_path == "src/parser/json_parser.py"
        assert correlation.location.line_number == 55

    @pytest.mark.asyncio
    async def test_correlate_from_file_no_path(self, mock_env_vars):
        """Test file correlation returns None when no path."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        event = NormalizedEvent(
            id="event-004",
            source=EventSource.SENTRY,
            external_id="ext-004",
            event_type=EventType.ERROR,
            title="Unknown Error",
            severity=Severity.ERROR,
            fingerprint="fp-004",
            created_at=datetime.utcnow(),
            # No file_path
        )

        correlation = await correlator._correlate_from_file(event)

        assert correlation is None

    @pytest.mark.asyncio
    async def test_correlate_from_component(self, mock_env_vars):
        """Test correlation from UI component name."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test component files
            components_dir = Path(tmpdir) / "components"
            components_dir.mkdir()
            (components_dir / "UserProfile.tsx").write_text("export default function UserProfile() {}")
            (components_dir / "UserProfile.test.tsx").write_text("test file")

            correlator = ErrorCorrelator(codebase_path=tmpdir, use_llm=False)

            event = NormalizedEvent(
                id="event-005",
                source=EventSource.FULLSTORY,
                external_id="ext-005",
                event_type=EventType.ERROR,
                title="Component Error",
                severity=Severity.ERROR,
                fingerprint="fp-005",
                created_at=datetime.utcnow(),
                component="UserProfile",
            )

            correlations = await correlator._correlate_from_component(event)

            # Should find the component files
            assert len(correlations) >= 1
            assert any("UserProfile" in c.location.file_path for c in correlations)

    @pytest.mark.asyncio
    async def test_correlate_from_component_no_codebase(self, mock_env_vars):
        """Test component correlation returns empty when no codebase."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)  # No codebase_path

        event = NormalizedEvent(
            id="event-006",
            source=EventSource.FULLSTORY,
            external_id="ext-006",
            event_type=EventType.ERROR,
            title="Component Error",
            severity=Severity.ERROR,
            fingerprint="fp-006",
            created_at=datetime.utcnow(),
            component="SomeComponent",
        )

        correlations = await correlator._correlate_from_component(event)

        assert len(correlations) == 0

    @pytest.mark.asyncio
    async def test_correlate_event_full(self, mock_env_vars):
        """Test full event correlation."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, StackFrame, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        event = NormalizedEvent(
            id="event-007",
            source=EventSource.SENTRY,
            external_id="ext-007",
            event_type=EventType.ERROR,
            title="Full Error",
            severity=Severity.ERROR,
            fingerprint="fp-007",
            created_at=datetime.utcnow(),
            file_path="src/main.py",
            line_number=100,
            stack_frames=[
                StackFrame(
                    filename="src/main.py",
                    function="main",
                    lineno=100,
                    in_app=True,
                ),
            ],
        )

        correlations = await correlator.correlate_event(event, include_semantic=False)

        assert len(correlations) >= 1
        # Should be sorted by confidence
        assert all(
            correlations[i].confidence_score >= correlations[i + 1].confidence_score
            for i in range(len(correlations) - 1)
        )

    @pytest.mark.asyncio
    async def test_detect_patterns_basic(self, mock_env_vars):
        """Test basic pattern detection."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        # Create events with same fingerprint
        events = [
            NormalizedEvent(
                id=f"event-{i}",
                source=EventSource.SENTRY,
                external_id=f"ext-{i}",
                event_type=EventType.ERROR,
                title="TypeError: Cannot read property",
                message="Cannot read property 'name' of undefined",
                severity=Severity.ERROR,
                fingerprint="fp-common",  # Same fingerprint
                created_at=datetime(2024, 1, 1, i, 0, 0),
                error_type="TypeError",
                file_path="src/user.py",
            )
            for i in range(5)
        ]

        patterns = await correlator.detect_patterns(events, min_occurrences=3)

        assert len(patterns) >= 1
        pattern = patterns[0]
        assert pattern.occurrence_count == 5
        assert pattern.error_type == "TypeError"

    @pytest.mark.asyncio
    async def test_detect_patterns_min_occurrences(self, mock_env_vars):
        """Test pattern detection respects min_occurrences."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        # Create 2 events (below threshold)
        events = [
            NormalizedEvent(
                id=f"event-{i}",
                source=EventSource.SENTRY,
                external_id=f"ext-{i}",
                event_type=EventType.ERROR,
                title="Rare Error",
                severity=Severity.ERROR,
                fingerprint="fp-rare",
                created_at=datetime.utcnow(),
            )
            for i in range(2)
        ]

        patterns = await correlator.detect_patterns(events, min_occurrences=3)

        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_get_code_snippet(self, mock_env_vars):
        """Test getting code snippet from file."""
        from src.core.correlator import ErrorCorrelator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\n")

            correlator = ErrorCorrelator(codebase_path=tmpdir, use_llm=False)

            snippet = await correlator._get_code_snippet("test.py", 4, context_lines=2)

            assert snippet is not None
            assert "line 2" in snippet
            assert "line 3" in snippet
            assert "line 4" in snippet
            assert "line 5" in snippet
            assert "line 6" in snippet

    @pytest.mark.asyncio
    async def test_get_code_snippet_caches(self, mock_env_vars):
        """Test that code snippets are cached."""
        from src.core.correlator import ErrorCorrelator

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "cached.py"
            test_file.write_text("line 1\nline 2\nline 3\n")

            correlator = ErrorCorrelator(codebase_path=tmpdir, use_llm=False)

            # First call
            await correlator._get_code_snippet("cached.py", 2)

            # Should be cached
            assert str(Path(tmpdir) / "cached.py") in correlator._file_cache

    @pytest.mark.asyncio
    async def test_get_code_snippet_file_not_found(self, mock_env_vars):
        """Test code snippet returns None for missing file."""
        from src.core.correlator import ErrorCorrelator

        with tempfile.TemporaryDirectory() as tmpdir:
            correlator = ErrorCorrelator(codebase_path=tmpdir, use_llm=False)

            snippet = await correlator._get_code_snippet("nonexistent.py", 10)

            assert snippet is None

    @pytest.mark.asyncio
    async def test_get_code_snippet_no_line_number(self, mock_env_vars):
        """Test code snippet returns None without line number."""
        from src.core.correlator import ErrorCorrelator

        with tempfile.TemporaryDirectory() as tmpdir:
            correlator = ErrorCorrelator(codebase_path=tmpdir, use_llm=False)

            snippet = await correlator._get_code_snippet("test.py", None)

            assert snippet is None

    @pytest.mark.asyncio
    async def test_get_git_blame_caches(self, mock_env_vars):
        """Test that git blame results are cached."""
        from src.core.correlator import ErrorCorrelator

        with tempfile.TemporaryDirectory() as tmpdir:
            correlator = ErrorCorrelator(codebase_path=tmpdir, use_llm=False)

            # Pre-populate cache
            correlator._git_cache["test.py:10"] = {
                "author": "dev@test.com",
                "commit": "abc123",
            }

            result = await correlator._get_git_blame("test.py", 10)

            assert result["author"] == "dev@test.com"

    @pytest.mark.asyncio
    async def test_correlate_semantic_without_client(self, mock_env_vars):
        """Test semantic correlation returns empty without LLM client."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        correlator = ErrorCorrelator(use_llm=False)

        event = NormalizedEvent(
            id="event-010",
            source=EventSource.SENTRY,
            external_id="ext-010",
            event_type=EventType.ERROR,
            title="Error",
            severity=Severity.ERROR,
            fingerprint="fp-010",
            created_at=datetime.utcnow(),
        )

        correlations = await correlator._correlate_semantic(event, [])

        assert len(correlations) == 0

    @pytest.mark.asyncio
    async def test_enhance_pattern_with_llm(self, mock_env_vars):
        """Test pattern enhancement with LLM."""
        from src.core.correlator import ErrorCorrelator, ErrorPattern
        from src.core.normalizer import NormalizedEvent, EventType, Severity, EventSource

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='''{
            "name": "Enhanced Pattern Name",
            "root_cause": "Missing null check",
            "recommended_fix": "Add validation"
        }''')]

        with patch("anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = AsyncMock()
            mock_client.messages.create = AsyncMock(return_value=mock_response)
            mock_anthropic.return_value = mock_client

            correlator = ErrorCorrelator(use_llm=True)
            correlator.client = mock_client

            pattern = ErrorPattern(
                id="p1",
                name="Original Name",
                description="Original description",
                occurrence_count=5,
            )

            events = [
                NormalizedEvent(
                    id=f"e{i}",
                    source=EventSource.SENTRY,
                    external_id=f"ext-{i}",
                    event_type=EventType.ERROR,
                    title="Error",
                    message="Error message",
                    severity=Severity.ERROR,
                    fingerprint="fp",
                    created_at=datetime.utcnow(),
                )
                for i in range(3)
            ]

            result = await correlator._enhance_pattern_with_llm(pattern, events)

            assert result.name == "Enhanced Pattern Name"
            assert result.root_cause == "Missing null check"
            assert result.recommended_fix == "Add validation"


class TestErrorCorrelatorIntegration:
    """Integration tests for ErrorCorrelator."""

    @pytest.mark.asyncio
    async def test_full_correlation_pipeline(self, mock_env_vars):
        """Test full correlation pipeline without LLM."""
        from src.core.correlator import ErrorCorrelator
        from src.core.normalizer import NormalizedEvent, StackFrame, EventType, Severity, EventSource

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source files
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            (src_dir / "handler.py").write_text("""
def process_request(data):
    # Line 3
    result = validate(data)
    return result
""")

            correlator = ErrorCorrelator(codebase_path=tmpdir, use_llm=False)

            event = NormalizedEvent(
                id="event-full",
                source=EventSource.SENTRY,
                external_id="ext-full",
                event_type=EventType.ERROR,
                title="Processing Error",
                message="Failed to process request",
                severity=Severity.ERROR,
                fingerprint="fp-full",
                created_at=datetime.utcnow(),
                file_path="src/handler.py",
                line_number=4,
                function_name="process_request",
                stack_frames=[
                    StackFrame(
                        filename="src/handler.py",
                        function="process_request",
                        lineno=4,
                        in_app=True,
                    ),
                ],
            )

            correlations = await correlator.correlate_event(event, include_semantic=False)

            assert len(correlations) >= 1
            # Direct file correlation should be present
            assert any(c.location.file_path == "src/handler.py" for c in correlations)
