"""Tests for the Failure Patterns API module (src/api/failure_patterns.py).

Covers:
- Pattern CRUD endpoints (list, get, create, delete, activate)
- Prediction endpoint
- Outcome recording endpoint
- Feature extraction endpoint
- Training job endpoints
- Accuracy statistics endpoint
- Pattern matching logic
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from fastapi import HTTPException


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_supabase():
    """Create a mock Supabase client."""
    mock = MagicMock()
    mock.request = AsyncMock()
    mock.insert = AsyncMock()
    mock.update = AsyncMock()
    mock.rpc = AsyncMock()
    return mock


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    request = MagicMock()
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-agent"}
    request.query_params = {}
    return request


@pytest.fixture
def mock_user_context():
    """Create a mock UserContext."""
    from src.api.security.auth import UserContext, AuthMethod

    return UserContext(
        user_id="user-123",
        organization_id=str(uuid.uuid4()),
        email="test@example.com",
        auth_method=AuthMethod.API_KEY,
    )


@pytest.fixture
def sample_pattern_data():
    """Create sample pattern data."""
    return {
        "id": str(uuid.uuid4()),
        "project_id": str(uuid.uuid4()),
        "pattern_type": "file_change",
        "pattern_definition": {
            "file_patterns": ["src/auth/*"],
            "test_patterns": ["tests/auth/*"],
        },
        "description": "Auth file changes cause test failures",
        "target_tests": ["test_login", "test_logout"],
        "occurrences": 5,
        "predictions_made": 10,
        "predictions_correct": 8,
        "confidence": 0.80,
        "is_active": True,
        "discovered_by": "algorithm",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def sample_commit_features():
    """Create sample commit features."""
    return {
        "commit_sha": "abc123def456",
        "author_email": "dev@example.com",
        "author_name": "Test Developer",
        "commit_timestamp": datetime.now(UTC).isoformat(),
        "day_of_week": 4,  # Friday
        "hour_of_day": 15,  # 3pm
        "files_changed": 5,
        "lines_added": 100,
        "lines_deleted": 50,
        "total_lines_changed": 150,
        "changed_files": ["src/auth/login.py", "src/auth/logout.py"],
        "changed_directories": ["src/auth"],
        "file_extensions": ["py"],
        "is_dependency_update": False,
        "touches_tests": False,
        "touches_config": False,
        "touches_migrations": False,
        "is_merge_commit": False,
        "is_revert": False,
    }


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestExtractCommitFeatures:
    """Tests for _extract_commit_features helper function."""

    def test_extract_basic_features(self, mock_env_vars):
        """Test extracting features from basic commit data."""
        from src.api.failure_patterns import _extract_commit_features

        commit_data = {
            "sha": "abc123",
            "files": ["src/main.py", "tests/test_main.py"],
            "timestamp": "2024-01-15T14:30:00Z",
            "author_email": "dev@example.com",
            "lines_added": 50,
            "lines_deleted": 20,
        }

        features = _extract_commit_features(commit_data)

        assert features["commit_sha"] == "abc123"
        assert features["files_changed"] == 2
        assert features["total_lines_changed"] == 70
        assert features["touches_tests"] is True
        assert "src" in features["changed_directories"]

    def test_extract_dependency_update(self, mock_env_vars):
        """Test detecting dependency updates."""
        from src.api.failure_patterns import _extract_commit_features

        commit_data = {
            "sha": "abc123",
            "files": ["package.json", "package-lock.json"],
            "timestamp": None,
        }

        features = _extract_commit_features(commit_data)

        assert features["is_dependency_update"] is True

    def test_extract_config_changes(self, mock_env_vars):
        """Test detecting config file changes."""
        from src.api.failure_patterns import _extract_commit_features

        commit_data = {
            "sha": "abc123",
            "files": ["config.yml", ".env.example"],
            "timestamp": None,
        }

        features = _extract_commit_features(commit_data)

        assert features["touches_config"] is True

    def test_extract_revert_commit(self, mock_env_vars):
        """Test detecting revert commits."""
        from src.api.failure_patterns import _extract_commit_features

        commit_data = {
            "sha": "abc123",
            "files": ["src/main.py"],
            "message": "Revert: Fix bug in main",
            "timestamp": None,
        }

        features = _extract_commit_features(commit_data)

        assert features["is_revert"] is True


class TestMatchPattern:
    """Tests for _match_pattern helper function."""

    def test_match_file_change_pattern(self, mock_env_vars):
        """Test matching file change patterns."""
        from src.api.failure_patterns import _match_pattern

        pattern = {
            "pattern_type": "file_change",
            "pattern_definition": {
                "file_patterns": ["src/auth/*"],
            },
        }

        features = {
            "changed_files": ["src/auth/login.py"],
        }

        matches, score, reason = _match_pattern(pattern, features)

        assert matches is True
        assert score == 1.0
        assert "login.py" in reason

    def test_match_author_pattern(self, mock_env_vars):
        """Test matching author patterns."""
        from src.api.failure_patterns import _match_pattern

        pattern = {
            "pattern_type": "author",
            "pattern_definition": {
                "author_email": "risky@example.com",
            },
        }

        # Matching author
        features = {"author_email": "risky@example.com"}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is True

        # Non-matching author
        features = {"author_email": "safe@example.com"}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is False

    def test_match_time_of_day_pattern(self, mock_env_vars):
        """Test matching time of day patterns."""
        from src.api.failure_patterns import _match_pattern

        pattern = {
            "pattern_type": "time_of_day",
            "pattern_definition": {
                "days": [4],  # Friday
                "hours": [14, 15, 16, 17],  # 2pm-5pm
            },
        }

        # Friday 3pm - should match
        features = {"day_of_week": 4, "hour_of_day": 15}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is True

        # Monday 3pm - should not match
        features = {"day_of_week": 0, "hour_of_day": 15}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is False

    def test_match_size_pattern(self, mock_env_vars):
        """Test matching size patterns."""
        from src.api.failure_patterns import _match_pattern

        pattern = {
            "pattern_type": "size",
            "pattern_definition": {
                "min_lines": 200,
                "min_files": 10,
            },
        }

        # Large commit - should match
        features = {"total_lines_changed": 250, "files_changed": 5}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is True

        # Small commit - should not match
        features = {"total_lines_changed": 50, "files_changed": 2}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is False

    def test_match_dependency_pattern(self, mock_env_vars):
        """Test matching dependency patterns."""
        from src.api.failure_patterns import _match_pattern

        pattern = {
            "pattern_type": "dependency",
            "pattern_definition": {},
        }

        # Dependency update - should match
        features = {"is_dependency_update": True}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is True

        # No dependency update - should not match
        features = {"is_dependency_update": False}
        matches, score, reason = _match_pattern(pattern, features)
        assert matches is False

    def test_match_flaky_pattern(self, mock_env_vars):
        """Test flaky patterns always match."""
        from src.api.failure_patterns import _match_pattern

        pattern = {
            "pattern_type": "flaky",
            "pattern_definition": {},
        }

        features = {}
        matches, score, reason = _match_pattern(pattern, features)

        assert matches is True
        assert score == 0.5  # Lower score for flaky


# ============================================================================
# Pattern Endpoint Tests
# ============================================================================

class TestListPatterns:
    """Tests for list_patterns endpoint."""

    @pytest.mark.asyncio
    async def test_list_patterns_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_pattern_data):
        """Test successful pattern listing."""
        from src.api.failure_patterns import list_patterns

        project_id = sample_pattern_data["project_id"]
        mock_supabase.request.return_value = {"data": [sample_pattern_data], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await list_patterns(
                mock_request,
                project_id=project_id,
                min_confidence=0.0,  # Pass actual value, not Query object
                limit=50,
                user=mock_user_context,
            )

        assert len(result) == 1
        assert result[0].pattern_type == "file_change"

    @pytest.mark.asyncio
    async def test_list_patterns_with_filters(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_pattern_data):
        """Test pattern listing with filters."""
        from src.api.failure_patterns import list_patterns

        project_id = sample_pattern_data["project_id"]
        mock_supabase.request.return_value = {"data": [sample_pattern_data], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await list_patterns(
                mock_request,
                project_id=project_id,
                pattern_type="file_change",
                is_active=True,
                min_confidence=0.5,
                limit=50,
                user=mock_user_context,
            )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_patterns_table_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test graceful handling when table doesn't exist."""
        from src.api.failure_patterns import list_patterns

        project_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": None, "error": "42P01: table does not exist"}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await list_patterns(
                mock_request,
                project_id=project_id,
                min_confidence=0.0,
                limit=50,
                user=mock_user_context
            )

        assert result == []


class TestGetPattern:
    """Tests for get_pattern endpoint."""

    @pytest.mark.asyncio
    async def test_get_pattern_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_pattern_data):
        """Test successful pattern retrieval."""
        from src.api.failure_patterns import get_pattern

        pattern_id = sample_pattern_data["id"]
        mock_supabase.request.return_value = {"data": [sample_pattern_data], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await get_pattern(pattern_id, mock_request, user=mock_user_context)

        assert result.id == pattern_id
        assert result.pattern_type == "file_change"

    @pytest.mark.asyncio
    async def test_get_pattern_not_found(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test pattern not found error."""
        from src.api.failure_patterns import get_pattern

        pattern_id = str(uuid.uuid4())
        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase):
            with pytest.raises(HTTPException) as exc_info:
                await get_pattern(pattern_id, mock_request, user=mock_user_context)

            assert exc_info.value.status_code == 404


class TestCreatePattern:
    """Tests for create_pattern endpoint."""

    @pytest.mark.asyncio
    async def test_create_pattern_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_pattern_data):
        """Test successful pattern creation."""
        from src.api.failure_patterns import create_pattern, CreatePatternRequest, PatternDefinition

        project_id = str(uuid.uuid4())
        create_request = CreatePatternRequest(
            pattern_type="file_change",
            pattern_definition=PatternDefinition(
                file_patterns=["src/auth/*"],
            ),
            description="Auth changes cause failures",
            confidence=0.7,
        )

        mock_supabase.insert.return_value = {"data": [sample_pattern_data], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await create_pattern(
                mock_request,
                pattern=create_request,
                project_id=project_id,
                user=mock_user_context,
            )

        assert result.pattern_type == "file_change"

    @pytest.mark.asyncio
    async def test_create_pattern_invalid_type(self, mock_env_vars):
        """Test pattern type validation."""
        from src.api.failure_patterns import CreatePatternRequest, PatternDefinition
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreatePatternRequest(
                pattern_type="invalid_type",
                pattern_definition=PatternDefinition(),
            )


class TestDeactivatePattern:
    """Tests for deactivate_pattern endpoint."""

    @pytest.mark.asyncio
    async def test_deactivate_pattern_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful pattern deactivation."""
        from src.api.failure_patterns import deactivate_pattern

        pattern_id = str(uuid.uuid4())
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase):
            result = await deactivate_pattern(pattern_id, mock_request, user=mock_user_context)

        assert result["success"] is True
        assert result["is_active"] is False


class TestActivatePattern:
    """Tests for activate_pattern endpoint."""

    @pytest.mark.asyncio
    async def test_activate_pattern_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful pattern activation."""
        from src.api.failure_patterns import activate_pattern

        pattern_id = str(uuid.uuid4())
        mock_supabase.update.return_value = {"data": [], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase):
            result = await activate_pattern(pattern_id, mock_request, user=mock_user_context)

        assert result["success"] is True
        assert result["is_active"] is True


# ============================================================================
# Prediction Endpoint Tests
# ============================================================================

class TestPredictFailure:
    """Tests for predict_failure endpoint."""

    @pytest.mark.asyncio
    async def test_predict_failure_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_pattern_data, sample_commit_features):
        """Test successful failure prediction."""
        from src.api.failure_patterns import predict_failure

        project_id = sample_pattern_data["project_id"]
        commit_sha = "abc123"

        mock_supabase.request.side_effect = [
            {"data": [sample_commit_features], "error": None},  # Features
            {"data": [sample_pattern_data], "error": None},  # Patterns
        ]

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await predict_failure(
                mock_request,
                project_id=project_id,
                commit_sha=commit_sha,
                user=mock_user_context,
            )

        assert result.commit_sha == commit_sha
        assert 0 <= result.failure_probability <= 1
        assert result.confidence_level in ["high", "medium", "low"]

    @pytest.mark.asyncio
    async def test_predict_failure_no_features(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test prediction when commit features not found."""
        from src.api.failure_patterns import predict_failure

        project_id = str(uuid.uuid4())
        commit_sha = "nonexistent"

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            with pytest.raises(HTTPException) as exc_info:
                await predict_failure(
                    mock_request,
                    project_id=project_id,
                    commit_sha=commit_sha,
                    user=mock_user_context,
                )

            assert exc_info.value.status_code == 404


# ============================================================================
# Feature Extraction Endpoint Tests
# ============================================================================

class TestExtractFeatures:
    """Tests for extract_features endpoint."""

    @pytest.mark.asyncio
    async def test_extract_features_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful feature extraction."""
        from src.api.failure_patterns import extract_features, CommitFeatures

        project_id = str(uuid.uuid4())
        features = CommitFeatures(
            commit_sha="abc123",
            author_email="dev@example.com",
            files_changed=5,
            lines_added=100,
            lines_deleted=50,
        )

        mock_supabase.request.return_value = {"data": [], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await extract_features(
                mock_request,
                features=features,
                project_id=project_id,
                user=mock_user_context,
            )

        assert result["success"] is True
        assert result["commit_sha"] == "abc123"


# ============================================================================
# Outcome Recording Endpoint Tests
# ============================================================================

class TestRecordOutcome:
    """Tests for record_outcome endpoint."""

    @pytest.mark.asyncio
    async def test_record_outcome_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful outcome recording."""
        from src.api.failure_patterns import record_outcome, RecordOutcomeRequest

        project_id = str(uuid.uuid4())
        outcome_request = RecordOutcomeRequest(
            commit_sha="abc123",
            test_name="test_login",
            outcome="passed",
        )

        mock_supabase.request.return_value = {"data": [], "error": None}
        mock_supabase.rpc.return_value = {"data": True, "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await record_outcome(
                mock_request,
                outcome=outcome_request,
                project_id=project_id,
                user=mock_user_context,
            )

        assert result["success"] is True

    def test_outcome_validation(self, mock_env_vars):
        """Test outcome value validation."""
        from src.api.failure_patterns import RecordOutcomeRequest
        from pydantic import ValidationError

        # Valid outcomes
        RecordOutcomeRequest(commit_sha="abc", test_name="test", outcome="passed")
        RecordOutcomeRequest(commit_sha="abc", test_name="test", outcome="failed")
        RecordOutcomeRequest(commit_sha="abc", test_name="test", outcome="skipped")
        RecordOutcomeRequest(commit_sha="abc", test_name="test", outcome="error")

        # Invalid outcome
        with pytest.raises(ValidationError):
            RecordOutcomeRequest(commit_sha="abc", test_name="test", outcome="invalid")


# ============================================================================
# Training Job Endpoint Tests
# ============================================================================

class TestTrainPatterns:
    """Tests for train_patterns endpoint."""

    @pytest.mark.asyncio
    async def test_train_patterns_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful training job initiation."""
        from src.api.failure_patterns import train_patterns
        from fastapi import BackgroundTasks

        project_id = str(uuid.uuid4())
        job_data = {
            "id": str(uuid.uuid4()),
            "job_type": "incremental",
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
        }

        mock_supabase.insert.return_value = {"data": [job_data], "error": None}
        background_tasks = BackgroundTasks()

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await train_patterns(
                mock_request,
                background_tasks,
                project_id=project_id,
                job_type="incremental",
                days=30,
                user=mock_user_context,
            )

        assert result.status == "pending"
        assert result.job_type == "incremental"

    @pytest.mark.asyncio
    async def test_train_patterns_invalid_job_type(self, mock_env_vars, mock_request, mock_user_context):
        """Test invalid job type validation."""
        from src.api.failure_patterns import train_patterns
        from fastapi import BackgroundTasks

        project_id = str(uuid.uuid4())
        background_tasks = BackgroundTasks()

        with patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):
            with pytest.raises(HTTPException) as exc_info:
                await train_patterns(
                    mock_request,
                    background_tasks,
                    project_id=project_id,
                    job_type="invalid_type",
                    user=mock_user_context,
                )

            assert exc_info.value.status_code == 400


class TestGetTrainingStatus:
    """Tests for get_training_status endpoint."""

    @pytest.mark.asyncio
    async def test_get_training_status_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context):
        """Test successful training status retrieval."""
        from src.api.failure_patterns import get_training_status

        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "job_type": "incremental",
            "status": "running",
            "progress": 50.0,
            "commits_analyzed": 25,
            "test_runs_analyzed": 100,
            "patterns_discovered": 3,
            "patterns_updated": 2,
            "patterns_deactivated": 0,
            "created_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
        }

        mock_supabase.request.return_value = {"data": [job_data], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase):
            result = await get_training_status(job_id, mock_request, user=mock_user_context)

        assert result.id == job_id
        assert result.status == "running"
        assert result.progress == 50.0


# ============================================================================
# Accuracy Statistics Endpoint Tests
# ============================================================================

class TestGetAccuracyStats:
    """Tests for get_accuracy_stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_accuracy_stats_success(self, mock_env_vars, mock_supabase, mock_request, mock_user_context, sample_pattern_data):
        """Test successful accuracy stats retrieval."""
        from src.api.failure_patterns import get_accuracy_stats

        project_id = sample_pattern_data["project_id"]
        mock_supabase.request.return_value = {"data": [sample_pattern_data], "error": None}

        with patch("src.api.failure_patterns.get_supabase_client", return_value=mock_supabase), \
             patch("src.api.failure_patterns.validate_project_ownership", AsyncMock()):

            result = await get_accuracy_stats(mock_request, project_id=project_id, user=mock_user_context)

        assert result.project_id == project_id
        assert result.total_patterns >= 0
        assert 0 <= result.overall_accuracy <= 1


# ============================================================================
# Model Validation Tests
# ============================================================================

class TestFailurePatternModel:
    """Tests for FailurePattern model validation."""

    def test_valid_pattern(self, mock_env_vars):
        """Test creating a valid pattern."""
        from src.api.failure_patterns import FailurePattern

        pattern = FailurePattern(
            id=str(uuid.uuid4()),
            project_id=str(uuid.uuid4()),
            pattern_type="file_change",
            pattern_definition={"file_patterns": ["*.py"]},
            occurrences=5,
            predictions_made=10,
            predictions_correct=8,
            accuracy=0.8,
            confidence=0.75,
            is_active=True,
            discovered_by="algorithm",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        assert pattern.accuracy == 0.8
        assert pattern.is_active is True


class TestCommitFeaturesModel:
    """Tests for CommitFeatures model validation."""

    def test_valid_commit_features(self, mock_env_vars):
        """Test creating valid commit features."""
        from src.api.failure_patterns import CommitFeatures

        features = CommitFeatures(
            commit_sha="abc123",
            author_email="dev@example.com",
            files_changed=5,
            lines_added=100,
            lines_deleted=50,
        )

        assert features.commit_sha == "abc123"
        assert features.files_changed == 5


class TestPatternDefinitionModel:
    """Tests for PatternDefinition model validation."""

    def test_file_change_definition(self, mock_env_vars):
        """Test file change pattern definition."""
        from src.api.failure_patterns import PatternDefinition

        definition = PatternDefinition(
            file_patterns=["src/*.py"],
            test_patterns=["tests/*.py"],
        )

        assert len(definition.file_patterns) == 1

    def test_time_of_day_definition(self, mock_env_vars):
        """Test time of day pattern definition."""
        from src.api.failure_patterns import PatternDefinition

        definition = PatternDefinition(
            hours=[14, 15, 16, 17],
            days=[4],  # Friday
        )

        assert 4 in definition.days
        assert 15 in definition.hours

    def test_size_definition(self, mock_env_vars):
        """Test size pattern definition."""
        from src.api.failure_patterns import PatternDefinition

        definition = PatternDefinition(
            min_lines=200,
            min_files=10,
        )

        assert definition.min_lines == 200
        assert definition.min_files == 10
