"""Tests for parameterized testing models."""

from datetime import datetime

import pytest
from pydantic import ValidationError as PydanticValidationError


class TestDataSourceType:
    """Tests for DataSourceType enum."""

    def test_data_source_types_exist(self, mock_env_vars):
        """Test that all data source types are defined."""
        from src.parameterized.models import DataSourceType

        assert DataSourceType.INLINE == "inline"
        assert DataSourceType.CSV == "csv"
        assert DataSourceType.JSON == "json"
        assert DataSourceType.ENV == "env"
        assert DataSourceType.DATABASE == "database"
        assert DataSourceType.API == "api"


class TestIterationMode:
    """Tests for IterationMode enum."""

    def test_iteration_modes_exist(self, mock_env_vars):
        """Test that all iteration modes are defined."""
        from src.parameterized.models import IterationMode

        assert IterationMode.SEQUENTIAL == "sequential"
        assert IterationMode.PARALLEL == "parallel"
        assert IterationMode.RANDOM == "random"
        assert IterationMode.FIRST_FAILURE == "first_failure"


class TestParameterSet:
    """Tests for ParameterSet model."""

    def test_parameter_set_creation(self, mock_env_vars):
        """Test creating a ParameterSet."""
        from src.parameterized.models import ParameterSet

        ps = ParameterSet(
            name="admin_login",
            values={"username": "admin", "password": "admin123"},
            description="Test admin login",
            tags=["admin", "auth"],
        )

        assert ps.name == "admin_login"
        assert ps.values["username"] == "admin"
        assert len(ps.tags) == 2

    def test_parameter_set_empty_values_rejected(self, mock_env_vars):
        """Test that empty values dict is rejected."""
        from src.parameterized.models import ParameterSet

        with pytest.raises(PydanticValidationError):
            ParameterSet(name="empty", values={})

    def test_parameter_set_empty_name_rejected(self, mock_env_vars):
        """Test that empty name is rejected."""
        from src.parameterized.models import ParameterSet

        with pytest.raises(PydanticValidationError):
            ParameterSet(name="", values={"key": "value"})

    def test_parameter_set_name_validation(self, mock_env_vars):
        """Test that name is stripped."""
        from src.parameterized.models import ParameterSet

        ps = ParameterSet(name="  test name  ", values={"key": "value"})

        assert ps.name == "test name"

    def test_parameter_set_get_safe_name(self, mock_env_vars):
        """Test get_safe_name method."""
        from src.parameterized.models import ParameterSet

        ps = ParameterSet(
            name="Admin Login Test - Special!",
            values={"key": "value"},
        )

        safe_name = ps.get_safe_name()

        assert safe_name == "admin_login_test_special"
        assert " " not in safe_name
        assert "-" not in safe_name
        assert "!" not in safe_name

    def test_parameter_set_get_safe_name_underscores(self, mock_env_vars):
        """Test that consecutive underscores are collapsed."""
        from src.parameterized.models import ParameterSet

        ps = ParameterSet(name="Test___Name", values={"key": "value"})

        assert ps.get_safe_name() == "test_name"

    def test_parameter_set_skip(self, mock_env_vars):
        """Test skip functionality."""
        from src.parameterized.models import ParameterSet

        ps = ParameterSet(
            name="skipped_test",
            values={"key": "value"},
            skip=True,
            skip_reason="Known issue #123",
        )

        assert ps.skip is True
        assert ps.skip_reason == "Known issue #123"


class TestDataSource:
    """Tests for DataSource model."""

    def test_data_source_inline(self, mock_env_vars):
        """Test inline data source creation."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.INLINE,
            data=[
                {"username": "admin", "password": "admin123"},
                {"username": "user", "password": "user123"},
            ],
        )

        assert ds.type == DataSourceType.INLINE
        assert len(ds.data) == 2

    def test_data_source_inline_requires_data(self, mock_env_vars):
        """Test that inline type requires data field."""
        from src.parameterized.models import DataSource, DataSourceType

        with pytest.raises(PydanticValidationError, match="data"):
            DataSource(type=DataSourceType.INLINE)

    def test_data_source_csv(self, mock_env_vars):
        """Test CSV data source creation."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.CSV,
            path="test_data/users.csv",
            delimiter=",",
            encoding="utf-8",
        )

        assert ds.type == DataSourceType.CSV
        assert ds.path == "test_data/users.csv"

    def test_data_source_csv_requires_path(self, mock_env_vars):
        """Test that CSV type requires path field."""
        from src.parameterized.models import DataSource, DataSourceType

        with pytest.raises(PydanticValidationError, match="path"):
            DataSource(type=DataSourceType.CSV)

    def test_data_source_json(self, mock_env_vars):
        """Test JSON data source creation."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.JSON,
            path="test_data/users.json",
        )

        assert ds.type == DataSourceType.JSON
        assert ds.path == "test_data/users.json"

    def test_data_source_json_requires_path(self, mock_env_vars):
        """Test that JSON type requires path field."""
        from src.parameterized.models import DataSource, DataSourceType

        with pytest.raises(PydanticValidationError, match="path"):
            DataSource(type=DataSourceType.JSON)

    def test_data_source_env_with_prefix(self, mock_env_vars):
        """Test ENV data source with prefix."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.ENV,
            env_prefix="TEST_USER_",
        )

        assert ds.type == DataSourceType.ENV
        assert ds.env_prefix == "TEST_USER_"

    def test_data_source_env_with_mapping(self, mock_env_vars):
        """Test ENV data source with mapping."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.ENV,
            env_mapping={"MY_USER": "username", "MY_PASS": "password"},
        )

        assert ds.type == DataSourceType.ENV
        assert len(ds.env_mapping) == 2

    def test_data_source_env_requires_prefix_or_mapping(self, mock_env_vars):
        """Test that ENV type requires prefix or mapping."""
        from src.parameterized.models import DataSource, DataSourceType

        with pytest.raises(PydanticValidationError):
            DataSource(type=DataSourceType.ENV)

    def test_data_source_with_mapping(self, mock_env_vars):
        """Test data source with field mapping."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.INLINE,
            data=[{"user_name": "admin", "user_pass": "123"}],
            mapping={"user_name": "username", "user_pass": "password"},
        )

        assert ds.mapping is not None

    def test_data_source_with_filter(self, mock_env_vars):
        """Test data source with filter."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.INLINE,
            data=[{"role": "admin"}, {"role": "user"}],
            filter="role == 'admin'",
        )

        assert ds.filter == "role == 'admin'"

    def test_data_source_with_limit(self, mock_env_vars):
        """Test data source with limit."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.INLINE,
            data=[{"a": 1}, {"a": 2}, {"a": 3}],
            limit=2,
        )

        assert ds.limit == 2

    def test_data_source_limit_must_be_positive(self, mock_env_vars):
        """Test that limit must be positive."""
        from src.parameterized.models import DataSource, DataSourceType

        with pytest.raises(PydanticValidationError):
            DataSource(
                type=DataSourceType.INLINE,
                data=[{"a": 1}],
                limit=0,
            )

    def test_data_source_get_inline_data_no_mapping(self, mock_env_vars):
        """Test get_inline_data without mapping."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.INLINE,
            data=[{"username": "admin"}, {"username": "user"}],
        )

        data = ds.get_inline_data()

        assert len(data) == 2
        assert data[0]["username"] == "admin"

    def test_data_source_get_inline_data_with_mapping(self, mock_env_vars):
        """Test get_inline_data with mapping."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.INLINE,
            data=[{"user": "admin", "extra": "value"}],
            mapping={"user": "username"},
        )

        data = ds.get_inline_data()

        assert data[0]["username"] == "admin"
        assert data[0]["extra"] == "value"  # Unmapped fields preserved

    def test_data_source_get_inline_data_empty(self, mock_env_vars):
        """Test get_inline_data with empty data."""
        from src.parameterized.models import DataSource, DataSourceType

        ds = DataSource(
            type=DataSourceType.CSV,
            path="test.csv",
        )

        data = ds.get_inline_data()

        assert data == []


class TestTestStep:
    """Tests for TestStep model."""

    def test_test_step_creation(self, mock_env_vars):
        """Test creating a TestStep."""
        from src.parameterized.models import TestStep

        step = TestStep(
            action="fill",
            target="#username",
            value="{{username}}",
            timeout=5000,
            description="Fill username field",
        )

        assert step.action == "fill"
        assert step.target == "#username"
        assert step.value == "{{username}}"

    def test_test_step_get_parameter_placeholders(self, mock_env_vars):
        """Test getting placeholders from step."""
        from src.parameterized.models import TestStep

        step = TestStep(
            action="fill",
            target="#{{field_id}}",
            value="{{value}}",
            description="Fill {{field_name}}",
        )

        placeholders = step.get_parameter_placeholders()

        assert "field_id" in placeholders
        assert "value" in placeholders
        assert "field_name" in placeholders

    def test_test_step_no_placeholders(self, mock_env_vars):
        """Test step with no placeholders."""
        from src.parameterized.models import TestStep

        step = TestStep(action="click", target="#submit")

        placeholders = step.get_parameter_placeholders()

        assert len(placeholders) == 0


class TestTestAssertion:
    """Tests for TestAssertion model."""

    def test_test_assertion_creation(self, mock_env_vars):
        """Test creating a TestAssertion."""
        from src.parameterized.models import TestAssertion

        assertion = TestAssertion(
            type="url_contains",
            expected="{{expected_url}}",
            timeout=3000,
            soft=False,
        )

        assert assertion.type == "url_contains"
        assert assertion.expected == "{{expected_url}}"

    def test_test_assertion_get_parameter_placeholders(self, mock_env_vars):
        """Test getting placeholders from assertion."""
        from src.parameterized.models import TestAssertion

        assertion = TestAssertion(
            type="text_equals",
            target="#{{element_id}}",
            expected="{{expected_text}}",
            description="Check {{element_name}} text",
        )

        placeholders = assertion.get_parameter_placeholders()

        assert "element_id" in placeholders
        assert "expected_text" in placeholders
        assert "element_name" in placeholders

    def test_test_assertion_soft(self, mock_env_vars):
        """Test soft assertion flag."""
        from src.parameterized.models import TestAssertion

        assertion = TestAssertion(type="visible", target="#element", soft=True)

        assert assertion.soft is True


class TestParameterizedTest:
    """Tests for ParameterizedTest model."""

    def test_parameterized_test_creation(self, mock_env_vars):
        """Test creating a ParameterizedTest."""
        from src.parameterized.models import (
            DataSource,
            DataSourceType,
            ParameterizedTest,
            TestAssertion,
            TestStep,
        )

        test = ParameterizedTest(
            id="test-001",
            name="Login Test",
            description="Test login with different users",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"username": "admin", "password": "123"}],
            ),
            steps=[
                TestStep(action="fill", target="#username", value="{{username}}"),
                TestStep(action="fill", target="#password", value="{{password}}"),
                TestStep(action="click", target="#submit"),
            ],
            assertions=[
                TestAssertion(type="url_contains", expected="/dashboard"),
            ],
        )

        assert test.id == "test-001"
        assert test.name == "Login Test"
        assert len(test.steps) == 3
        assert len(test.assertions) == 1

    def test_parameterized_test_with_parameter_sets(self, mock_env_vars):
        """Test ParameterizedTest with explicit parameter sets."""
        from src.parameterized.models import ParameterizedTest, ParameterSet

        test = ParameterizedTest(
            name="Test with sets",
            parameter_sets=[
                ParameterSet(name="admin", values={"role": "admin"}),
                ParameterSet(name="user", values={"role": "user"}),
            ],
        )

        assert len(test.parameter_sets) == 2

    def test_parameterized_test_cannot_have_both_sources(self, mock_env_vars):
        """Test that both data_source and parameter_sets is rejected."""
        from src.parameterized.models import (
            DataSource,
            DataSourceType,
            ParameterizedTest,
            ParameterSet,
        )

        with pytest.raises(PydanticValidationError, match="Cannot specify both"):
            ParameterizedTest(
                name="Invalid test",
                data_source=DataSource(
                    type=DataSourceType.INLINE,
                    data=[{"a": 1}],
                ),
                parameter_sets=[ParameterSet(name="set1", values={"a": 1})],
            )

    def test_parameterized_test_get_all_placeholders(self, mock_env_vars):
        """Test getting all parameter placeholders."""
        from src.parameterized.models import (
            ParameterizedTest,
            TestAssertion,
            TestStep,
        )

        test = ParameterizedTest(
            name="Test",
            steps=[
                TestStep(action="fill", target="#user", value="{{username}}"),
            ],
            assertions=[
                TestAssertion(type="url_contains", expected="{{expected_url}}"),
            ],
            setup=[
                TestStep(action="goto", target="{{base_url}}"),
            ],
            teardown=[
                TestStep(action="click", target="#logout"),
            ],
        )

        placeholders = test.get_all_parameter_placeholders()

        assert "username" in placeholders
        assert "expected_url" in placeholders
        assert "base_url" in placeholders

    def test_parameterized_test_is_parameterized(self, mock_env_vars):
        """Test is_parameterized method."""
        from src.parameterized.models import (
            DataSource,
            DataSourceType,
            ParameterizedTest,
            ParameterSet,
        )

        # With data source
        test1 = ParameterizedTest(
            name="Test1",
            data_source=DataSource(type=DataSourceType.INLINE, data=[{"a": 1}]),
        )
        assert test1.is_parameterized() is True

        # With parameter sets
        test2 = ParameterizedTest(
            name="Test2",
            parameter_sets=[ParameterSet(name="set", values={"a": 1})],
        )
        assert test2.is_parameterized() is True

        # Without either
        test3 = ParameterizedTest(name="Test3")
        assert test3.is_parameterized() is False

    def test_parameterized_test_to_dict(self, mock_env_vars):
        """Test to_dict method."""
        from src.parameterized.models import ParameterizedTest, TestStep

        test = ParameterizedTest(
            id="test-001",
            name="Test",
            steps=[TestStep(action="click", target="#btn")],
        )

        result = test.to_dict()

        assert result["id"] == "test-001"
        assert result["name"] == "Test"
        assert len(result["steps"]) == 1


class TestParameterSetResult:
    """Tests for ParameterSetResult model."""

    def test_parameter_set_result_creation(self, mock_env_vars):
        """Test creating a ParameterSetResult."""
        from src.parameterized.models import ParameterSet, ParameterSetResult

        ps = ParameterSet(name="test", values={"a": 1})

        result = ParameterSetResult(
            parameter_set=ps,
            status="passed",
            duration_ms=1500.0,
            assertions_passed=5,
            assertions_failed=0,
        )

        assert result.status == "passed"
        assert result.duration_ms == 1500.0
        assert result.assertions_passed == 5

    def test_parameter_set_result_failed(self, mock_env_vars):
        """Test failed ParameterSetResult."""
        from src.parameterized.models import ParameterSet, ParameterSetResult

        ps = ParameterSet(name="failed_test", values={"a": 1})

        result = ParameterSetResult(
            parameter_set=ps,
            status="failed",
            error="Element not found",
            error_step=2,
            retry_count=3,
        )

        assert result.status == "failed"
        assert result.error == "Element not found"
        assert result.error_step == 2
        assert result.retry_count == 3


class TestParameterizedResult:
    """Tests for ParameterizedResult model."""

    def test_parameterized_result_creation(self, mock_env_vars):
        """Test creating a ParameterizedResult."""
        from src.parameterized.models import ParameterizedResult

        result = ParameterizedResult(
            test_id="test-001",
            test_name="Login Test",
            total_iterations=10,
            passed=8,
            failed=1,
            skipped=1,
            duration_ms=15000.0,
        )

        assert result.test_name == "Login Test"
        assert result.total_iterations == 10

    def test_parameterized_result_success_rate(self, mock_env_vars):
        """Test success_rate calculation."""
        from src.parameterized.models import ParameterizedResult

        result = ParameterizedResult(
            test_name="Test",
            total_iterations=10,
            passed=7,
            failed=2,
            skipped=1,
        )

        assert result.success_rate == 70.0

    def test_parameterized_result_success_rate_zero_iterations(self, mock_env_vars):
        """Test success_rate with zero iterations."""
        from src.parameterized.models import ParameterizedResult

        result = ParameterizedResult(test_name="Test", total_iterations=0)

        assert result.success_rate == 0.0

    def test_parameterized_result_overall_status_passed(self, mock_env_vars):
        """Test overall_status when all passed."""
        from src.parameterized.models import ParameterizedResult

        result = ParameterizedResult(
            test_name="Test",
            total_iterations=5,
            passed=5,
            failed=0,
            skipped=0,
        )

        assert result.overall_status == "passed"

    def test_parameterized_result_overall_status_failed(self, mock_env_vars):
        """Test overall_status when any failed."""
        from src.parameterized.models import ParameterizedResult

        result = ParameterizedResult(
            test_name="Test",
            total_iterations=5,
            passed=4,
            failed=1,
            skipped=0,
        )

        assert result.overall_status == "failed"

    def test_parameterized_result_overall_status_skipped(self, mock_env_vars):
        """Test overall_status when all skipped."""
        from src.parameterized.models import ParameterizedResult

        result = ParameterizedResult(
            test_name="Test",
            total_iterations=5,
            passed=0,
            failed=0,
            skipped=5,
        )

        assert result.overall_status == "skipped"

    def test_parameterized_result_to_dict(self, mock_env_vars):
        """Test to_dict method."""
        from src.parameterized.models import ParameterizedResult

        result = ParameterizedResult(
            test_id="test-001",
            test_name="Test",
            total_iterations=10,
            passed=9,
            failed=1,
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            completed_at=datetime(2024, 1, 15, 10, 5, 0),
        )

        d = result.to_dict()

        assert d["test_id"] == "test-001"
        assert d["success_rate"] == 90.0
        assert d["overall_status"] == "failed"
        assert "2024-01-15" in d["started_at"]


class TestValidationError:
    """Tests for ValidationError model."""

    def test_validation_error_creation(self, mock_env_vars):
        """Test creating a ValidationError."""
        from src.parameterized.models import ValidationError

        error = ValidationError(
            parameter="username",
            message="Parameter 'username' is required",
            location="step[0].value",
            severity="error",
        )

        assert error.parameter == "username"
        assert error.severity == "error"


class TestParameterValidationResult:
    """Tests for ParameterValidationResult model."""

    def test_create_valid_result(self, mock_env_vars):
        """Test create_valid class method."""
        from src.parameterized.models import ParameterValidationResult

        result = ParameterValidationResult.create_valid(
            required={"username", "password"},
            provided={"username", "password", "extra"},
            unused={"extra"},
        )

        assert result.valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1  # unused parameter warning

    def test_create_invalid_result(self, mock_env_vars):
        """Test create_invalid class method."""
        from src.parameterized.models import ParameterValidationResult

        result = ParameterValidationResult.create_invalid(
            required={"username", "password"},
            provided={"username"},
            missing={"password"},
        )

        assert result.valid is False
        assert len(result.errors) >= 1
        assert "password" in result.missing_parameters


class TestExpandedTest:
    """Tests for ExpandedTest model."""

    def test_expanded_test_creation(self, mock_env_vars):
        """Test creating an ExpandedTest."""
        from src.parameterized.models import ExpandedTest, ParameterSet

        ps = ParameterSet(name="admin", values={"username": "admin"})

        expanded = ExpandedTest(
            original_test_id="test-001",
            iteration_index=0,
            parameter_set=ps,
            name="Login Test [admin]",
            steps=[{"action": "click", "target": "#submit"}],
            assertions=[{"type": "visible", "target": "#welcome"}],
        )

        assert expanded.name == "Login Test [admin]"
        assert expanded.iteration_index == 0

    def test_expanded_test_to_test_spec(self, mock_env_vars):
        """Test to_test_spec method."""
        from src.parameterized.models import ExpandedTest, ParameterSet

        ps = ParameterSet(name="user1", values={"id": 1})

        expanded = ExpandedTest(
            original_test_id="test-001",
            iteration_index=2,
            parameter_set=ps,
            name="Test [user1]",
            description="Test for user1",
            steps=[{"action": "click", "target": "#btn"}],
        )

        spec = expanded.to_test_spec()

        assert spec["id"] == "test-001_2"
        assert spec["name"] == "Test [user1]"
        assert spec["metadata"]["parameterized"] is True
        assert spec["metadata"]["parameter_set_name"] == "user1"

    def test_expanded_test_to_test_spec_no_original_id(self, mock_env_vars):
        """Test to_test_spec without original_test_id."""
        from src.parameterized.models import ExpandedTest, ParameterSet

        ps = ParameterSet(name="test", values={"a": 1})

        expanded = ExpandedTest(
            iteration_index=5,
            parameter_set=ps,
            name="Test",
        )

        spec = expanded.to_test_spec()

        assert spec["id"] == "test_5"
