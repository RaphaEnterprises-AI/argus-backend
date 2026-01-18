"""Tests for parameterized testing engine."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestParameterizationError:
    """Tests for ParameterizationError exception."""

    def test_parameterization_error_creation(self, mock_env_vars):
        """Test creating a ParameterizationError."""
        from src.parameterized.engine import ParameterizationError

        error = ParameterizationError("Test error")

        assert str(error) == "Test error"


class TestParameterizationEngine:
    """Tests for ParameterizationEngine class."""

    def test_engine_initialization_default(self, mock_env_vars):
        """Test engine initialization with defaults."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine()

        assert engine.strict_validation is True
        assert engine.allow_undefined_params is False

    def test_engine_initialization_custom(self, mock_env_vars):
        """Test engine initialization with custom settings."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine(
            strict_validation=False,
            allow_undefined_params=True,
        )

        assert engine.strict_validation is False
        assert engine.allow_undefined_params is True

    def test_substitute_values_basic(self, mock_env_vars):
        """Test basic placeholder substitution."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine()

        result = engine.substitute_values(
            "Hello {{name}}, your role is {{role}}",
            {"name": "Alice", "role": "admin"},
        )

        assert result == "Hello Alice, your role is admin"

    def test_substitute_values_empty_template(self, mock_env_vars):
        """Test substitution with empty template."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine()

        result = engine.substitute_values("", {"name": "Alice"})

        assert result == ""

    def test_substitute_values_no_placeholders(self, mock_env_vars):
        """Test substitution with no placeholders."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine()

        result = engine.substitute_values("No placeholders here", {"name": "Alice"})

        assert result == "No placeholders here"

    def test_substitute_values_undefined_param_kept(self, mock_env_vars):
        """Test that undefined params are kept when allowed."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine(allow_undefined_params=True)

        result = engine.substitute_values(
            "{{defined}} and {{undefined}}",
            {"defined": "value"},
        )

        assert result == "value and {{undefined}}"

    def test_substitute_values_undefined_param_warning(self, mock_env_vars):
        """Test that undefined params log warning when not allowed."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine(allow_undefined_params=False)

        result = engine.substitute_values("{{undefined}}", {})

        # Should keep the placeholder
        assert result == "{{undefined}}"

    def test_substitute_values_numeric(self, mock_env_vars):
        """Test substitution with numeric values."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine()

        result = engine.substitute_values(
            "Count: {{count}}, Price: {{price}}",
            {"count": 42, "price": 99.99},
        )

        assert "42" in result
        assert "99.99" in result

    def test_expand_test_basic(self, mock_env_vars):
        """Test basic test expansion."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            ParameterSet,
            TestStep,
            TestAssertion,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            id="test-001",
            name="Login Test",
            description="Test {{user_type}} login",
            steps=[
                TestStep(action="fill", target="#username", value="{{username}}"),
                TestStep(action="fill", target="#password", value="{{password}}"),
            ],
            assertions=[
                TestAssertion(type="url_contains", expected="{{expected_url}}"),
            ],
        )

        params = ParameterSet(
            name="admin",
            values={
                "username": "admin",
                "password": "admin123",
                "expected_url": "/dashboard",
                "user_type": "admin",
            },
        )

        expanded = engine.expand_test(test, params)

        assert expanded.name == "Login Test [admin]"
        assert expanded.description == "Test admin login"
        assert expanded.steps[0]["value"] == "admin"
        assert expanded.steps[1]["value"] == "admin123"
        assert expanded.assertions[0]["expected"] == "/dashboard"

    def test_expand_test_with_setup_teardown(self, mock_env_vars):
        """Test expansion includes setup and teardown."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            ParameterSet,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            setup=[TestStep(action="goto", target="{{base_url}}/setup")],
            steps=[TestStep(action="click", target="#btn")],
            teardown=[TestStep(action="click", target="#logout")],
        )

        params = ParameterSet(name="test", values={"base_url": "https://example.com"})

        expanded = engine.expand_test(test, params)

        assert "https://example.com/setup" in str(expanded.setup)

    def test_expand_test_missing_param_strict(self, mock_env_vars):
        """Test expansion fails on missing param in strict mode."""
        from src.parameterized.engine import ParameterizationEngine, ParameterizationError
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine(strict_validation=True)

        test = ParameterizedTest(
            name="Test",
            steps=[TestStep(action="fill", target="#field", value="{{missing_param}}")],
        )

        params = ParameterSet(name="test", values={"other_param": "value"})

        with pytest.raises(ParameterizationError, match="validation failed"):
            engine.expand_test(test, params)

    def test_expand_test_missing_param_lenient(self, mock_env_vars):
        """Test expansion continues on missing param in lenient mode."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine(strict_validation=False)

        test = ParameterizedTest(
            name="Test",
            steps=[TestStep(action="fill", target="#field", value="{{missing_param}}")],
        )

        params = ParameterSet(name="test", values={"other_param": "value"})

        expanded = engine.expand_test(test, params)

        # Should still expand but keep placeholder
        assert expanded is not None

    def test_generate_test_name(self, mock_env_vars):
        """Test test name generation."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterSet

        engine = ParameterizationEngine()

        params = ParameterSet(name="Admin Login - Special!", values={"a": 1})
        name = engine._generate_test_name("Base Test", params)

        assert "Base Test" in name
        assert "admin_login_special" in name.lower()

    def test_validate_parameters_valid(self, mock_env_vars):
        """Test parameter validation - valid case."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            steps=[TestStep(action="fill", target="#user", value="{{username}}")],
        )

        params = ParameterSet(name="test", values={"username": "admin"})

        result = engine.validate_parameters(test, params)

        assert result.valid is True

    def test_validate_parameters_missing(self, mock_env_vars):
        """Test parameter validation - missing params."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            steps=[TestStep(action="fill", target="#user", value="{{username}}")],
        )

        params = ParameterSet(name="test", values={"other": "value"})

        result = engine.validate_parameters(test, params)

        assert result.valid is False
        assert "username" in result.missing_parameters

    def test_validate_parameters_unused(self, mock_env_vars):
        """Test parameter validation - unused params."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            steps=[TestStep(action="fill", target="#user", value="{{username}}")],
        )

        params = ParameterSet(
            name="test",
            values={"username": "admin", "extra": "unused"},
        )

        result = engine.validate_parameters(test, params)

        assert result.valid is True
        assert "extra" in result.unused_parameters

    def test_generate_test_matrix_basic(self, mock_env_vars):
        """Test basic test matrix generation."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            id="test-001",
            name="Login Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[
                    {"username": "admin", "password": "admin123"},
                    {"username": "user", "password": "user123"},
                ],
            ),
            steps=[
                TestStep(action="fill", target="#username", value="{{username}}"),
                TestStep(action="fill", target="#password", value="{{password}}"),
            ],
        )

        expanded = engine.generate_test_matrix(test)

        assert len(expanded) == 2
        assert expanded[0].iteration_index == 0
        assert expanded[1].iteration_index == 1
        assert expanded[0].steps[0]["value"] == "admin"
        assert expanded[1].steps[0]["value"] == "user"

    def test_generate_test_matrix_with_override(self, mock_env_vars):
        """Test matrix generation with data source override."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"a": 1}],  # Original
            ),
            steps=[TestStep(action="fill", target="#field", value="{{a}}")],
        )

        override = DataSource(
            type=DataSourceType.INLINE,
            data=[{"a": 10}, {"a": 20}],  # Override
        )

        expanded = engine.generate_test_matrix(test, override)

        assert len(expanded) == 2
        assert expanded[0].steps[0]["value"] == "10"

    def test_generate_test_matrix_with_parameter_sets(self, mock_env_vars):
        """Test matrix generation with explicit parameter sets."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            parameter_sets=[
                ParameterSet(name="set1", values={"value": "first"}),
                ParameterSet(name="set2", values={"value": "second"}),
            ],
            steps=[TestStep(action="fill", target="#field", value="{{value}}")],
        )

        expanded = engine.generate_test_matrix(test)

        assert len(expanded) == 2

    def test_generate_test_matrix_skips_skipped(self, mock_env_vars):
        """Test that skipped parameter sets are excluded."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            parameter_sets=[
                ParameterSet(name="active", values={"a": 1}),
                ParameterSet(name="skipped", values={"a": 2}, skip=True),
            ],
            steps=[TestStep(action="click", target="#btn")],
        )

        expanded = engine.generate_test_matrix(test)

        assert len(expanded) == 1
        assert expanded[0].parameter_set.name == "active"

    def test_generate_test_matrix_non_parameterized(self, mock_env_vars):
        """Test matrix generation for non-parameterized test."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, TestStep, ParameterSet

        engine = ParameterizationEngine()

        # Non-parameterized test still needs at least one parameter set
        test = ParameterizedTest(
            name="Simple Test",
            parameter_sets=[ParameterSet(name="default", values={"placeholder": "value"})],
            steps=[TestStep(action="click", target="#btn")],
        )

        expanded = engine.generate_test_matrix(test)

        assert len(expanded) == 1
        assert expanded[0].parameter_set.name == "default"

    def test_generate_test_matrix_no_data(self, mock_env_vars):
        """Test matrix generation raises error when parameterized but no data."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.data_sources import DataSourceError
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"a": 1}],  # Will be cleared
            ),
        )
        test.data_source.data = []  # Clear data

        # Data source raises error when data is empty/missing
        with pytest.raises(DataSourceError, match="Inline data source requires 'data' field"):
            engine.generate_test_matrix(test)

    def test_generate_test_matrix_random_mode(self, mock_env_vars):
        """Test matrix generation with random iteration mode."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            IterationMode,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            iteration_mode=IterationMode.RANDOM,
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"a": i} for i in range(10)],
            ),
            steps=[TestStep(action="click", target="#btn")],
        )

        expanded = engine.generate_test_matrix(test)

        assert len(expanded) == 10

    def test_validate_test_spec_valid(self, mock_env_vars):
        """Test test spec validation - valid case."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"username": "admin"}],
            ),
            steps=[TestStep(action="fill", target="#user", value="{{username}}")],
        )

        result = engine.validate_test_spec(test)

        assert result.valid is True

    def test_validate_test_spec_missing_params(self, mock_env_vars):
        """Test test spec validation - data missing required params."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"other": "value"}],  # Missing username
            ),
            steps=[TestStep(action="fill", target="#user", value="{{username}}")],
        )

        result = engine.validate_test_spec(test)

        assert result.valid is False

    def test_validate_test_spec_parameter_sets(self, mock_env_vars):
        """Test test spec validation with parameter sets."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest, ParameterSet, TestStep

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            parameter_sets=[
                ParameterSet(name="valid", values={"username": "admin"}),
                ParameterSet(name="invalid", values={"other": "value"}),  # Missing
            ],
            steps=[TestStep(action="fill", target="#user", value="{{username}}")],
        )

        result = engine.validate_test_spec(test)

        assert result.valid is False

    def test_preview_expanded_tests(self, mock_env_vars):
        """Test preview of expanded tests."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            TestStep,
            TestAssertion,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"a": i} for i in range(10)],
            ),
            steps=[TestStep(action="click", target="#btn")],
            assertions=[TestAssertion(type="visible", target="#result")],
        )

        previews = engine.preview_expanded_tests(test, limit=3)

        assert len(previews) == 3
        assert "name" in previews[0]
        assert "parameter_set" in previews[0]
        assert "step_count" in previews[0]
        assert "assertion_count" in previews[0]

    def test_create_empty_result(self, mock_env_vars):
        """Test creating empty result structure."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest

        engine = ParameterizationEngine()

        test = ParameterizedTest(id="test-001", name="My Test")

        result = engine.create_empty_result(test)

        assert result.test_id == "test-001"
        assert result.test_name == "My Test"
        assert result.total_iterations == 0
        assert result.passed == 0

    def test_create_empty_result_generates_id(self, mock_env_vars):
        """Test that empty result generates ID if missing."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import ParameterizedTest

        engine = ParameterizationEngine()

        test = ParameterizedTest(name="Test No ID")

        result = engine.create_empty_result(test)

        assert result.test_id is not None

    def test_merge_results_empty(self, mock_env_vars):
        """Test merging empty results list."""
        from src.parameterized.engine import ParameterizationEngine

        engine = ParameterizationEngine()

        merged = engine.merge_results([])

        assert merged.test_name == "empty"
        assert merged.total_iterations == 0

    def test_merge_results_multiple(self, mock_env_vars):
        """Test merging multiple results."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedResult,
            ParameterSetResult,
            ParameterSet,
        )

        engine = ParameterizationEngine()

        ps1 = ParameterSet(name="test1", values={"a": 1})
        ps2 = ParameterSet(name="test2", values={"a": 2})

        results = [
            ParameterizedResult(
                test_id="test-001",
                test_name="Test",
                total_iterations=5,
                passed=4,
                failed=1,
                skipped=0,
                duration_ms=1000.0,
                results=[
                    ParameterSetResult(parameter_set=ps1, status="passed"),
                ],
                started_at=datetime(2024, 1, 15, 10, 0, 0),
                completed_at=datetime(2024, 1, 15, 10, 1, 0),
            ),
            ParameterizedResult(
                test_id="test-001",
                test_name="Test",
                total_iterations=3,
                passed=3,
                failed=0,
                skipped=0,
                duration_ms=500.0,
                results=[
                    ParameterSetResult(parameter_set=ps2, status="passed"),
                ],
                started_at=datetime(2024, 1, 15, 10, 2, 0),
                completed_at=datetime(2024, 1, 15, 10, 3, 0),
            ),
        ]

        merged = engine.merge_results(results)

        assert merged.total_iterations == 8
        assert merged.passed == 7
        assert merged.failed == 1
        assert merged.duration_ms == 1500.0
        assert len(merged.results) == 2


class TestExpandParameterizedTest:
    """Tests for expand_parameterized_test convenience function."""

    def test_expand_from_dict(self, mock_env_vars):
        """Test expanding from dictionary spec."""
        from src.parameterized.engine import expand_parameterized_test

        test_spec = {
            "name": "Login Test",
            "data_source": {
                "type": "inline",
                "data": [
                    {"username": "admin", "password": "admin123"},
                    {"username": "user", "password": "user123"},
                ],
            },
            "steps": [
                {"action": "fill", "target": "#username", "value": "{{username}}"},
                {"action": "fill", "target": "#password", "value": "{{password}}"},
            ],
        }

        expanded = expand_parameterized_test(test_spec)

        assert len(expanded) == 2
        assert expanded[0]["metadata"]["parameterized"] is True

    def test_expand_with_override_data_source(self, mock_env_vars):
        """Test expanding with data source override."""
        from src.parameterized.engine import expand_parameterized_test

        test_spec = {
            "name": "Test",
            "steps": [{"action": "fill", "target": "#field", "value": "{{value}}"}],
        }

        data_source = {
            "type": "inline",
            "data": [{"value": "first"}, {"value": "second"}],
        }

        expanded = expand_parameterized_test(test_spec, data_source)

        assert len(expanded) == 2

    def test_expand_with_parameters_in_spec(self, mock_env_vars):
        """Test expanding with parameters field in spec."""
        from src.parameterized.engine import expand_parameterized_test

        test_spec = {
            "name": "Test",
            "parameters": {
                "data_source": {
                    "type": "inline",
                    "data": [{"a": 1}, {"a": 2}],
                },
            },
            "steps": [{"action": "click", "target": "#btn"}],
        }

        expanded = expand_parameterized_test(test_spec)

        assert len(expanded) == 2


class TestParameterizationEngineIterationModes:
    """Tests for different iteration modes."""

    def test_sequential_mode_preserves_order(self, mock_env_vars):
        """Test sequential mode preserves order."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            IterationMode,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            iteration_mode=IterationMode.SEQUENTIAL,
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"i": 0}, {"i": 1}, {"i": 2}],
            ),
            steps=[TestStep(action="click", target="#btn")],
        )

        expanded = engine.generate_test_matrix(test)

        # Check order is preserved
        for idx, exp in enumerate(expanded):
            assert exp.parameter_set.values["i"] == idx

    def test_parallel_mode_same_tests(self, mock_env_vars):
        """Test parallel mode generates same tests (just signals execution)."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            IterationMode,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            iteration_mode=IterationMode.PARALLEL,
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"i": 0}, {"i": 1}, {"i": 2}],
            ),
            steps=[TestStep(action="click", target="#btn")],
        )

        expanded = engine.generate_test_matrix(test)

        assert len(expanded) == 3

    def test_first_failure_mode_same_tests(self, mock_env_vars):
        """Test first_failure mode generates same tests (affects execution)."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            IterationMode,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            iteration_mode=IterationMode.FIRST_FAILURE,
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[{"i": 0}, {"i": 1}],
            ),
            steps=[TestStep(action="click", target="#btn")],
        )

        expanded = engine.generate_test_matrix(test)

        assert len(expanded) == 2


class TestParameterizationEngineEdgeCases:
    """Edge case tests for ParameterizationEngine."""

    def test_expand_test_all_step_types(self, mock_env_vars):
        """Test expansion handles all step field types."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            ParameterSet,
            TestStep,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            steps=[
                TestStep(
                    action="fill",
                    target="{{target}}",
                    value="{{value}}",
                    description="Fill {{field_name}}",
                    timeout=5000,
                ),
            ],
        )

        params = ParameterSet(
            name="test",
            values={"target": "#input", "value": "hello", "field_name": "username"},
        )

        expanded = engine.expand_test(test, params)

        assert expanded.steps[0]["target"] == "#input"
        assert expanded.steps[0]["value"] == "hello"
        assert expanded.steps[0]["description"] == "Fill username"

    def test_expand_test_all_assertion_types(self, mock_env_vars):
        """Test expansion handles all assertion field types."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            ParameterSet,
            TestAssertion,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Test",
            assertions=[
                TestAssertion(
                    type="text_equals",
                    target="{{selector}}",
                    expected="{{expected}}",
                    description="Check {{element}}",
                ),
            ],
        )

        params = ParameterSet(
            name="test",
            values={"selector": "#msg", "expected": "Hello", "element": "message"},
        )

        expanded = engine.expand_test(test, params)

        assert expanded.assertions[0]["target"] == "#msg"
        assert expanded.assertions[0]["expected"] == "Hello"

    def test_validation_handles_complex_test(self, mock_env_vars):
        """Test validation handles complex test with many placeholders."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            ParameterSet,
            TestStep,
            TestAssertion,
        )

        engine = ParameterizationEngine()

        test = ParameterizedTest(
            name="Complex Test",
            setup=[TestStep(action="goto", target="{{base_url}}/setup")],
            steps=[
                TestStep(action="fill", target="#{{field1}}", value="{{value1}}"),
                TestStep(action="fill", target="#{{field2}}", value="{{value2}}"),
            ],
            assertions=[
                TestAssertion(type="visible", target="{{result_selector}}"),
            ],
            teardown=[TestStep(action="click", target="#{{logout_btn}}")],
        )

        # All params provided
        params = ParameterSet(
            name="test",
            values={
                "base_url": "https://example.com",
                "field1": "username",
                "value1": "admin",
                "field2": "password",
                "value2": "123",
                "result_selector": "#welcome",
                "logout_btn": "logout",
            },
        )

        result = engine.validate_parameters(test, params)

        assert result.valid is True

    def test_error_handling_in_expansion(self, mock_env_vars):
        """Test error handling during test expansion with partial data."""
        from src.parameterized.engine import ParameterizationEngine
        from src.parameterized.models import (
            ParameterizedTest,
            DataSource,
            DataSourceType,
            TestStep,
        )

        engine = ParameterizationEngine(strict_validation=False)

        # Test with data that has a missing parameter (username missing in one)
        test = ParameterizedTest(
            name="Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[
                    {"username": "admin", "role": "admin"},  # Valid, has username
                    {"role": "guest"},  # Valid, but missing username
                ],
            ),
            steps=[TestStep(action="fill", target="#user", value="{{username}}")],
        )

        # Should not raise - both parameter sets are valid
        # (missing placeholders are left as-is in non-strict mode)
        expanded = engine.generate_test_matrix(test)

        # Both data items create valid ParameterSets with non-empty values
        assert len(expanded) == 2
