"""Core parameterization engine for data-driven testing.

This module provides the ParameterizationEngine class which handles:
- Expanding parameterized tests into individual test instances
- Substituting parameter placeholders in test steps and assertions
- Generating test matrices from data sources
- Validating parameter configurations

The engine supports {{param}} placeholder syntax for parameter substitution
in test steps, assertions, and other string fields.
"""

import re
import uuid
from typing import Any

import structlog

from src.parameterized.data_sources import BaseDataSource, DataSourceFactory
from src.parameterized.models import (
    DataSource,
    ExpandedTest,
    IterationMode,
    ParameterizedResult,
    ParameterizedTest,
    ParameterSet,
    ParameterValidationResult,
    TestAssertion,
    TestStep,
    ValidationError,
)

logger = structlog.get_logger()


class ParameterizationError(Exception):
    """Exception raised when parameterization operations fail."""

    pass


class ParameterizationEngine:
    """Engine for expanding and executing parameterized tests.

    The engine handles the core logic of data-driven testing:
    - Loading parameter data from various sources
    - Expanding test templates into concrete test instances
    - Substituting placeholders with actual values
    - Validating parameter configurations

    Example:
        engine = ParameterizationEngine()

        test_spec = ParameterizedTest(
            name="Login Test",
            data_source=DataSource(
                type=DataSourceType.INLINE,
                data=[
                    {"username": "admin", "password": "admin123"},
                    {"username": "user", "password": "user123"},
                ]
            ),
            steps=[
                TestStep(action="fill", target="#username", value="{{username}}"),
                TestStep(action="fill", target="#password", value="{{password}}"),
            ]
        )

        expanded_tests = engine.generate_test_matrix(test_spec)
        for test in expanded_tests:
            # Execute test...
            pass
    """

    # Regex pattern for parameter placeholders
    PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")

    def __init__(
        self,
        strict_validation: bool = True,
        allow_undefined_params: bool = False,
    ):
        """Initialize the parameterization engine.

        Args:
            strict_validation: If True, raise errors on validation failures
            allow_undefined_params: If True, leave undefined placeholders unchanged
        """
        self.strict_validation = strict_validation
        self.allow_undefined_params = allow_undefined_params
        self._data_source_cache: dict[str, BaseDataSource] = {}

    def expand_test(
        self,
        test_spec: ParameterizedTest,
        params: ParameterSet,
    ) -> ExpandedTest:
        """Expand a single test with a specific parameter set.

        Takes a parameterized test template and a parameter set, then returns
        a fully expanded test with all placeholders replaced.

        Args:
            test_spec: The parameterized test template
            params: Parameter set to apply

        Returns:
            ExpandedTest with all placeholders substituted

        Raises:
            ParameterizationError: If required parameters are missing
        """
        logger.debug(
            "Expanding test with parameters",
            test_name=test_spec.name,
            param_set=params.name,
        )

        # Validate parameters
        validation = self.validate_parameters(test_spec, params)
        if not validation.valid and self.strict_validation:
            error_msgs = [e.message for e in validation.errors]
            raise ParameterizationError(
                f"Parameter validation failed: {'; '.join(error_msgs)}"
            )

        # Generate unique test name
        test_name = self._generate_test_name(test_spec.name, params)

        # Substitute placeholders in all steps
        expanded_steps = [
            self._substitute_step(step, params.values)
            for step in test_spec.steps
        ]

        # Substitute in assertions
        expanded_assertions = [
            self._substitute_assertion(assertion, params.values)
            for assertion in test_spec.assertions
        ]

        # Substitute in setup/teardown
        expanded_setup = [
            self._substitute_step(step, params.values)
            for step in test_spec.setup
        ]
        expanded_teardown = [
            self._substitute_step(step, params.values)
            for step in test_spec.teardown
        ]

        # Substitute in description
        description = test_spec.description
        if description:
            description = self.substitute_values(description, params.values)

        return ExpandedTest(
            original_test_id=test_spec.id,
            iteration_index=0,  # Will be set by generate_test_matrix
            parameter_set=params,
            name=test_name,
            description=description,
            steps=expanded_steps,
            assertions=expanded_assertions,
            setup=expanded_setup,
            teardown=expanded_teardown,
            metadata={
                "timeout": test_spec.timeout,
                "retry_count": test_spec.retry_count,
                "original_name": test_spec.name,
            },
        )

    def substitute_values(
        self,
        template: str,
        params: dict[str, Any],
    ) -> str:
        """Replace {{param}} placeholders with actual values.

        Args:
            template: String containing {{param}} placeholders
            params: Dictionary of parameter name to value mappings

        Returns:
            String with all placeholders replaced

        Example:
            >>> engine.substitute_values("Hello {{name}}!", {"name": "World"})
            "Hello World!"
        """
        if not template:
            return template

        def replace_placeholder(match: re.Match) -> str:
            param_name = match.group(1)
            if param_name in params:
                return str(params[param_name])
            elif self.allow_undefined_params:
                return match.group(0)  # Keep original placeholder
            else:
                logger.warning(
                    "Undefined parameter in template",
                    param=param_name,
                    template=template[:50],
                )
                return match.group(0)

        return self.PLACEHOLDER_PATTERN.sub(replace_placeholder, template)

    def _substitute_step(
        self,
        step: TestStep,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute placeholders in a test step.

        Args:
            step: Test step with potential placeholders
            params: Parameter values

        Returns:
            Dictionary representation of step with substituted values
        """
        step_dict = step.model_dump(exclude_none=True)

        # Substitute in string fields
        for field in ["target", "value", "description"]:
            if field in step_dict and isinstance(step_dict[field], str):
                step_dict[field] = self.substitute_values(step_dict[field], params)

        return step_dict

    def _substitute_assertion(
        self,
        assertion: TestAssertion,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Substitute placeholders in a test assertion.

        Args:
            assertion: Test assertion with potential placeholders
            params: Parameter values

        Returns:
            Dictionary representation of assertion with substituted values
        """
        assertion_dict = assertion.model_dump(exclude_none=True)

        # Substitute in string fields
        for field in ["target", "expected", "description"]:
            if field in assertion_dict and isinstance(assertion_dict[field], str):
                assertion_dict[field] = self.substitute_values(
                    assertion_dict[field], params
                )

        return assertion_dict

    def generate_test_matrix(
        self,
        test: ParameterizedTest,
        data_source: DataSource | None = None,
    ) -> list[ExpandedTest]:
        """Generate all test combinations from a parameterized test.

        Creates expanded test instances for each parameter set, either from
        the test's configured data source or an override.

        Args:
            test: Parameterized test specification
            data_source: Optional override data source

        Returns:
            List of ExpandedTest instances

        Raises:
            ParameterizationError: If no parameter data is available
        """
        logger.info(
            "Generating test matrix",
            test_name=test.name,
            iteration_mode=test.iteration_mode,
        )

        # Get parameter sets
        parameter_sets = self._get_parameter_sets(test, data_source)

        if not parameter_sets:
            if test.is_parameterized():
                raise ParameterizationError(
                    f"No parameter sets available for test '{test.name}'"
                )
            # Non-parameterized test - return single instance with empty params
            logger.debug("Test is not parameterized, returning single instance")
            return [
                ExpandedTest(
                    original_test_id=test.id,
                    iteration_index=0,
                    parameter_set=ParameterSet(name="default", values={}),
                    name=test.name,
                    description=test.description,
                    steps=[s.model_dump(exclude_none=True) for s in test.steps],
                    assertions=[a.model_dump(exclude_none=True) for a in test.assertions],
                    setup=[s.model_dump(exclude_none=True) for s in test.setup],
                    teardown=[s.model_dump(exclude_none=True) for s in test.teardown],
                    metadata={"timeout": test.timeout, "retry_count": test.retry_count},
                )
            ]

        # Apply iteration mode
        parameter_sets = self._apply_iteration_mode(
            parameter_sets, test.iteration_mode
        )

        # Expand test for each parameter set
        expanded_tests = []
        for index, params in enumerate(parameter_sets):
            if params.skip:
                logger.debug(
                    "Skipping parameter set",
                    param_set=params.name,
                    reason=params.skip_reason,
                )
                continue

            try:
                expanded = self.expand_test(test, params)
                expanded.iteration_index = index
                expanded_tests.append(expanded)
            except ParameterizationError as e:
                logger.error(
                    "Failed to expand test",
                    test_name=test.name,
                    param_set=params.name,
                    error=str(e),
                )
                if self.strict_validation:
                    raise

        logger.info(
            "Test matrix generated",
            test_name=test.name,
            total_combinations=len(expanded_tests),
        )

        return expanded_tests

    def _get_parameter_sets(
        self,
        test: ParameterizedTest,
        data_source_override: DataSource | None = None,
    ) -> list[ParameterSet]:
        """Get parameter sets for a test.

        Args:
            test: Parameterized test specification
            data_source_override: Optional data source to use instead of test's

        Returns:
            List of ParameterSet objects
        """
        # Use explicit parameter sets if provided
        if test.parameter_sets and not data_source_override:
            return test.parameter_sets

        # Use data source
        data_source_config = data_source_override or test.data_source
        if not data_source_config:
            return []

        # Create data source handler
        source = DataSourceFactory.create(data_source_config)
        return source.load()

    def _apply_iteration_mode(
        self,
        parameter_sets: list[ParameterSet],
        mode: IterationMode,
    ) -> list[ParameterSet]:
        """Apply iteration mode to parameter sets.

        Args:
            parameter_sets: List of parameter sets
            mode: Iteration mode to apply

        Returns:
            Potentially reordered list of parameter sets
        """
        if mode == IterationMode.RANDOM:
            import random

            shuffled = parameter_sets.copy()
            random.shuffle(shuffled)
            return shuffled
        elif mode == IterationMode.SEQUENTIAL:
            return parameter_sets
        elif mode == IterationMode.PARALLEL:
            # Parallel mode doesn't change order, just signals execution strategy
            return parameter_sets
        elif mode == IterationMode.FIRST_FAILURE:
            # First failure mode doesn't change order, affects execution
            return parameter_sets
        else:
            return parameter_sets

    def validate_parameters(
        self,
        test: ParameterizedTest,
        params: ParameterSet,
    ) -> ParameterValidationResult:
        """Validate that all required parameters are provided.

        Checks that:
        - All placeholders in the test have corresponding parameter values
        - Reports unused parameters as warnings

        Args:
            test: Parameterized test specification
            params: Parameter set to validate

        Returns:
            ParameterValidationResult with validation details
        """
        # Get all placeholders used in the test
        required_params = test.get_all_parameter_placeholders()
        provided_params = set(params.values.keys())

        # Find missing parameters
        missing = required_params - provided_params

        # Find unused parameters
        unused = provided_params - required_params

        if missing:
            return ParameterValidationResult.create_invalid(
                required=required_params,
                provided=provided_params,
                missing=missing,
            )

        return ParameterValidationResult.create_valid(
            required=required_params,
            provided=provided_params,
            unused=unused,
        )

    def validate_test_spec(
        self,
        test: ParameterizedTest,
    ) -> ParameterValidationResult:
        """Validate a parameterized test specification.

        Validates the test structure and its data source configuration
        without executing the test.

        Args:
            test: Parameterized test to validate

        Returns:
            ParameterValidationResult with validation details
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # Validate data source if configured
        if test.data_source:
            source_errors = DataSourceFactory.validate_config(test.data_source)
            for error in source_errors:
                errors.append(
                    ValidationError(
                        parameter="data_source",
                        message=error,
                        severity="error",
                    )
                )

        # Get required parameters
        required_params = test.get_all_parameter_placeholders()

        # If we have parameter sets, validate each one
        if test.parameter_sets:
            for i, param_set in enumerate(test.parameter_sets):
                provided = set(param_set.values.keys())
                missing = required_params - provided
                if missing:
                    errors.append(
                        ValidationError(
                            parameter=f"parameter_sets[{i}]",
                            message=f"Missing parameters: {', '.join(missing)}",
                            location=f"parameter_sets[{i}]",
                            severity="error",
                        )
                    )

        # If we have a data source, try to load and validate
        if test.data_source and not errors:
            try:
                source = DataSourceFactory.create(test.data_source)
                param_sets = source.load()

                if not param_sets:
                    warnings.append(
                        ValidationError(
                            parameter="data_source",
                            message="Data source returned no parameter sets",
                            severity="warning",
                        )
                    )
                else:
                    # Validate first parameter set as sample
                    sample = param_sets[0]
                    provided = set(sample.values.keys())
                    missing = required_params - provided
                    if missing:
                        errors.append(
                            ValidationError(
                                parameter="data_source",
                                message=f"Data source missing parameters: {', '.join(missing)}",
                                severity="error",
                            )
                        )
            except Exception as e:
                errors.append(
                    ValidationError(
                        parameter="data_source",
                        message=f"Failed to load data source: {str(e)}",
                        severity="error",
                    )
                )

        if errors:
            return ParameterValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                required_parameters=list(required_params),
                provided_parameters=[],
                missing_parameters=[],
                unused_parameters=[],
            )

        return ParameterValidationResult(
            valid=True,
            errors=[],
            warnings=warnings,
            required_parameters=list(required_params),
            provided_parameters=[],
            missing_parameters=[],
            unused_parameters=[],
        )

    def _generate_test_name(
        self,
        base_name: str,
        params: ParameterSet,
    ) -> str:
        """Generate a unique test name from base name and parameters.

        Args:
            base_name: Original test name
            params: Parameter set being used

        Returns:
            Unique test name incorporating parameter info
        """
        safe_param_name = params.get_safe_name()
        return f"{base_name} [{safe_param_name}]"

    def preview_expanded_tests(
        self,
        test: ParameterizedTest,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Preview expanded tests without full execution.

        Useful for showing users what tests will be generated before
        actually running them.

        Args:
            test: Parameterized test to preview
            limit: Maximum number of previews to generate

        Returns:
            List of preview dictionaries with test info
        """
        expanded = self.generate_test_matrix(test)
        previews = []

        for i, exp_test in enumerate(expanded[:limit]):
            previews.append({
                "index": i,
                "name": exp_test.name,
                "description": exp_test.description,
                "parameter_set": {
                    "name": exp_test.parameter_set.name,
                    "values": exp_test.parameter_set.values,
                },
                "step_count": len(exp_test.steps),
                "assertion_count": len(exp_test.assertions),
            })

        return previews

    def create_empty_result(
        self,
        test: ParameterizedTest,
    ) -> ParameterizedResult:
        """Create an empty result structure for a parameterized test.

        Args:
            test: Parameterized test

        Returns:
            ParameterizedResult with initialized counters
        """
        return ParameterizedResult(
            test_id=test.id or str(uuid.uuid4()),
            test_name=test.name,
            total_iterations=0,
            passed=0,
            failed=0,
            skipped=0,
            duration_ms=0.0,
            results=[],
        )

    def merge_results(
        self,
        results: list[ParameterizedResult],
    ) -> ParameterizedResult:
        """Merge multiple parameterized results into one.

        Useful for aggregating results from parallel execution.

        Args:
            results: List of results to merge

        Returns:
            Combined ParameterizedResult
        """
        if not results:
            return ParameterizedResult(
                test_name="empty",
                total_iterations=0,
            )

        merged = ParameterizedResult(
            test_id=results[0].test_id,
            test_name=results[0].test_name,
            total_iterations=sum(r.total_iterations for r in results),
            passed=sum(r.passed for r in results),
            failed=sum(r.failed for r in results),
            skipped=sum(r.skipped for r in results),
            duration_ms=sum(r.duration_ms for r in results),
            results=[
                result
                for r in results
                for result in r.results
            ],
            started_at=min((r.started_at for r in results if r.started_at), default=None),
            completed_at=max((r.completed_at for r in results if r.completed_at), default=None),
        )

        return merged


def expand_parameterized_test(
    test_spec: dict[str, Any],
    data_source: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convenience function to expand a parameterized test from dictionaries.

    Args:
        test_spec: Test specification as dictionary
        data_source: Optional data source configuration override

    Returns:
        List of expanded test dictionaries
    """
    # Parse test spec
    test = ParameterizedTest(**test_spec)

    # Parse data source if provided
    ds = None
    if data_source:
        ds = DataSource(**data_source)
    elif "parameters" in test_spec and "data_source" in test_spec["parameters"]:
        ds = DataSource(**test_spec["parameters"]["data_source"])

    # Generate test matrix
    engine = ParameterizationEngine()
    expanded = engine.generate_test_matrix(test, ds)

    # Convert to dictionaries
    return [exp.to_test_spec() for exp in expanded]
