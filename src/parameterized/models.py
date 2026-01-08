"""Pydantic models for parameterized/data-driven testing.

This module defines the core data models for parameterized tests:
- ParameterSet: A named set of parameter values
- DataSource: Configuration for loading test data
- ParameterizedTest: A test specification with parameters
- ParameterizedResult: Results from parameterized test execution
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class DataSourceType(str, Enum):
    """Types of data sources for parameterized tests."""

    INLINE = "inline"  # Direct values in test spec
    CSV = "csv"  # Load from CSV file
    JSON = "json"  # Load from JSON file
    ENV = "env"  # Load from environment variables
    DATABASE = "database"  # Load from database query
    API = "api"  # Load from API endpoint


class IterationMode(str, Enum):
    """How to iterate through parameter sets."""

    SEQUENTIAL = "sequential"  # Run tests in order
    PARALLEL = "parallel"  # Run tests in parallel (if supported)
    RANDOM = "random"  # Randomize order
    FIRST_FAILURE = "first_failure"  # Stop on first failure


class ParameterSet(BaseModel):
    """A named set of parameter values for a single test iteration.

    Attributes:
        name: Human-readable name for this parameter set
        values: Dictionary mapping parameter names to their values
        description: Optional description explaining this test case
        tags: Optional tags for filtering/categorization
        skip: Whether to skip this parameter set
        skip_reason: Reason for skipping (if skip is True)

    Example:
        {
            "name": "admin_login",
            "values": {"username": "admin", "password": "admin123", "expected": "dashboard"},
            "description": "Test admin user login flow",
            "tags": ["admin", "critical"]
        }
    """

    name: str = Field(..., min_length=1, description="Human-readable name for this parameter set")
    values: dict[str, Any] = Field(..., description="Parameter name to value mapping")
    description: Optional[str] = Field(None, description="Description of this test case")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    skip: bool = Field(False, description="Whether to skip this parameter set")
    skip_reason: Optional[str] = Field(None, description="Reason for skipping")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is valid for test naming conventions."""
        # Replace spaces and special chars with underscores for test method names
        return v.strip()

    @field_validator("values")
    @classmethod
    def validate_values(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure values dict is not empty."""
        if not v:
            raise ValueError("Parameter values cannot be empty")
        return v

    def get_safe_name(self) -> str:
        """Get a sanitized name suitable for test method naming."""
        # Replace non-alphanumeric chars with underscores
        import re

        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name)
        # Remove consecutive underscores
        safe_name = re.sub(r"_+", "_", safe_name)
        # Remove leading/trailing underscores
        return safe_name.strip("_").lower()


class DataSource(BaseModel):
    """Configuration for loading test parameter data.

    Attributes:
        type: Type of data source (inline, csv, json, env)
        data: Inline data (for inline type)
        path: Path to data file (for csv, json types)
        mapping: Optional mapping of source fields to parameter names
        filter: Optional filter expression to select data
        limit: Optional limit on number of parameter sets
        encoding: File encoding (default: utf-8)
        delimiter: CSV delimiter (default: ,)
        env_prefix: Prefix for environment variables
        env_mapping: Mapping of env var names to parameter names

    Example (inline):
        {
            "type": "inline",
            "data": [
                {"username": "admin", "password": "admin123"},
                {"username": "user", "password": "user123"}
            ]
        }

    Example (CSV):
        {
            "type": "csv",
            "path": "test_data/users.csv",
            "mapping": {"user_name": "username", "user_pass": "password"},
            "filter": "role == 'admin'"
        }
    """

    type: DataSourceType = Field(..., description="Type of data source")
    data: Optional[list[dict[str, Any]]] = Field(None, description="Inline data")
    path: Optional[str] = Field(None, description="Path to data file")
    mapping: Optional[dict[str, str]] = Field(
        None, description="Source field to parameter name mapping"
    )
    filter: Optional[str] = Field(None, description="Filter expression")
    limit: Optional[int] = Field(None, ge=1, description="Limit number of parameter sets")
    encoding: str = Field("utf-8", description="File encoding")
    delimiter: str = Field(",", description="CSV delimiter")
    env_prefix: Optional[str] = Field(None, description="Environment variable prefix")
    env_mapping: Optional[dict[str, str]] = Field(
        None, description="Env var to parameter mapping"
    )
    # Database/API specific fields
    query: Optional[str] = Field(None, description="Database query or API endpoint")
    headers: Optional[dict[str, str]] = Field(None, description="API headers")

    @model_validator(mode="after")
    def validate_data_source(self) -> "DataSource":
        """Validate that required fields are provided based on type."""
        if self.type == DataSourceType.INLINE:
            if not self.data:
                raise ValueError("Inline data source requires 'data' field with list of values")
        elif self.type in (DataSourceType.CSV, DataSourceType.JSON):
            if not self.path:
                raise ValueError(f"{self.type.value} data source requires 'path' field")
        elif self.type == DataSourceType.ENV:
            if not self.env_mapping and not self.env_prefix:
                raise ValueError(
                    "Environment data source requires 'env_mapping' or 'env_prefix'"
                )
        return self

    def get_inline_data(self) -> list[dict[str, Any]]:
        """Get inline data, applying mapping if specified."""
        if self.data is None:
            return []

        if not self.mapping:
            return self.data

        # Apply field mapping
        mapped_data = []
        for item in self.data:
            mapped_item = {}
            for source_field, target_field in self.mapping.items():
                if source_field in item:
                    mapped_item[target_field] = item[source_field]
            # Include unmapped fields
            for key, value in item.items():
                if key not in self.mapping:
                    mapped_item[key] = value
            mapped_data.append(mapped_item)

        return mapped_data


class TestStep(BaseModel):
    """A single step in a test.

    Supports parameter placeholders in the format {{param_name}}.
    """

    action: str = Field(..., description="Action to perform (click, fill, etc.)")
    target: Optional[str] = Field(None, description="Target selector")
    value: Optional[str] = Field(None, description="Value for the action")
    timeout: Optional[int] = Field(None, ge=0, description="Timeout in milliseconds")
    wait_after: Optional[int] = Field(None, ge=0, description="Wait time after action")
    description: Optional[str] = Field(None, description="Step description")

    def get_parameter_placeholders(self) -> set[str]:
        """Extract all parameter placeholders from this step."""
        import re

        placeholders: set[str] = set()
        pattern = r"\{\{(\w+)\}\}"

        for field_value in [self.target, self.value, self.description]:
            if field_value:
                matches = re.findall(pattern, str(field_value))
                placeholders.update(matches)

        return placeholders


class TestAssertion(BaseModel):
    """A test assertion with parameter support.

    Supports parameter placeholders in the format {{param_name}}.
    """

    type: str = Field(..., description="Assertion type (visible, url_contains, etc.)")
    target: Optional[str] = Field(None, description="Target selector")
    expected: Optional[str] = Field(None, description="Expected value")
    timeout: Optional[int] = Field(None, ge=0, description="Timeout for assertion")
    description: Optional[str] = Field(None, description="Assertion description")
    soft: bool = Field(False, description="Whether this is a soft assertion")

    def get_parameter_placeholders(self) -> set[str]:
        """Extract all parameter placeholders from this assertion."""
        import re

        placeholders: set[str] = set()
        pattern = r"\{\{(\w+)\}\}"

        for field_value in [self.target, self.expected, self.description]:
            if field_value:
                matches = re.findall(pattern, str(field_value))
                placeholders.update(matches)

        return placeholders


class ParameterizedTest(BaseModel):
    """A test specification with parameterization support.

    Attributes:
        id: Unique test identifier
        name: Human-readable test name
        description: Test description
        base_test: Optional reference to a base test to extend
        data_source: Configuration for loading parameter data
        parameter_sets: Explicit parameter sets (alternative to data_source)
        iteration_mode: How to iterate through parameters
        steps: Test steps with parameter placeholders
        assertions: Test assertions with parameter placeholders
        setup: Setup steps to run before each iteration
        teardown: Teardown steps to run after each iteration
        metadata: Additional test metadata
        timeout: Default timeout for steps
        retry_count: Number of retries on failure
        fail_fast: Stop on first failure

    Example:
        {
            "name": "Login Test",
            "data_source": {
                "type": "inline",
                "data": [
                    {"username": "admin", "password": "admin123", "expected": "dashboard"},
                    {"username": "user", "password": "user123", "expected": "home"}
                ]
            },
            "steps": [
                {"action": "fill", "target": "#username", "value": "{{username}}"},
                {"action": "fill", "target": "#password", "value": "{{password}}"},
                {"action": "click", "target": "#submit"}
            ],
            "assertions": [
                {"type": "url_contains", "expected": "{{expected}}"}
            ]
        }
    """

    id: Optional[str] = Field(None, description="Unique test identifier")
    name: str = Field(..., min_length=1, description="Test name")
    description: Optional[str] = Field(None, description="Test description")
    base_test: Optional[str] = Field(None, description="Base test ID to extend")
    data_source: Optional[DataSource] = Field(None, description="Data source configuration")
    parameter_sets: Optional[list[ParameterSet]] = Field(
        None, description="Explicit parameter sets"
    )
    iteration_mode: IterationMode = Field(
        IterationMode.SEQUENTIAL, description="Iteration mode"
    )
    steps: list[TestStep] = Field(default_factory=list, description="Test steps")
    assertions: list[TestAssertion] = Field(
        default_factory=list, description="Test assertions"
    )
    setup: list[TestStep] = Field(default_factory=list, description="Setup steps")
    teardown: list[TestStep] = Field(default_factory=list, description="Teardown steps")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timeout: int = Field(30000, ge=0, description="Default timeout in milliseconds")
    retry_count: int = Field(0, ge=0, description="Number of retries on failure")
    fail_fast: bool = Field(False, description="Stop on first failure")

    @model_validator(mode="after")
    def validate_parameters(self) -> "ParameterizedTest":
        """Validate that either data_source or parameter_sets is provided."""
        if not self.data_source and not self.parameter_sets:
            # This is fine - a non-parameterized test
            pass
        elif self.data_source and self.parameter_sets:
            raise ValueError(
                "Cannot specify both 'data_source' and 'parameter_sets'. Use one or the other."
            )
        return self

    def get_all_parameter_placeholders(self) -> set[str]:
        """Get all parameter placeholders used in this test."""
        placeholders: set[str] = set()

        for step in self.steps + self.setup + self.teardown:
            placeholders.update(step.get_parameter_placeholders())

        for assertion in self.assertions:
            placeholders.update(assertion.get_parameter_placeholders())

        return placeholders

    def is_parameterized(self) -> bool:
        """Check if this test has parameters."""
        return bool(self.data_source or self.parameter_sets)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(exclude_none=True)


class ParameterSetResult(BaseModel):
    """Result from executing a single parameter set.

    Attributes:
        parameter_set: The parameter set that was executed
        status: Execution status (passed, failed, skipped)
        duration_ms: Execution duration in milliseconds
        error: Error message if failed
        screenshots: Screenshots captured during execution
        logs: Execution logs
        assertions_passed: Number of passed assertions
        assertions_failed: Number of failed assertions
    """

    parameter_set: ParameterSet = Field(..., description="Parameter set that was executed")
    status: str = Field(..., description="Execution status")
    duration_ms: float = Field(0.0, ge=0, description="Duration in milliseconds")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_step: Optional[int] = Field(None, description="Step number where error occurred")
    screenshots: list[str] = Field(default_factory=list, description="Base64 screenshots")
    logs: list[str] = Field(default_factory=list, description="Execution logs")
    assertions_passed: int = Field(0, ge=0, description="Passed assertions count")
    assertions_failed: int = Field(0, ge=0, description="Failed assertions count")
    retry_count: int = Field(0, ge=0, description="Number of retries attempted")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ParameterizedResult(BaseModel):
    """Aggregated results from executing a parameterized test.

    Attributes:
        test_id: ID of the parameterized test
        test_name: Name of the test
        total_iterations: Total number of parameter sets
        passed: Number of passed iterations
        failed: Number of failed iterations
        skipped: Number of skipped iterations
        duration_ms: Total execution time
        results: Individual results per parameter set
        started_at: Start timestamp
        completed_at: Completion timestamp
    """

    test_id: Optional[str] = Field(None, description="Test identifier")
    test_name: str = Field(..., description="Test name")
    total_iterations: int = Field(0, ge=0, description="Total iterations")
    passed: int = Field(0, ge=0, description="Passed iterations")
    failed: int = Field(0, ge=0, description="Failed iterations")
    skipped: int = Field(0, ge=0, description="Skipped iterations")
    duration_ms: float = Field(0.0, ge=0, description="Total duration")
    results: list[ParameterSetResult] = Field(
        default_factory=list, description="Per-iteration results"
    )
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_iterations == 0:
            return 0.0
        return (self.passed / self.total_iterations) * 100

    @property
    def overall_status(self) -> str:
        """Get overall test status."""
        if self.failed > 0:
            return "failed"
        if self.skipped == self.total_iterations:
            return "skipped"
        return "passed"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "total_iterations": self.total_iterations,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.success_rate,
            "overall_status": self.overall_status,
            "duration_ms": self.duration_ms,
            "results": [r.model_dump() for r in self.results],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class ValidationError(BaseModel):
    """A parameter validation error.

    Attributes:
        parameter: Parameter name with the issue
        message: Error message
        location: Where the error occurred (step index, assertion index)
        severity: Error severity (error, warning)
    """

    parameter: str = Field(..., description="Parameter name")
    message: str = Field(..., description="Error message")
    location: Optional[str] = Field(None, description="Location of error")
    severity: str = Field("error", description="Error severity")


class ParameterValidationResult(BaseModel):
    """Result of parameter validation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        required_parameters: Set of required parameter names
        provided_parameters: Set of provided parameter names
        missing_parameters: Parameters used but not provided
        unused_parameters: Parameters provided but not used
    """

    valid: bool = Field(..., description="Whether validation passed")
    errors: list[ValidationError] = Field(default_factory=list, description="Errors")
    warnings: list[ValidationError] = Field(default_factory=list, description="Warnings")
    required_parameters: list[str] = Field(
        default_factory=list, description="Required parameters"
    )
    provided_parameters: list[str] = Field(
        default_factory=list, description="Provided parameters"
    )
    missing_parameters: list[str] = Field(
        default_factory=list, description="Missing parameters"
    )
    unused_parameters: list[str] = Field(
        default_factory=list, description="Unused parameters"
    )

    @classmethod
    def create_valid(
        cls,
        required: set[str],
        provided: set[str],
        unused: Optional[set[str]] = None,
    ) -> "ParameterValidationResult":
        """Create a valid validation result."""
        warnings = []
        if unused:
            for param in unused:
                warnings.append(
                    ValidationError(
                        parameter=param,
                        message=f"Parameter '{param}' is provided but not used in test",
                        severity="warning",
                    )
                )

        return cls(
            valid=True,
            errors=[],
            warnings=warnings,
            required_parameters=list(required),
            provided_parameters=list(provided),
            missing_parameters=[],
            unused_parameters=list(unused or set()),
        )

    @classmethod
    def create_invalid(
        cls,
        required: set[str],
        provided: set[str],
        missing: set[str],
        errors: Optional[list[ValidationError]] = None,
    ) -> "ParameterValidationResult":
        """Create an invalid validation result."""
        all_errors = errors or []

        for param in missing:
            all_errors.append(
                ValidationError(
                    parameter=param,
                    message=f"Required parameter '{param}' is not provided",
                    severity="error",
                )
            )

        return cls(
            valid=False,
            errors=all_errors,
            warnings=[],
            required_parameters=list(required),
            provided_parameters=list(provided),
            missing_parameters=list(missing),
            unused_parameters=[],
        )


class ExpandedTest(BaseModel):
    """A single expanded test instance from a parameterized test.

    This represents one iteration of a parameterized test with all
    placeholders replaced by actual values.
    """

    original_test_id: Optional[str] = Field(None, description="Original test ID")
    iteration_index: int = Field(..., ge=0, description="Iteration index")
    parameter_set: ParameterSet = Field(..., description="Parameter set used")
    name: str = Field(..., description="Generated test name")
    description: Optional[str] = Field(None, description="Generated description")
    steps: list[dict[str, Any]] = Field(default_factory=list, description="Expanded steps")
    assertions: list[dict[str, Any]] = Field(
        default_factory=list, description="Expanded assertions"
    )
    setup: list[dict[str, Any]] = Field(default_factory=list, description="Expanded setup")
    teardown: list[dict[str, Any]] = Field(
        default_factory=list, description="Expanded teardown"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    def to_test_spec(self) -> dict[str, Any]:
        """Convert to a standard test specification dict."""
        return {
            "id": f"{self.original_test_id}_{self.iteration_index}"
            if self.original_test_id
            else f"test_{self.iteration_index}",
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "assertions": self.assertions,
            "setup": self.setup,
            "teardown": self.teardown,
            "metadata": {
                **self.metadata,
                "parameterized": True,
                "iteration_index": self.iteration_index,
                "parameter_set_name": self.parameter_set.name,
                "parameter_values": self.parameter_set.values,
            },
        }
