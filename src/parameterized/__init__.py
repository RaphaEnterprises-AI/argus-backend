"""Parameterized/data-driven testing module for the E2E Testing Agent.

This module provides functionality for:
- Defining parameterized test specifications with data sources
- Loading test data from inline values, CSV, JSON, or environment variables
- Expanding single tests into multiple parameterized versions
- Generating test matrices for comprehensive coverage

Example usage:
    from src.parameterized import ParameterizationEngine, ParameterizedTest

    engine = ParameterizationEngine()

    test_spec = {
        "name": "Login Test",
        "parameters": {
            "data_source": {
                "type": "inline",
                "data": [
                    {"username": "admin", "password": "admin123", "expected": "dashboard"},
                    {"username": "user", "password": "user123", "expected": "home"}
                ]
            }
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

    expanded_tests = engine.generate_test_matrix(test_spec, test_spec["parameters"]["data_source"])
"""

from src.parameterized.data_sources import (
    BaseDataSource,
    CSVDataSource,
    DataSourceFactory,
    EnvironmentDataSource,
    InlineDataSource,
    JSONDataSource,
)
from src.parameterized.engine import ParameterizationEngine
from src.parameterized.executor import (
    ExecutionResult,
    ExecutionStatus,
    IterationResult,
    IterationStatus,
    ParameterizedTestExecutor,
    resolve_parameters,
)
from src.parameterized.models import (
    DataSource,
    DataSourceType,
    IterationMode,
    ParameterizedResult,
    ParameterizedTest,
    ParameterSet,
    ParameterSetResult,
    ParameterValidationResult,
    ValidationError,
)

__all__ = [
    # Models
    "ParameterSet",
    "DataSource",
    "DataSourceType",
    "IterationMode",
    "ParameterizedTest",
    "ParameterizedResult",
    "ParameterSetResult",
    "ValidationError",
    "ParameterValidationResult",
    # Data Sources
    "BaseDataSource",
    "InlineDataSource",
    "CSVDataSource",
    "JSONDataSource",
    "EnvironmentDataSource",
    "DataSourceFactory",
    # Engine
    "ParameterizationEngine",
    # Executor
    "ParameterizedTestExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    "IterationResult",
    "IterationStatus",
    "resolve_parameters",
]
