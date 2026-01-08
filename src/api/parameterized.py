"""API router for parameterized/data-driven test operations.

This module provides REST API endpoints for:
- Expanding parameterized tests into individual test instances
- Previewing expanded tests before execution
- Validating parameter configurations
- Importing test data from CSV/JSON files

All endpoints are prefixed with /api/v1/parameterized.
"""

import base64
import csv
import json
from datetime import datetime, timezone
from io import StringIO
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
import structlog

from src.parameterized.data_sources import (
    CSVDataSource,
    DataSourceError,
    DataSourceFactory,
    JSONDataSource,
)
from src.parameterized.engine import ParameterizationEngine, ParameterizationError
from src.parameterized.models import (
    DataSource,
    DataSourceType,
    ExpandedTest,
    IterationMode,
    ParameterizedTest,
    ParameterSet,
    ParameterValidationResult,
    TestAssertion,
    TestStep,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/parameterized", tags=["Parameterized Tests"])


# =============================================================================
# Request/Response Models
# =============================================================================


class TestStepRequest(BaseModel):
    """Test step for API requests."""

    action: str = Field(..., description="Action to perform")
    target: Optional[str] = Field(None, description="Target selector")
    value: Optional[str] = Field(None, description="Value for the action")
    timeout: Optional[int] = Field(None, description="Timeout in ms")
    description: Optional[str] = Field(None, description="Step description")


class TestAssertionRequest(BaseModel):
    """Test assertion for API requests."""

    type: str = Field(..., description="Assertion type")
    target: Optional[str] = Field(None, description="Target selector")
    expected: Optional[str] = Field(None, description="Expected value")
    description: Optional[str] = Field(None, description="Assertion description")


class DataSourceRequest(BaseModel):
    """Data source configuration for API requests."""

    type: str = Field(..., description="Data source type: inline, csv, json, env")
    data: Optional[list[dict[str, Any]]] = Field(None, description="Inline data")
    path: Optional[str] = Field(None, description="Path to data file")
    mapping: Optional[dict[str, str]] = Field(None, description="Field mapping")
    filter: Optional[str] = Field(None, description="Filter expression")
    limit: Optional[int] = Field(None, description="Limit results")
    delimiter: Optional[str] = Field(",", description="CSV delimiter")


class ParameterizedTestRequest(BaseModel):
    """Request model for parameterized test operations."""

    id: Optional[str] = Field(None, description="Test identifier")
    name: str = Field(..., description="Test name")
    description: Optional[str] = Field(None, description="Test description")
    data_source: Optional[DataSourceRequest] = Field(
        None, description="Data source configuration"
    )
    parameter_sets: Optional[list[dict[str, Any]]] = Field(
        None, description="Explicit parameter sets"
    )
    iteration_mode: Optional[str] = Field(
        "sequential", description="Iteration mode"
    )
    steps: list[TestStepRequest] = Field(..., description="Test steps")
    assertions: Optional[list[TestAssertionRequest]] = Field(
        None, description="Test assertions"
    )
    setup: Optional[list[TestStepRequest]] = Field(None, description="Setup steps")
    teardown: Optional[list[TestStepRequest]] = Field(
        None, description="Teardown steps"
    )
    timeout: Optional[int] = Field(30000, description="Default timeout in ms")


class ExpandRequest(BaseModel):
    """Request to expand a parameterized test."""

    test: ParameterizedTestRequest = Field(..., description="Test to expand")
    data_source_override: Optional[DataSourceRequest] = Field(
        None, description="Override data source"
    )
    limit: Optional[int] = Field(None, description="Limit expanded tests")


class PreviewRequest(BaseModel):
    """Request to preview expanded tests."""

    test: ParameterizedTestRequest = Field(..., description="Test to preview")
    limit: int = Field(5, ge=1, le=50, description="Number of previews")


class ValidateRequest(BaseModel):
    """Request to validate parameter configuration."""

    test: ParameterizedTestRequest = Field(..., description="Test to validate")


class ImportDataRequest(BaseModel):
    """Request to import data from content."""

    content: str = Field(..., description="File content (base64 for binary)")
    format: str = Field(..., description="Format: csv or json")
    mapping: Optional[dict[str, str]] = Field(None, description="Field mapping")
    filter: Optional[str] = Field(None, description="Filter expression")
    delimiter: Optional[str] = Field(",", description="CSV delimiter")


class ExpandedTestResponse(BaseModel):
    """Response for a single expanded test."""

    index: int
    name: str
    description: Optional[str]
    parameter_set_name: str
    parameter_values: dict[str, Any]
    steps: list[dict[str, Any]]
    assertions: list[dict[str, Any]]
    setup: list[dict[str, Any]]
    teardown: list[dict[str, Any]]


class ExpandResponse(BaseModel):
    """Response from expand operation."""

    success: bool
    test_name: str
    total_expanded: int
    expanded_tests: list[ExpandedTestResponse]
    iteration_mode: str
    warnings: list[str] = []
    error: Optional[str] = None


class PreviewResponse(BaseModel):
    """Response from preview operation."""

    success: bool
    test_name: str
    total_combinations: int
    previews: list[dict[str, Any]]
    data_source_type: Optional[str] = None
    warnings: list[str] = []


class ValidateResponse(BaseModel):
    """Response from validate operation."""

    valid: bool
    required_parameters: list[str]
    provided_parameters: list[str]
    missing_parameters: list[str]
    unused_parameters: list[str]
    errors: list[dict[str, str]]
    warnings: list[dict[str, str]]


class ImportDataResponse(BaseModel):
    """Response from import data operation."""

    success: bool
    parameter_sets_count: int
    parameter_sets: list[dict[str, Any]]
    detected_fields: list[str]
    sample_values: dict[str, Any]
    warnings: list[str] = []
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================


def _convert_to_parameterized_test(request: ParameterizedTestRequest) -> ParameterizedTest:
    """Convert API request model to internal ParameterizedTest model.

    Args:
        request: API request model

    Returns:
        ParameterizedTest model
    """
    # Convert data source if present
    data_source = None
    if request.data_source:
        data_source = DataSource(
            type=DataSourceType(request.data_source.type),
            data=request.data_source.data,
            path=request.data_source.path,
            mapping=request.data_source.mapping,
            filter=request.data_source.filter,
            limit=request.data_source.limit,
            delimiter=request.data_source.delimiter or ",",
        )

    # Convert parameter sets if present
    parameter_sets = None
    if request.parameter_sets:
        parameter_sets = [
            ParameterSet(
                name=ps.get("name", f"set_{i}"),
                values={k: v for k, v in ps.items() if k != "name"},
                description=ps.get("description"),
                tags=ps.get("tags", []),
                skip=ps.get("skip", False),
                skip_reason=ps.get("skip_reason"),
            )
            for i, ps in enumerate(request.parameter_sets)
        ]

    # Convert steps
    steps = [
        TestStep(
            action=s.action,
            target=s.target,
            value=s.value,
            timeout=s.timeout,
            description=s.description,
        )
        for s in request.steps
    ]

    # Convert assertions
    assertions = []
    if request.assertions:
        assertions = [
            TestAssertion(
                type=a.type,
                target=a.target,
                expected=a.expected,
                description=a.description,
            )
            for a in request.assertions
        ]

    # Convert setup/teardown
    setup = []
    if request.setup:
        setup = [
            TestStep(
                action=s.action,
                target=s.target,
                value=s.value,
                timeout=s.timeout,
            )
            for s in request.setup
        ]

    teardown = []
    if request.teardown:
        teardown = [
            TestStep(
                action=s.action,
                target=s.target,
                value=s.value,
                timeout=s.timeout,
            )
            for s in request.teardown
        ]

    return ParameterizedTest(
        id=request.id,
        name=request.name,
        description=request.description,
        data_source=data_source,
        parameter_sets=parameter_sets,
        iteration_mode=IterationMode(request.iteration_mode or "sequential"),
        steps=steps,
        assertions=assertions,
        setup=setup,
        teardown=teardown,
        timeout=request.timeout or 30000,
    )


def _convert_expanded_to_response(expanded: ExpandedTest) -> ExpandedTestResponse:
    """Convert ExpandedTest to API response model.

    Args:
        expanded: ExpandedTest model

    Returns:
        ExpandedTestResponse model
    """
    return ExpandedTestResponse(
        index=expanded.iteration_index,
        name=expanded.name,
        description=expanded.description,
        parameter_set_name=expanded.parameter_set.name,
        parameter_values=expanded.parameter_set.values,
        steps=expanded.steps,
        assertions=expanded.assertions,
        setup=expanded.setup,
        teardown=expanded.teardown,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/expand", response_model=ExpandResponse)
async def expand_test(request: ExpandRequest):
    """Expand a parameterized test into multiple concrete test instances.

    Takes a test template with parameter placeholders and a data source,
    then generates individual test instances with all placeholders replaced.

    Args:
        request: ExpandRequest with test specification and optional data source override

    Returns:
        ExpandResponse with all expanded tests

    Raises:
        HTTPException: If expansion fails

    Example:
        ```json
        {
            "test": {
                "name": "Login Test",
                "data_source": {
                    "type": "inline",
                    "data": [
                        {"username": "admin", "password": "admin123"},
                        {"username": "user", "password": "user123"}
                    ]
                },
                "steps": [
                    {"action": "fill", "target": "#username", "value": "{{username}}"},
                    {"action": "fill", "target": "#password", "value": "{{password}}"}
                ]
            }
        }
        ```
    """
    try:
        logger.info("Expanding parameterized test", test_name=request.test.name)

        # Convert to internal model
        test = _convert_to_parameterized_test(request.test)

        # Handle data source override
        data_source_override = None
        if request.data_source_override:
            data_source_override = DataSource(
                type=DataSourceType(request.data_source_override.type),
                data=request.data_source_override.data,
                path=request.data_source_override.path,
                mapping=request.data_source_override.mapping,
                filter=request.data_source_override.filter,
                limit=request.data_source_override.limit,
            )

        # Create engine and expand
        engine = ParameterizationEngine(strict_validation=True)
        expanded_tests = engine.generate_test_matrix(test, data_source_override)

        # Apply limit if specified
        if request.limit and request.limit < len(expanded_tests):
            expanded_tests = expanded_tests[: request.limit]

        # Convert to response
        expanded_responses = [
            _convert_expanded_to_response(exp) for exp in expanded_tests
        ]

        logger.info(
            "Test expansion complete",
            test_name=request.test.name,
            expanded_count=len(expanded_responses),
        )

        return ExpandResponse(
            success=True,
            test_name=request.test.name,
            total_expanded=len(expanded_responses),
            expanded_tests=expanded_responses,
            iteration_mode=test.iteration_mode.value,
        )

    except ParameterizationError as e:
        logger.error("Parameterization error", error=str(e))
        return ExpandResponse(
            success=False,
            test_name=request.test.name,
            total_expanded=0,
            expanded_tests=[],
            iteration_mode="sequential",
            error=str(e),
        )
    except Exception as e:
        logger.exception("Failed to expand test", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preview", response_model=PreviewResponse)
async def preview_expanded_tests(request: PreviewRequest):
    """Preview expanded tests without generating full test code.

    Shows what tests will be generated from a parameterized test
    without actually expanding all steps and assertions.

    Args:
        request: PreviewRequest with test specification and preview limit

    Returns:
        PreviewResponse with test previews

    Example:
        ```json
        {
            "test": {
                "name": "Login Test",
                "data_source": {"type": "inline", "data": [...]},
                "steps": [...]
            },
            "limit": 10
        }
        ```
    """
    try:
        logger.info("Previewing parameterized test", test_name=request.test.name)

        # Convert to internal model
        test = _convert_to_parameterized_test(request.test)

        # Create engine and preview
        engine = ParameterizationEngine(strict_validation=False)
        previews = engine.preview_expanded_tests(test, limit=request.limit)

        # Get total count
        try:
            all_expanded = engine.generate_test_matrix(test)
            total = len(all_expanded)
        except Exception:
            total = len(previews)

        data_source_type = None
        if test.data_source:
            data_source_type = test.data_source.type.value

        return PreviewResponse(
            success=True,
            test_name=request.test.name,
            total_combinations=total,
            previews=previews,
            data_source_type=data_source_type,
        )

    except Exception as e:
        logger.exception("Failed to preview test", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate", response_model=ValidateResponse)
async def validate_parameters(request: ValidateRequest):
    """Validate parameter configuration for a test.

    Checks that all parameter placeholders in the test have
    corresponding values in the data source.

    Args:
        request: ValidateRequest with test specification

    Returns:
        ValidateResponse with validation results

    Example:
        ```json
        {
            "test": {
                "name": "Login Test",
                "data_source": {"type": "inline", "data": [{"username": "admin"}]},
                "steps": [
                    {"action": "fill", "target": "#username", "value": "{{username}}"},
                    {"action": "fill", "target": "#password", "value": "{{password}}"}
                ]
            }
        }
        ```
        // Returns: missing_parameters: ["password"]
    """
    try:
        logger.info("Validating parameter configuration", test_name=request.test.name)

        # Convert to internal model
        test = _convert_to_parameterized_test(request.test)

        # Create engine and validate
        engine = ParameterizationEngine()
        result = engine.validate_test_spec(test)

        return ValidateResponse(
            valid=result.valid,
            required_parameters=result.required_parameters,
            provided_parameters=result.provided_parameters,
            missing_parameters=result.missing_parameters,
            unused_parameters=result.unused_parameters,
            errors=[
                {"parameter": e.parameter, "message": e.message}
                for e in result.errors
            ],
            warnings=[
                {"parameter": w.parameter, "message": w.message}
                for w in result.warnings
            ],
        )

    except Exception as e:
        logger.exception("Failed to validate parameters", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import-data", response_model=ImportDataResponse)
async def import_data(request: ImportDataRequest):
    """Import test data from CSV or JSON content.

    Parses data from provided content and returns parameter sets
    that can be used with parameterized tests.

    Args:
        request: ImportDataRequest with file content and format

    Returns:
        ImportDataResponse with parsed parameter sets

    Example (CSV):
        ```json
        {
            "content": "username,password\nadmin,admin123\nuser,user123",
            "format": "csv"
        }
        ```
    """
    try:
        logger.info("Importing test data", format=request.format)

        # Decode content if base64
        try:
            content = base64.b64decode(request.content).decode("utf-8")
        except Exception:
            content = request.content

        # Create appropriate data source config
        data_source_config = DataSource(
            type=DataSourceType(request.format),
            mapping=request.mapping,
            filter=request.filter,
            delimiter=request.delimiter or ",",
            data=[],  # Will be populated below
        )

        parameter_sets: list[ParameterSet] = []

        if request.format == "csv":
            source = CSVDataSource(data_source_config)
            parameter_sets = source.load_from_string(content)
        elif request.format == "json":
            source = JSONDataSource(data_source_config)
            parameter_sets = source.load_from_string(content)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format: {request.format}. Use 'csv' or 'json'.",
            )

        if not parameter_sets:
            return ImportDataResponse(
                success=True,
                parameter_sets_count=0,
                parameter_sets=[],
                detected_fields=[],
                sample_values={},
                warnings=["No data found in content"],
            )

        # Get detected fields from first parameter set
        detected_fields = list(parameter_sets[0].values.keys())

        # Get sample values
        sample_values = parameter_sets[0].values if parameter_sets else {}

        logger.info(
            "Data import complete",
            count=len(parameter_sets),
            fields=detected_fields,
        )

        return ImportDataResponse(
            success=True,
            parameter_sets_count=len(parameter_sets),
            parameter_sets=[
                {"name": ps.name, **ps.values} for ps in parameter_sets
            ],
            detected_fields=detected_fields,
            sample_values=sample_values,
        )

    except DataSourceError as e:
        logger.error("Data source error", error=str(e))
        return ImportDataResponse(
            success=False,
            parameter_sets_count=0,
            parameter_sets=[],
            detected_fields=[],
            sample_values={},
            error=str(e),
        )
    except Exception as e:
        logger.exception("Failed to import data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import-data/upload")
async def import_data_file(
    file: UploadFile = File(...),
    mapping: Optional[str] = None,
    filter_expr: Optional[str] = None,
):
    """Import test data from uploaded CSV or JSON file.

    Args:
        file: Uploaded file (CSV or JSON)
        mapping: JSON string of field mapping
        filter_expr: Filter expression

    Returns:
        ImportDataResponse with parsed parameter sets
    """
    try:
        # Determine format from filename
        filename = file.filename or ""
        if filename.endswith(".csv"):
            data_format = "csv"
        elif filename.endswith(".json"):
            data_format = "json"
        else:
            raise HTTPException(
                status_code=400,
                detail="File must be CSV or JSON (based on extension)",
            )

        # Read content
        content = await file.read()
        content_str = content.decode("utf-8")

        # Parse mapping if provided
        field_mapping = None
        if mapping:
            field_mapping = json.loads(mapping)

        # Create import request
        request = ImportDataRequest(
            content=content_str,
            format=data_format,
            mapping=field_mapping,
            filter=filter_expr,
        )

        # Delegate to import_data
        return await import_data(request)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to import file", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-sources")
async def list_data_sources():
    """List supported data source types.

    Returns:
        Dictionary of supported data source types and their configurations
    """
    return {
        "success": True,
        "data_sources": [
            {
                "type": "inline",
                "description": "Direct values in test specification",
                "required_fields": ["data"],
                "optional_fields": ["mapping", "filter", "limit"],
                "example": {
                    "type": "inline",
                    "data": [
                        {"username": "admin", "password": "admin123"},
                        {"username": "user", "password": "user123"},
                    ],
                },
            },
            {
                "type": "csv",
                "description": "Load from CSV file",
                "required_fields": ["path"],
                "optional_fields": ["mapping", "filter", "limit", "delimiter", "encoding"],
                "example": {
                    "type": "csv",
                    "path": "test_data/users.csv",
                    "delimiter": ",",
                },
            },
            {
                "type": "json",
                "description": "Load from JSON file",
                "required_fields": ["path"],
                "optional_fields": ["mapping", "filter", "limit", "encoding"],
                "example": {
                    "type": "json",
                    "path": "test_data/users.json",
                },
            },
            {
                "type": "env",
                "description": "Load from environment variables",
                "required_fields": ["env_prefix or env_mapping"],
                "optional_fields": ["mapping"],
                "example": {
                    "type": "env",
                    "env_prefix": "TEST_USER_",
                },
            },
        ],
        "supported_filters": [
            "field == 'value'",
            "field != 'value'",
            "field > 10",
            "field < 10",
            "field >= 10",
            "field <= 10",
            "field in ['a', 'b', 'c']",
        ],
    }


@router.get("/iteration-modes")
async def list_iteration_modes():
    """List supported iteration modes.

    Returns:
        Dictionary of supported iteration modes
    """
    return {
        "success": True,
        "iteration_modes": [
            {
                "mode": "sequential",
                "description": "Run tests in order (default)",
                "use_case": "Standard test execution",
            },
            {
                "mode": "parallel",
                "description": "Run tests in parallel (if supported)",
                "use_case": "Speed up execution with independent tests",
            },
            {
                "mode": "random",
                "description": "Randomize test order",
                "use_case": "Detect order-dependent bugs",
            },
            {
                "mode": "first_failure",
                "description": "Stop on first failure",
                "use_case": "Quick feedback during development",
            },
        ],
    }
