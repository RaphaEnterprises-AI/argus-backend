"""API router for parameterized/data-driven test operations.

This module provides REST API endpoints for:
- Creating and managing parameterized test definitions
- Expanding parameterized tests into individual test instances
- Previewing expanded tests before execution
- Validating parameter configurations
- Importing test data from CSV/JSON files
- Storing execution results

Now with Supabase persistence for production use.

All endpoints are prefixed with /api/v1/parameterized.
"""

import base64
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.integrations.supabase import get_supabase
from src.parameterized.data_sources import (
    CSVDataSource,
    DataSourceError,
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
    TestAssertion,
    TestStep,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/parameterized", tags=["Parameterized Tests"])


# =============================================================================
# In-Memory Storage (fallback when Supabase not configured)
# =============================================================================

_parameterized_tests: dict[str, dict] = {}
_parameter_sets: dict[str, list[dict]] = {}  # test_id -> list of sets
_execution_results: dict[str, list[dict]] = {}  # test_id -> list of results


# =============================================================================
# Supabase Helper Functions
# =============================================================================

async def _get_test_from_db(test_id: str) -> dict | None:
    """Get a parameterized test from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        return _parameterized_tests.get(test_id)

    result = await supabase.select(
        "parameterized_tests",
        columns="*",
        filters={"id": test_id},
        limit=1
    )
    return result[0] if result else None


async def _list_tests_from_db(
    project_id: str | None = None,
    limit: int = 50
) -> list[dict]:
    """List parameterized tests from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        tests = list(_parameterized_tests.values())
        if project_id:
            tests = [t for t in tests if t.get("project_id") == project_id]
        return tests[:limit]

    filters = {}
    if project_id:
        filters["project_id"] = project_id

    return await supabase.select(
        "parameterized_tests",
        columns="*",
        filters=filters if filters else None,
        order_by="created_at",
        ascending=False,
        limit=limit
    )


async def _save_test_to_db(test: dict) -> bool:
    """Save a parameterized test to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        _parameterized_tests[test["id"]] = test
        return True

    return await supabase.insert("parameterized_tests", [test])


async def _update_test_in_db(test_id: str, updates: dict) -> bool:
    """Update a parameterized test in Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if test_id in _parameterized_tests:
            _parameterized_tests[test_id].update(updates)
            return True
        return False

    return await supabase.update("parameterized_tests", updates, {"id": test_id})


async def _delete_test_from_db(test_id: str) -> bool:
    """Delete a parameterized test from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        if test_id in _parameterized_tests:
            del _parameterized_tests[test_id]
            return True
        return False

    return await supabase.delete("parameterized_tests", {"id": test_id})


async def _get_parameter_sets_from_db(test_id: str) -> list[dict]:
    """Get parameter sets for a test from Supabase."""
    supabase = await get_supabase()
    if not supabase:
        return _parameter_sets.get(test_id, [])

    return await supabase.select(
        "parameter_sets",
        columns="*",
        filters={"parameterized_test_id": test_id},
        order_by="order_index",
        ascending=True
    )


async def _save_parameter_set_to_db(param_set: dict) -> bool:
    """Save a parameter set to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        test_id = param_set.get("parameterized_test_id")
        if test_id not in _parameter_sets:
            _parameter_sets[test_id] = []
        _parameter_sets[test_id].append(param_set)
        return True

    return await supabase.insert("parameter_sets", [param_set])


async def _delete_parameter_sets_for_test(test_id: str) -> bool:
    """Delete all parameter sets for a test."""
    supabase = await get_supabase()
    if not supabase:
        if test_id in _parameter_sets:
            del _parameter_sets[test_id]
        return True

    return await supabase.delete("parameter_sets", {"parameterized_test_id": test_id})


async def _save_execution_result_to_db(result: dict) -> bool:
    """Save an execution result to Supabase."""
    supabase = await get_supabase()
    if not supabase:
        test_id = result.get("parameterized_test_id")
        if test_id not in _execution_results:
            _execution_results[test_id] = []
        _execution_results[test_id].append(result)
        return True

    return await supabase.insert("parameterized_results", [result])


async def _list_execution_results_from_db(test_id: str, limit: int = 20) -> list[dict]:
    """List execution results for a test."""
    supabase = await get_supabase()
    if not supabase:
        return _execution_results.get(test_id, [])[-limit:]

    return await supabase.select(
        "parameterized_results",
        columns="*",
        filters={"parameterized_test_id": test_id},
        order_by="created_at",
        ascending=False,
        limit=limit
    )


# =============================================================================
# Request/Response Models
# =============================================================================


class TestStepRequest(BaseModel):
    """Test step for API requests."""

    action: str = Field(..., description="Action to perform")
    target: str | None = Field(None, description="Target selector")
    value: str | None = Field(None, description="Value for the action")
    timeout: int | None = Field(None, description="Timeout in ms")
    description: str | None = Field(None, description="Step description")


class TestAssertionRequest(BaseModel):
    """Test assertion for API requests."""

    type: str = Field(..., description="Assertion type")
    target: str | None = Field(None, description="Target selector")
    expected: str | None = Field(None, description="Expected value")
    description: str | None = Field(None, description="Assertion description")


class DataSourceRequest(BaseModel):
    """Data source configuration for API requests."""

    type: str = Field(..., description="Data source type: inline, csv, json, env")
    data: list[dict[str, Any]] | None = Field(None, description="Inline data")
    path: str | None = Field(None, description="Path to data file")
    mapping: dict[str, str] | None = Field(None, description="Field mapping")
    filter: str | None = Field(None, description="Filter expression")
    limit: int | None = Field(None, description="Limit results")
    delimiter: str | None = Field(",", description="CSV delimiter")


class ParameterizedTestRequest(BaseModel):
    """Request model for parameterized test operations."""

    id: str | None = Field(None, description="Test identifier")
    name: str = Field(..., description="Test name")
    description: str | None = Field(None, description="Test description")
    data_source: DataSourceRequest | None = Field(
        None, description="Data source configuration"
    )
    parameter_sets: list[dict[str, Any]] | None = Field(
        None, description="Explicit parameter sets"
    )
    iteration_mode: str | None = Field(
        "sequential", description="Iteration mode"
    )
    steps: list[TestStepRequest] = Field(..., description="Test steps")
    assertions: list[TestAssertionRequest] | None = Field(
        None, description="Test assertions"
    )
    setup: list[TestStepRequest] | None = Field(None, description="Setup steps")
    teardown: list[TestStepRequest] | None = Field(
        None, description="Teardown steps"
    )
    timeout: int | None = Field(30000, description="Default timeout in ms")


class ExpandRequest(BaseModel):
    """Request to expand a parameterized test."""

    test: ParameterizedTestRequest = Field(..., description="Test to expand")
    data_source_override: DataSourceRequest | None = Field(
        None, description="Override data source"
    )
    limit: int | None = Field(None, description="Limit expanded tests")


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
    mapping: dict[str, str] | None = Field(None, description="Field mapping")
    filter: str | None = Field(None, description="Filter expression")
    delimiter: str | None = Field(",", description="CSV delimiter")


class ExpandedTestResponse(BaseModel):
    """Response for a single expanded test."""

    index: int
    name: str
    description: str | None
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
    error: str | None = None


class PreviewResponse(BaseModel):
    """Response from preview operation."""

    success: bool
    test_name: str
    total_combinations: int
    previews: list[dict[str, Any]]
    data_source_type: str | None = None
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
    error: str | None = None


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
    mapping: str | None = None,
    filter_expr: str | None = None,
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


# =============================================================================
# CRUD Endpoints for Parameterized Test Management
# =============================================================================


class TestCreateRequest(BaseModel):
    """Request to create a new parameterized test."""

    project_id: str = Field(..., description="Project ID this test belongs to")
    name: str = Field(..., description="Test name")
    description: str | None = Field(None, description="Test description")
    data_source_type: str = Field(..., description="Data source type: inline, csv, json, env, database, api")
    data_source_config: dict[str, Any] = Field(..., description="Data source configuration")
    parameter_schema: dict[str, str] | None = Field(None, description="Parameter schema")
    steps: list[dict[str, Any]] = Field(default_factory=list, description="Test steps")
    assertions: list[dict[str, Any]] | None = Field(None, description="Test assertions")
    setup: list[dict[str, Any]] | None = Field(None, description="Setup steps")
    teardown: list[dict[str, Any]] | None = Field(None, description="Teardown steps")
    iteration_mode: str = Field("sequential", description="Iteration mode")
    max_parallel: int = Field(5, description="Max parallel iterations")
    timeout_per_iteration_ms: int = Field(60000, description="Timeout per iteration in ms")


class TestUpdateRequest(BaseModel):
    """Request to update a parameterized test."""

    name: str | None = None
    description: str | None = None
    data_source_type: str | None = None
    data_source_config: dict[str, Any] | None = None
    parameter_schema: dict[str, str] | None = None
    steps: list[dict[str, Any]] | None = None
    assertions: list[dict[str, Any]] | None = None
    setup: list[dict[str, Any]] | None = None
    teardown: list[dict[str, Any]] | None = None
    iteration_mode: str | None = None
    max_parallel: int | None = None
    timeout_per_iteration_ms: int | None = None


class TestResponse(BaseModel):
    """Response for parameterized test."""

    id: str
    project_id: str
    name: str
    description: str | None = None
    data_source_type: str
    data_source_config: dict[str, Any]
    parameter_schema: dict[str, str] | None = None
    steps: list[dict[str, Any]]
    assertions: list[dict[str, Any]] = []
    setup: list[dict[str, Any]] = []
    teardown: list[dict[str, Any]] = []
    iteration_mode: str = "sequential"
    max_parallel: int = 5
    timeout_per_iteration_ms: int = 60000
    created_at: str | None = None
    updated_at: str | None = None


class ParameterSetCreateRequest(BaseModel):
    """Request to create a parameter set."""

    name: str = Field(..., description="Parameter set name")
    description: str | None = Field(None, description="Description")
    values: dict[str, Any] = Field(..., description="Parameter values")
    tags: list[str] = Field(default_factory=list, description="Tags")
    skip: bool = Field(False, description="Skip this set")
    skip_reason: str | None = Field(None, description="Reason for skipping")


class ParameterSetResponse(BaseModel):
    """Response for parameter set."""

    id: str
    parameterized_test_id: str
    name: str
    description: str | None = None
    values: dict[str, Any]
    tags: list[str] = []
    skip: bool = False
    skip_reason: str | None = None
    order_index: int = 0
    created_at: str | None = None


class ExecutionResultResponse(BaseModel):
    """Response for execution result."""

    id: str
    parameterized_test_id: str
    test_run_id: str | None = None
    schedule_run_id: str | None = None
    total_iterations: int
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    iteration_results: list[dict[str, Any]] = []
    created_at: str | None = None


@router.post("/tests", response_model=TestResponse, status_code=201)
async def create_test(request: TestCreateRequest):
    """Create a new parameterized test.

    Args:
        request: Test creation request

    Returns:
        Created test
    """
    test_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    test_data = {
        "id": test_id,
        "project_id": request.project_id,
        "name": request.name,
        "description": request.description,
        "data_source_type": request.data_source_type,
        "data_source_config": request.data_source_config,
        "parameter_schema": request.parameter_schema,
        "steps": request.steps,
        "assertions": request.assertions or [],
        "setup": request.setup or [],
        "teardown": request.teardown or [],
        "iteration_mode": request.iteration_mode,
        "max_parallel": request.max_parallel,
        "timeout_per_iteration_ms": request.timeout_per_iteration_ms,
        "created_at": now,
        "updated_at": now,
    }

    success = await _save_test_to_db(test_data)
    if not success:
        logger.warning("Failed to persist test to Supabase, using in-memory")

    logger.info("Created parameterized test", test_id=test_id, name=request.name)

    return TestResponse(**test_data)


@router.get("/tests", response_model=list[TestResponse])
async def list_tests(
    project_id: str | None = Query(None, description="Filter by project"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results"),
):
    """List parameterized tests.

    Args:
        project_id: Optional project filter
        limit: Maximum results to return

    Returns:
        List of tests
    """
    tests = await _list_tests_from_db(project_id=project_id, limit=limit)
    return [TestResponse(**t) for t in tests]


@router.get("/tests/{test_id}", response_model=TestResponse)
async def get_test(test_id: str):
    """Get a parameterized test by ID.

    Args:
        test_id: Test ID

    Returns:
        Test details

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    return TestResponse(**test)


@router.patch("/tests/{test_id}", response_model=TestResponse)
async def update_test(test_id: str, request: TestUpdateRequest):
    """Update a parameterized test.

    Args:
        test_id: Test ID
        request: Update request

    Returns:
        Updated test

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    updates["updated_at"] = datetime.now(UTC).isoformat()

    success = await _update_test_in_db(test_id, updates)
    if not success:
        logger.warning("Failed to persist test update to Supabase")

    test.update(updates)
    logger.info("Updated parameterized test", test_id=test_id)

    return TestResponse(**test)


@router.delete("/tests/{test_id}", status_code=204)
async def delete_test(test_id: str):
    """Delete a parameterized test.

    Args:
        test_id: Test ID

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    # Delete parameter sets first
    await _delete_parameter_sets_for_test(test_id)

    success = await _delete_test_from_db(test_id)
    if not success:
        logger.warning("Failed to delete test from Supabase")

    logger.info("Deleted parameterized test", test_id=test_id)


@router.post("/tests/{test_id}/parameter-sets", response_model=ParameterSetResponse, status_code=201)
async def create_parameter_set(test_id: str, request: ParameterSetCreateRequest):
    """Add a parameter set to a test.

    Args:
        test_id: Test ID
        request: Parameter set creation request

    Returns:
        Created parameter set

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    # Get existing sets to determine order
    existing_sets = await _get_parameter_sets_from_db(test_id)
    order_index = len(existing_sets)

    set_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    param_set_data = {
        "id": set_id,
        "parameterized_test_id": test_id,
        "name": request.name,
        "description": request.description,
        "values": request.values,
        "tags": request.tags,
        "skip": request.skip,
        "skip_reason": request.skip_reason,
        "order_index": order_index,
        "created_at": now,
    }

    success = await _save_parameter_set_to_db(param_set_data)
    if not success:
        logger.warning("Failed to persist parameter set to Supabase")

    logger.info("Created parameter set", set_id=set_id, test_id=test_id)

    return ParameterSetResponse(**param_set_data)


@router.get("/tests/{test_id}/parameter-sets", response_model=list[ParameterSetResponse])
async def list_parameter_sets(test_id: str):
    """List parameter sets for a test.

    Args:
        test_id: Test ID

    Returns:
        List of parameter sets

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    sets = await _get_parameter_sets_from_db(test_id)
    return [ParameterSetResponse(**s) for s in sets]


@router.delete("/tests/{test_id}/parameter-sets", status_code=204)
async def delete_all_parameter_sets(test_id: str):
    """Delete all parameter sets for a test.

    Args:
        test_id: Test ID

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    await _delete_parameter_sets_for_test(test_id)
    logger.info("Deleted all parameter sets", test_id=test_id)


@router.get("/tests/{test_id}/results", response_model=list[ExecutionResultResponse])
async def list_execution_results(
    test_id: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
):
    """List execution results for a test.

    Args:
        test_id: Test ID
        limit: Maximum results to return

    Returns:
        List of execution results

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    results = await _list_execution_results_from_db(test_id, limit=limit)
    return [ExecutionResultResponse(**r) for r in results]


@router.post("/tests/{test_id}/execute", response_model=dict)
async def execute_test(test_id: str, dry_run: bool = Query(False, description="Preview without executing")):
    """Execute a parameterized test (legacy endpoint, preview only).

    Args:
        test_id: Test ID
        dry_run: If true, preview expansion without executing

    Returns:
        Execution result or preview

    Raises:
        HTTPException: If test not found
    """
    test = await _get_test_from_db(test_id)
    if not test:
        raise HTTPException(status_code=404, detail=f"Test {test_id} not found")

    # Get parameter sets
    param_sets = await _get_parameter_sets_from_db(test_id)

    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "test_id": test_id,
            "test_name": test.get("name"),
            "total_iterations": len(param_sets),
            "parameter_sets": [
                {"name": s.get("name"), "values": s.get("values"), "skip": s.get("skip", False)}
                for s in param_sets
            ],
            "iteration_mode": test.get("iteration_mode", "sequential"),
        }

    # Create execution result
    result_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    execution_result = {
        "id": result_id,
        "parameterized_test_id": test_id,
        "total_iterations": len(param_sets),
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "duration_ms": None,
        "started_at": now,
        "completed_at": None,
        "iteration_results": [],
        "created_at": now,
    }

    await _save_execution_result_to_db(execution_result)

    logger.info("Started parameterized test execution", test_id=test_id, result_id=result_id)

    return {
        "success": True,
        "dry_run": False,
        "result_id": result_id,
        "test_id": test_id,
        "test_name": test.get("name"),
        "total_iterations": len(param_sets),
        "status": "started",
        "message": "Execution started. Poll /results endpoint for updates.",
    }


# =============================================================================
# Parameterized Test Execution Engine
# =============================================================================


class ExecuteTestRequest(BaseModel):
    """Request to execute a parameterized test with full options."""

    test_id: str = Field(..., description="ID of the parameterized test to execute")
    parameter_sets: list[dict[str, Any]] | None = Field(
        None, description="Optional inline parameter sets (overrides stored sets)"
    )
    selected_set_ids: list[str] | None = Field(
        None, description="Run only these parameter set IDs (filters stored sets)"
    )
    app_url: str = Field(..., description="Application URL to test against")
    browser: str = Field("chromium", description="Browser to use (chromium, firefox, webkit)")
    environment: str = Field("staging", description="Environment name")
    iteration_mode: str = Field("sequential", description="Iteration mode (sequential, parallel, random)")
    max_parallel: int = Field(5, ge=1, le=20, description="Max parallel workers for parallel mode")
    timeout_per_iteration_ms: int = Field(60000, ge=1000, le=300000, description="Timeout per iteration in ms")
    stop_on_failure: bool = Field(False, description="Stop execution on first failure")
    retry_failed_iterations: int = Field(0, ge=0, le=3, description="Number of retries for failed iterations")
    triggered_by: str | None = Field(None, description="User or system triggering the execution")
    trigger_type: str = Field("manual", description="Trigger type (manual, scheduled, webhook, ci)")


class ExecuteTestResponse(BaseModel):
    """Response from test execution."""

    success: bool
    result_id: str
    test_id: str
    test_name: str
    total_iterations: int
    passed: int
    failed: int
    skipped: int
    error: int
    status: str
    duration_ms: int
    avg_iteration_ms: float | None = None
    min_iteration_ms: int | None = None
    max_iteration_ms: int | None = None
    iteration_results: list[dict[str, Any]]
    failure_summary: dict[str, Any] | None = None
    started_at: str | None = None
    completed_at: str | None = None


@router.post("/execute", response_model=ExecuteTestResponse)
async def execute_parameterized_test(request: ExecuteTestRequest):
    """Execute a parameterized test with full execution engine.

    This endpoint provides the complete parameterized test execution capability:
    - Resolves {{param}} placeholders with actual values from parameter sets
    - Iterates through all parameter sets (sequential, parallel, or random)
    - Executes each iteration via the browser pool API
    - Tracks results per iteration with detailed metrics
    - Supports retries, timeouts, and stop-on-failure
    - Persists results to Supabase for dashboard viewing

    Args:
        request: ExecuteTestRequest with test_id, parameter_sets, and execution options

    Returns:
        ExecuteTestResponse with aggregated results from all iterations

    Example:
        ```json
        {
            "test_id": "uuid-of-parameterized-test",
            "app_url": "http://localhost:3000",
            "browser": "chromium",
            "iteration_mode": "sequential",
            "selected_set_ids": ["set-1", "set-2"]
        }
        ```
    """
    from src.config import get_settings
    settings = get_settings()
    from src.parameterized.executor import ParameterizedTestExecutor
    from src.parameterized.models import IterationMode, ParameterizedTest, ParameterSet

    # Get test from database
    test_data = await _get_test_from_db(request.test_id)
    if not test_data:
        raise HTTPException(status_code=404, detail=f"Test {request.test_id} not found")

    # Get parameter sets (use provided or fetch from DB)
    if request.parameter_sets:
        param_sets_data = request.parameter_sets
    else:
        param_sets_data = await _get_parameter_sets_from_db(request.test_id)

    if not param_sets_data:
        raise HTTPException(
            status_code=400,
            detail="No parameter sets available. Provide parameter_sets or create them for this test."
        )

    # Convert test data to ParameterizedTest model
    test = ParameterizedTest(
        id=test_data.get("id"),
        name=test_data.get("name", "Unnamed Test"),
        description=test_data.get("description"),
        steps=[
            TestStep(**s) if isinstance(s, dict) and "action" in s else TestStep(action="custom", description=str(s))
            for s in (test_data.get("steps") or [])
        ],
        assertions=[
            TestAssertion(**a) if isinstance(a, dict) and "type" in a else TestAssertion(type="custom", description=str(a))
            for a in (test_data.get("assertions") or [])
        ],
        setup=[
            TestStep(**s) if isinstance(s, dict) and "action" in s else TestStep(action="custom", description=str(s))
            for s in (test_data.get("setup") or [])
        ],
        teardown=[
            TestStep(**s) if isinstance(s, dict) and "action" in s else TestStep(action="custom", description=str(s))
            for s in (test_data.get("teardown") or [])
        ],
        iteration_mode=IterationMode(test_data.get("iteration_mode", "sequential")),
        timeout=test_data.get("timeout_per_iteration_ms", 60000),
    )

    # Convert parameter sets to models
    param_sets = []
    for ps in param_sets_data:
        param_sets.append(ParameterSet(
            name=ps.get("name", f"set_{len(param_sets)}"),
            values=ps.get("values", {}),
            description=ps.get("description"),
            tags=ps.get("tags", []),
            skip=ps.get("skip", False),
            skip_reason=ps.get("skip_reason"),
        ))

    # Determine browser API URL
    browser_api_url = getattr(settings, "BROWSER_API_URL", None)
    if not browser_api_url:
        # Default to internal endpoint
        base_url = getattr(settings, "BASE_URL", "http://localhost:8000")
        browser_api_url = f"{base_url}/api/v1/browser/test"

    logger.info(
        "Executing parameterized test",
        test_id=request.test_id,
        test_name=test.name,
        total_sets=len(param_sets),
        browser_api_url=browser_api_url,
        iteration_mode=request.iteration_mode,
    )

    # Create executor
    executor = ParameterizedTestExecutor(
        browser_api_url=browser_api_url,
        timeout_per_iteration_ms=request.timeout_per_iteration_ms,
        max_retries=request.retry_failed_iterations,
        stop_on_failure=request.stop_on_failure,
        max_parallel=request.max_parallel,
    )

    # Execute the test
    try:
        result = await executor.execute(
            test=test,
            parameter_sets=param_sets,
            app_url=request.app_url,
            browser=request.browser,
            environment=request.environment,
            test_id=request.test_id,
            triggered_by=request.triggered_by,
            trigger_type=request.trigger_type,
            iteration_mode=request.iteration_mode,
            selected_set_ids=request.selected_set_ids,
        )
    except Exception as e:
        logger.exception("Parameterized test execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

    # Persist result to database
    result_data = {
        "id": result.id,
        "parameterized_test_id": result.parameterized_test_id,
        "test_run_id": result.test_run_id,
        "schedule_run_id": result.schedule_run_id,
        "total_iterations": result.total_iterations,
        "passed": result.passed,
        "failed": result.failed,
        "skipped": result.skipped,
        "error": result.error,
        "duration_ms": result.duration_ms,
        "avg_iteration_ms": result.avg_iteration_ms,
        "min_iteration_ms": result.min_iteration_ms,
        "max_iteration_ms": result.max_iteration_ms,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "iteration_mode": result.iteration_mode,
        "parallel_workers": result.parallel_workers,
        "status": result.status.value,
        "iteration_results": [r.to_dict() for r in result.iteration_results],
        "failure_summary": result.failure_summary,
        "environment": result.environment,
        "browser": result.browser,
        "app_url": result.app_url,
        "triggered_by": result.triggered_by,
        "trigger_type": result.trigger_type,
        "metadata": result.metadata,
        "created_at": datetime.now(UTC).isoformat(),
    }

    await _save_execution_result_to_db(result_data)

    # Also save individual iteration results
    supabase = await get_supabase()
    if supabase:
        for iteration in result.iteration_results:
            iteration_data = {
                "id": iteration.id,
                "parameterized_result_id": result.id,
                "parameter_set_id": iteration.parameter_set_id,
                "iteration_index": iteration.iteration_index,
                "parameter_values": iteration.parameter_set.values,
                "status": iteration.status.value,
                "started_at": iteration.started_at.isoformat() if iteration.started_at else None,
                "completed_at": iteration.completed_at.isoformat() if iteration.completed_at else None,
                "duration_ms": iteration.duration_ms,
                "step_results": iteration.step_results,
                "error_message": iteration.error_message,
                "error_stack": iteration.error_stack,
                "error_screenshot_url": iteration.error_screenshot_url,
                "assertions_passed": iteration.assertions_passed,
                "assertions_failed": iteration.assertions_failed,
                "retry_count": iteration.retry_count,
                "is_retry": iteration.is_retry,
                "original_iteration_id": iteration.original_iteration_id,
                "metadata": iteration.metadata,
                "created_at": datetime.now(UTC).isoformat(),
            }
            try:
                await supabase.insert("iteration_results", [iteration_data])
            except Exception as e:
                logger.warning("Failed to save iteration result", iteration_id=iteration.id, error=str(e))

    # Update test's last run status
    await _update_test_in_db(request.test_id, {
        "last_run_at": datetime.now(UTC).isoformat(),
        "last_run_status": result.status.value,
    })

    logger.info(
        "Parameterized test execution completed",
        test_id=request.test_id,
        result_id=result.id,
        status=result.status.value,
        passed=result.passed,
        failed=result.failed,
        duration_ms=result.duration_ms,
    )

    return ExecuteTestResponse(
        success=result.status.value in ("passed", "completed"),
        result_id=result.id,
        test_id=request.test_id,
        test_name=test.name,
        total_iterations=result.total_iterations,
        passed=result.passed,
        failed=result.failed,
        skipped=result.skipped,
        error=result.error,
        status=result.status.value,
        duration_ms=result.duration_ms,
        avg_iteration_ms=result.avg_iteration_ms,
        min_iteration_ms=result.min_iteration_ms,
        max_iteration_ms=result.max_iteration_ms,
        iteration_results=[r.to_dict() for r in result.iteration_results],
        failure_summary=result.failure_summary if result.failure_summary else None,
        started_at=result.started_at.isoformat() if result.started_at else None,
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
    )


class ExecuteInlineRequest(BaseModel):
    """Request to execute an inline parameterized test (no pre-saved test required)."""

    name: str = Field(..., description="Test name")
    steps: list[dict[str, Any]] = Field(..., description="Test steps with {{param}} placeholders")
    assertions: list[dict[str, Any]] | None = Field(None, description="Test assertions")
    parameter_sets: list[dict[str, Any]] = Field(..., description="Parameter sets to iterate")
    app_url: str = Field(..., description="Application URL to test against")
    browser: str = Field("chromium", description="Browser to use")
    iteration_mode: str = Field("sequential", description="Iteration mode")
    timeout_per_iteration_ms: int = Field(60000, description="Timeout per iteration in ms")
    stop_on_failure: bool = Field(False, description="Stop on first failure")


@router.post("/execute/inline", response_model=ExecuteTestResponse)
async def execute_inline_parameterized_test(request: ExecuteInlineRequest):
    """Execute an inline parameterized test without pre-saving.

    This endpoint allows executing a parameterized test directly without first
    creating it in the database. Useful for one-off testing or CI/CD integration.

    Args:
        request: ExecuteInlineRequest with test definition and parameter sets

    Returns:
        ExecuteTestResponse with aggregated results

    Example:
        ```json
        {
            "name": "Login Test",
            "steps": [
                {"action": "fill", "target": "#username", "value": "{{username}}"},
                {"action": "fill", "target": "#password", "value": "{{password}}"},
                {"action": "click", "target": "#submit"}
            ],
            "assertions": [
                {"type": "url_contains", "expected": "{{expected_url}}"}
            ],
            "parameter_sets": [
                {"name": "admin", "values": {"username": "admin", "password": "admin123", "expected_url": "dashboard"}},
                {"name": "user", "values": {"username": "user", "password": "user123", "expected_url": "home"}}
            ],
            "app_url": "http://localhost:3000"
        }
        ```
    """
    from src.config import get_settings
    settings = get_settings()
    from src.parameterized.executor import ParameterizedTestExecutor
    from src.parameterized.models import IterationMode, ParameterizedTest, ParameterSet

    # Create test model
    test = ParameterizedTest(
        id=str(uuid.uuid4()),
        name=request.name,
        steps=[
            TestStep(**s) if isinstance(s, dict) and "action" in s else TestStep(action="custom", description=str(s))
            for s in request.steps
        ],
        assertions=[
            TestAssertion(**a) if isinstance(a, dict) and "type" in a else TestAssertion(type="custom", description=str(a))
            for a in (request.assertions or [])
        ],
        iteration_mode=IterationMode(request.iteration_mode),
        timeout=request.timeout_per_iteration_ms,
    )

    # Create parameter sets
    param_sets = []
    for ps in request.parameter_sets:
        param_sets.append(ParameterSet(
            name=ps.get("name", f"set_{len(param_sets)}"),
            values=ps.get("values", {}),
            skip=ps.get("skip", False),
        ))

    # Get browser API URL
    browser_api_url = getattr(settings, "BROWSER_API_URL", None)
    if not browser_api_url:
        base_url = getattr(settings, "BASE_URL", "http://localhost:8000")
        browser_api_url = f"{base_url}/api/v1/browser/test"

    # Create and execute
    executor = ParameterizedTestExecutor(
        browser_api_url=browser_api_url,
        timeout_per_iteration_ms=request.timeout_per_iteration_ms,
        stop_on_failure=request.stop_on_failure,
    )

    try:
        result = await executor.execute(
            test=test,
            parameter_sets=param_sets,
            app_url=request.app_url,
            browser=request.browser,
            iteration_mode=request.iteration_mode,
        )
    except Exception as e:
        logger.exception("Inline parameterized test execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

    return ExecuteTestResponse(
        success=result.status.value in ("passed", "completed"),
        result_id=result.id,
        test_id=test.id or "",
        test_name=test.name,
        total_iterations=result.total_iterations,
        passed=result.passed,
        failed=result.failed,
        skipped=result.skipped,
        error=result.error,
        status=result.status.value,
        duration_ms=result.duration_ms,
        avg_iteration_ms=result.avg_iteration_ms,
        min_iteration_ms=result.min_iteration_ms,
        max_iteration_ms=result.max_iteration_ms,
        iteration_results=[r.to_dict() for r in result.iteration_results],
        failure_summary=result.failure_summary if result.failure_summary else None,
        started_at=result.started_at.isoformat() if result.started_at else None,
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
    )
