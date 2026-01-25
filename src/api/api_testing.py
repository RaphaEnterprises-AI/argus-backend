"""API Testing endpoints.

Provides REST endpoints for:
- Discovering API endpoints from OpenAPI/Swagger specs
- Generating test cases using AI
- Executing API tests with validation
- Managing test results and history

Follows patterns from src/api/browser.py for test execution.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import httpx
import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field, HttpUrl

from src.agents.api_tester import APITesterAgent
from src.api.projects import verify_project_access
from src.api.teams import get_current_user, log_audit
from src.api.tests import get_project_org_id
from src.config import get_settings
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/api-tests", tags=["API Testing"])

settings = get_settings()

# ============================================================================
# Request/Response Models
# ============================================================================

AuthType = Literal["none", "bearer", "basic", "api_key", "oauth2"]
TestType = Literal["functional", "negative", "boundary", "security", "performance", "integration"]
TestStatus = Literal["passed", "failed", "error", "skipped", "timeout"]
PriorityType = Literal["critical", "high", "medium", "low"]


class DiscoverEndpointsRequest(BaseModel):
    """Request to discover API endpoints from OpenAPI spec."""
    openapi_url: str = Field(..., description="URL to OpenAPI/Swagger spec (JSON or YAML)")
    project_id: str = Field(..., description="Project ID to associate discovered endpoints")


class DiscoveredEndpoint(BaseModel):
    """A discovered API endpoint."""
    id: str
    path: str
    method: str
    operation_id: str | None
    summary: str | None
    description: str | None
    tags: list[str]
    parameters: list[dict]
    request_body_schema: dict | None
    response_schemas: dict
    auth_type: str


class DiscoverEndpointsResponse(BaseModel):
    """Response from endpoint discovery."""
    success: bool
    session_id: str
    endpoints_discovered: int
    endpoints: list[DiscoveredEndpoint]
    spec_title: str | None
    spec_version: str | None
    errors: list[str]


class GenerateTestsRequest(BaseModel):
    """Request to generate test cases for endpoints."""
    project_id: str = Field(..., description="Project ID")
    endpoint_ids: list[str] | None = Field(None, description="Specific endpoint IDs to generate tests for")
    test_types: list[TestType] = Field(
        default=["functional", "negative", "boundary"],
        description="Types of tests to generate"
    )
    include_security_tests: bool = Field(False, description="Include security/injection tests")
    max_tests_per_endpoint: int = Field(5, ge=1, le=20, description="Maximum tests per endpoint")


class GeneratedTestCase(BaseModel):
    """A generated test case."""
    id: str
    name: str
    description: str
    endpoint: str
    method: str
    test_type: str
    headers: dict
    query_params: dict
    body: dict | None
    expected_status: int
    expected_body_schema: dict | None
    tags: list[str]


class GenerateTestsResponse(BaseModel):
    """Response from test generation."""
    success: bool
    tests_generated: int
    test_cases: list[GeneratedTestCase]
    generation_time_ms: int


class RunTestsRequest(BaseModel):
    """Request to execute API tests."""
    project_id: str = Field(..., description="Project ID")
    test_ids: list[str] | None = Field(None, description="Specific test IDs to run (all active if not specified)")
    base_url: str = Field(..., description="Base URL for API requests")
    auth_token: str | None = Field(None, description="Bearer token for authentication")
    auth_type: AuthType = Field("none", description="Authentication type")
    auth_config: dict = Field(default_factory=dict, description="Authentication configuration")
    environment: str = Field("test", description="Environment name for result tracking")
    parallel: bool = Field(False, description="Run tests in parallel")
    stop_on_failure: bool = Field(False, description="Stop execution on first failure")
    timeout_ms: int = Field(30000, ge=1000, le=120000, description="Request timeout in milliseconds")


class TestResultSummary(BaseModel):
    """Summary of a single test result."""
    test_id: str
    test_name: str
    status: TestStatus
    duration_ms: int
    response_status: int | None
    response_time_ms: int | None
    error_message: str | None
    schema_valid: bool | None


class RunTestsResponse(BaseModel):
    """Response from test execution."""
    success: bool
    run_id: str
    total_tests: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total_duration_ms: int
    results: list[TestResultSummary]


class APITestCaseResponse(BaseModel):
    """Full API test case details."""
    id: str
    project_id: str
    endpoint_id: str | None
    name: str
    description: str | None
    endpoint: str
    method: str
    headers: dict
    query_params: dict
    path_params: dict
    body: dict | None
    auth_type: str
    expected_status: int
    expected_body_schema: dict | None
    max_latency_ms: int
    test_type: str
    tags: list[str]
    priority: str
    source: str
    is_active: bool
    last_run_status: str | None
    created_at: str
    updated_at: str | None


class APITestResultResponse(BaseModel):
    """Full API test result details."""
    id: str
    test_case_id: str
    test_run_id: str | None
    status: str
    started_at: str
    completed_at: str | None
    duration_ms: int | None
    request_url: str
    request_method: str
    request_headers: dict
    request_body: dict | None
    response_status: int | None
    response_headers: dict
    response_body: Any | None
    response_time_ms: int | None
    status_code_valid: bool | None
    schema_valid: bool | None
    schema_errors: list[str]
    error_message: str | None
    environment: str
    created_at: str


class CreateTestCaseRequest(BaseModel):
    """Request to create a new API test case."""
    project_id: str = Field(..., description="Project ID")
    endpoint_id: str | None = Field(None, description="Associated endpoint ID")
    name: str = Field(..., min_length=1, max_length=255, description="Test name")
    description: str | None = Field(None, description="Test description")
    endpoint: str = Field(..., description="API endpoint path")
    method: str = Field(..., description="HTTP method")
    headers: dict = Field(default_factory=dict, description="Request headers")
    query_params: dict = Field(default_factory=dict, description="Query parameters")
    path_params: dict = Field(default_factory=dict, description="Path parameters")
    body: dict | None = Field(None, description="Request body")
    auth_type: AuthType = Field("none", description="Authentication type")
    expected_status: int = Field(200, description="Expected status code")
    expected_body_schema: dict | None = Field(None, description="Expected response schema")
    max_latency_ms: int = Field(5000, description="Maximum latency in ms")
    test_type: TestType = Field("functional", description="Test type")
    tags: list[str] = Field(default_factory=list, description="Tags")
    priority: PriorityType = Field("medium", description="Priority")


class UpdateTestCaseRequest(BaseModel):
    """Request to update an API test case."""
    name: str | None = None
    description: str | None = None
    headers: dict | None = None
    query_params: dict | None = None
    path_params: dict | None = None
    body: dict | None = None
    auth_type: AuthType | None = None
    expected_status: int | None = None
    expected_body_schema: dict | None = None
    max_latency_ms: int | None = None
    test_type: TestType | None = None
    tags: list[str] | None = None
    priority: PriorityType | None = None
    is_active: bool | None = None


# ============================================================================
# OpenAPI Parser
# ============================================================================

class OpenAPIParser:
    """Parser for OpenAPI 3.0 and Swagger 2.0 specs."""

    def __init__(self):
        self.spec: dict = {}
        self.version: str = ""
        self.title: str = ""
        self.api_version: str = ""

    async def parse_from_url(self, url: str) -> dict:
        """Fetch and parse OpenAPI spec from URL."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.text
            # Try JSON first, then YAML
            try:
                self.spec = json.loads(content)
            except json.JSONDecodeError:
                import yaml
                self.spec = yaml.safe_load(content)

        self._extract_metadata()
        return self.spec

    def parse_from_dict(self, spec: dict) -> dict:
        """Parse OpenAPI spec from dictionary."""
        self.spec = spec
        self._extract_metadata()
        return self.spec

    def _extract_metadata(self):
        """Extract spec metadata."""
        if "openapi" in self.spec:
            self.version = self.spec.get("openapi", "3.0.0")
        elif "swagger" in self.spec:
            self.version = self.spec.get("swagger", "2.0")

        info = self.spec.get("info", {})
        self.title = info.get("title", "")
        self.api_version = info.get("version", "")

    def get_endpoints(self) -> list[dict]:
        """Extract all endpoints from the spec."""
        endpoints = []
        paths = self.spec.get("paths", {})

        for path, path_item in paths.items():
            for method in ["get", "post", "put", "delete", "patch", "head", "options"]:
                if method in path_item:
                    operation = path_item[method]
                    endpoint = self._parse_operation(path, method.upper(), operation)
                    endpoints.append(endpoint)

        return endpoints

    def _parse_operation(self, path: str, method: str, operation: dict) -> dict:
        """Parse a single operation into endpoint dict."""
        # Extract parameters
        parameters = []
        for param in operation.get("parameters", []):
            parameters.append({
                "name": param.get("name"),
                "in": param.get("in"),  # path, query, header, cookie
                "required": param.get("required", False),
                "schema": param.get("schema", {}),
                "description": param.get("description", ""),
            })

        # Extract request body (OpenAPI 3.0)
        request_body_schema = None
        request_content_types = []
        if "requestBody" in operation:
            request_body = operation["requestBody"]
            content = request_body.get("content", {})
            for content_type, media in content.items():
                request_content_types.append(content_type)
                if not request_body_schema and "schema" in media:
                    request_body_schema = self._resolve_schema(media["schema"])

        # Extract responses
        response_schemas = {}
        response_content_types = []
        for status_code, response in operation.get("responses", {}).items():
            content = response.get("content", {})
            for content_type, media in content.items():
                if content_type not in response_content_types:
                    response_content_types.append(content_type)
                if "schema" in media:
                    response_schemas[status_code] = self._resolve_schema(media["schema"])

        # Extract security requirements
        security_schemes = []
        auth_type = "none"
        security = operation.get("security", self.spec.get("security", []))
        if security:
            for sec_req in security:
                for scheme_name in sec_req.keys():
                    security_schemes.append(scheme_name)
                    # Determine auth type from scheme
                    schemes = self.spec.get("components", {}).get("securitySchemes", {})
                    if scheme_name in schemes:
                        scheme = schemes[scheme_name]
                        scheme_type = scheme.get("type", "")
                        if scheme_type == "http" and scheme.get("scheme") == "bearer":
                            auth_type = "bearer"
                        elif scheme_type == "http" and scheme.get("scheme") == "basic":
                            auth_type = "basic"
                        elif scheme_type == "apiKey":
                            auth_type = "api_key"
                        elif scheme_type == "oauth2":
                            auth_type = "oauth2"

        return {
            "path": path,
            "method": method,
            "operation_id": operation.get("operationId"),
            "summary": operation.get("summary"),
            "description": operation.get("description"),
            "tags": operation.get("tags", []),
            "parameters": parameters,
            "request_body_schema": request_body_schema,
            "request_content_types": request_content_types,
            "response_schemas": response_schemas,
            "response_content_types": response_content_types,
            "security_schemes": security_schemes,
            "auth_type": auth_type,
            "is_deprecated": operation.get("deprecated", False),
        }

    def _resolve_schema(self, schema: dict, depth: int = 0) -> dict:
        """Resolve $ref references in schema."""
        if depth > 10:  # Prevent infinite recursion
            return schema

        if "$ref" in schema:
            ref_path = schema["$ref"]
            # Handle local references like "#/components/schemas/User"
            if ref_path.startswith("#/"):
                parts = ref_path[2:].split("/")
                resolved = self.spec
                for part in parts:
                    resolved = resolved.get(part, {})
                return self._resolve_schema(resolved, depth + 1)

        # Resolve nested schemas
        if "properties" in schema:
            schema = schema.copy()
            schema["properties"] = {
                k: self._resolve_schema(v, depth + 1)
                for k, v in schema["properties"].items()
            }

        if "items" in schema:
            schema = schema.copy()
            schema["items"] = self._resolve_schema(schema["items"], depth + 1)

        return schema


# ============================================================================
# Test Generator using AI
# ============================================================================

class AITestGenerator:
    """Generate test cases using Claude AI."""

    def __init__(self):
        self.agent = APITesterAgent()

    async def generate_tests(
        self,
        endpoint: dict,
        test_types: list[str],
        max_tests: int = 5,
        include_security: bool = False,
    ) -> list[dict]:
        """Generate test cases for an endpoint using AI."""
        import anthropic

        prompt = self._build_generation_prompt(endpoint, test_types, max_tests, include_security)

        try:
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )

            response = client.messages.create(
                model="claude-sonnet-4-5-20250514",
                max_tokens=4096,
                temperature=0.2,
                system="""You are an expert API test engineer. Generate comprehensive test cases for API endpoints.

Output valid JSON array of test cases. Each test case should include:
- name: Descriptive test name
- description: What the test validates
- test_type: functional, negative, boundary, security, or performance
- headers: Request headers object
- query_params: Query parameters object
- body: Request body (if applicable)
- expected_status: Expected HTTP status code
- expected_body_contains: Key paths that should exist in response
- tags: Relevant tags

Generate diverse test cases covering:
1. Happy path (successful requests)
2. Error cases (4xx responses)
3. Edge cases (empty values, nulls, boundary values)
4. Security tests (if requested): injection attempts, auth bypass""",
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            # Parse the response
            content = response.content[0].text
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            test_cases = json.loads(content.strip())
            return test_cases[:max_tests]

        except Exception as e:
            logger.error("AI test generation failed", error=str(e))
            # Return basic test cases as fallback
            return self._generate_basic_tests(endpoint, test_types, max_tests)

    def _build_generation_prompt(
        self,
        endpoint: dict,
        test_types: list[str],
        max_tests: int,
        include_security: bool,
    ) -> str:
        """Build the prompt for test generation."""
        return f"""Generate {max_tests} API test cases for the following endpoint:

**Endpoint:** {endpoint['method']} {endpoint['path']}
**Summary:** {endpoint.get('summary', 'N/A')}
**Description:** {endpoint.get('description', 'N/A')}

**Parameters:**
{json.dumps(endpoint.get('parameters', []), indent=2)}

**Request Body Schema:**
{json.dumps(endpoint.get('request_body_schema'), indent=2) if endpoint.get('request_body_schema') else 'None'}

**Response Schemas:**
{json.dumps(endpoint.get('response_schemas', {}), indent=2)}

**Test Types to Generate:** {', '.join(test_types)}
**Include Security Tests:** {include_security}

Generate a JSON array of test case objects. Be creative with edge cases and error scenarios."""

    def _generate_basic_tests(
        self,
        endpoint: dict,
        test_types: list[str],
        max_tests: int,
    ) -> list[dict]:
        """Generate basic test cases without AI."""
        tests = []

        # Happy path test
        if "functional" in test_types:
            tests.append({
                "name": f"Test {endpoint['method']} {endpoint['path']} - Success",
                "description": f"Verify {endpoint['method']} {endpoint['path']} returns successful response",
                "test_type": "functional",
                "headers": {"Content-Type": "application/json"},
                "query_params": {},
                "body": None,
                "expected_status": 200,
                "tags": ["functional", "happy-path"],
            })

        # 404 test
        if "negative" in test_types:
            tests.append({
                "name": f"Test {endpoint['method']} {endpoint['path']} - Not Found",
                "description": "Verify endpoint returns 404 for non-existent resource",
                "test_type": "negative",
                "headers": {"Content-Type": "application/json"},
                "query_params": {},
                "body": None,
                "expected_status": 404,
                "tags": ["negative", "error-handling"],
            })

        # Unauthorized test
        if "negative" in test_types:
            tests.append({
                "name": f"Test {endpoint['method']} {endpoint['path']} - Unauthorized",
                "description": "Verify endpoint returns 401 without authentication",
                "test_type": "negative",
                "headers": {},
                "query_params": {},
                "body": None,
                "expected_status": 401,
                "tags": ["negative", "auth"],
            })

        return tests[:max_tests]


# ============================================================================
# Test Executor
# ============================================================================

class APITestExecutor:
    """Execute API tests with validation."""

    def __init__(
        self,
        base_url: str,
        auth_type: str = "none",
        auth_token: str | None = None,
        auth_config: dict | None = None,
        timeout_ms: int = 30000,
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_type = auth_type
        self.auth_token = auth_token
        self.auth_config = auth_config or {}
        self.timeout = timeout_ms / 1000

    def _get_auth_headers(self) -> dict:
        """Get authentication headers based on auth type."""
        headers = {}
        if self.auth_type == "bearer" and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.auth_type == "basic":
            import base64
            username = self.auth_config.get("username", "")
            password = self.auth_config.get("password", "")
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        elif self.auth_type == "api_key":
            key_name = self.auth_config.get("key_name", "X-API-Key")
            key_value = self.auth_config.get("key_value", self.auth_token)
            key_location = self.auth_config.get("key_location", "header")
            if key_location == "header" and key_value:
                headers[key_name] = key_value
        return headers

    async def execute_test(self, test_case: dict) -> dict:
        """Execute a single API test case."""
        start_time = time.time()
        result = {
            "test_case_id": test_case.get("id"),
            "status": "error",
            "started_at": datetime.now(UTC).isoformat(),
            "request_url": "",
            "request_method": test_case.get("method", "GET"),
            "request_headers": {},
            "request_body": None,
            "response_status": None,
            "response_headers": {},
            "response_body": None,
            "response_time_ms": None,
            "response_size_bytes": None,
            "status_code_valid": None,
            "schema_valid": None,
            "schema_errors": [],
            "latency_valid": None,
            "error_message": None,
        }

        try:
            # Build URL with path params
            path = test_case.get("endpoint", "/")
            path_params = test_case.get("path_params", {})
            for param_name, param_value in path_params.items():
                path = path.replace(f"{{{param_name}}}", str(param_value))

            url = f"{self.base_url}{path}"
            result["request_url"] = url

            # Build headers
            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            headers.update(test_case.get("headers", {}))
            result["request_headers"] = headers

            # Build query params
            query_params = test_case.get("query_params", {})

            # Build body
            body = test_case.get("body")
            result["request_body"] = body

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                method = test_case.get("method", "GET").upper()
                request_start = time.time()

                if method == "GET":
                    response = await client.get(url, headers=headers, params=query_params)
                elif method == "POST":
                    response = await client.post(url, headers=headers, params=query_params, json=body)
                elif method == "PUT":
                    response = await client.put(url, headers=headers, params=query_params, json=body)
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers, params=query_params)
                elif method == "PATCH":
                    response = await client.patch(url, headers=headers, params=query_params, json=body)
                elif method == "HEAD":
                    response = await client.head(url, headers=headers, params=query_params)
                elif method == "OPTIONS":
                    response = await client.options(url, headers=headers, params=query_params)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response_time_ms = int((time.time() - request_start) * 1000)

                # Capture response
                result["response_status"] = response.status_code
                result["response_headers"] = dict(response.headers)
                result["response_time_ms"] = response_time_ms
                result["response_size_bytes"] = len(response.content)

                try:
                    result["response_body"] = response.json()
                except Exception:
                    result["response_body"] = response.text[:10000]  # Limit size

                # Validate status code
                expected_status = test_case.get("expected_status", 200)
                result["status_code_valid"] = response.status_code == expected_status

                # Validate latency
                max_latency = test_case.get("max_latency_ms", 5000)
                result["latency_valid"] = response_time_ms <= max_latency

                # Validate schema
                expected_schema = test_case.get("expected_body_schema")
                if expected_schema:
                    schema_result = self._validate_schema(result["response_body"], expected_schema)
                    result["schema_valid"] = schema_result["valid"]
                    result["schema_errors"] = schema_result["errors"]
                else:
                    result["schema_valid"] = True

                # Determine overall status
                if result["status_code_valid"] and result["schema_valid"] and result["latency_valid"]:
                    result["status"] = "passed"
                else:
                    result["status"] = "failed"
                    if not result["status_code_valid"]:
                        result["error_message"] = f"Expected status {expected_status}, got {response.status_code}"
                    elif not result["schema_valid"]:
                        result["error_message"] = f"Schema validation failed: {result['schema_errors']}"
                    elif not result["latency_valid"]:
                        result["error_message"] = f"Latency {response_time_ms}ms exceeds max {max_latency}ms"

        except httpx.TimeoutException:
            result["status"] = "timeout"
            result["error_message"] = f"Request timed out after {self.timeout}s"
        except httpx.ConnectError as e:
            result["status"] = "error"
            result["error_message"] = f"Connection failed: {str(e)}"
        except Exception as e:
            result["status"] = "error"
            result["error_message"] = str(e)
            logger.exception("Test execution error", test_id=test_case.get("id"))

        # Calculate duration
        result["duration_ms"] = int((time.time() - start_time) * 1000)
        result["completed_at"] = datetime.now(UTC).isoformat()

        return result

    def _validate_schema(self, data: Any, schema: dict) -> dict:
        """Validate data against JSON schema."""
        errors = []

        try:
            # Basic type validation
            expected_type = schema.get("type")
            if expected_type:
                type_valid = self._check_type(data, expected_type)
                if not type_valid:
                    errors.append(f"Expected type {expected_type}, got {type(data).__name__}")

            # Required fields
            if schema.get("required") and isinstance(data, dict):
                for field in schema["required"]:
                    if field not in data:
                        errors.append(f"Missing required field: {field}")

            # Properties validation
            if schema.get("properties") and isinstance(data, dict):
                for prop, prop_schema in schema["properties"].items():
                    if prop in data:
                        nested_result = self._validate_schema(data[prop], prop_schema)
                        errors.extend([f"{prop}.{e}" for e in nested_result["errors"]])

            # Array items validation
            if schema.get("items") and isinstance(data, list):
                for i, item in enumerate(data):
                    item_result = self._validate_schema(item, schema["items"])
                    errors.extend([f"[{i}].{e}" for e in item_result["errors"]])

        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")

        return {"valid": len(errors) == 0, "errors": errors}

    def _check_type(self, data: Any, expected_type: str) -> bool:
        """Check if data matches expected JSON schema type."""
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(data, expected)
        return True


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/discover", response_model=DiscoverEndpointsResponse)
async def discover_api_endpoints(body: DiscoverEndpointsRequest, request: Request):
    """
    Discover API endpoints from an OpenAPI/Swagger specification.

    Parses the spec and stores discovered endpoints in the database for test generation.
    """
    logger.info("Discovering API endpoints", openapi_url=body.openapi_url, project_id=body.project_id)

    # Get current user and verify project access
    user = await get_current_user(request)
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Create discovery session
    session_id = str(uuid4())
    session_result = await supabase.request(
        "/api_discovery_sessions",
        method="POST",
        body={
            "id": session_id,
            "project_id": body.project_id,
            "openapi_url": body.openapi_url,
            "status": "parsing",
        }
    )

    errors = []
    endpoints = []
    spec_title = None
    spec_version = None

    try:
        # Parse OpenAPI spec
        parser = OpenAPIParser()
        await parser.parse_from_url(body.openapi_url)

        spec_title = parser.title
        spec_version = parser.api_version
        raw_endpoints = parser.get_endpoints()

        # Store endpoints in database
        for ep in raw_endpoints:
            endpoint_id = str(uuid4())
            endpoint_data = {
                "id": endpoint_id,
                "project_id": body.project_id,
                "path": ep["path"],
                "method": ep["method"],
                "operation_id": ep.get("operation_id"),
                "summary": ep.get("summary"),
                "description": ep.get("description"),
                "parameters": ep.get("parameters", []),
                "request_body_schema": ep.get("request_body_schema"),
                "request_content_types": ep.get("request_content_types", []),
                "response_schemas": ep.get("response_schemas", {}),
                "response_content_types": ep.get("response_content_types", []),
                "security_schemes": ep.get("security_schemes", []),
                "auth_type": ep.get("auth_type", "none"),
                "discovered_from": "openapi",
                "spec_version": parser.version,
                "tags": ep.get("tags", []),
                "is_deprecated": ep.get("is_deprecated", False),
            }

            result = await supabase.request(
                "/api_endpoints",
                method="POST",
                body=endpoint_data,
            )

            if result.get("error"):
                errors.append(f"Failed to store endpoint {ep['method']} {ep['path']}: {result['error']}")
            else:
                endpoints.append(DiscoveredEndpoint(
                    id=endpoint_id,
                    path=ep["path"],
                    method=ep["method"],
                    operation_id=ep.get("operation_id"),
                    summary=ep.get("summary"),
                    description=ep.get("description"),
                    tags=ep.get("tags", []),
                    parameters=ep.get("parameters", []),
                    request_body_schema=ep.get("request_body_schema"),
                    response_schemas=ep.get("response_schemas", {}),
                    auth_type=ep.get("auth_type", "none"),
                ))

        # Update session as completed
        await supabase.request(
            f"/api_discovery_sessions?id=eq.{session_id}",
            method="PATCH",
            body={
                "status": "completed",
                "endpoints_discovered": len(endpoints),
                "api_title": spec_title,
                "api_version": spec_version,
                "spec_version": parser.version,
                "completed_at": datetime.now(UTC).isoformat(),
                "errors": errors,
            }
        )

        # Log audit
        org_id = await get_project_org_id(body.project_id)
        await log_audit(
            organization_id=org_id,
            user_id=user["user_id"],
            user_email=user.get("email"),
            action="api_endpoints.discover",
            resource_type="api_endpoints",
            resource_id=session_id,
            description=f"Discovered {len(endpoints)} API endpoints from OpenAPI spec",
            metadata={
                "project_id": body.project_id,
                "openapi_url": body.openapi_url,
                "endpoints_discovered": len(endpoints),
            },
            request=request,
        )

        logger.info(
            "API endpoint discovery completed",
            session_id=session_id,
            endpoints_discovered=len(endpoints),
            errors=len(errors),
        )

        return DiscoverEndpointsResponse(
            success=len(errors) == 0,
            session_id=session_id,
            endpoints_discovered=len(endpoints),
            endpoints=endpoints,
            spec_title=spec_title,
            spec_version=spec_version,
            errors=errors,
        )

    except httpx.HTTPStatusError as e:
        error_msg = f"Failed to fetch OpenAPI spec: HTTP {e.response.status_code}"
        logger.error(error_msg, url=body.openapi_url)
        await supabase.request(
            f"/api_discovery_sessions?id=eq.{session_id}",
            method="PATCH",
            body={"status": "failed", "errors": [error_msg]}
        )
        raise HTTPException(status_code=400, detail=error_msg)

    except Exception as e:
        error_msg = f"Failed to parse OpenAPI spec: {str(e)}"
        logger.exception(error_msg, url=body.openapi_url)
        await supabase.request(
            f"/api_discovery_sessions?id=eq.{session_id}",
            method="PATCH",
            body={"status": "failed", "errors": [error_msg]}
        )
        raise HTTPException(status_code=400, detail=error_msg)


@router.post("/generate", response_model=GenerateTestsResponse)
async def generate_api_tests(body: GenerateTestsRequest, request: Request):
    """
    Generate test cases for API endpoints using AI.

    Creates intelligent test cases covering happy paths, error cases, edge cases,
    and optionally security tests.
    """
    logger.info("Generating API tests", project_id=body.project_id, endpoint_ids=body.endpoint_ids)

    start_time = time.time()

    # Get current user and verify project access
    user = await get_current_user(request)
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Get endpoints to generate tests for
    if body.endpoint_ids:
        endpoint_ids_str = ",".join([f'"{eid}"' for eid in body.endpoint_ids])
        endpoint_result = await supabase.request(
            f"/api_endpoints?id=in.({endpoint_ids_str})&project_id=eq.{body.project_id}&is_active=eq.true"
        )
    else:
        endpoint_result = await supabase.request(
            f"/api_endpoints?project_id=eq.{body.project_id}&is_active=eq.true"
        )

    if not endpoint_result.get("data"):
        raise HTTPException(status_code=404, detail="No active endpoints found")

    endpoints = endpoint_result["data"]
    generator = AITestGenerator()
    generated_tests = []

    for endpoint in endpoints:
        try:
            # Generate tests using AI
            test_cases = await generator.generate_tests(
                endpoint=endpoint,
                test_types=body.test_types,
                max_tests=body.max_tests_per_endpoint,
                include_security=body.include_security_tests,
            )

            # Store test cases in database
            for tc in test_cases:
                test_id = str(uuid4())
                test_data = {
                    "id": test_id,
                    "project_id": body.project_id,
                    "endpoint_id": endpoint["id"],
                    "name": tc.get("name", f"Test {endpoint['method']} {endpoint['path']}"),
                    "description": tc.get("description"),
                    "endpoint": endpoint["path"],
                    "method": endpoint["method"],
                    "headers": tc.get("headers", {}),
                    "query_params": tc.get("query_params", {}),
                    "path_params": tc.get("path_params", {}),
                    "body": tc.get("body"),
                    "auth_type": endpoint.get("auth_type", "none"),
                    "expected_status": tc.get("expected_status", 200),
                    "expected_body_schema": tc.get("expected_body_schema"),
                    "max_latency_ms": tc.get("max_latency_ms", 5000),
                    "test_type": tc.get("test_type", "functional"),
                    "tags": tc.get("tags", []),
                    "priority": tc.get("priority", "medium"),
                    "source": "ai_generated",
                    "generated_by": "claude",
                    "is_active": True,
                }

                result = await supabase.request(
                    "/api_test_cases",
                    method="POST",
                    body=test_data,
                )

                if not result.get("error"):
                    generated_tests.append(GeneratedTestCase(
                        id=test_id,
                        name=test_data["name"],
                        description=test_data.get("description", ""),
                        endpoint=test_data["endpoint"],
                        method=test_data["method"],
                        test_type=test_data["test_type"],
                        headers=test_data["headers"],
                        query_params=test_data["query_params"],
                        body=test_data.get("body"),
                        expected_status=test_data["expected_status"],
                        expected_body_schema=test_data.get("expected_body_schema"),
                        tags=test_data["tags"],
                    ))

        except Exception as e:
            logger.error(
                "Failed to generate tests for endpoint",
                endpoint_id=endpoint["id"],
                error=str(e),
            )

    generation_time_ms = int((time.time() - start_time) * 1000)

    # Log audit
    org_id = await get_project_org_id(body.project_id)
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="api_tests.generate",
        resource_type="api_test_cases",
        resource_id=body.project_id,
        description=f"Generated {len(generated_tests)} API test cases",
        metadata={
            "tests_generated": len(generated_tests),
            "endpoints_processed": len(endpoints),
            "test_types": body.test_types,
        },
        request=request,
    )

    logger.info(
        "API test generation completed",
        tests_generated=len(generated_tests),
        generation_time_ms=generation_time_ms,
    )

    return GenerateTestsResponse(
        success=True,
        tests_generated=len(generated_tests),
        test_cases=generated_tests,
        generation_time_ms=generation_time_ms,
    )


@router.post("/run", response_model=RunTestsResponse)
async def run_api_tests(body: RunTestsRequest, request: Request):
    """
    Execute API tests and return results.

    Runs the specified tests (or all active tests) against the provided base URL
    and stores results in the database.
    """
    logger.info(
        "Running API tests",
        project_id=body.project_id,
        base_url=body.base_url,
        test_ids=body.test_ids,
    )

    start_time = time.time()

    # Get current user and verify project access
    user = await get_current_user(request)
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Get test cases to run
    if body.test_ids:
        test_ids_str = ",".join([f'"{tid}"' for tid in body.test_ids])
        test_result = await supabase.request(
            f"/api_test_cases?id=in.({test_ids_str})&project_id=eq.{body.project_id}"
        )
    else:
        test_result = await supabase.request(
            f"/api_test_cases?project_id=eq.{body.project_id}&is_active=eq.true"
        )

    if not test_result.get("data"):
        raise HTTPException(status_code=404, detail="No test cases found")

    test_cases = test_result["data"]
    run_id = str(uuid4())

    # Create test executor
    executor = APITestExecutor(
        base_url=body.base_url,
        auth_type=body.auth_type,
        auth_token=body.auth_token,
        auth_config=body.auth_config,
        timeout_ms=body.timeout_ms,
    )

    results = []
    passed = 0
    failed = 0
    errors = 0
    skipped = 0

    # Execute tests
    if body.parallel:
        # Parallel execution
        tasks = [executor.execute_test(tc) for tc in test_cases]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        for tc, result in zip(test_cases, results_raw):
            if isinstance(result, Exception):
                result = {
                    "test_case_id": tc["id"],
                    "status": "error",
                    "error_message": str(result),
                    "started_at": datetime.now(UTC).isoformat(),
                    "completed_at": datetime.now(UTC).isoformat(),
                    "duration_ms": 0,
                    "request_url": "",
                    "request_method": tc["method"],
                }
            results.append((tc, result))
    else:
        # Sequential execution
        for tc in test_cases:
            result = await executor.execute_test(tc)
            results.append((tc, result))

            if body.stop_on_failure and result["status"] in ("failed", "error"):
                # Mark remaining tests as skipped
                for remaining_tc in test_cases[test_cases.index(tc) + 1:]:
                    results.append((remaining_tc, {
                        "test_case_id": remaining_tc["id"],
                        "status": "skipped",
                        "error_message": "Skipped due to previous failure",
                        "started_at": datetime.now(UTC).isoformat(),
                        "completed_at": datetime.now(UTC).isoformat(),
                        "duration_ms": 0,
                        "request_url": "",
                        "request_method": remaining_tc["method"],
                    }))
                break

    # Store results and build response
    result_summaries = []
    for tc, result in results:
        # Store result in database
        result_data = {
            "id": str(uuid4()),
            "project_id": body.project_id,
            "test_case_id": tc["id"],
            "test_run_id": run_id,
            "status": result["status"],
            "started_at": result.get("started_at"),
            "completed_at": result.get("completed_at"),
            "duration_ms": result.get("duration_ms"),
            "request_url": result.get("request_url", ""),
            "request_method": result.get("request_method", tc["method"]),
            "request_headers": result.get("request_headers", {}),
            "request_body": result.get("request_body"),
            "response_status": result.get("response_status"),
            "response_headers": result.get("response_headers", {}),
            "response_body": result.get("response_body"),
            "response_size_bytes": result.get("response_size_bytes"),
            "response_time_ms": result.get("response_time_ms"),
            "status_code_valid": result.get("status_code_valid"),
            "schema_valid": result.get("schema_valid"),
            "schema_errors": result.get("schema_errors", []),
            "latency_valid": result.get("latency_valid"),
            "error_message": result.get("error_message"),
            "base_url": body.base_url,
            "environment": body.environment,
        }

        await supabase.request(
            "/api_test_results",
            method="POST",
            body=result_data,
        )

        # Update test case last run info
        await supabase.request(
            f"/api_test_cases?id=eq.{tc['id']}",
            method="PATCH",
            body={
                "last_run_at": datetime.now(UTC).isoformat(),
                "last_run_status": result["status"],
            }
        )

        # Count results
        if result["status"] == "passed":
            passed += 1
        elif result["status"] == "failed":
            failed += 1
        elif result["status"] == "error":
            errors += 1
        elif result["status"] == "skipped":
            skipped += 1

        result_summaries.append(TestResultSummary(
            test_id=tc["id"],
            test_name=tc["name"],
            status=result["status"],
            duration_ms=result.get("duration_ms", 0),
            response_status=result.get("response_status"),
            response_time_ms=result.get("response_time_ms"),
            error_message=result.get("error_message"),
            schema_valid=result.get("schema_valid"),
        ))

    total_duration_ms = int((time.time() - start_time) * 1000)

    # Log audit
    org_id = await get_project_org_id(body.project_id)
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="api_tests.run",
        resource_type="api_test_results",
        resource_id=run_id,
        description=f"Executed {len(test_cases)} API tests: {passed} passed, {failed} failed",
        metadata={
            "project_id": body.project_id,
            "total_tests": len(test_cases),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "duration_ms": total_duration_ms,
        },
        request=request,
    )

    logger.info(
        "API test run completed",
        run_id=run_id,
        total=len(test_cases),
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        duration_ms=total_duration_ms,
    )

    return RunTestsResponse(
        success=failed == 0 and errors == 0,
        run_id=run_id,
        total_tests=len(test_cases),
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        total_duration_ms=total_duration_ms,
        results=result_summaries,
    )


@router.get("/{project_id}/test-cases", response_model=list[APITestCaseResponse])
async def list_api_test_cases(
    project_id: str,
    request: Request,
    test_type: TestType | None = None,
    status: str | None = None,
    is_active: bool | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """List API test cases for a project."""
    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Build query
    query = f"/api_test_cases?project_id=eq.{project_id}&order=created_at.desc"
    if test_type:
        query += f"&test_type=eq.{test_type}"
    if status:
        query += f"&last_run_status=eq.{status}"
    if is_active is not None:
        query += f"&is_active=eq.{str(is_active).lower()}"
    query += f"&limit={limit}&offset={offset}"

    result = await supabase.request(query)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return [
        APITestCaseResponse(
            id=tc["id"],
            project_id=tc["project_id"],
            endpoint_id=tc.get("endpoint_id"),
            name=tc["name"],
            description=tc.get("description"),
            endpoint=tc["endpoint"],
            method=tc["method"],
            headers=tc.get("headers", {}),
            query_params=tc.get("query_params", {}),
            path_params=tc.get("path_params", {}),
            body=tc.get("body"),
            auth_type=tc.get("auth_type", "none"),
            expected_status=tc.get("expected_status", 200),
            expected_body_schema=tc.get("expected_body_schema"),
            max_latency_ms=tc.get("max_latency_ms", 5000),
            test_type=tc.get("test_type", "functional"),
            tags=tc.get("tags", []),
            priority=tc.get("priority", "medium"),
            source=tc.get("source", "manual"),
            is_active=tc.get("is_active", True),
            last_run_status=tc.get("last_run_status"),
            created_at=tc["created_at"],
            updated_at=tc.get("updated_at"),
        )
        for tc in result.get("data", [])
    ]


@router.get("/{project_id}/results", response_model=list[APITestResultResponse])
async def list_api_test_results(
    project_id: str,
    request: Request,
    test_case_id: str | None = None,
    status: TestStatus | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Get historical API test results for a project."""
    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    # Build query
    query = f"/api_test_results?project_id=eq.{project_id}&order=started_at.desc"
    if test_case_id:
        query += f"&test_case_id=eq.{test_case_id}"
    if status:
        query += f"&status=eq.{status}"
    query += f"&limit={limit}&offset={offset}"

    result = await supabase.request(query)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return [
        APITestResultResponse(
            id=r["id"],
            test_case_id=r["test_case_id"],
            test_run_id=r.get("test_run_id"),
            status=r["status"],
            started_at=r["started_at"],
            completed_at=r.get("completed_at"),
            duration_ms=r.get("duration_ms"),
            request_url=r["request_url"],
            request_method=r["request_method"],
            request_headers=r.get("request_headers", {}),
            request_body=r.get("request_body"),
            response_status=r.get("response_status"),
            response_headers=r.get("response_headers", {}),
            response_body=r.get("response_body"),
            response_time_ms=r.get("response_time_ms"),
            status_code_valid=r.get("status_code_valid"),
            schema_valid=r.get("schema_valid"),
            schema_errors=r.get("schema_errors", []),
            error_message=r.get("error_message"),
            environment=r.get("environment", "test"),
            created_at=r["created_at"],
        )
        for r in result.get("data", [])
    ]


@router.post("/test-cases", response_model=APITestCaseResponse)
async def create_api_test_case(body: CreateTestCaseRequest, request: Request):
    """Create a new API test case."""
    user = await get_current_user(request)
    await verify_project_access(body.project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    test_id = str(uuid4())
    test_data = {
        "id": test_id,
        "project_id": body.project_id,
        "endpoint_id": body.endpoint_id,
        "name": body.name,
        "description": body.description,
        "endpoint": body.endpoint,
        "method": body.method.upper(),
        "headers": body.headers,
        "query_params": body.query_params,
        "path_params": body.path_params,
        "body": body.body,
        "auth_type": body.auth_type,
        "expected_status": body.expected_status,
        "expected_body_schema": body.expected_body_schema,
        "max_latency_ms": body.max_latency_ms,
        "test_type": body.test_type,
        "tags": body.tags,
        "priority": body.priority,
        "source": "manual",
        "is_active": True,
    }

    result = await supabase.request(
        "/api_test_cases",
        method="POST",
        body=test_data,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    created = result["data"][0] if result.get("data") else test_data

    org_id = await get_project_org_id(body.project_id)
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="api_test_case.create",
        resource_type="api_test_cases",
        resource_id=test_id,
        description=f"Created API test case '{body.name}'",
        metadata={"project_id": body.project_id, "name": body.name},
        request=request,
    )

    return APITestCaseResponse(
        id=created["id"],
        project_id=created["project_id"],
        endpoint_id=created.get("endpoint_id"),
        name=created["name"],
        description=created.get("description"),
        endpoint=created["endpoint"],
        method=created["method"],
        headers=created.get("headers", {}),
        query_params=created.get("query_params", {}),
        path_params=created.get("path_params", {}),
        body=created.get("body"),
        auth_type=created.get("auth_type", "none"),
        expected_status=created.get("expected_status", 200),
        expected_body_schema=created.get("expected_body_schema"),
        max_latency_ms=created.get("max_latency_ms", 5000),
        test_type=created.get("test_type", "functional"),
        tags=created.get("tags", []),
        priority=created.get("priority", "medium"),
        source=created.get("source", "manual"),
        is_active=created.get("is_active", True),
        last_run_status=None,
        created_at=created["created_at"],
        updated_at=None,
    )


@router.patch("/test-cases/{test_id}", response_model=APITestCaseResponse)
async def update_api_test_case(test_id: str, body: UpdateTestCaseRequest, request: Request):
    """Update an API test case."""
    user = await get_current_user(request)

    supabase = get_supabase_client()

    # Get test case to verify access
    test_result = await supabase.request(f"/api_test_cases?id=eq.{test_id}")
    if not test_result.get("data"):
        raise HTTPException(status_code=404, detail="Test case not found")

    test_case = test_result["data"][0]
    await verify_project_access(test_case["project_id"], user["user_id"], user.get("email"), request)

    # Build update data
    update_data = {}
    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description
    if body.headers is not None:
        update_data["headers"] = body.headers
    if body.query_params is not None:
        update_data["query_params"] = body.query_params
    if body.path_params is not None:
        update_data["path_params"] = body.path_params
    if body.body is not None:
        update_data["body"] = body.body
    if body.auth_type is not None:
        update_data["auth_type"] = body.auth_type
    if body.expected_status is not None:
        update_data["expected_status"] = body.expected_status
    if body.expected_body_schema is not None:
        update_data["expected_body_schema"] = body.expected_body_schema
    if body.max_latency_ms is not None:
        update_data["max_latency_ms"] = body.max_latency_ms
    if body.test_type is not None:
        update_data["test_type"] = body.test_type
    if body.tags is not None:
        update_data["tags"] = body.tags
    if body.priority is not None:
        update_data["priority"] = body.priority
    if body.is_active is not None:
        update_data["is_active"] = body.is_active

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = await supabase.request(
        f"/api_test_cases?id=eq.{test_id}",
        method="PATCH",
        body=update_data,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    # Fetch updated test case
    updated_result = await supabase.request(f"/api_test_cases?id=eq.{test_id}")
    updated = updated_result["data"][0]

    org_id = await get_project_org_id(test_case["project_id"])
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="api_test_case.update",
        resource_type="api_test_cases",
        resource_id=test_id,
        description=f"Updated API test case '{updated['name']}'",
        metadata={"updated_fields": list(update_data.keys())},
        request=request,
    )

    return APITestCaseResponse(
        id=updated["id"],
        project_id=updated["project_id"],
        endpoint_id=updated.get("endpoint_id"),
        name=updated["name"],
        description=updated.get("description"),
        endpoint=updated["endpoint"],
        method=updated["method"],
        headers=updated.get("headers", {}),
        query_params=updated.get("query_params", {}),
        path_params=updated.get("path_params", {}),
        body=updated.get("body"),
        auth_type=updated.get("auth_type", "none"),
        expected_status=updated.get("expected_status", 200),
        expected_body_schema=updated.get("expected_body_schema"),
        max_latency_ms=updated.get("max_latency_ms", 5000),
        test_type=updated.get("test_type", "functional"),
        tags=updated.get("tags", []),
        priority=updated.get("priority", "medium"),
        source=updated.get("source", "manual"),
        is_active=updated.get("is_active", True),
        last_run_status=updated.get("last_run_status"),
        created_at=updated["created_at"],
        updated_at=updated.get("updated_at"),
    )


@router.delete("/test-cases/{test_id}")
async def delete_api_test_case(test_id: str, request: Request):
    """Delete an API test case."""
    user = await get_current_user(request)

    supabase = get_supabase_client()

    # Get test case to verify access
    test_result = await supabase.request(f"/api_test_cases?id=eq.{test_id}")
    if not test_result.get("data"):
        raise HTTPException(status_code=404, detail="Test case not found")

    test_case = test_result["data"][0]
    await verify_project_access(test_case["project_id"], user["user_id"], user.get("email"), request)

    result = await supabase.request(
        f"/api_test_cases?id=eq.{test_id}",
        method="DELETE",
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    org_id = await get_project_org_id(test_case["project_id"])
    await log_audit(
        organization_id=org_id,
        user_id=user["user_id"],
        user_email=user.get("email"),
        action="api_test_case.delete",
        resource_type="api_test_cases",
        resource_id=test_id,
        description=f"Deleted API test case '{test_case['name']}'",
        metadata={"project_id": test_case["project_id"], "name": test_case["name"]},
        request=request,
    )

    return {"success": True, "message": "Test case deleted"}


@router.get("/{project_id}/endpoints", response_model=list[DiscoveredEndpoint])
async def list_api_endpoints(
    project_id: str,
    request: Request,
    is_active: bool | None = None,
    method: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """List discovered API endpoints for a project."""
    user = await get_current_user(request)
    await verify_project_access(project_id, user["user_id"], user.get("email"), request)

    supabase = get_supabase_client()

    query = f"/api_endpoints?project_id=eq.{project_id}&order=path.asc"
    if is_active is not None:
        query += f"&is_active=eq.{str(is_active).lower()}"
    if method:
        query += f"&method=eq.{method.upper()}"
    query += f"&limit={limit}&offset={offset}"

    result = await supabase.request(query)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return [
        DiscoveredEndpoint(
            id=ep["id"],
            path=ep["path"],
            method=ep["method"],
            operation_id=ep.get("operation_id"),
            summary=ep.get("summary"),
            description=ep.get("description"),
            tags=ep.get("tags", []),
            parameters=ep.get("parameters", []),
            request_body_schema=ep.get("request_body_schema"),
            response_schemas=ep.get("response_schemas", {}),
            auth_type=ep.get("auth_type", "none"),
        )
        for ep in result.get("data", [])
    ]
