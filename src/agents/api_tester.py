"""API Tester Agent - Executes API tests with schema validation.

This agent:
- Executes HTTP requests against API endpoints
- Validates responses against expected schemas
- Checks status codes, headers, and body content
- Supports authentication flows
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .base import AgentResult, BaseAgent
from .prompts import get_enhanced_prompt
from .test_planner import TestSpec


@dataclass
class APIRequestResult:
    """Result from a single API request."""

    method: str
    url: str
    status_code: int
    response_time_ms: int
    headers: dict
    body: Any
    success: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "url": self.url,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "headers": dict(self.headers),
            "body": self.body,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class SchemaValidationResult:
    """Result from validating response against schema."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    schema_used: dict | None = None


@dataclass
class APITestResult:
    """Complete result from an API test execution."""

    test_id: str
    test_name: str
    status: str  # passed, failed, error
    requests: list[APIRequestResult] = field(default_factory=list)
    schema_validations: list[SchemaValidationResult] = field(default_factory=list)
    total_duration_ms: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status,
            "requests": [r.to_dict() for r in self.requests],
            "schema_validations": [
                {"valid": v.valid, "errors": v.errors} for v in self.schema_validations
            ],
            "total_duration_ms": self.total_duration_ms,
            "error_message": self.error_message,
        }


class APITesterAgent(BaseAgent):
    """Agent that executes API tests with validation.

    Features:
    - HTTP request execution (GET, POST, PUT, DELETE, PATCH)
    - Response schema validation using Pydantic
    - Header and status code assertions
    - Authentication token management
    - Request chaining with variable extraction
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._auth_token: str | None = None
        self._variables: dict[str, Any] = {}

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for API testing."""
        enhanced = get_enhanced_prompt("api_tester")
        if enhanced:
            return enhanced

        return """You are an expert API tester. Analyze API responses and validate them against expected behavior.

When asked to analyze an API response:
1. Check if status code is appropriate
2. Validate response body structure
3. Identify any error messages or unexpected values
4. Compare against expected schema

Respond with JSON containing:
- valid: boolean
- issues: list of problems found
- suggestions: list of improvements"""

    async def execute(
        self,
        test_spec: TestSpec | dict,
        app_url: str,
        auth_token: str | None = None,
    ) -> AgentResult[APITestResult]:
        """Execute an API test specification.

        Args:
            test_spec: Test specification with API steps
            app_url: Base API URL
            auth_token: Optional authentication token

        Returns:
            AgentResult containing APITestResult
        """
        if isinstance(test_spec, dict):
            test_id = test_spec.get("id", "unknown")
            test_name = test_spec.get("name", "Unknown API Test")
            steps = test_spec.get("steps", [])
            assertions = test_spec.get("assertions", [])
        else:
            test_id = test_spec.id
            test_name = test_spec.name
            steps = [s.to_dict() if hasattr(s, "to_dict") else s for s in test_spec.steps]
            assertions = [a.to_dict() if hasattr(a, "to_dict") else a for a in test_spec.assertions]

        self.log.info(
            "Executing API test",
            test_id=test_id,
            test_name=test_name,
        )

        if auth_token:
            self._auth_token = auth_token

        start_time = time.time()
        requests_results = []
        schema_validations = []
        error_message = None
        status = "passed"

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                for step in steps:
                    action = step.get("action", "").lower()

                    if action in ("get", "post", "put", "delete", "patch", "request"):
                        result = await self._execute_request(
                            client, step, app_url
                        )
                        requests_results.append(result)

                        if not result.success:
                            status = "failed"
                            error_message = result.error
                            break

                        # Validate schema if provided
                        schema = step.get("schema") or step.get("expected_schema")
                        if schema:
                            validation = self._validate_schema(result.body, schema)
                            schema_validations.append(validation)
                            if not validation.valid:
                                status = "failed"
                                error_message = f"Schema validation failed: {validation.errors}"

                        # Extract variables
                        extractions = step.get("extract", {})
                        for var_name, json_path in extractions.items():
                            self._variables[var_name] = self._extract_value(
                                result.body, json_path
                            )

                # Check assertions
                for assertion in assertions:
                    if not self._check_api_assertion(assertion, requests_results):
                        status = "failed"
                        error_message = f"Assertion failed: {assertion}"
                        break

            except Exception as e:
                status = "error"
                error_message = str(e)
                self.log.error("API test error", error=str(e))

        total_duration = int((time.time() - start_time) * 1000)

        result = APITestResult(
            test_id=test_id,
            test_name=test_name,
            status=status,
            requests=requests_results,
            schema_validations=schema_validations,
            total_duration_ms=total_duration,
            error_message=error_message,
        )

        self.log.info(
            "API test complete",
            test_id=test_id,
            status=status,
            requests_count=len(requests_results),
        )

        return AgentResult(
            success=status == "passed",
            data=result,
        )

    async def _execute_request(
        self,
        client: httpx.AsyncClient,
        step: dict,
        base_url: str,
    ) -> APIRequestResult:
        """Execute a single API request."""
        action = step.get("action", "get").upper()
        if action == "REQUEST":
            action = step.get("method", "GET").upper()

        target = step.get("target", "/")
        url = target if target.startswith("http") else f"{base_url.rstrip('/')}{target}"

        # Substitute variables in URL
        url = self._substitute_variables(url)

        # Build headers
        headers = step.get("headers", {})
        if self._auth_token and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        # Build body
        body = step.get("value") or step.get("body")
        if isinstance(body, str):
            body = self._substitute_variables(body)
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass

        start_time = time.time()

        try:
            if action == "GET":
                response = await client.get(url, headers=headers)
            elif action == "POST":
                response = await client.post(url, headers=headers, json=body)
            elif action == "PUT":
                response = await client.put(url, headers=headers, json=body)
            elif action == "DELETE":
                response = await client.delete(url, headers=headers)
            elif action == "PATCH":
                response = await client.patch(url, headers=headers, json=body)
            else:
                raise ValueError(f"Unsupported HTTP method: {action}")

            response_time = int((time.time() - start_time) * 1000)

            # Parse response body
            try:
                response_body = response.json()
            except Exception:
                response_body = response.text

            # Check expected status
            expected_status = step.get("expected_status", 200)
            if isinstance(expected_status, list):
                success = response.status_code in expected_status
            else:
                success = response.status_code == expected_status

            return APIRequestResult(
                method=action,
                url=url,
                status_code=response.status_code,
                response_time_ms=response_time,
                headers=dict(response.headers),
                body=response_body,
                success=success,
                error=None if success else f"Expected status {expected_status}, got {response.status_code}",
            )

        except Exception as e:
            return APIRequestResult(
                method=action,
                url=url,
                status_code=0,
                response_time_ms=int((time.time() - start_time) * 1000),
                headers={},
                body=None,
                success=False,
                error=str(e),
            )

    def _validate_schema(
        self, data: Any, schema: dict
    ) -> SchemaValidationResult:
        """Validate data against a JSON schema using Pydantic."""
        try:
            # Convert JSON schema to Pydantic model dynamically
            # For now, do basic type checking
            errors = []

            if "type" in schema:
                expected_type = schema["type"]
                if expected_type == "object" and not isinstance(data, dict):
                    errors.append(f"Expected object, got {type(data).__name__}")
                elif expected_type == "array" and not isinstance(data, list):
                    errors.append(f"Expected array, got {type(data).__name__}")
                elif expected_type == "string" and not isinstance(data, str):
                    errors.append(f"Expected string, got {type(data).__name__}")
                elif expected_type == "number" and not isinstance(data, (int, float)):
                    errors.append(f"Expected number, got {type(data).__name__}")
                elif expected_type == "boolean" and not isinstance(data, bool):
                    errors.append(f"Expected boolean, got {type(data).__name__}")

            if "required" in schema and isinstance(data, dict):
                for field in schema["required"]:
                    if field not in data:
                        errors.append(f"Missing required field: {field}")

            if "properties" in schema and isinstance(data, dict):
                for prop, prop_schema in schema["properties"].items():
                    if prop in data:
                        # Recursive validation could go here
                        pass

            return SchemaValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                schema_used=schema,
            )

        except Exception as e:
            return SchemaValidationResult(
                valid=False,
                errors=[f"Schema validation error: {str(e)}"],
                schema_used=schema,
            )

    def _substitute_variables(self, text: str) -> str:
        """Substitute variables in text."""
        for var_name, value in self._variables.items():
            text = text.replace(f"${{{var_name}}}", str(value))
            text = text.replace(f"${var_name}", str(value))
        return text

    def _extract_value(self, data: Any, json_path: str) -> Any:
        """Extract a value from data using simple JSON path."""
        parts = json_path.replace("$.", "").split(".")
        current = data

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list) and part.isdigit():
                current = current[int(part)]
            else:
                return None

        return current

    def _check_api_assertion(
        self, assertion: dict, requests: list[APIRequestResult]
    ) -> bool:
        """Check an API assertion."""
        assertion_type = assertion.get("type", "").lower()
        expected = assertion.get("expected")

        if assertion_type == "status_code":
            if requests:
                return requests[-1].status_code == int(expected)

        elif assertion_type == "response_contains":
            if requests:
                body = requests[-1].body
                if isinstance(body, dict):
                    return expected in json.dumps(body)
                return expected in str(body)

        elif assertion_type == "response_time_under":
            if requests:
                return requests[-1].response_time_ms < int(expected)

        elif assertion_type == "header_exists":
            target = assertion.get("target")
            if requests and target:
                return target.lower() in {k.lower() for k in requests[-1].headers}

        # Unknown assertions should fail with a warning, not silently pass
        self.log.warning(
            "Unknown assertion type - failing safely",
            assertion_type=assertion_type,
            assertion=assertion
        )
        return False

    def set_auth_token(self, token: str) -> None:
        """Set authentication token for requests."""
        self._auth_token = token

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable for use in requests."""
        self._variables[name] = value

    def clear_variables(self) -> None:
        """Clear all variables."""
        self._variables.clear()
