"""Tests for the API Tester agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


class TestAPIRequestResult:
    """Tests for APIRequestResult dataclass."""

    def test_api_request_result_creation(self, mock_env_vars):
        """Test creating an APIRequestResult."""
        from src.agents.api_tester import APIRequestResult

        request = APIRequestResult(
            method="GET",
            url="https://api.example.com/users",
            status_code=200,
            response_time_ms=150,
            headers={"Content-Type": "application/json"},
            body={"users": []},
            success=True,
        )

        assert request.method == "GET"
        assert request.status_code == 200
        assert request.response_time_ms == 150
        assert request.success is True


class TestAPITestResult:
    """Tests for APITestResult dataclass."""

    def test_api_test_result(self, mock_env_vars):
        """Test creating APITestResult."""
        from src.agents.api_tester import APITestResult

        result = APITestResult(
            test_id="api-001",
            test_name="User API Test",
            status="passed",
            total_duration_ms=500,
        )

        assert result.status == "passed"


class TestAPITesterAgent:
    """Tests for APITesterAgent class."""

    @pytest.fixture
    def api_agent(self, mock_env_vars):
        """Create API tester agent."""
        from src.agents.api_tester import APITesterAgent

        return APITesterAgent()

    @pytest.fixture
    def mock_httpx_response(self):
        """Create mock httpx response."""
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = {"id": 1, "name": "Test User"}
        response.text = '{"id": 1, "name": "Test User"}'
        return response

    def test_agent_initialization(self, api_agent):
        """Test APITesterAgent initialization."""
        assert api_agent._auth_token is None
        assert api_agent._variables == {}

    def test_set_auth_token(self, api_agent):
        """Test setting auth token."""
        api_agent.set_auth_token("test-token")
        assert api_agent._auth_token == "test-token"

    def test_set_variable(self, api_agent):
        """Test setting variables."""
        api_agent.set_variable("user_id", 123)
        assert api_agent._variables["user_id"] == 123

    def test_clear_variables(self, api_agent):
        """Test clearing variables."""
        api_agent.set_variable("key", "value")
        api_agent.clear_variables()
        assert api_agent._variables == {}

    @pytest.mark.asyncio
    async def test_execute_success(self, api_agent, sample_api_test_spec):
        """Test successful API test execution."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = {"name": "Test User"}
            mock_response.text = '{"name": "Test User"}'

            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            result = await api_agent.execute(
                test_spec=sample_api_test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is True

    def test_check_assertion_status_code(self, api_agent):
        """Test status_code assertion."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/users",
                status_code=200,
                response_time_ms=100,
                headers={},
                body={},
                success=True,
            ),
        ]

        assertion = {"type": "status_code", "expected": "200"}

        result = api_agent._check_api_assertion(assertion, requests)

        assert result is True

    def test_check_assertion_response_contains(self, api_agent):
        """Test response_contains assertion."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/users",
                status_code=200,
                response_time_ms=100,
                headers={},
                body={"name": "Test User"},
                success=True,
            ),
        ]

        assertion = {"type": "response_contains", "expected": "Test User"}

        result = api_agent._check_api_assertion(assertion, requests)

        assert result is True

    def test_check_assertion_response_time(self, api_agent):
        """Test response_time_under assertion."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/users",
                status_code=200,
                response_time_ms=100,
                headers={},
                body={},
                success=True,
            ),
        ]

        assertion = {"type": "response_time_under", "expected": "500"}

        result = api_agent._check_api_assertion(assertion, requests)

        assert result is True

    def test_check_assertion_header_exists(self, api_agent):
        """Test header_exists assertion."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/users",
                status_code=200,
                response_time_ms=100,
                headers={"Content-Type": "application/json"},
                body={},
                success=True,
            ),
        ]

        assertion = {"type": "header_exists", "target": "Content-Type"}

        result = api_agent._check_api_assertion(assertion, requests)

        assert result is True

    def test_check_assertion_unknown_type(self, api_agent):
        """Test unknown assertion type fails safely."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/users",
                status_code=200,
                response_time_ms=100,
                headers={},
                body={},
                success=True,
            ),
        ]

        assertion = {"type": "unknown_type", "expected": "value"}

        result = api_agent._check_api_assertion(assertion, requests)

        # Unknown assertions should fail (not pass silently)
        assert result is False


class TestAPITesterEdgeCases:
    """Edge case tests for APITesterAgent."""

    @pytest.fixture
    def api_agent(self, mock_env_vars):
        """Create API tester agent."""
        from src.agents.api_tester import APITesterAgent

        return APITesterAgent()

    @pytest.mark.asyncio
    async def test_empty_response_body(self, api_agent):
        """Test handling of empty response body."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.side_effect = Exception("No JSON body")
            mock_response.text = ""

            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "empty-response-test",
                "name": "Empty Response Test",
                "steps": [
                    {"action": "get", "target": "/api/empty", "expected_status": 204},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is True
            assert result.data.requests[0].body == ""

    @pytest.mark.asyncio
    async def test_empty_response_null_body(self, api_agent):
        """Test handling of null/None response body."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = None
            mock_response.text = "null"

            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "null-response-test",
                "name": "Null Response Test",
                "steps": [
                    {"action": "get", "target": "/api/null", "expected_status": 200},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is True
            assert result.data.requests[0].body is None

    @pytest.mark.asyncio
    async def test_timeout_error(self, api_agent):
        """Test handling of request timeout."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Connection timed out"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "timeout-test",
                "name": "Timeout Test",
                "steps": [
                    {"action": "get", "target": "/api/slow-endpoint"},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is False
            assert result.data.status == "failed"
            assert result.data.requests[0].success is False
            assert "timed out" in result.data.requests[0].error.lower()

    @pytest.mark.asyncio
    async def test_read_timeout(self, api_agent):
        """Test handling of read timeout."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout("Read timed out"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "read-timeout-test",
                "name": "Read Timeout Test",
                "steps": [
                    {"action": "post", "target": "/api/data", "body": {"key": "value"}},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is False
            assert result.data.requests[0].status_code == 0

    @pytest.mark.asyncio
    async def test_invalid_status_code_4xx(self, api_agent):
        """Test handling of 4xx client error status codes."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = {"error": "Not Found"}
            mock_response.text = '{"error": "Not Found"}'

            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "404-test",
                "name": "404 Error Test",
                "steps": [
                    {"action": "get", "target": "/api/nonexistent", "expected_status": 200},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is False
            assert result.data.status == "failed"
            assert result.data.requests[0].status_code == 404
            assert "Expected status 200" in result.data.requests[0].error

    @pytest.mark.asyncio
    async def test_invalid_status_code_5xx(self, api_agent):
        """Test handling of 5xx server error status codes."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = {"error": "Service Unavailable"}
            mock_response.text = '{"error": "Service Unavailable"}'

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "503-test",
                "name": "503 Error Test",
                "steps": [
                    {"action": "post", "target": "/api/process", "body": {}},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is False
            assert result.data.requests[0].status_code == 503

    @pytest.mark.asyncio
    async def test_expected_error_status_codes(self, api_agent):
        """Test that expected error status codes are treated as success."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = {"error": "Resource not found"}
            mock_response.text = '{"error": "Resource not found"}'

            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "expected-404-test",
                "name": "Expected 404 Test",
                "steps": [
                    {"action": "get", "target": "/api/deleted", "expected_status": 404},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is True
            assert result.data.requests[0].success is True

    @pytest.mark.asyncio
    async def test_expected_status_code_list(self, api_agent):
        """Test handling of multiple expected status codes."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = {"id": 123}
            mock_response.text = '{"id": 123}'

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "multi-status-test",
                "name": "Multi Status Test",
                "steps": [
                    {"action": "post", "target": "/api/create", "expected_status": [200, 201]},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_malformed_json_response(self, api_agent):
        """Test handling of malformed JSON response."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.side_effect = Exception("Invalid JSON")
            mock_response.text = "{ invalid json content"

            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "malformed-json-test",
                "name": "Malformed JSON Test",
                "steps": [
                    {"action": "get", "target": "/api/broken"},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            # Should still succeed since we fall back to text
            assert result.success is True
            assert result.data.requests[0].body == "{ invalid json content"

    @pytest.mark.asyncio
    async def test_html_response_instead_of_json(self, api_agent):
        """Test handling of HTML response when JSON expected."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.json.side_effect = Exception("Not JSON")
            mock_response.text = "<html><body>Error Page</body></html>"

            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "html-response-test",
                "name": "HTML Response Test",
                "steps": [
                    {"action": "get", "target": "/api/misconfigured"},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is True
            assert "<html>" in result.data.requests[0].body

    @pytest.mark.asyncio
    async def test_network_connection_error(self, api_agent):
        """Test handling of network connection failure."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "connection-error-test",
                "name": "Connection Error Test",
                "steps": [
                    {"action": "get", "target": "/api/users"},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is False
            assert result.data.status == "failed"
            assert result.data.requests[0].status_code == 0
            assert "Connection refused" in result.data.requests[0].error

    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self, api_agent):
        """Test handling of DNS resolution failure."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("Name or service not known")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "dns-failure-test",
                "name": "DNS Failure Test",
                "steps": [
                    {"action": "get", "target": "/api/health"},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://nonexistent.invalid.domain",
            )

            assert result.success is False
            assert result.data.requests[0].success is False

    @pytest.mark.asyncio
    async def test_ssl_certificate_error(self, api_agent):
        """Test handling of SSL certificate errors."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.ConnectError("SSL: CERTIFICATE_VERIFY_FAILED")
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "ssl-error-test",
                "name": "SSL Error Test",
                "steps": [
                    {"action": "get", "target": "/api/secure"},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://self-signed.example.com",
            )

            assert result.success is False
            assert "SSL" in result.data.requests[0].error or "CERTIFICATE" in result.data.requests[0].error

    @pytest.mark.asyncio
    async def test_empty_steps_list(self, api_agent):
        """Test handling of test spec with no steps."""
        test_spec = {
            "id": "no-steps-test",
            "name": "No Steps Test",
            "steps": [],
            "assertions": [],
        }

        result = await api_agent.execute(
            test_spec=test_spec,
            app_url="https://api.example.com",
        )

        assert result.success is True
        assert len(result.data.requests) == 0

    @pytest.mark.asyncio
    async def test_unsupported_http_method(self, api_agent):
        """Test handling of unsupported HTTP method.

        The agent uses 'request' action type with a 'method' field for custom methods.
        Using an unsupported method should result in an error.
        """
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "unsupported-method-test",
                "name": "Unsupported Method Test",
                "steps": [
                    {"action": "request", "method": "INVALID_METHOD", "target": "/api/test"},
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            assert result.success is False
            assert "Unsupported HTTP method" in result.data.requests[0].error

    @pytest.mark.asyncio
    async def test_request_body_json_decode_error(self, api_agent):
        """Test handling of invalid JSON in request body string."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = {"success": True}
            mock_response.text = '{"success": true}'

            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            test_spec = {
                "id": "invalid-body-json-test",
                "name": "Invalid Body JSON Test",
                "steps": [
                    {
                        "action": "post",
                        "target": "/api/data",
                        "body": "not valid json {",
                    },
                ],
                "assertions": [],
            }

            result = await api_agent.execute(
                test_spec=test_spec,
                app_url="https://api.example.com",
            )

            # Should still proceed with the raw string body
            assert result.data is not None

    def test_assertion_empty_requests_list(self, api_agent):
        """Test assertion checking with empty requests list."""
        assertion = {"type": "status_code", "expected": "200"}

        result = api_agent._check_api_assertion(assertion, [])

        assert result is False

    def test_response_contains_assertion_with_empty_body(self, api_agent):
        """Test response_contains assertion with empty/None body."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/empty",
                status_code=200,
                response_time_ms=100,
                headers={},
                body=None,
                success=True,
            ),
        ]

        assertion = {"type": "response_contains", "expected": "test"}

        result = api_agent._check_api_assertion(assertion, requests)

        assert result is False

    def test_header_exists_assertion_with_empty_headers(self, api_agent):
        """Test header_exists assertion with empty headers dict."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/no-headers",
                status_code=200,
                response_time_ms=100,
                headers={},
                body={},
                success=True,
            ),
        ]

        assertion = {"type": "header_exists", "target": "Content-Type"}

        result = api_agent._check_api_assertion(assertion, requests)

        assert result is False

    def test_response_time_assertion_with_slow_response(self, api_agent):
        """Test response_time_under assertion fails for slow response."""
        from src.agents.api_tester import APIRequestResult

        requests = [
            APIRequestResult(
                method="GET",
                url="/api/slow",
                status_code=200,
                response_time_ms=5000,
                headers={},
                body={},
                success=True,
            ),
        ]

        assertion = {"type": "response_time_under", "expected": "1000"}

        result = api_agent._check_api_assertion(assertion, requests)

        assert result is False

    def test_schema_validation_with_missing_required_fields(self, api_agent):
        """Test schema validation detects missing required fields."""
        schema = {
            "type": "object",
            "required": ["id", "name", "email"],
        }

        data = {"id": 1, "name": "Test"}

        result = api_agent._validate_schema(data, schema)

        assert result.valid is False
        assert any("email" in error for error in result.errors)

    def test_schema_validation_with_wrong_type(self, api_agent):
        """Test schema validation detects wrong data type."""
        schema = {"type": "object"}

        data = ["not", "an", "object"]

        result = api_agent._validate_schema(data, schema)

        assert result.valid is False
        assert any("Expected object" in error for error in result.errors)

    def test_schema_validation_with_array_type(self, api_agent):
        """Test schema validation for array type mismatch."""
        schema = {"type": "array"}

        data = {"not": "an array"}

        result = api_agent._validate_schema(data, schema)

        assert result.valid is False
        assert any("Expected array" in error for error in result.errors)

    def test_variable_extraction_from_nested_response(self, api_agent):
        """Test variable extraction from deeply nested JSON."""
        data = {
            "user": {
                "profile": {
                    "id": 12345,
                },
            },
        }

        result = api_agent._extract_value(data, "$.user.profile.id")

        assert result == 12345

    def test_variable_extraction_missing_path(self, api_agent):
        """Test variable extraction with non-existent path."""
        data = {"user": {"name": "Test"}}

        result = api_agent._extract_value(data, "$.user.profile.id")

        assert result is None

    def test_variable_extraction_from_array(self, api_agent):
        """Test variable extraction from array index."""
        data = {"users": [{"id": 1}, {"id": 2}, {"id": 3}]}

        result = api_agent._extract_value(data, "$.users.0.id")

        assert result == 1

    def test_variable_substitution_in_url(self, api_agent):
        """Test variable substitution in URL strings."""
        api_agent.set_variable("user_id", 123)
        api_agent.set_variable("org", "acme")

        url = "/api/orgs/${org}/users/${user_id}"
        result = api_agent._substitute_variables(url)

        assert result == "/api/orgs/acme/users/123"

    def test_variable_substitution_brace_syntax(self, api_agent):
        """Test variable substitution with ${var} syntax."""
        api_agent.set_variable("token", "abc123")

        text = "Bearer ${token}"
        result = api_agent._substitute_variables(text)

        assert result == "Bearer abc123"
