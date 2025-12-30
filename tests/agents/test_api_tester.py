"""Tests for the API Tester agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


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
