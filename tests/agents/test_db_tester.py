"""Tests for the database tester module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Patch target for Anthropic client in base agent
ANTHROPIC_PATCH = 'anthropic.Anthropic'


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self, mock_env_vars):
        """Test QueryResult creation."""
        from src.agents.db_tester import QueryResult

        result = QueryResult(
            query="SELECT * FROM users",
            rows=[{"id": 1, "name": "Alice"}],
            row_count=1,
            execution_time_ms=50,
            success=True,
        )

        assert result.query == "SELECT * FROM users"
        assert result.row_count == 1
        assert result.success is True
        assert result.error is None

    def test_query_result_with_error(self, mock_env_vars):
        """Test QueryResult with error."""
        from src.agents.db_tester import QueryResult

        result = QueryResult(
            query="SELECT * FROM nonexistent",
            rows=[],
            row_count=0,
            execution_time_ms=10,
            success=False,
            error="Table not found",
        )

        assert result.success is False
        assert result.error == "Table not found"

    def test_query_result_to_dict(self, mock_env_vars):
        """Test QueryResult to_dict method."""
        from src.agents.db_tester import QueryResult

        result = QueryResult(
            query="SELECT id FROM users",
            rows=[{"id": i} for i in range(15)],  # More than 10 rows
            row_count=15,
            execution_time_ms=100,
            success=True,
        )

        dict_result = result.to_dict()

        assert len(dict_result["rows"]) == 10  # Truncated to 10
        assert dict_result["row_count"] == 15  # Original count preserved


class TestDataValidationResult:
    """Tests for DataValidationResult dataclass."""

    def test_validation_result_creation(self, mock_env_vars):
        """Test DataValidationResult creation."""
        from src.agents.db_tester import DataValidationResult

        result = DataValidationResult(
            validation_type="exists",
            table="users",
            passed=True,
            expected="record exists",
            actual="1 record(s) found",
        )

        assert result.validation_type == "exists"
        assert result.passed is True

    def test_validation_result_failed(self, mock_env_vars):
        """Test failed DataValidationResult."""
        from src.agents.db_tester import DataValidationResult

        result = DataValidationResult(
            validation_type="count",
            table="orders",
            passed=False,
            expected=5,
            actual=3,
            error="Expected 5, got 3",
        )

        assert result.passed is False
        assert result.error is not None

    def test_validation_result_to_dict(self, mock_env_vars):
        """Test DataValidationResult to_dict method."""
        from src.agents.db_tester import DataValidationResult

        result = DataValidationResult(
            validation_type="relationship",
            table="orders",
            passed=True,
            expected="no orphans",
            actual="0 orphans",
        )

        dict_result = result.to_dict()

        assert dict_result["validation_type"] == "relationship"
        assert dict_result["table"] == "orders"
        assert dict_result["passed"] is True


class TestDBTestResult:
    """Tests for DBTestResult dataclass."""

    def test_db_test_result_creation(self, mock_env_vars):
        """Test DBTestResult creation."""
        from src.agents.db_tester import DBTestResult

        result = DBTestResult(
            test_id="test-001",
            test_name="User Creation Test",
            status="passed",
        )

        assert result.test_id == "test-001"
        assert result.status == "passed"
        assert result.queries == []
        assert result.validations == []

    def test_db_test_result_with_queries(self, mock_env_vars):
        """Test DBTestResult with queries."""
        from src.agents.db_tester import DBTestResult, QueryResult

        query = QueryResult(
            query="SELECT COUNT(*) FROM users",
            rows=[{"count": 10}],
            row_count=1,
            execution_time_ms=50,
            success=True,
        )

        result = DBTestResult(
            test_id="test-002",
            test_name="Count Test",
            status="passed",
            queries=[query],
            total_duration_ms=100,
        )

        assert len(result.queries) == 1
        assert result.total_duration_ms == 100

    def test_db_test_result_to_dict(self, mock_env_vars):
        """Test DBTestResult to_dict method."""
        from src.agents.db_tester import DataValidationResult, DBTestResult, QueryResult

        result = DBTestResult(
            test_id="test-003",
            test_name="Full Test",
            status="failed",
            queries=[
                QueryResult("SELECT 1", [], 0, 10, True)
            ],
            validations=[
                DataValidationResult("exists", "users", False, "exists", "not found")
            ],
            error_message="Validation failed",
        )

        dict_result = result.to_dict()

        assert dict_result["status"] == "failed"
        assert len(dict_result["queries"]) == 1
        assert len(dict_result["validations"]) == 1


class TestDBTesterAgent:
    """Tests for DBTesterAgent class."""

    def test_agent_creation(self, mock_env_vars):
        """Test DBTesterAgent creation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            assert agent._database_url is None
            assert agent._engine is None

    def test_agent_creation_with_url(self, mock_env_vars):
        """Test DBTesterAgent creation with database URL."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent(database_url="postgresql://localhost/test")

            assert agent._database_url == "postgresql://localhost/test"

    def test_get_system_prompt(self, mock_env_vars):
        """Test system prompt generation."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            prompt = agent._get_system_prompt()

            assert "database" in prompt.lower()
            assert "validate" in prompt.lower()

    @pytest.mark.asyncio
    async def test_execute_no_connection(self, mock_env_vars):
        """Test execute without database connection."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            test_spec = {
                "id": "test-001",
                "name": "Test",
                "steps": [],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.success is False
            assert result.data.status == "error"
            assert "No database connection" in result.data.error_message

    @pytest.mark.asyncio
    async def test_execute_with_dict_spec(self, mock_env_vars):
        """Test execute with dict test specification."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()  # Mock engine to bypass connection
            agent._disconnect = AsyncMock()  # Mock async disconnect

            test_spec = {
                "id": "test-001",
                "name": "Dict Test",
                "steps": [],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.test_id == "test-001"
            assert result.data.test_name == "Dict Test"

    @pytest.mark.asyncio
    async def test_execute_query_action(self, mock_env_vars):
        """Test execute with query action."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            # Mock the internal methods
            agent._engine = MagicMock()
            agent._session = MagicMock()

            async def mock_execute_query(query):
                from src.agents.db_tester import QueryResult
                return QueryResult(
                    query=query,
                    rows=[{"id": 1}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query = mock_execute_query
            agent._disconnect = AsyncMock()

            test_spec = {
                "id": "test-001",
                "name": "Query Test",
                "steps": [
                    {"action": "query", "target": "SELECT * FROM users"},
                ],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert len(result.data.queries) == 1
            assert result.data.queries[0].success is True

    @pytest.mark.asyncio
    async def test_execute_validate_exists_action(self, mock_env_vars):
        """Test execute with validate_exists action."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DataValidationResult, DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()

            async def mock_validate_exists(table, conditions):
                return DataValidationResult(
                    validation_type="exists",
                    table=table,
                    passed=True,
                    expected="record exists",
                    actual="1 record(s) found",
                )

            agent._validate_exists = mock_validate_exists
            agent._disconnect = AsyncMock()

            test_spec = {
                "id": "test-001",
                "name": "Exists Test",
                "steps": [
                    {"action": "validate_exists", "target": "users", "value": {"id": 1}},
                ],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert len(result.data.validations) == 1
            assert result.data.validations[0].passed is True

    @pytest.mark.asyncio
    async def test_execute_validate_count_action(self, mock_env_vars):
        """Test execute with validate_count action."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DataValidationResult, DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()

            async def mock_validate_count(table, expected_count, conditions):
                return DataValidationResult(
                    validation_type="count",
                    table=table,
                    passed=True,
                    expected=expected_count,
                    actual=expected_count,
                )

            agent._validate_count = mock_validate_count
            agent._disconnect = AsyncMock()

            test_spec = {
                "id": "test-001",
                "name": "Count Test",
                "steps": [
                    {"action": "validate_count", "target": "users", "value": 5},
                ],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert len(result.data.validations) == 1
            assert result.data.validations[0].validation_type == "count"

    @pytest.mark.asyncio
    async def test_execute_validate_relationship_action(self, mock_env_vars):
        """Test execute with validate_relationship action."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DataValidationResult, DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()

            async def mock_validate_relationship(source_table, relationship):
                return DataValidationResult(
                    validation_type="relationship",
                    table=source_table,
                    passed=True,
                    expected="no orphaned records",
                    actual="0 orphaned record(s)",
                )

            agent._validate_relationship = mock_validate_relationship
            agent._disconnect = AsyncMock()

            test_spec = {
                "id": "test-001",
                "name": "Relationship Test",
                "steps": [
                    {
                        "action": "validate_relationship",
                        "target": "orders",
                        "value": {
                            "target_table": "users",
                            "source_column": "user_id",
                            "target_column": "id",
                        },
                    },
                ],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert len(result.data.validations) == 1

    @pytest.mark.asyncio
    async def test_execute_with_test_spec_object(self, mock_env_vars):
        """Test execute with TestSpec object."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent
            from src.agents.test_planner import TestSpec

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            test_spec = TestSpec(
                id="test-obj-001",
                name="Object Test",
                type="db",
                priority="high",
                description="Test with object",
                steps=[],
                assertions=[],
            )

            result = await agent.execute(test_spec)

            assert result.data.test_id == "test-obj-001"
            assert result.data.test_name == "Object Test"

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, mock_env_vars):
        """Test execute handles exceptions during step execution."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            # Mock _execute_query to raise an exception
            async def failing_query(query):
                raise Exception("Query execution failed")

            agent._execute_query = failing_query

            test_spec = {
                "id": "test-001",
                "name": "Exception Test",
                "steps": [{"action": "query", "target": "SELECT 1"}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.status == "error"
            assert "Query execution failed" in result.data.error_message

    @pytest.mark.asyncio
    async def test_check_db_assertion_row_exists(self, mock_env_vars):
        """Test _check_db_assertion with row_exists type."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DataValidationResult, DBTesterAgent

            agent = DBTesterAgent()

            async def mock_validate_exists(table, conditions):
                return DataValidationResult(
                    validation_type="exists",
                    table=table,
                    passed=True,
                    expected="record exists",
                    actual="1 record(s) found",
                )

            agent._validate_exists = mock_validate_exists

            assertion = {
                "type": "row_exists",
                "target": "users",
                "expected": {"id": 1},
            }

            result = await agent._check_db_assertion(assertion)

            assert result.validation_type == "exists"
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_db_assertion_row_count(self, mock_env_vars):
        """Test _check_db_assertion with row_count type."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DataValidationResult, DBTesterAgent

            agent = DBTesterAgent()

            async def mock_validate_count(table, expected_count, conditions):
                return DataValidationResult(
                    validation_type="count",
                    table=table,
                    passed=True,
                    expected=expected_count,
                    actual=expected_count,
                )

            agent._validate_count = mock_validate_count

            assertion = {
                "type": "row_count",
                "target": "users",
                "expected": 10,
            }

            result = await agent._check_db_assertion(assertion)

            assert result.validation_type == "count"

    @pytest.mark.asyncio
    async def test_check_db_assertion_query_returns(self, mock_env_vars):
        """Test _check_db_assertion with query_returns type."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query(query):
                return QueryResult(
                    query=query,
                    rows=[{"id": 1}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query = mock_execute_query

            assertion = {
                "type": "query_returns",
                "target": "SELECT * FROM users WHERE active = true",
                "expected": "rows",
            }

            result = await agent._check_db_assertion(assertion)

            assert result.validation_type == "query_returns"
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_db_assertion_unknown_type(self, mock_env_vars):
        """Test _check_db_assertion with unknown type."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            assertion = {
                "type": "unknown_type",
                "target": "users",
            }

            result = await agent._check_db_assertion(assertion)

            assert result.passed is False
            assert "Unknown assertion type" in result.error

    @pytest.mark.asyncio
    async def test_get_table_schema(self, mock_env_vars):
        """Test get_table_schema method."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query(query):
                return QueryResult(
                    query=query,
                    rows=[
                        {"column_name": "id", "data_type": "integer", "is_nullable": "NO"},
                        {"column_name": "name", "data_type": "varchar", "is_nullable": "YES"},
                    ],
                    row_count=2,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query = mock_execute_query

            result = await agent.get_table_schema("users")

            assert result["table"] == "users"
            assert len(result["columns"]) == 2

    @pytest.mark.asyncio
    async def test_get_table_stats(self, mock_env_vars):
        """Test get_table_stats method."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query(query):
                return QueryResult(
                    query=query,
                    rows=[{"total": 100}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query = mock_execute_query

            result = await agent.get_table_stats("users")

            assert result["table"] == "users"
            assert result["row_count"] == 100
