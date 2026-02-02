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

            async def mock_execute_query_parameterized(query, params):
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

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent.get_table_schema("users")

            assert result["table"] == "users"
            assert len(result["columns"]) == 2

    @pytest.mark.asyncio
    async def test_get_table_stats(self, mock_env_vars):
        """Test get_table_stats method."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query_parameterized(query, params):
                return QueryResult(
                    query=query,
                    rows=[{"total": 100}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent.get_table_stats("users")

            assert result["table"] == "users"
            assert result["row_count"] == 100


class TestDBTesterEdgeCases:
    """Edge case tests for DBTesterAgent."""

    # =========================================================================
    # Empty Input Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_execute_empty_test_spec(self, mock_env_vars):
        """Test execute with empty test specification."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            test_spec = {
                "id": "",
                "name": "",
                "steps": [],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.test_id == ""
            assert result.data.test_name == ""
            assert result.data.status == "passed"  # No steps = no failures
            assert len(result.data.queries) == 0
            assert len(result.data.validations) == 0

    @pytest.mark.asyncio
    async def test_execute_empty_query(self, mock_env_vars):
        """Test execute with empty query string."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            async def mock_execute_query(query):
                if not query.strip():
                    return QueryResult(
                        query=query,
                        rows=[],
                        row_count=0,
                        execution_time_ms=0,
                        success=False,
                        error="Empty query",
                    )
                return QueryResult(
                    query=query,
                    rows=[],
                    row_count=0,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query = mock_execute_query

            test_spec = {
                "id": "test-empty-query",
                "name": "Empty Query Test",
                "steps": [{"action": "query", "target": ""}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.status == "failed"
            assert result.data.queries[0].success is False
            assert "Empty query" in result.data.queries[0].error

    @pytest.mark.asyncio
    async def test_validate_exists_empty_table(self, mock_env_vars):
        """Test validate_exists with empty table name."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            result = await agent._validate_exists(table="", conditions={})

            assert result.passed is False
            assert result.error is not None
            assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_validate_count_empty_conditions(self, mock_env_vars):
        """Test validate_count with empty conditions."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query_parameterized(query, params):
                return QueryResult(
                    query=query,
                    rows=[{"cnt": 5}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent._validate_count(
                table="users",
                expected_count=5,
                conditions={},
            )

            assert result.passed is True
            assert result.actual == 5

    # =========================================================================
    # Invalid SQL Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_execute_invalid_sql_syntax(self, mock_env_vars):
        """Test execute with invalid SQL syntax."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            async def mock_execute_query(query):
                return QueryResult(
                    query=query,
                    rows=[],
                    row_count=0,
                    execution_time_ms=5,
                    success=False,
                    error="syntax error at or near 'SELEKT'",
                )

            agent._execute_query = mock_execute_query

            test_spec = {
                "id": "test-invalid-sql",
                "name": "Invalid SQL Test",
                "steps": [{"action": "query", "target": "SELEKT * FORM users"}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.status == "failed"
            assert "syntax error" in result.data.error_message

    @pytest.mark.asyncio
    async def test_validate_exists_sql_injection_attempt(self, mock_env_vars):
        """Test that SQL injection attempts are blocked."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            # Attempt SQL injection via table name
            result = await agent._validate_exists(
                table="users; DROP TABLE users; --",
                conditions={},
            )

            assert result.passed is False
            assert result.error is not None
            assert "Invalid SQL identifier" in result.error

    @pytest.mark.asyncio
    async def test_validate_exists_sql_injection_in_column(self, mock_env_vars):
        """Test that SQL injection in column names is blocked."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            # Attempt SQL injection via column name
            result = await agent._validate_exists(
                table="users",
                conditions={"id; DROP TABLE users; --": 1},
            )

            assert result.passed is False
            assert result.error is not None
            assert "Invalid SQL identifier" in result.error

    @pytest.mark.asyncio
    async def test_validate_relationship_invalid_identifiers(self, mock_env_vars):
        """Test validate_relationship with invalid SQL identifiers."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            result = await agent._validate_relationship(
                source_table="orders",
                relationship={
                    "target_table": "users' OR '1'='1",
                    "source_column": "user_id",
                    "target_column": "id",
                },
            )

            assert result.passed is False
            assert "Invalid SQL identifier" in result.error

    @pytest.mark.asyncio
    async def test_get_table_schema_invalid_table_name(self, mock_env_vars):
        """Test get_table_schema with invalid table name."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            result = await agent.get_table_schema("users; DELETE FROM users;")

            assert "error" in result
            assert "Invalid table name" in result["error"]

    @pytest.mark.asyncio
    async def test_get_table_stats_invalid_table_name(self, mock_env_vars):
        """Test get_table_stats with invalid table name."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            result = await agent.get_table_stats("users--")

            assert "error" in result
            assert "Invalid SQL identifier" in result["error"]

    # =========================================================================
    # Connection Failure Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_connect_invalid_url(self, mock_env_vars):
        """Test connection with invalid database URL."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            with patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_create_engine:
                mock_create_engine.side_effect = Exception("Invalid database URL")

                with pytest.raises(Exception) as exc_info:
                    await agent._connect("invalid://url")

                assert "Invalid database URL" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connect_unreachable_host(self, mock_env_vars):
        """Test connection to unreachable host."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            with patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_create_engine:
                mock_create_engine.side_effect = Exception(
                    "could not connect to server: Connection refused"
                )

                with pytest.raises(Exception) as exc_info:
                    await agent._connect("postgresql://nonexistent-host:5432/db")

                assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_connection_lost_during_query(self, mock_env_vars):
        """Test handling of connection loss during query execution."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            async def mock_execute_query(query):
                raise Exception("Connection reset by peer")

            agent._execute_query = mock_execute_query

            test_spec = {
                "id": "test-conn-lost",
                "name": "Connection Lost Test",
                "steps": [{"action": "query", "target": "SELECT 1"}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.status == "error"
            assert "Connection reset by peer" in result.data.error_message

    @pytest.mark.asyncio
    async def test_execute_authentication_failure(self, mock_env_vars):
        """Test handling of authentication failure."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            with patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_create_engine:
                mock_create_engine.side_effect = Exception(
                    "password authentication failed for user 'test'"
                )

                test_spec = {
                    "id": "test-auth-fail",
                    "name": "Auth Failure Test",
                    "steps": [{"action": "query", "target": "SELECT 1"}],
                    "assertions": [],
                }

                result = await agent.execute(
                    test_spec,
                    database_url="postgresql://user:wrongpass@localhost/db",
                )

                assert result.data.status == "error"
                assert "password authentication failed" in result.data.error_message

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, mock_env_vars):
        """Test disconnect when no connection exists."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = None
            agent._session = None

            # Should not raise any exceptions
            await agent._disconnect()

            assert agent._engine is None
            assert agent._session is None

    # =========================================================================
    # Timeout Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_execute_query_timeout(self, mock_env_vars):
        """Test query execution timeout."""
        with patch(ANTHROPIC_PATCH):
            import asyncio

            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            async def mock_slow_query(query):
                raise asyncio.TimeoutError("Query timed out after 30 seconds")

            agent._execute_query = mock_slow_query

            test_spec = {
                "id": "test-timeout",
                "name": "Timeout Test",
                "steps": [{"action": "query", "target": "SELECT pg_sleep(60)"}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.status == "error"
            assert "timed out" in result.data.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_connection_timeout(self, mock_env_vars):
        """Test connection timeout."""
        with patch(ANTHROPIC_PATCH):
            import asyncio

            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()

            with patch("sqlalchemy.ext.asyncio.create_async_engine") as mock_create_engine:
                mock_create_engine.side_effect = asyncio.TimeoutError(
                    "Connection timed out"
                )

                test_spec = {
                    "id": "test-conn-timeout",
                    "name": "Connection Timeout Test",
                    "steps": [{"action": "query", "target": "SELECT 1"}],
                    "assertions": [],
                }

                result = await agent.execute(
                    test_spec,
                    database_url="postgresql://slow-host:5432/db",
                )

                assert result.data.status == "error"
                assert "timed out" in result.data.error_message.lower()

    @pytest.mark.asyncio
    async def test_validate_exists_timeout(self, mock_env_vars):
        """Test validate_exists with timeout - returns failed QueryResult."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_timeout_query(query, params):
                # Simulate timeout being caught by _execute_query_parameterized
                # and returned as a failed QueryResult
                return QueryResult(
                    query=query,
                    rows=[],
                    row_count=0,
                    execution_time_ms=30000,
                    success=False,
                    error="Query execution timed out after 30 seconds",
                )

            agent._execute_query_parameterized = mock_timeout_query

            result = await agent._validate_exists(
                table="large_table",
                conditions={"id": 1},
            )

            assert result.passed is False
            assert result.error is not None
            # The actual error from validate_exists is "query failed"
            assert "query failed" in result.actual.lower()

    # =========================================================================
    # Large Result Set Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_query_result_truncation(self, mock_env_vars):
        """Test that QueryResult truncates large result sets in to_dict()."""
        from src.agents.db_tester import QueryResult

        # Create result with 100 rows
        large_rows = [{"id": i, "name": f"User {i}"} for i in range(100)]

        result = QueryResult(
            query="SELECT * FROM users",
            rows=large_rows,
            row_count=100,
            execution_time_ms=500,
            success=True,
        )

        dict_result = result.to_dict()

        # Rows should be truncated to 10 in to_dict output
        assert len(dict_result["rows"]) == 10
        # Original row_count preserved
        assert dict_result["row_count"] == 100
        # Original object retains all rows
        assert len(result.rows) == 100

    @pytest.mark.asyncio
    async def test_execute_large_result_set(self, mock_env_vars):
        """Test execute with a query returning many rows."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            large_rows = [{"id": i} for i in range(10000)]

            async def mock_execute_query(query):
                return QueryResult(
                    query=query,
                    rows=large_rows,
                    row_count=10000,
                    execution_time_ms=5000,
                    success=True,
                )

            agent._execute_query = mock_execute_query

            test_spec = {
                "id": "test-large-result",
                "name": "Large Result Test",
                "steps": [{"action": "query", "target": "SELECT * FROM big_table"}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.status == "passed"
            assert result.data.queries[0].row_count == 10000
            # Verify to_dict truncation
            assert len(result.data.queries[0].to_dict()["rows"]) == 10

    @pytest.mark.asyncio
    async def test_validate_count_large_table(self, mock_env_vars):
        """Test validate_count on a large table."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query_parameterized(query, params):
                return QueryResult(
                    query=query,
                    rows=[{"cnt": 1000000}],
                    row_count=1,
                    execution_time_ms=2000,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent._validate_count(
                table="large_table",
                expected_count=1000000,
                conditions={},
            )

            assert result.passed is True
            assert result.actual == 1000000

    @pytest.mark.asyncio
    async def test_get_table_schema_many_columns(self, mock_env_vars):
        """Test get_table_schema with a table having many columns."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            # Simulate a table with 200 columns
            many_columns = [
                {"column_name": f"col_{i}", "data_type": "varchar", "is_nullable": "YES"}
                for i in range(200)
            ]

            async def mock_execute_query_parameterized(query, params):
                return QueryResult(
                    query=query,
                    rows=many_columns,
                    row_count=200,
                    execution_time_ms=50,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent.get_table_schema("wide_table")

            assert result["table"] == "wide_table"
            assert len(result["columns"]) == 200

    @pytest.mark.asyncio
    async def test_memory_handling_very_large_result(self, mock_env_vars):
        """Test that very large result sets don't cause memory issues."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            # Simulate a large text field in each row
            large_text = "x" * 1000  # 1KB per field
            rows = [{"id": i, "data": large_text} for i in range(1000)]

            async def mock_execute_query(query):
                return QueryResult(
                    query=query,
                    rows=rows,
                    row_count=1000,
                    execution_time_ms=3000,
                    success=True,
                )

            agent._execute_query = mock_execute_query

            test_spec = {
                "id": "test-memory",
                "name": "Memory Test",
                "steps": [{"action": "query", "target": "SELECT * FROM text_table"}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            assert result.data.status == "passed"
            # to_dict should truncate to 10 rows
            dict_result = result.data.queries[0].to_dict()
            assert len(dict_result["rows"]) == 10

    # =========================================================================
    # Additional Edge Cases
    # =========================================================================

    @pytest.mark.asyncio
    async def test_execute_null_values_in_conditions(self, mock_env_vars):
        """Test validate_exists with NULL values in conditions."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query_parameterized(query, params):
                return QueryResult(
                    query=query,
                    rows=[{"cnt": 0}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent._validate_exists(
                table="users",
                conditions={"deleted_at": None},
            )

            # NULL conditions should still work (though may not match as expected)
            assert result.validation_type == "exists"

    @pytest.mark.asyncio
    async def test_execute_special_characters_in_values(self, mock_env_vars):
        """Test validate_exists with special characters in condition values."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query_parameterized(query, params):
                # Verify parameterized query doesn't include raw value
                assert "O'Brien" not in query  # Value should be parameterized
                return QueryResult(
                    query=query,
                    rows=[{"cnt": 1}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent._validate_exists(
                table="users",
                conditions={"name": "O'Brien"},  # Special character
            )

            assert result.validation_type == "exists"
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_execute_unicode_in_values(self, mock_env_vars):
        """Test handling of unicode values in conditions."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query_parameterized(query, params):
                return QueryResult(
                    query=query,
                    rows=[{"cnt": 1}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent._validate_exists(
                table="users",
                conditions={"name": "test user"},  # Unicode characters
            )

            assert result.passed is True

    @pytest.mark.asyncio
    async def test_execute_schema_qualified_table_name(self, mock_env_vars):
        """Test validation with schema-qualified table name."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query_parameterized(query, params):
                # Verify schema-qualified name is properly quoted
                assert '"public"."users"' in query
                return QueryResult(
                    query=query,
                    rows=[{"cnt": 5}],
                    row_count=1,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query_parameterized = mock_execute_query_parameterized

            result = await agent._validate_count(
                table="public.users",
                expected_count=5,
                conditions={},
            )

            assert result.passed is True

    @pytest.mark.asyncio
    async def test_check_db_assertion_query_returns_empty(self, mock_env_vars):
        """Test query_returns assertion expecting empty result."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent, QueryResult

            agent = DBTesterAgent()

            async def mock_execute_query(query):
                return QueryResult(
                    query=query,
                    rows=[],
                    row_count=0,
                    execution_time_ms=10,
                    success=True,
                )

            agent._execute_query = mock_execute_query

            assertion = {
                "type": "query_returns",
                "target": "SELECT * FROM users WHERE deleted = true",
                "expected": "no_rows",
            }

            result = await agent._check_db_assertion(assertion)

            assert result.passed is True  # 0 rows when expecting no_rows

    @pytest.mark.asyncio
    async def test_execute_missing_action_in_step(self, mock_env_vars):
        """Test execute with step missing action field."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            test_spec = {
                "id": "test-no-action",
                "name": "Missing Action Test",
                "steps": [{"target": "SELECT 1"}],  # Missing 'action' key
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            # Should pass since unknown action is ignored
            assert result.data.status == "passed"

    @pytest.mark.asyncio
    async def test_execute_unknown_action_type(self, mock_env_vars):
        """Test execute with unknown action type."""
        with patch(ANTHROPIC_PATCH):
            from src.agents.db_tester import DBTesterAgent

            agent = DBTesterAgent()
            agent._engine = MagicMock()
            agent._disconnect = AsyncMock()

            test_spec = {
                "id": "test-unknown-action",
                "name": "Unknown Action Test",
                "steps": [{"action": "unknown_action", "target": "users"}],
                "assertions": [],
            }

            result = await agent.execute(test_spec)

            # Unknown actions should be skipped
            assert result.data.status == "passed"
            assert len(result.data.queries) == 0
            assert len(result.data.validations) == 0


class TestValidateSqlIdentifier:
    """Tests for validate_sql_identifier function."""

    def test_valid_simple_identifier(self, mock_env_vars):
        """Test valid simple identifier."""
        from src.agents.db_tester import validate_sql_identifier

        result = validate_sql_identifier("users")
        assert result == '"users"'

    def test_valid_underscore_identifier(self, mock_env_vars):
        """Test valid identifier with underscore."""
        from src.agents.db_tester import validate_sql_identifier

        result = validate_sql_identifier("user_profiles")
        assert result == '"user_profiles"'

    def test_valid_schema_qualified(self, mock_env_vars):
        """Test valid schema-qualified identifier."""
        from src.agents.db_tester import validate_sql_identifier

        result = validate_sql_identifier("public.users")
        assert result == '"public"."users"'

    def test_empty_identifier(self, mock_env_vars):
        """Test empty identifier raises ValueError."""
        from src.agents.db_tester import validate_sql_identifier

        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier("")

        assert "cannot be empty" in str(exc_info.value)

    def test_invalid_special_characters(self, mock_env_vars):
        """Test identifier with special characters raises ValueError."""
        from src.agents.db_tester import validate_sql_identifier

        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier("users; DROP TABLE users;")

        assert "Invalid SQL identifier" in str(exc_info.value)

    def test_invalid_starting_with_number(self, mock_env_vars):
        """Test identifier starting with number raises ValueError."""
        from src.agents.db_tester import validate_sql_identifier

        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier("123users")

        assert "Invalid SQL identifier" in str(exc_info.value)

    def test_invalid_hyphen_in_identifier(self, mock_env_vars):
        """Test identifier with hyphen raises ValueError."""
        from src.agents.db_tester import validate_sql_identifier

        with pytest.raises(ValueError) as exc_info:
            validate_sql_identifier("user-profiles")

        assert "Invalid SQL identifier" in str(exc_info.value)

    def test_valid_identifier_with_numbers(self, mock_env_vars):
        """Test valid identifier containing numbers."""
        from src.agents.db_tester import validate_sql_identifier

        result = validate_sql_identifier("users2")
        assert result == '"users2"'

    def test_valid_identifier_starting_with_underscore(self, mock_env_vars):
        """Test valid identifier starting with underscore."""
        from src.agents.db_tester import validate_sql_identifier

        result = validate_sql_identifier("_temp_table")
        assert result == '"_temp_table"'
