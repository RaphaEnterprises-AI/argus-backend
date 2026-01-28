"""Database Tester Agent - Validates database state and data integrity.

This agent:
- Connects to databases via SQLAlchemy
- Validates data after operations
- Checks constraints and relationships
- Verifies data integrity
"""

import time
from dataclasses import dataclass, field
from typing import Any

from .base import AgentCapability, AgentResult, BaseAgent
from .prompts import get_enhanced_prompt
from .test_planner import TestSpec


@dataclass
class QueryResult:
    """Result from a database query."""

    query: str
    rows: list[dict]
    row_count: int
    execution_time_ms: int
    success: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "success": self.success,
            "error": self.error,
            "rows": self.rows[:10],  # Limit rows in output
        }


@dataclass
class DataValidationResult:
    """Result from data validation."""

    validation_type: str
    table: str
    passed: bool
    expected: Any
    actual: Any
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "validation_type": self.validation_type,
            "table": self.table,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "error": self.error,
        }


@dataclass
class DBTestResult:
    """Complete result from a database test."""

    test_id: str
    test_name: str
    status: str  # passed, failed, error
    queries: list[QueryResult] = field(default_factory=list)
    validations: list[DataValidationResult] = field(default_factory=list)
    total_duration_ms: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status,
            "queries": [q.to_dict() for q in self.queries],
            "validations": [v.to_dict() for v in self.validations],
            "total_duration_ms": self.total_duration_ms,
            "error_message": self.error_message,
        }


class DBTesterAgent(BaseAgent):
    """Agent that validates database state and data integrity.

    Features:
    - SQLAlchemy-based database access
    - Data existence validation
    - Constraint checking
    - Relationship verification
    - Before/after state comparison
    """

    # RAP-231: Agent capabilities for A2A discovery
    # Note: DB testing uses generic capabilities - may add DB-specific ones later
    CAPABILITIES = [
        AgentCapability.SCHEMA_VALIDATION,
    ]

    def __init__(self, database_url: str | None = None, **kwargs):
        """Initialize with optional database URL.

        Args:
            database_url: SQLAlchemy database URL
        """
        super().__init__(**kwargs)
        self._database_url = database_url
        self._engine = None
        self._session = None

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for database testing."""
        enhanced = get_enhanced_prompt("db_tester")
        if enhanced:
            return enhanced

        return """You are an expert database tester. Analyze database queries and data to validate integrity.

When asked to analyze database state:
1. Check if expected data exists
2. Validate relationships between tables
3. Verify constraints are met
4. Identify any data anomalies

Respond with JSON containing:
- valid: boolean
- issues: list of problems found
- recommendations: list of fixes"""

    async def execute(
        self,
        test_spec: TestSpec | dict,
        database_url: str | None = None,
    ) -> AgentResult[DBTestResult]:
        """Execute a database test specification.

        Args:
            test_spec: Test specification with DB steps
            database_url: Optional database URL override

        Returns:
            AgentResult containing DBTestResult
        """
        db_url = database_url or self._database_url

        if isinstance(test_spec, dict):
            test_id = test_spec.get("id", "unknown")
            test_name = test_spec.get("name", "Unknown DB Test")
            steps = test_spec.get("steps", [])
            assertions = test_spec.get("assertions", [])
        else:
            test_id = test_spec.id
            test_name = test_spec.name
            steps = [s.to_dict() if hasattr(s, "to_dict") else s for s in test_spec.steps]
            assertions = [a.to_dict() if hasattr(a, "to_dict") else a for a in test_spec.assertions]

        self.log.info(
            "Executing database test",
            test_id=test_id,
            test_name=test_name,
        )

        start_time = time.time()
        queries = []
        validations = []
        error_message = None
        status = "passed"

        try:
            # Initialize database connection
            if db_url:
                await self._connect(db_url)

            if not self._engine:
                return AgentResult(
                    success=False,
                    error="No database connection available",
                    data=DBTestResult(
                        test_id=test_id,
                        test_name=test_name,
                        status="error",
                        error_message="No database connection",
                    ),
                )

            # Execute steps
            for step in steps:
                action = step.get("action", "").lower()

                if action == "query":
                    query_result = await self._execute_query(step.get("target", ""))
                    queries.append(query_result)

                    if not query_result.success:
                        status = "failed"
                        error_message = query_result.error
                        break

                elif action == "validate_exists":
                    validation = await self._validate_exists(
                        table=step.get("target"),
                        conditions=step.get("value", {}),
                    )
                    validations.append(validation)

                    if not validation.passed:
                        status = "failed"
                        error_message = f"Validation failed: {validation.error}"

                elif action == "validate_count":
                    validation = await self._validate_count(
                        table=step.get("target"),
                        expected_count=int(step.get("value", 0)),
                        conditions=step.get("conditions", {}),
                    )
                    validations.append(validation)

                    if not validation.passed:
                        status = "failed"
                        error_message = f"Count validation failed: {validation.error}"

                elif action == "validate_relationship":
                    validation = await self._validate_relationship(
                        source_table=step.get("target"),
                        relationship=step.get("value", {}),
                    )
                    validations.append(validation)

                    if not validation.passed:
                        status = "failed"

            # Check assertions
            for assertion in assertions:
                validation = await self._check_db_assertion(assertion)
                validations.append(validation)
                if not validation.passed:
                    status = "failed"
                    error_message = f"Assertion failed: {assertion}"

        except Exception as e:
            status = "error"
            error_message = str(e)
            self.log.error("Database test error", error=str(e))

        finally:
            await self._disconnect()

        total_duration = int((time.time() - start_time) * 1000)

        result = DBTestResult(
            test_id=test_id,
            test_name=test_name,
            status=status,
            queries=queries,
            validations=validations,
            total_duration_ms=total_duration,
            error_message=error_message,
        )

        self.log.info(
            "Database test complete",
            test_id=test_id,
            status=status,
        )

        return AgentResult(
            success=status == "passed",
            data=result,
        )

    async def _connect(self, database_url: str) -> None:
        """Connect to the database."""
        try:
            from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
            from sqlalchemy.orm import sessionmaker

            # Convert to async URL if needed
            if database_url.startswith("postgresql://"):
                database_url = database_url.replace(
                    "postgresql://", "postgresql+asyncpg://", 1
                )

            self._engine = create_async_engine(database_url)
            async_session = sessionmaker(
                self._engine, class_=AsyncSession, expire_on_commit=False
            )
            self._session = async_session()

            self.log.info("Connected to database")

        except ImportError:
            self.log.warning("SQLAlchemy async not available, using sync mode")
            from sqlalchemy import create_engine

            self._engine = create_engine(database_url)

        except Exception as e:
            self.log.error("Database connection failed", error=str(e))
            raise

    async def _disconnect(self) -> None:
        """Disconnect from the database."""
        if self._session:
            await self._session.close()
            self._session = None

        if self._engine:
            await self._engine.dispose()
            self._engine = None

    async def _execute_query(self, query: str) -> QueryResult:
        """Execute a raw SQL query."""
        start_time = time.time()

        try:
            from sqlalchemy import text

            if self._session:
                result = await self._session.execute(text(query))
                rows = [dict(row._mapping) for row in result.fetchall()]
            else:
                with self._engine.connect() as conn:
                    result = conn.execute(text(query))
                    rows = [dict(row._mapping) for row in result.fetchall()]

            return QueryResult(
                query=query,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=int((time.time() - start_time) * 1000),
                success=True,
            )

        except Exception as e:
            return QueryResult(
                query=query,
                rows=[],
                row_count=0,
                execution_time_ms=int((time.time() - start_time) * 1000),
                success=False,
                error=str(e),
            )

    async def _validate_exists(
        self, table: str, conditions: dict
    ) -> DataValidationResult:
        """Validate that a record exists."""
        where_clauses = " AND ".join(
            f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
            for k, v in conditions.items()
        )
        query = f"SELECT COUNT(*) as cnt FROM {table}"
        if where_clauses:
            query += f" WHERE {where_clauses}"

        result = await self._execute_query(query)

        if not result.success:
            return DataValidationResult(
                validation_type="exists",
                table=table,
                passed=False,
                expected="record exists",
                actual="query failed",
                error=result.error,
            )

        count = result.rows[0]["cnt"] if result.rows else 0
        passed = count > 0

        return DataValidationResult(
            validation_type="exists",
            table=table,
            passed=passed,
            expected="record exists",
            actual=f"{count} record(s) found",
            error=None if passed else "No matching records found",
        )

    async def _validate_count(
        self, table: str, expected_count: int, conditions: dict
    ) -> DataValidationResult:
        """Validate record count."""
        where_clauses = " AND ".join(
            f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}"
            for k, v in conditions.items()
        )
        query = f"SELECT COUNT(*) as cnt FROM {table}"
        if where_clauses:
            query += f" WHERE {where_clauses}"

        result = await self._execute_query(query)

        if not result.success:
            return DataValidationResult(
                validation_type="count",
                table=table,
                passed=False,
                expected=expected_count,
                actual="query failed",
                error=result.error,
            )

        actual_count = result.rows[0]["cnt"] if result.rows else 0
        passed = actual_count == expected_count

        return DataValidationResult(
            validation_type="count",
            table=table,
            passed=passed,
            expected=expected_count,
            actual=actual_count,
            error=None if passed else f"Expected {expected_count}, got {actual_count}",
        )

    async def _validate_relationship(
        self, source_table: str, relationship: dict
    ) -> DataValidationResult:
        """Validate a foreign key relationship."""
        target_table = relationship.get("target_table")
        source_column = relationship.get("source_column")
        target_column = relationship.get("target_column")

        # Check for orphaned records
        query = f"""
            SELECT COUNT(*) as orphan_count
            FROM {source_table} s
            LEFT JOIN {target_table} t ON s.{source_column} = t.{target_column}
            WHERE s.{source_column} IS NOT NULL AND t.{target_column} IS NULL
        """

        result = await self._execute_query(query)

        if not result.success:
            return DataValidationResult(
                validation_type="relationship",
                table=source_table,
                passed=False,
                expected="no orphaned records",
                actual="query failed",
                error=result.error,
            )

        orphan_count = result.rows[0]["orphan_count"] if result.rows else 0
        passed = orphan_count == 0

        return DataValidationResult(
            validation_type="relationship",
            table=source_table,
            passed=passed,
            expected="no orphaned records",
            actual=f"{orphan_count} orphaned record(s)",
            error=None if passed else f"Found {orphan_count} orphaned records",
        )

    async def _check_db_assertion(self, assertion: dict) -> DataValidationResult:
        """Check a database assertion."""
        assertion_type = assertion.get("type", "").lower()
        target = assertion.get("target")
        expected = assertion.get("expected")

        if assertion_type == "row_exists":
            return await self._validate_exists(
                table=target,
                conditions=expected if isinstance(expected, dict) else {},
            )

        elif assertion_type == "row_count":
            return await self._validate_count(
                table=target,
                expected_count=int(expected),
                conditions={},
            )

        elif assertion_type == "query_returns":
            result = await self._execute_query(target)
            passed = result.row_count > 0 if expected == "rows" else result.row_count == 0

            return DataValidationResult(
                validation_type="query_returns",
                table="custom_query",
                passed=passed,
                expected=expected,
                actual=f"{result.row_count} rows",
            )

        return DataValidationResult(
            validation_type=assertion_type,
            table=target or "unknown",
            passed=False,
            expected=expected,
            actual=None,
            error=f"Unknown assertion type: {assertion_type}",
        )

    async def get_table_schema(self, table_name: str) -> dict:
        """Get schema information for a table."""
        query = f"""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position
        """

        result = await self._execute_query(query)

        if result.success:
            return {
                "table": table_name,
                "columns": result.rows,
            }
        return {"table": table_name, "error": result.error}

    async def get_table_stats(self, table_name: str) -> dict:
        """Get statistics for a table."""
        count_result = await self._execute_query(
            f"SELECT COUNT(*) as total FROM {table_name}"
        )

        return {
            "table": table_name,
            "row_count": count_result.rows[0]["total"] if count_result.success else 0,
        }
