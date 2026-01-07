"""Database Analyzer - SQL queries, migrations, and schema analysis.

Understands:
- SQL queries (SELECT, INSERT, UPDATE, DELETE)
- ORM patterns (SQLAlchemy, Prisma, TypeORM)
- Migration files
- Index usage
- SQL injection risks
"""

import re
import logging
from pathlib import Path
from typing import Optional

from .base import (
    BaseAnalyzer,
    ComponentInfo,
    ComponentType,
    QueryInfo,
    Issue,
    Severity,
)
from src.indexer import ParsedFile

logger = logging.getLogger(__name__)


class DatabaseAnalyzer(BaseAnalyzer):
    """Analyzer for database queries, migrations, and schema."""

    @property
    def analyzer_type(self) -> str:
        return "database"

    def get_file_patterns(self) -> list[str]:
        return [
            # SQL files
            "**/*.sql",
            # Migrations
            "**/migrations/**/*.py",
            "**/migrations/**/*.sql",
            "**/migrate/**/*.py",
            # ORM models
            "**/models/**/*.py",
            "**/models/**/*.ts",
            "**/entities/**/*.ts",
            # Prisma
            "**/prisma/**/*.prisma",
            # Repositories
            "**/repositories/**/*.py",
            "**/repositories/**/*.ts",
        ]

    def analyze_file(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze a database-related file."""
        components: list[ComponentInfo] = []

        file_path = parsed.file_path.lower()

        if file_path.endswith(".sql"):
            components.extend(self._analyze_sql_file(parsed))
        elif file_path.endswith(".prisma"):
            components.extend(self._analyze_prisma(parsed))
        elif "migration" in file_path:
            components.extend(self._analyze_migration(parsed))
        elif "model" in file_path or "entities" in file_path:
            components.extend(self._analyze_orm_models(parsed))
        else:
            # Look for SQL queries in code
            components.extend(self._analyze_embedded_sql(parsed))

        return components

    def _analyze_sql_file(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze a raw SQL file."""
        components: list[ComponentInfo] = []
        content = parsed.content
        lines = content.split("\n")

        # Parse SQL statements
        statements = self._split_sql_statements(content)

        for i, statement in enumerate(statements):
            statement = statement.strip()
            if not statement:
                continue

            query_type = self._get_query_type(statement)
            tables = self._extract_tables(statement)

            # Calculate approximate line number
            line_number = 1
            pos = 0
            for j, stmt in enumerate(statements[:i]):
                pos += len(stmt) + 1
            line_number = content[:pos].count("\n") + 1

            query = QueryInfo(
                query_type=query_type,
                tables=tables,
                file_path=parsed.file_path,
                line_number=line_number,
                raw_query=statement[:500],  # Truncate for storage
                columns=self._extract_columns(statement),
                joins=self._extract_joins(statement),
                uses_parameterization=True,  # SQL files typically safe
            )

            # Check for issues
            query.issues.extend(self._check_sql_issues(statement, query))

            # Convert to component
            component = ComponentInfo(
                name=f"{query_type} on {', '.join(tables[:3])}",
                component_type=ComponentType.QUERY,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + statement.count("\n"),
                issues=query.issues,
            )
            components.append(component)

        return components

    def _split_sql_statements(self, content: str) -> list[str]:
        """Split SQL content into individual statements."""
        # Simple split by semicolon (doesn't handle all edge cases)
        statements = []
        current = []

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("--"):
                continue
            current.append(line)
            if line.endswith(";"):
                statements.append(" ".join(current))
                current = []

        if current:
            statements.append(" ".join(current))

        return statements

    def _get_query_type(self, statement: str) -> str:
        """Determine the type of SQL query."""
        statement_upper = statement.upper().strip()

        if statement_upper.startswith("SELECT"):
            return "SELECT"
        elif statement_upper.startswith("INSERT"):
            return "INSERT"
        elif statement_upper.startswith("UPDATE"):
            return "UPDATE"
        elif statement_upper.startswith("DELETE"):
            return "DELETE"
        elif statement_upper.startswith("CREATE TABLE"):
            return "CREATE_TABLE"
        elif statement_upper.startswith("CREATE INDEX"):
            return "CREATE_INDEX"
        elif statement_upper.startswith("ALTER TABLE"):
            return "ALTER_TABLE"
        elif statement_upper.startswith("DROP"):
            return "DROP"
        elif statement_upper.startswith("CREATE"):
            return "CREATE"

        return "UNKNOWN"

    def _extract_tables(self, statement: str) -> list[str]:
        """Extract table names from SQL statement."""
        tables = []

        # FROM clause
        from_match = re.search(r'\bFROM\s+([^\s,;()]+)', statement, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1).strip('`"[]'))

        # JOIN clauses
        for join_match in re.finditer(r'\bJOIN\s+([^\s,;()]+)', statement, re.IGNORECASE):
            tables.append(join_match.group(1).strip('`"[]'))

        # INTO clause (INSERT)
        into_match = re.search(r'\bINTO\s+([^\s,;()]+)', statement, re.IGNORECASE)
        if into_match:
            tables.append(into_match.group(1).strip('`"[]'))

        # UPDATE clause
        update_match = re.search(r'\bUPDATE\s+([^\s,;()]+)', statement, re.IGNORECASE)
        if update_match:
            tables.append(update_match.group(1).strip('`"[]'))

        # CREATE TABLE
        create_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)', statement, re.IGNORECASE)
        if create_match:
            tables.append(create_match.group(1).strip('`"[]'))

        return list(set(tables))

    def _extract_columns(self, statement: str) -> list[str]:
        """Extract column names from SELECT statement."""
        columns = []

        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', statement, re.IGNORECASE | re.DOTALL)
        if select_match:
            cols_str = select_match.group(1)
            if cols_str.strip() != "*":
                for col in cols_str.split(","):
                    col = col.strip()
                    # Handle aliases
                    if " AS " in col.upper():
                        col = col.split(" AS ")[0].strip()
                    # Remove table prefix
                    if "." in col:
                        col = col.split(".")[-1]
                    columns.append(col.strip('`"[]'))

        return columns

    def _extract_joins(self, statement: str) -> list[str]:
        """Extract JOIN information from statement."""
        joins = []

        join_pattern = r'((?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL)?\s*JOIN)\s+([^\s]+)'
        for match in re.finditer(join_pattern, statement, re.IGNORECASE):
            join_type = match.group(1).strip().upper()
            table = match.group(2).strip('`"[]')
            joins.append(f"{join_type} {table}")

        return joins

    def _check_sql_issues(self, statement: str, query: QueryInfo) -> list[Issue]:
        """Check for SQL quality and security issues."""
        issues = []

        # SELECT *
        if re.search(r'SELECT\s+\*\s+FROM', statement, re.IGNORECASE):
            issues.append(Issue(
                severity=Severity.WARNING,
                message="SELECT * can be inefficient - consider specifying columns",
                file_path=query.file_path,
                line_number=query.line_number,
                code="SQL_SELECT_STAR",
            ))

        # Missing WHERE on UPDATE/DELETE
        if query.query_type in ["UPDATE", "DELETE"]:
            if not re.search(r'\bWHERE\b', statement, re.IGNORECASE):
                issues.append(Issue(
                    severity=Severity.CRITICAL,
                    message=f"{query.query_type} without WHERE clause - dangerous!",
                    file_path=query.file_path,
                    line_number=query.line_number,
                    code="SQL_NO_WHERE_CLAUSE",
                ))

        # N+1 query pattern hint
        if query.query_type == "SELECT" and len(query.joins) == 0:
            # Simple heuristic - might indicate N+1
            if any(col in statement.lower() for col in ["user_id", "parent_id", "foreign"]):
                issues.append(Issue(
                    severity=Severity.INFO,
                    message="Consider using JOIN instead of separate queries (N+1 risk)",
                    file_path=query.file_path,
                    line_number=query.line_number,
                    code="SQL_POTENTIAL_N_PLUS_ONE",
                ))

        # LIKE without index consideration
        if re.search(r"LIKE\s+['\"]%", statement, re.IGNORECASE):
            issues.append(Issue(
                severity=Severity.WARNING,
                message="LIKE with leading wildcard cannot use index",
                file_path=query.file_path,
                line_number=query.line_number,
                code="SQL_LIKE_LEADING_WILDCARD",
            ))

        return issues

    def _analyze_prisma(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Prisma schema file."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find models
        model_pattern = r'model\s+(\w+)\s*\{([^}]+)\}'

        for match in re.finditer(model_pattern, content):
            name = match.group(1)
            body = match.group(2)
            line_number = content[:match.start()].count("\n") + 1

            # Extract fields
            fields = []
            for line in body.split("\n"):
                line = line.strip()
                if line and not line.startswith("//") and not line.startswith("@@"):
                    field_match = re.match(r'(\w+)\s+(\w+)', line)
                    if field_match:
                        fields.append(field_match.group(1))

            component = ComponentInfo(
                name=name,
                component_type=ComponentType.MODEL,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + body.count("\n"),
                props=fields,
            )

            # Check for issues
            if "@@index" not in body and len(fields) > 3:
                component.issues.append(Issue(
                    severity=Severity.INFO,
                    message=f"Model {name} may benefit from indexes",
                    file_path=parsed.file_path,
                    line_number=line_number,
                    code="PRISMA_NO_INDEX",
                ))

            components.append(component)

        return components

    def _analyze_migration(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze migration files."""
        components: list[ComponentInfo] = []
        content = parsed.content
        file_name = Path(parsed.file_path).name

        # Extract migration name/version from filename
        version_match = re.search(r'(\d+)', file_name)
        version = version_match.group(1) if version_match else "unknown"

        component = ComponentInfo(
            name=f"Migration {version}",
            component_type=ComponentType.MIGRATION,
            file_path=parsed.file_path,
            start_line=1,
            end_line=content.count("\n") + 1,
        )

        # Check for risky operations
        risky_patterns = [
            (r'DROP\s+TABLE', "DROP TABLE - destructive operation"),
            (r'DROP\s+COLUMN', "DROP COLUMN - potential data loss"),
            (r'TRUNCATE', "TRUNCATE - removes all data"),
            (r'DELETE\s+FROM.*(?!WHERE)', "DELETE without WHERE"),
        ]

        for pattern, message in risky_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                component.issues.append(Issue(
                    severity=Severity.WARNING,
                    message=message,
                    file_path=parsed.file_path,
                    code="MIGRATION_RISKY_OPERATION",
                ))

        # Check for lack of rollback
        if "def down" not in content and "async down" not in content and "DROP" not in content:
            if "def up" in content or "async up" in content:
                component.issues.append(Issue(
                    severity=Severity.INFO,
                    message="Migration may lack rollback (down) method",
                    file_path=parsed.file_path,
                    code="MIGRATION_NO_ROLLBACK",
                ))

        components.append(component)
        return components

    def _analyze_orm_models(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze ORM model definitions (SQLAlchemy, TypeORM)."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # SQLAlchemy models
        if "sqlalchemy" in content.lower() or "declarative_base" in content:
            components.extend(self._analyze_sqlalchemy(parsed))

        # TypeORM entities
        if "@Entity" in content:
            components.extend(self._analyze_typeorm(parsed))

        return components

    def _analyze_sqlalchemy(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze SQLAlchemy models."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find model classes
        for cls in parsed.get_classes():
            name = cls.name
            if not name:
                continue

            # Check if it's a model (inherits from Base or has __tablename__)
            if "__tablename__" in cls.text or "Column(" in cls.text:
                # Extract columns
                columns = []
                for match in re.finditer(r'(\w+)\s*=\s*Column\(', cls.text):
                    columns.append(match.group(1))

                component = ComponentInfo(
                    name=name,
                    component_type=ComponentType.MODEL,
                    file_path=parsed.file_path,
                    start_line=cls.start_line,
                    end_line=cls.end_line,
                    props=columns,
                )

                # Check for issues
                if "relationship" not in cls.text and len(columns) > 0:
                    # Check if there are foreign keys without relationships
                    if "ForeignKey" in cls.text:
                        component.issues.append(Issue(
                            severity=Severity.INFO,
                            message=f"Model {name} has ForeignKey but no relationship defined",
                            file_path=parsed.file_path,
                            line_number=cls.start_line,
                            code="SQLALCHEMY_NO_RELATIONSHIP",
                        ))

                components.append(component)

        return components

    def _analyze_typeorm(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze TypeORM entities."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find @Entity classes
        entity_pattern = r'@Entity\([^)]*\)\s*(?:export\s+)?class\s+(\w+)'

        for match in re.finditer(entity_pattern, content):
            name = match.group(1)
            line_number = content[:match.start()].count("\n") + 1

            # Find class body
            class_start = match.end()
            brace_count = 0
            class_end = class_start

            for i, char in enumerate(content[class_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        class_end = class_start + i
                        break

            class_body = content[class_start:class_end]

            # Extract columns
            columns = []
            for col_match in re.finditer(r'@Column\([^)]*\)\s*(\w+)', class_body):
                columns.append(col_match.group(1))

            component = ComponentInfo(
                name=name,
                component_type=ComponentType.MODEL,
                file_path=parsed.file_path,
                start_line=line_number,
                end_line=line_number + class_body.count("\n"),
                props=columns,
            )

            components.append(component)

        return components

    def _analyze_embedded_sql(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Find SQL queries embedded in code."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Pattern for SQL strings
        sql_patterns = [
            r'(?:execute|query|run)\s*\(\s*["\']([^"\']+SELECT[^"\']+)["\']',
            r'(?:execute|query|run)\s*\(\s*["\']([^"\']+INSERT[^"\']+)["\']',
            r'(?:execute|query|run)\s*\(\s*["\']([^"\']+UPDATE[^"\']+)["\']',
            r'(?:execute|query|run)\s*\(\s*["\']([^"\']+DELETE[^"\']+)["\']',
            r'"""([^"]*(?:SELECT|INSERT|UPDATE|DELETE)[^"]*)"""',
            r"'''([^']*(?:SELECT|INSERT|UPDATE|DELETE)[^']*)'''",
        ]

        for pattern in sql_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE | re.DOTALL):
                sql = match.group(1)
                line_number = content[:match.start()].count("\n") + 1

                query_type = self._get_query_type(sql)
                tables = self._extract_tables(sql)

                # Check for SQL injection
                sql_injection_risk = self._check_sql_injection(sql, content[match.start():match.end() + 100])

                query = QueryInfo(
                    query_type=query_type,
                    tables=tables,
                    file_path=parsed.file_path,
                    line_number=line_number,
                    raw_query=sql[:200],
                    uses_parameterization=not sql_injection_risk,
                    sql_injection_risk=sql_injection_risk,
                )

                if sql_injection_risk:
                    query.issues.append(Issue(
                        severity=Severity.CRITICAL,
                        message="Potential SQL injection vulnerability",
                        file_path=parsed.file_path,
                        line_number=line_number,
                        suggestion="Use parameterized queries instead of string concatenation",
                        code="SQL_INJECTION_RISK",
                    ))

                component = ComponentInfo(
                    name=f"Embedded {query_type}",
                    component_type=ComponentType.QUERY,
                    file_path=parsed.file_path,
                    start_line=line_number,
                    end_line=line_number + sql.count("\n"),
                    issues=query.issues,
                )
                components.append(component)

        return components

    def _check_sql_injection(self, sql: str, context: str) -> bool:
        """Check if SQL query might be vulnerable to injection."""
        # String concatenation patterns
        injection_patterns = [
            r'\+\s*\w+\s*\+',  # "SELECT " + var + " FROM"
            r'\$\{[^}]+\}',    # `SELECT ${var}`
            r'%s',             # Format string (OK if parameterized)
            r'\.format\(',     # .format() on SQL string
            r'f"[^"]*\{',      # f-string with SQL
            r"f'[^']*\{",      # f-string with SQL
        ]

        # Check for parameterized queries (safe patterns)
        safe_patterns = [
            r'\?\s*,',         # Placeholder ?
            r':\w+',           # Named parameter :name
            r'%\(\w+\)s',      # %(name)s style
        ]

        # If it has safe patterns, probably OK
        for pattern in safe_patterns:
            if re.search(pattern, context):
                return False

        # Check for dangerous patterns
        for pattern in injection_patterns:
            if re.search(pattern, context):
                return True

        return False

    def analyze(self):
        """Override to also extract queries."""
        result = super().analyze()

        # Extract QueryInfo from components
        queries = []
        for component in result.components:
            if component.component_type == ComponentType.QUERY:
                query_type = component.name.replace("Embedded ", "").split(" on ")[0]
                queries.append(QueryInfo(
                    query_type=query_type,
                    tables=[],
                    file_path=component.file_path,
                    line_number=component.start_line,
                    issues=component.issues,
                ))

        result.queries = queries
        return result
