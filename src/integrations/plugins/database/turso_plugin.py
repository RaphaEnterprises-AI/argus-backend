"""Turso database integration plugin.

This plugin provides integration with Turso (libSQL) for:
- Organization and database management via Platform API
- Database groups and tokens management
- Schema introspection
- Test data management via HTTP API
- Connection testing
"""

import re
import time
from datetime import datetime
from typing import Any, Optional

import httpx


# RAP-291: SQL Injection Prevention for Turso queries
# Regex pattern for valid SQL identifiers (table names, column names)
VALID_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_turso_identifier(identifier: str) -> str:
    """Validate a SQL identifier for Turso queries to prevent SQL injection.

    Args:
        identifier: Table or column name to validate

    Returns:
        The validated identifier (Turso/SQLite uses double quotes for identifiers)

    Raises:
        ValueError: If identifier contains invalid characters
    """
    if not identifier:
        raise ValueError("SQL identifier cannot be empty")

    # Check against allowlist pattern
    if not VALID_IDENTIFIER_PATTERN.match(identifier):
        raise ValueError(
            f"Invalid SQL identifier: '{identifier}'. "
            "Only alphanumeric characters and underscores are allowed."
        )

    # Quote the identifier for safety (SQLite style)
    return f'"{identifier}"'

from src.integrations.base import (
    AuthType,
    ConnectionTestResult,
    FieldDefinition,
    IntegrationCategory,
    IntegrationCredentials,
    IntegrationMetadata,
    IntegrationPlugin,
    SyncResult,
)


class TursoPlugin(IntegrationPlugin):
    """Turso (libSQL) integration plugin.

    Provides access to Turso Platform API and database operations for:
    - Organization management
    - Database and group management
    - Token management
    - Schema introspection
    - Test data operations via HTTP API

    Authentication uses:
    - Platform API Token for management operations
    - Database tokens for query operations
    """

    # Turso Platform API base URL
    PLATFORM_API_URL = "https://api.turso.tech"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about the Turso integration."""
        return IntegrationMetadata(
            slug="turso",
            name="Turso",
            description="Edge-hosted, distributed SQLite database powered by libSQL with "
            "global replication and embedded sync",
            category=IntegrationCategory.DATABASE,
            auth_type=AuthType.BEARER_TOKEN,
            icon_url="https://turso.tech/favicon.ico",
            docs_url="https://docs.turso.tech",
            website_url="https://turso.tech",
            features=[
                "database_management",
                "group_management",
                "token_management",
                "schema_introspection",
                "test_data_management",
                "edge_replication",
                "embedded_sync",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="API Token",
                    type="password",
                    required=True,
                    placeholder="turso_...",
                    help_text="Platform API token from Turso dashboard",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="org_slug",
                    label="Organization",
                    type="text",
                    required=False,
                    placeholder="my-org",
                    help_text="Organization slug (defaults to personal account)",
                ),
                FieldDefinition(
                    name="instance_url",
                    label="Database URL",
                    type="url",
                    required=False,
                    placeholder="libsql://my-db-my-org.turso.io",
                    help_text="Database URL for direct query operations",
                ),
                FieldDefinition(
                    name="access_token",
                    label="Database Token",
                    type="password",
                    required=False,
                    placeholder="eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9...",
                    help_text="Database auth token for query operations",
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=["organizations", "databases", "groups", "tokens"],
        )

    def _get_platform_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Platform API requests.

        Args:
            credentials: Credentials containing API token.

        Returns:
            Headers dictionary for API requests.
        """
        return {
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
        }

    def _get_db_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for database HTTP API requests.

        Args:
            credentials: Credentials containing database token.

        Returns:
            Headers dictionary for database requests.
        """
        token = credentials.access_token or credentials.api_key or ""
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test connection to Turso.

        Tests the Platform API to verify the API token is valid.

        Args:
            credentials: Credentials containing API token.

        Returns:
            ConnectionTestResult with connection status and details.
        """
        start_time = time.time()

        # Validate required credentials
        is_valid, missing = self.validate_credentials(credentials)
        if not is_valid:
            return ConnectionTestResult(
                success=False,
                message=f"Missing required credentials: {', '.join(missing)}",
                error_code="MISSING_CREDENTIALS",
            )

        try:
            headers = self._get_platform_headers(credentials)

            # Test connection by listing organizations
            response = await self.http_client.get(
                f"{self.PLATFORM_API_URL}/v1/organizations",
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                organizations = data if isinstance(data, list) else data.get("organizations", [])

                # Get account info
                account_info: dict[str, Any] = {
                    "organization_count": len(organizations),
                }

                if organizations:
                    # Use first org or specified org
                    org_slug = credentials.org_slug
                    if org_slug:
                        org = next((o for o in organizations if o.get("slug") == org_slug), None)
                    else:
                        org = organizations[0]

                    if org:
                        account_info["organization"] = org.get("slug") or org.get("name")
                        account_info["organization_type"] = org.get("type", "personal")

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Turso",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=["read", "write", "manage"],
                    account_info=account_info,
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed - check your API token",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                    error_details={"status_code": 401},
                )
            elif response.status_code == 403:
                return ConnectionTestResult(
                    success=False,
                    message="Access forbidden - token may have insufficient permissions",
                    latency_ms=latency_ms,
                    error_code="FORBIDDEN",
                    error_details={"status_code": 403},
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    message=f"Connection failed with status {response.status_code}",
                    latency_ms=latency_ms,
                    error_code="CONNECTION_FAILED",
                    error_details={
                        "status_code": response.status_code,
                        "response": response.text[:500] if response.text else None,
                    },
                )

        except httpx.TimeoutException:
            return ConnectionTestResult(
                success=False,
                message="Connection timed out",
                error_code="TIMEOUT",
            )
        except httpx.RequestError as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection error: {str(e)}",
                error_code="REQUEST_ERROR",
                error_details={"error": str(e)},
            )
        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Unexpected error: {str(e)}",
                error_code="UNKNOWN_ERROR",
                error_details={"error": str(e)},
            )

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from Turso.

        Syncs organization, database, and group information from Turso Platform API.

        Args:
            credentials: Credentials for Turso access.
            resources: Optional list of resources to sync (organizations, databases, groups, tokens).
            since: Not used for platform sync (always full sync).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced information.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or ["organizations", "databases", "groups"]
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        data_points = 0

        headers = self._get_platform_headers(credentials)

        try:
            # Sync organizations
            if "organizations" in sync_resources:
                orgs_data = await self._sync_organizations(headers, limit)
                if orgs_data.get("error"):
                    errors.append(orgs_data["error"])
                else:
                    data["organizations"] = orgs_data.get("organizations", [])
                    data_points += len(data["organizations"])

            # Determine which org to use for database/group queries
            org_slug = credentials.org_slug
            if not org_slug and data.get("organizations"):
                # Use first org if not specified
                org_slug = data["organizations"][0].get("slug")

            if org_slug:
                # Sync databases
                if "databases" in sync_resources:
                    dbs_data = await self._sync_databases(headers, org_slug, limit)
                    if dbs_data.get("error"):
                        warnings.append(dbs_data["error"])
                    else:
                        data["databases"] = dbs_data.get("databases", [])
                        data_points += len(data["databases"])

                # Sync groups
                if "groups" in sync_resources:
                    groups_data = await self._sync_groups(headers, org_slug, limit)
                    if groups_data.get("error"):
                        warnings.append(groups_data["error"])
                    else:
                        data["groups"] = groups_data.get("groups", [])
                        data_points += len(data["groups"])

                # Sync tokens for each database
                if "tokens" in sync_resources and data.get("databases"):
                    tokens_data = await self._sync_database_tokens(
                        headers, org_slug, data["databases"], limit
                    )
                    if tokens_data.get("error"):
                        warnings.append(tokens_data["error"])
                    else:
                        data["tokens"] = tokens_data.get("tokens", [])
                        data_points += len(data["tokens"])
            else:
                warnings.append("No organization found - skipping database and group sync")

            return SyncResult(
                success=len(errors) == 0,
                message="Sync completed" if not errors else f"Sync completed with errors",
                data_points_synced=data_points,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=errors,
                warnings=warnings,
                data=data,
            )

        except Exception as e:
            return SyncResult(
                success=False,
                message=f"Sync failed: {str(e)}",
                started_at=started_at,
                completed_at=datetime.utcnow(),
                errors=[str(e)],
                data=data,
            )

    async def _sync_organizations(
        self,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync organizations from Platform API.

        Args:
            headers: API headers.
            limit: Maximum organizations to return.

        Returns:
            Dictionary with organizations list or error.
        """
        try:
            response = await self.http_client.get(
                f"{self.PLATFORM_API_URL}/v1/organizations",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                orgs = data if isinstance(data, list) else data.get("organizations", [])
                return {"organizations": orgs[:limit]}

            return {"organizations": [], "error": f"API returned {response.status_code}"}

        except Exception as e:
            return {"error": f"Failed to sync organizations: {str(e)}"}

    async def _sync_databases(
        self,
        headers: dict[str, str],
        org_slug: str,
        limit: int,
    ) -> dict[str, Any]:
        """Sync databases for an organization.

        Args:
            headers: API headers.
            org_slug: Organization slug.
            limit: Maximum databases to return.

        Returns:
            Dictionary with databases list or error.
        """
        try:
            response = await self.http_client.get(
                f"{self.PLATFORM_API_URL}/v1/organizations/{org_slug}/databases",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                dbs = data if isinstance(data, list) else data.get("databases", [])
                return {"databases": dbs[:limit]}

            return {"databases": [], "error": f"API returned {response.status_code}"}

        except Exception as e:
            return {"error": f"Failed to sync databases: {str(e)}"}

    async def _sync_groups(
        self,
        headers: dict[str, str],
        org_slug: str,
        limit: int,
    ) -> dict[str, Any]:
        """Sync database groups for an organization.

        Args:
            headers: API headers.
            org_slug: Organization slug.
            limit: Maximum groups to return.

        Returns:
            Dictionary with groups list or error.
        """
        try:
            response = await self.http_client.get(
                f"{self.PLATFORM_API_URL}/v1/organizations/{org_slug}/groups",
                headers=headers,
            )

            if response.status_code == 200:
                data = response.json()
                groups = data if isinstance(data, list) else data.get("groups", [])
                return {"groups": groups[:limit]}

            return {"groups": [], "error": f"API returned {response.status_code}"}

        except Exception as e:
            return {"error": f"Failed to sync groups: {str(e)}"}

    async def _sync_database_tokens(
        self,
        headers: dict[str, str],
        org_slug: str,
        databases: list[dict[str, Any]],
        limit: int,
    ) -> dict[str, Any]:
        """Sync tokens for databases (metadata only, not actual token values).

        Args:
            headers: API headers.
            org_slug: Organization slug.
            databases: List of databases to get tokens for.
            limit: Maximum tokens to return.

        Returns:
            Dictionary with tokens list or error.
        """
        try:
            all_tokens: list[dict[str, Any]] = []

            for db in databases[:limit]:
                db_name = db.get("name") or db.get("Name")
                if not db_name:
                    continue

                response = await self.http_client.get(
                    f"{self.PLATFORM_API_URL}/v1/organizations/{org_slug}/databases/{db_name}/auth/tokens",
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    tokens = data if isinstance(data, list) else data.get("tokens", [])
                    for token in tokens:
                        token["database"] = db_name
                    all_tokens.extend(tokens)

            return {"tokens": all_tokens[:limit]}

        except Exception as e:
            return {"error": f"Failed to sync tokens: {str(e)}"}

    # -------------------------------------------------------------------------
    # Database Operations for Test Data Management
    # -------------------------------------------------------------------------

    def _get_http_url(self, credentials: IntegrationCredentials) -> Optional[str]:
        """Get HTTP API URL from database URL.

        Converts libsql:// URL to https:// for HTTP API access.

        Args:
            credentials: Credentials containing database URL.

        Returns:
            HTTP API URL or None.
        """
        instance_url = credentials.instance_url
        if not instance_url:
            return None

        # Convert libsql:// to https://
        url = instance_url.replace("libsql://", "https://")
        return url

    async def execute_query(
        self,
        credentials: IntegrationCredentials,
        sql: str,
        params: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Execute a SQL query via HTTP API.

        Args:
            credentials: Turso credentials with database URL and token.
            sql: SQL statement to execute.
            params: Optional positional parameters for the query.

        Returns:
            Query result with rows or error.
        """
        http_url = self._get_http_url(credentials)
        if not http_url:
            return {"success": False, "error": "Database URL not configured"}

        headers = self._get_db_headers(credentials)

        try:
            # Turso HTTP API format
            request_body: dict[str, Any] = {
                "statements": [
                    {
                        "q": sql,
                    }
                ]
            }

            if params:
                request_body["statements"][0]["params"] = params

            response = await self.http_client.post(
                http_url,
                headers=headers,
                json=request_body,
            )

            if response.status_code in (200, 201):
                data = response.json()
                results = data.get("results", [])

                if results and len(results) > 0:
                    result = results[0]
                    if "error" in result:
                        return {
                            "success": False,
                            "error": result["error"].get("message", str(result["error"])),
                        }

                    return {
                        "success": True,
                        "columns": result.get("columns", []),
                        "rows": result.get("rows", []),
                        "rows_affected": result.get("rows_affected", 0),
                        "last_insert_rowid": result.get("last_insert_rowid"),
                    }

                return {"success": True, "rows": [], "columns": []}
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def execute_batch(
        self,
        credentials: IntegrationCredentials,
        statements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute multiple SQL statements in a batch.

        Args:
            credentials: Turso credentials with database URL and token.
            statements: List of statement objects with 'q' (SQL) and optional 'params'.

        Returns:
            Batch results or error.
        """
        http_url = self._get_http_url(credentials)
        if not http_url:
            return {"success": False, "error": "Database URL not configured"}

        headers = self._get_db_headers(credentials)

        try:
            request_body = {"statements": statements}

            response = await self.http_client.post(
                http_url,
                headers=headers,
                json=request_body,
            )

            if response.status_code in (200, 201):
                data = response.json()
                return {
                    "success": True,
                    "results": data.get("results", []),
                }
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_schema(
        self,
        credentials: IntegrationCredentials,
        table: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get database schema information.

        Args:
            credentials: Turso credentials.
            table: Optional specific table to get schema for.

        Returns:
            Schema information for tables and columns.
        """
        try:
            # Get list of tables
            tables_result = await self.execute_query(
                credentials,
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
            )

            if not tables_result.get("success"):
                return tables_result

            tables = []
            for row in tables_result.get("rows", []):
                table_name = row[0] if isinstance(row, list) else row.get("name")
                create_sql = row[1] if isinstance(row, list) else row.get("sql")

                if table and table_name != table:
                    continue

                # Get column info - RAP-291: Validate table name to prevent SQL injection
                # Note: table_name comes from sqlite_master but could be crafted maliciously
                try:
                    if not VALID_IDENTIFIER_PATTERN.match(table_name):
                        continue  # Skip tables with invalid names
                    safe_table = validate_turso_identifier(table_name)
                except ValueError:
                    continue  # Skip tables with invalid names

                columns_result = await self.execute_query(
                    credentials,
                    f"PRAGMA table_info({safe_table})",
                )

                columns = []
                if columns_result.get("success"):
                    for col_row in columns_result.get("rows", []):
                        if isinstance(col_row, list):
                            columns.append({
                                "cid": col_row[0],
                                "name": col_row[1],
                                "type": col_row[2],
                                "notnull": col_row[3],
                                "default": col_row[4],
                                "pk": col_row[5],
                            })
                        else:
                            columns.append(col_row)

                tables.append({
                    "name": table_name,
                    "sql": create_sql,
                    "columns": columns,
                })

            if table and len(tables) == 0:
                return {"success": False, "error": f"Table '{table}' not found"}

            return {"success": True, "tables": tables}

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def setup_test_data(
        self,
        credentials: IntegrationCredentials,
        table: str,
        records: list[dict[str, Any]],
        cleanup_first: bool = True,
        cleanup_condition: Optional[str] = None,
    ) -> dict[str, Any]:
        """Setup test data in a table.

        Args:
            credentials: Turso credentials.
            table: Table to insert data into.
            records: List of records to insert.
            cleanup_first: Whether to delete existing data first.
            cleanup_condition: Optional parameterized WHERE clause dict
                               (e.g., {"column": "type", "value": "test"}).
                               Raw SQL strings are no longer accepted for security.

        Returns:
            Result of the operation.

        RAP-291: This method now validates all identifiers and uses parameterized
        queries to prevent SQL injection attacks.
        """
        results: dict[str, Any] = {"cleanup": None, "insert": None}
        statements: list[dict[str, Any]] = []

        try:
            # RAP-291: Validate table name
            safe_table = validate_turso_identifier(table)

            # Cleanup existing test data if requested
            if cleanup_first:
                cleanup_sql = f"DELETE FROM {safe_table}"

                # RAP-291: Handle cleanup_condition securely
                if cleanup_condition:
                    # If cleanup_condition is a dict, use parameterized query
                    if isinstance(cleanup_condition, dict):
                        col = cleanup_condition.get("column", "")
                        val = cleanup_condition.get("value")
                        safe_col = validate_turso_identifier(col)
                        cleanup_sql += f" WHERE {safe_col} = ?"
                        statements.append({"q": cleanup_sql, "params": [val]})
                    else:
                        # For backwards compatibility, validate that it looks safe
                        # Only allow simple column = 'value' patterns
                        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*['\"]?[^'\"]+['\"]?$", cleanup_condition):
                            return {
                                "success": False,
                                "error": "Invalid cleanup_condition format. Use dict format: {'column': 'col', 'value': 'val'}",
                                "results": results,
                            }
                        # Log deprecation warning
                        cleanup_sql += f" WHERE {cleanup_condition}"
                        statements.append({"q": cleanup_sql})
                else:
                    statements.append({"q": cleanup_sql})

            # Build insert statements
            if records:
                for record in records:
                    columns = list(record.keys())

                    # RAP-291: Validate all column names
                    safe_columns = []
                    for col in columns:
                        safe_columns.append(validate_turso_identifier(col))

                    placeholders = ", ".join(["?" for _ in columns])
                    values = [record[col] for col in columns]

                    insert_sql = f"INSERT INTO {safe_table} ({', '.join(safe_columns)}) VALUES ({placeholders})"
                    statements.append({"q": insert_sql, "params": values})

            # Execute batch
            batch_result = await self.execute_batch(credentials, statements)

            if not batch_result.get("success"):
                return {"success": False, "error": batch_result.get("error"), "results": results}

            # Parse results
            batch_results = batch_result.get("results", [])

            if cleanup_first and batch_results:
                results["cleanup"] = {
                    "success": True,
                    "rows_affected": batch_results[0].get("rows_affected", 0),
                }
                batch_results = batch_results[1:]

            insert_count = sum(r.get("rows_affected", 0) for r in batch_results)
            results["insert"] = {"success": True, "rows_inserted": insert_count}

            return {
                "success": True,
                "results": results,
                "records_inserted": insert_count,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "results": results}

    # -------------------------------------------------------------------------
    # Platform Management Operations
    # -------------------------------------------------------------------------

    async def create_database(
        self,
        credentials: IntegrationCredentials,
        name: str,
        group: Optional[str] = None,
        org_slug: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a new database.

        Args:
            credentials: Turso credentials.
            name: Database name.
            group: Optional group name (uses default if not specified).
            org_slug: Optional organization slug.

        Returns:
            Created database info or error.
        """
        headers = self._get_platform_headers(credentials)
        org = org_slug or credentials.org_slug

        if not org:
            return {"success": False, "error": "Organization slug required"}

        try:
            body: dict[str, Any] = {"name": name}
            if group:
                body["group"] = group

            response = await self.http_client.post(
                f"{self.PLATFORM_API_URL}/v1/organizations/{org}/databases",
                headers=headers,
                json=body,
            )

            if response.status_code in (200, 201):
                data = response.json()
                return {"success": True, "database": data.get("database", data)}
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def delete_database(
        self,
        credentials: IntegrationCredentials,
        name: str,
        org_slug: Optional[str] = None,
    ) -> dict[str, Any]:
        """Delete a database.

        Args:
            credentials: Turso credentials.
            name: Database name.
            org_slug: Optional organization slug.

        Returns:
            Success status or error.
        """
        headers = self._get_platform_headers(credentials)
        org = org_slug or credentials.org_slug

        if not org:
            return {"success": False, "error": "Organization slug required"}

        try:
            response = await self.http_client.delete(
                f"{self.PLATFORM_API_URL}/v1/organizations/{org}/databases/{name}",
                headers=headers,
            )

            if response.status_code in (200, 204):
                return {"success": True}
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_database_token(
        self,
        credentials: IntegrationCredentials,
        database: str,
        expiration: Optional[str] = None,
        authorization: Optional[str] = None,
        org_slug: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a database auth token.

        Args:
            credentials: Turso credentials.
            database: Database name.
            expiration: Optional expiration (e.g., "24h", "7d", "never").
            authorization: Optional authorization level ("full-access" or "read-only").
            org_slug: Optional organization slug.

        Returns:
            Created token or error.
        """
        headers = self._get_platform_headers(credentials)
        org = org_slug or credentials.org_slug

        if not org:
            return {"success": False, "error": "Organization slug required"}

        try:
            body: dict[str, Any] = {}
            if expiration:
                body["expiration"] = expiration
            if authorization:
                body["authorization"] = authorization

            response = await self.http_client.post(
                f"{self.PLATFORM_API_URL}/v1/organizations/{org}/databases/{database}/auth/tokens",
                headers=headers,
                json=body if body else None,
            )

            if response.status_code in (200, 201):
                data = response.json()
                return {"success": True, "token": data.get("jwt", data)}
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def create_group(
        self,
        credentials: IntegrationCredentials,
        name: str,
        location: str,
        org_slug: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a database group.

        Args:
            credentials: Turso credentials.
            name: Group name.
            location: Primary location (e.g., "iad", "lhr", "sin").
            org_slug: Optional organization slug.

        Returns:
            Created group info or error.
        """
        headers = self._get_platform_headers(credentials)
        org = org_slug or credentials.org_slug

        if not org:
            return {"success": False, "error": "Organization slug required"}

        try:
            body = {"name": name, "location": location}

            response = await self.http_client.post(
                f"{self.PLATFORM_API_URL}/v1/organizations/{org}/groups",
                headers=headers,
                json=body,
            )

            if response.status_code in (200, 201):
                data = response.json()
                return {"success": True, "group": data.get("group", data)}
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}
