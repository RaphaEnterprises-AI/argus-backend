"""Supabase database integration plugin.

This plugin provides integration with Supabase for:
- Project and database management via Management API
- Schema introspection
- Test data management
- Connection testing
- Storage bucket management
"""

import time
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urljoin

import httpx

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


class SupabasePlugin(IntegrationPlugin):
    """Supabase integration plugin.

    Provides access to Supabase Management API and PostgREST API for:
    - Database operations and queries
    - Schema introspection
    - Storage bucket management
    - Project and organization info

    Authentication uses either:
    - Service Role Key for database operations (PostgREST API)
    - Management API Token for project management (Management API)
    """

    # PostgREST API base URL format
    POSTGREST_URL_FORMAT = "https://{project_ref}.supabase.co/rest/v1"

    # Management API base URL
    MANAGEMENT_API_URL = "https://api.supabase.com"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get metadata about the Supabase integration."""
        return IntegrationMetadata(
            slug="supabase",
            name="Supabase",
            description="Open source Firebase alternative with PostgreSQL database, authentication, "
            "storage, and edge functions",
            category=IntegrationCategory.DATABASE,
            auth_type=AuthType.SERVICE_ACCOUNT,
            icon_url="https://supabase.com/favicon/favicon-196x196.png",
            docs_url="https://supabase.com/docs",
            website_url="https://supabase.com",
            features=[
                "database_queries",
                "schema_introspection",
                "test_data_management",
                "storage_buckets",
                "rpc_functions",
                "project_management",
                "realtime_subscriptions",
            ],
            required_fields=[
                FieldDefinition(
                    name="instance_url",
                    label="Project URL",
                    type="url",
                    required=True,
                    placeholder="https://your-project.supabase.co",
                    help_text="Your Supabase project URL",
                ),
                FieldDefinition(
                    name="api_key",
                    label="Service Role Key",
                    type="password",
                    required=True,
                    placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    help_text="Service Role Key from Project Settings > API (not the anon key)",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="access_token",
                    label="Management API Token",
                    type="password",
                    required=False,
                    placeholder="sbp_...",
                    help_text="Personal access token for Management API operations",
                ),
            ],
            supports_webhooks=True,
            supports_sync=True,
            sync_resources=["tables", "functions", "storage_buckets", "policies", "projects"],
        )

    def _get_project_ref(self, credentials: IntegrationCredentials) -> str:
        """Extract project reference from instance URL.

        Args:
            credentials: Credentials containing instance_url.

        Returns:
            Project reference string (e.g., "xyzcompany" from "https://xyzcompany.supabase.co").
        """
        instance_url = credentials.instance_url or ""
        # Remove protocol and .supabase.co suffix
        url = instance_url.replace("https://", "").replace("http://", "")
        return url.split(".")[0] if "." in url else url

    def _get_postgrest_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for PostgREST API requests.

        Args:
            credentials: Credentials containing service role key.

        Returns:
            Headers dictionary for API requests.
        """
        return {
            "apikey": credentials.api_key or "",
            "Authorization": f"Bearer {credentials.api_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def _get_management_headers(self, credentials: IntegrationCredentials) -> dict[str, str]:
        """Get headers for Management API requests.

        Args:
            credentials: Credentials containing management access token.

        Returns:
            Headers dictionary for Management API requests.
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
        """Test connection to Supabase.

        Tests the PostgREST API endpoint to verify the service role key is valid.

        Args:
            credentials: Credentials containing project URL and service role key.

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
            instance_url = credentials.instance_url or ""
            rest_url = f"{instance_url.rstrip('/')}/rest/v1/"
            headers = self._get_postgrest_headers(credentials)

            # Test connection by checking the root endpoint
            response = await self.http_client.get(
                rest_url,
                headers=headers,
            )

            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                # Try to get project info via introspection
                tables = response.json() if response.text else []
                table_count = len(tables) if isinstance(tables, list) else 0

                # Check permissions by attempting to list tables
                permissions = ["read"]
                # Service role key typically has all permissions
                if credentials.api_key and "service_role" in str(credentials.api_key):
                    permissions = ["read", "write", "delete", "admin"]

                return ConnectionTestResult(
                    success=True,
                    message="Successfully connected to Supabase",
                    latency_ms=latency_ms,
                    api_version="v1",
                    permissions=permissions,
                    account_info={
                        "project_ref": self._get_project_ref(credentials),
                        "project_url": instance_url,
                        "table_count": table_count,
                    },
                )
            elif response.status_code == 401:
                return ConnectionTestResult(
                    success=False,
                    message="Authentication failed - check your Service Role Key",
                    latency_ms=latency_ms,
                    error_code="AUTH_FAILED",
                    error_details={"status_code": 401},
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
                message="Connection timed out - check if the project URL is correct",
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
        """Sync data from Supabase.

        Syncs database schema information including tables, functions, and storage buckets.

        Args:
            credentials: Credentials for Supabase access.
            resources: Optional list of resources to sync (tables, functions, storage_buckets).
            since: Not used for schema sync (always full sync).
            limit: Maximum items per resource type.

        Returns:
            SyncResult with synced schema information.
        """
        started_at = datetime.utcnow()
        sync_resources = resources or ["tables", "functions", "storage_buckets"]
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        data_points = 0

        instance_url = credentials.instance_url or ""
        headers = self._get_postgrest_headers(credentials)

        try:
            # Sync tables (schema introspection)
            if "tables" in sync_resources:
                tables_data = await self._sync_tables(instance_url, headers, limit)
                if tables_data.get("error"):
                    errors.append(tables_data["error"])
                else:
                    data["tables"] = tables_data.get("tables", [])
                    data_points += len(data["tables"])

            # Sync RPC functions
            if "functions" in sync_resources:
                functions_data = await self._sync_functions(instance_url, headers, limit)
                if functions_data.get("error"):
                    warnings.append(functions_data["error"])
                else:
                    data["functions"] = functions_data.get("functions", [])
                    data_points += len(data["functions"])

            # Sync storage buckets
            if "storage_buckets" in sync_resources:
                buckets_data = await self._sync_storage_buckets(instance_url, headers, limit)
                if buckets_data.get("error"):
                    warnings.append(buckets_data["error"])
                else:
                    data["storage_buckets"] = buckets_data.get("buckets", [])
                    data_points += len(data["storage_buckets"])

            # Sync policies if requested
            if "policies" in sync_resources:
                policies_data = await self._sync_policies(instance_url, headers, limit)
                if policies_data.get("error"):
                    warnings.append(policies_data["error"])
                else:
                    data["policies"] = policies_data.get("policies", [])
                    data_points += len(data["policies"])

            # Sync projects via Management API if token available
            if "projects" in sync_resources and credentials.access_token:
                projects_data = await self._sync_projects(credentials)
                if projects_data.get("error"):
                    warnings.append(projects_data["error"])
                else:
                    data["projects"] = projects_data.get("projects", [])
                    data_points += len(data["projects"])

            return SyncResult(
                success=len(errors) == 0,
                message="Sync completed" if not errors else f"Sync completed with errors: {errors}",
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

    async def _sync_tables(
        self,
        instance_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync table schema information.

        Args:
            instance_url: Supabase project URL.
            headers: API headers.
            limit: Maximum tables to return.

        Returns:
            Dictionary with tables list or error.
        """
        try:
            # Query pg_tables via PostgREST to get table information
            # This requires the pg_catalog access which service role has
            schema_url = f"{instance_url.rstrip('/')}/rest/v1/rpc/get_tables"

            # First try the RPC approach
            response = await self.http_client.post(
                schema_url,
                headers=headers,
                json={},
            )

            if response.status_code == 200:
                tables = response.json()
                return {"tables": tables[:limit] if isinstance(tables, list) else []}

            # Fallback: Get table names from root endpoint
            root_url = f"{instance_url.rstrip('/')}/rest/v1/"
            response = await self.http_client.get(root_url, headers=headers)

            if response.status_code == 200:
                # PostgREST returns OpenAPI spec at root
                data = response.json()
                if isinstance(data, dict) and "definitions" in data:
                    tables = [
                        {"name": name, "columns": list(schema.get("properties", {}).keys())}
                        for name, schema in data.get("definitions", {}).items()
                    ]
                    return {"tables": tables[:limit]}
                elif isinstance(data, list):
                    return {"tables": [{"name": t} for t in data[:limit]]}

            return {"tables": [], "error": "Could not retrieve table schema"}

        except Exception as e:
            return {"error": f"Failed to sync tables: {str(e)}"}

    async def _sync_functions(
        self,
        instance_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync database functions (RPC endpoints).

        Args:
            instance_url: Supabase project URL.
            headers: API headers.
            limit: Maximum functions to return.

        Returns:
            Dictionary with functions list or error.
        """
        try:
            # Try to get functions via RPC
            rpc_url = f"{instance_url.rstrip('/')}/rest/v1/rpc/get_functions"
            response = await self.http_client.post(
                rpc_url,
                headers=headers,
                json={},
            )

            if response.status_code == 200:
                functions = response.json()
                return {"functions": functions[:limit] if isinstance(functions, list) else []}

            # Functions list not available via direct query
            return {"functions": [], "error": "RPC function listing not available"}

        except Exception as e:
            return {"error": f"Failed to sync functions: {str(e)}"}

    async def _sync_storage_buckets(
        self,
        instance_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync storage bucket information.

        Args:
            instance_url: Supabase project URL.
            headers: API headers.
            limit: Maximum buckets to return.

        Returns:
            Dictionary with buckets list or error.
        """
        try:
            storage_url = f"{instance_url.rstrip('/')}/storage/v1/bucket"
            response = await self.http_client.get(storage_url, headers=headers)

            if response.status_code == 200:
                buckets = response.json()
                return {"buckets": buckets[:limit] if isinstance(buckets, list) else []}

            return {"buckets": [], "error": f"Storage API returned {response.status_code}"}

        except Exception as e:
            return {"error": f"Failed to sync storage buckets: {str(e)}"}

    async def _sync_policies(
        self,
        instance_url: str,
        headers: dict[str, str],
        limit: int,
    ) -> dict[str, Any]:
        """Sync RLS policies.

        Args:
            instance_url: Supabase project URL.
            headers: API headers.
            limit: Maximum policies to return.

        Returns:
            Dictionary with policies list or error.
        """
        try:
            # Try to get policies via RPC
            rpc_url = f"{instance_url.rstrip('/')}/rest/v1/rpc/get_policies"
            response = await self.http_client.post(
                rpc_url,
                headers=headers,
                json={},
            )

            if response.status_code == 200:
                policies = response.json()
                return {"policies": policies[:limit] if isinstance(policies, list) else []}

            return {"policies": [], "error": "RLS policy listing not available"}

        except Exception as e:
            return {"error": f"Failed to sync policies: {str(e)}"}

    async def _sync_projects(self, credentials: IntegrationCredentials) -> dict[str, Any]:
        """Sync projects from Management API.

        Args:
            credentials: Credentials with management access token.

        Returns:
            Dictionary with projects list or error.
        """
        try:
            headers = self._get_management_headers(credentials)
            url = f"{self.MANAGEMENT_API_URL}/v1/projects"

            response = await self.http_client.get(url, headers=headers)

            if response.status_code == 200:
                projects = response.json()
                return {"projects": projects if isinstance(projects, list) else []}

            return {"projects": [], "error": f"Management API returned {response.status_code}"}

        except Exception as e:
            return {"error": f"Failed to sync projects: {str(e)}"}

    # -------------------------------------------------------------------------
    # Database Operations for Test Data Management
    # -------------------------------------------------------------------------

    async def execute_query(
        self,
        credentials: IntegrationCredentials,
        table: str,
        operation: str = "select",
        data: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
        select: Optional[str] = "*",
        limit: Optional[int] = None,
        order: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute a database query via PostgREST.

        Args:
            credentials: Supabase credentials.
            table: Table name to query.
            operation: Operation type (select, insert, update, upsert, delete).
            data: Data for insert/update operations.
            filters: Filter conditions as key-value pairs.
            select: Columns to select (default "*").
            limit: Maximum rows to return.
            order: Order by column (e.g., "created_at.desc").

        Returns:
            Query result with data or error.
        """
        instance_url = credentials.instance_url or ""
        headers = self._get_postgrest_headers(credentials)
        url = f"{instance_url.rstrip('/')}/rest/v1/{table}"

        # Build query parameters
        params: dict[str, Any] = {}
        if select:
            params["select"] = select
        if limit:
            params["limit"] = str(limit)
        if order:
            params["order"] = order

        # Add filters
        if filters:
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Handle operators like {"gte": 10}
                    for op, val in value.items():
                        params[key] = f"{op}.{val}"
                else:
                    params[key] = f"eq.{value}"

        try:
            if operation == "select":
                response = await self.http_client.get(url, headers=headers, params=params)
            elif operation == "insert":
                response = await self.http_client.post(url, headers=headers, json=data, params=params)
            elif operation == "update":
                response = await self.http_client.patch(url, headers=headers, json=data, params=params)
            elif operation == "upsert":
                headers["Prefer"] = "resolution=merge-duplicates"
                response = await self.http_client.post(url, headers=headers, json=data, params=params)
            elif operation == "delete":
                response = await self.http_client.delete(url, headers=headers, params=params)
            else:
                return {"error": f"Unknown operation: {operation}"}

            if response.status_code in (200, 201, 204):
                return {
                    "success": True,
                    "data": response.json() if response.text else None,
                    "status_code": response.status_code,
                }
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def call_rpc(
        self,
        credentials: IntegrationCredentials,
        function_name: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Call a database RPC function.

        Args:
            credentials: Supabase credentials.
            function_name: Name of the RPC function to call.
            params: Parameters to pass to the function.

        Returns:
            Function result or error.
        """
        instance_url = credentials.instance_url or ""
        headers = self._get_postgrest_headers(credentials)
        url = f"{instance_url.rstrip('/')}/rest/v1/rpc/{function_name}"

        try:
            response = await self.http_client.post(
                url,
                headers=headers,
                json=params or {},
            )

            if response.status_code in (200, 201):
                return {
                    "success": True,
                    "data": response.json() if response.text else None,
                }
            else:
                return {
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code,
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def setup_test_data(
        self,
        credentials: IntegrationCredentials,
        table: str,
        records: list[dict[str, Any]],
        cleanup_first: bool = True,
        filter_column: Optional[str] = None,
        filter_values: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Setup test data in a table.

        Convenience method for test data management that optionally cleans up
        existing data before inserting new test records.

        Args:
            credentials: Supabase credentials.
            table: Table to insert data into.
            records: List of records to insert.
            cleanup_first: Whether to delete matching records first.
            filter_column: Column to filter on for cleanup.
            filter_values: Values to filter for cleanup.

        Returns:
            Result of the operation.
        """
        results: dict[str, Any] = {"cleanup": None, "insert": None}

        try:
            # Cleanup existing test data if requested
            if cleanup_first and filter_column and filter_values:
                for value in filter_values:
                    cleanup_result = await self.execute_query(
                        credentials,
                        table=table,
                        operation="delete",
                        filters={filter_column: value},
                    )
                    if not cleanup_result.get("success", True):
                        results["cleanup"] = cleanup_result
                        break
                results["cleanup"] = {"success": True, "deleted_count": len(filter_values)}

            # Insert test records
            insert_result = await self.execute_query(
                credentials,
                table=table,
                operation="insert",
                data=records,
            )
            results["insert"] = insert_result

            return {
                "success": insert_result.get("success", False),
                "results": results,
                "records_inserted": len(records) if insert_result.get("success") else 0,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "results": results}

    async def get_schema(
        self,
        credentials: IntegrationCredentials,
        table: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get database schema information.

        Args:
            credentials: Supabase credentials.
            table: Optional specific table to get schema for.

        Returns:
            Schema information for tables and columns.
        """
        instance_url = credentials.instance_url or ""
        headers = self._get_postgrest_headers(credentials)

        try:
            # Get OpenAPI spec which contains schema
            url = f"{instance_url.rstrip('/')}/rest/v1/"
            response = await self.http_client.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "definitions" in data:
                    definitions = data.get("definitions", {})

                    if table:
                        if table in definitions:
                            return {
                                "success": True,
                                "schema": {table: definitions[table]},
                            }
                        else:
                            return {
                                "success": False,
                                "error": f"Table '{table}' not found",
                            }

                    return {"success": True, "schema": definitions}

            return {"success": False, "error": "Could not retrieve schema"}

        except Exception as e:
            return {"success": False, "error": str(e)}
