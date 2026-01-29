"""AWS deployment integration plugin.

This module provides integration with AWS services for deployment tracking,
monitoring, and resource management including CloudWatch, ECS, and Lambda.

AWS SDK Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Optional

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


class AWSPlugin(IntegrationPlugin):
    """AWS cloud platform integration.

    Amazon Web Services integration for monitoring and managing
    cloud infrastructure including ECS containers, Lambda functions,
    and CloudWatch metrics.

    Features:
    - CloudWatch metrics monitoring
    - ECS service and task tracking
    - Lambda function management
    - EC2 instance monitoring (optional)

    Authentication:
    - Access Key ID + Secret Access Key
    - Supports region selection
    - IAM permissions required for each service

    Required IAM Permissions:
    - sts:GetCallerIdentity (for connection testing)
    - cloudwatch:GetMetricData, cloudwatch:ListMetrics
    - ecs:ListClusters, ecs:ListServices, ecs:DescribeServices
    - lambda:ListFunctions, lambda:GetFunction

    Usage:
        plugin = AWSPlugin()
        credentials = IntegrationCredentials(
            api_key="AKIAIOSFODNN7EXAMPLE",
            api_secret="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            extra={"region": "us-east-1"}
        )
        result = await plugin.test_connection(credentials)
    """

    DEFAULT_REGION = "us-east-1"

    @property
    def metadata(self) -> IntegrationMetadata:
        """Get AWS plugin metadata."""
        return IntegrationMetadata(
            slug="aws",
            name="Amazon Web Services",
            description="Monitor and manage AWS infrastructure including ECS, Lambda, and CloudWatch",
            category=IntegrationCategory.DEPLOYMENT,
            auth_type=AuthType.API_KEY,
            icon_url="https://aws.amazon.com/favicon.ico",
            docs_url="https://docs.aws.amazon.com/",
            website_url="https://aws.amazon.com",
            features=[
                "cloudwatch_metrics",
                "ecs_monitoring",
                "lambda_monitoring",
                "deployment_tracking",
                "resource_management",
            ],
            required_fields=[
                FieldDefinition(
                    name="api_key",
                    label="Access Key ID",
                    type="text",
                    required=True,
                    placeholder="AKIAIOSFODNN7EXAMPLE",
                    help_text="AWS Access Key ID from IAM",
                ),
                FieldDefinition(
                    name="api_secret",
                    label="Secret Access Key",
                    type="password",
                    required=True,
                    placeholder="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                    help_text="AWS Secret Access Key from IAM",
                ),
            ],
            optional_fields=[
                FieldDefinition(
                    name="extra.region",
                    label="AWS Region",
                    type="select",
                    required=False,
                    placeholder="us-east-1",
                    help_text="AWS region for API calls",
                    options=[
                        {"value": "us-east-1", "label": "US East (N. Virginia)"},
                        {"value": "us-east-2", "label": "US East (Ohio)"},
                        {"value": "us-west-1", "label": "US West (N. California)"},
                        {"value": "us-west-2", "label": "US West (Oregon)"},
                        {"value": "eu-west-1", "label": "EU (Ireland)"},
                        {"value": "eu-west-2", "label": "EU (London)"},
                        {"value": "eu-west-3", "label": "EU (Paris)"},
                        {"value": "eu-central-1", "label": "EU (Frankfurt)"},
                        {"value": "eu-north-1", "label": "EU (Stockholm)"},
                        {"value": "ap-northeast-1", "label": "Asia Pacific (Tokyo)"},
                        {"value": "ap-northeast-2", "label": "Asia Pacific (Seoul)"},
                        {"value": "ap-southeast-1", "label": "Asia Pacific (Singapore)"},
                        {"value": "ap-southeast-2", "label": "Asia Pacific (Sydney)"},
                        {"value": "ap-south-1", "label": "Asia Pacific (Mumbai)"},
                        {"value": "sa-east-1", "label": "South America (Sao Paulo)"},
                        {"value": "ca-central-1", "label": "Canada (Central)"},
                    ],
                ),
                FieldDefinition(
                    name="extra.session_token",
                    label="Session Token",
                    type="password",
                    required=False,
                    help_text="AWS Session Token (for temporary credentials)",
                ),
            ],
            supports_webhooks=False,
            supports_sync=True,
            sync_resources=["cloudwatch_metrics", "ecs_services", "lambda_functions"],
        )

    def _get_region(self, credentials: IntegrationCredentials) -> str:
        """Get the AWS region from credentials or default."""
        return credentials.extra.get("region", self.DEFAULT_REGION)

    def _get_boto3_session(self, credentials: IntegrationCredentials):
        """Create a boto3 session with the provided credentials.

        Returns:
            boto3.Session object
        """
        import boto3

        session_kwargs = {
            "aws_access_key_id": credentials.api_key,
            "aws_secret_access_key": credentials.api_secret,
            "region_name": self._get_region(credentials),
        }

        session_token = credentials.extra.get("session_token")
        if session_token:
            session_kwargs["aws_session_token"] = session_token

        return boto3.Session(**session_kwargs)

    async def test_connection(
        self,
        credentials: IntegrationCredentials,
    ) -> ConnectionTestResult:
        """Test the AWS API connection.

        Tests by calling STS get_caller_identity to validate credentials.

        Args:
            credentials: Credentials containing Access Key ID and Secret.

        Returns:
            ConnectionTestResult with success status and identity info.
        """
        if not credentials.api_key:
            return ConnectionTestResult(
                success=False,
                message="Access Key ID is required",
                error_code="MISSING_ACCESS_KEY",
            )

        if not credentials.api_secret:
            return ConnectionTestResult(
                success=False,
                message="Secret Access Key is required",
                error_code="MISSING_SECRET_KEY",
            )

        start_time = time.time()

        try:
            # Run boto3 call in thread pool since it's synchronous
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self._test_sts_connection,
                    credentials,
                )

            latency_ms = (time.time() - start_time) * 1000
            return ConnectionTestResult(
                success=result["success"],
                message=result["message"],
                latency_ms=latency_ms,
                api_version="boto3",
                permissions=result.get("permissions", []),
                account_info=result.get("account_info", {}),
                error_code=result.get("error_code"),
                error_details=result.get("error_details"),
            )

        except Exception as e:
            return ConnectionTestResult(
                success=False,
                message=f"Connection failed: {str(e)}",
                error_code="CONNECTION_ERROR",
                error_details={"error": str(e)},
            )

    def _test_sts_connection(self, credentials: IntegrationCredentials) -> dict[str, Any]:
        """Synchronous STS connection test (runs in thread pool)."""
        try:
            session = self._get_boto3_session(credentials)
            sts_client = session.client("sts")

            # Get caller identity to validate credentials
            identity = sts_client.get_caller_identity()

            # Check available permissions by testing each service
            permissions = ["sts:GetCallerIdentity"]

            # Test CloudWatch access
            try:
                cw_client = session.client("cloudwatch")
                cw_client.list_metrics(Limit=1)
                permissions.extend(["cloudwatch:ListMetrics", "cloudwatch:GetMetricData"])
            except Exception:
                pass

            # Test ECS access
            try:
                ecs_client = session.client("ecs")
                ecs_client.list_clusters(maxResults=1)
                permissions.extend(
                    ["ecs:ListClusters", "ecs:ListServices", "ecs:DescribeServices"]
                )
            except Exception:
                pass

            # Test Lambda access
            try:
                lambda_client = session.client("lambda")
                lambda_client.list_functions(MaxItems=1)
                permissions.extend(["lambda:ListFunctions", "lambda:GetFunction"])
            except Exception:
                pass

            return {
                "success": True,
                "message": "Successfully connected to AWS",
                "permissions": permissions,
                "account_info": {
                    "account_id": identity.get("Account"),
                    "user_id": identity.get("UserId"),
                    "arn": identity.get("Arn"),
                    "region": self._get_region(credentials),
                },
            }

        except Exception as e:
            error_str = str(e)
            error_code = "UNKNOWN_ERROR"

            if "InvalidClientTokenId" in error_str:
                error_code = "INVALID_ACCESS_KEY"
                message = "Invalid Access Key ID"
            elif "SignatureDoesNotMatch" in error_str:
                error_code = "INVALID_SECRET_KEY"
                message = "Invalid Secret Access Key"
            elif "ExpiredToken" in error_str:
                error_code = "EXPIRED_TOKEN"
                message = "AWS session token has expired"
            elif "AccessDenied" in error_str:
                error_code = "ACCESS_DENIED"
                message = "Access denied - check IAM permissions"
            else:
                message = f"AWS error: {error_str}"

            return {
                "success": False,
                "message": message,
                "error_code": error_code,
                "error_details": {"error": error_str},
            }

    async def sync_data(
        self,
        credentials: IntegrationCredentials,
        resources: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> SyncResult:
        """Sync data from AWS services.

        Syncs CloudWatch metrics, ECS services, and Lambda functions.

        Args:
            credentials: Credentials containing Access Key ID and Secret.
            resources: Optional list of resources to sync.
            since: Optional datetime to sync data since (for metrics).
            limit: Maximum number of items to sync per resource.

        Returns:
            SyncResult with synced AWS resources.
        """
        if not credentials.api_key or not credentials.api_secret:
            return SyncResult(
                success=False,
                message="Access Key ID and Secret Access Key are required",
            )

        started_at = datetime.utcnow()
        resources_to_sync = resources or self.metadata.sync_resources
        data: dict[str, Any] = {}
        errors: list[str] = []
        warnings: list[str] = []
        total_synced = 0

        try:
            # Run all sync operations in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                # Sync CloudWatch metrics
                if "cloudwatch_metrics" in resources_to_sync:
                    result = await loop.run_in_executor(
                        executor,
                        self._sync_cloudwatch_metrics,
                        credentials,
                        since,
                        limit,
                    )
                    data["cloudwatch_metrics"] = result["metrics"]
                    total_synced += len(result["metrics"])
                    if result.get("errors"):
                        errors.extend(result["errors"])

                # Sync ECS services
                if "ecs_services" in resources_to_sync:
                    result = await loop.run_in_executor(
                        executor,
                        self._sync_ecs_services,
                        credentials,
                        limit,
                    )
                    data["ecs_services"] = result["services"]
                    total_synced += len(result["services"])
                    if result.get("errors"):
                        errors.extend(result["errors"])

                # Sync Lambda functions
                if "lambda_functions" in resources_to_sync:
                    result = await loop.run_in_executor(
                        executor,
                        self._sync_lambda_functions,
                        credentials,
                        limit,
                    )
                    data["lambda_functions"] = result["functions"]
                    total_synced += len(result["functions"])
                    if result.get("errors"):
                        errors.extend(result["errors"])

            return SyncResult(
                success=len(errors) == 0,
                message=f"Synced {total_synced} items from AWS"
                if len(errors) == 0
                else f"Synced with {len(errors)} errors",
                data_points_synced=total_synced,
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

    def _sync_cloudwatch_metrics(
        self,
        credentials: IntegrationCredentials,
        since: Optional[datetime],
        limit: int,
    ) -> dict[str, Any]:
        """Sync CloudWatch metrics (synchronous, runs in thread pool)."""
        metrics: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            session = self._get_boto3_session(credentials)
            cw_client = session.client("cloudwatch")

            # Define key metrics to fetch
            metric_queries = [
                # ECS metrics
                {"namespace": "AWS/ECS", "name": "CPUUtilization"},
                {"namespace": "AWS/ECS", "name": "MemoryUtilization"},
                # Lambda metrics
                {"namespace": "AWS/Lambda", "name": "Invocations"},
                {"namespace": "AWS/Lambda", "name": "Errors"},
                {"namespace": "AWS/Lambda", "name": "Duration"},
                # EC2 metrics (if applicable)
                {"namespace": "AWS/EC2", "name": "CPUUtilization"},
                {"namespace": "AWS/EC2", "name": "NetworkIn"},
            ]

            # Calculate time range
            end_time = datetime.utcnow()
            start_time = since or (end_time - timedelta(hours=24))

            for metric_query in metric_queries[:limit]:
                try:
                    # List metrics to find dimensions
                    list_response = cw_client.list_metrics(
                        Namespace=metric_query["namespace"],
                        MetricName=metric_query["name"],
                        Limit=10,
                    )

                    for metric in list_response.get("Metrics", []):
                        dimensions = metric.get("Dimensions", [])

                        # Get metric statistics
                        stats_response = cw_client.get_metric_statistics(
                            Namespace=metric_query["namespace"],
                            MetricName=metric_query["name"],
                            Dimensions=dimensions,
                            StartTime=start_time,
                            EndTime=end_time,
                            Period=3600,  # 1 hour periods
                            Statistics=["Average", "Maximum", "Minimum"],
                        )

                        datapoints = stats_response.get("Datapoints", [])
                        if datapoints:
                            # Get the most recent datapoint
                            latest = max(datapoints, key=lambda x: x.get("Timestamp", datetime.min))

                            metrics.append(
                                {
                                    "namespace": metric_query["namespace"],
                                    "metric_name": metric_query["name"],
                                    "dimensions": {
                                        d["Name"]: d["Value"] for d in dimensions
                                    },
                                    "statistics": {
                                        "average": latest.get("Average"),
                                        "maximum": latest.get("Maximum"),
                                        "minimum": latest.get("Minimum"),
                                    },
                                    "timestamp": latest.get("Timestamp", "").isoformat()
                                    if isinstance(latest.get("Timestamp"), datetime)
                                    else str(latest.get("Timestamp")),
                                    "unit": latest.get("Unit"),
                                }
                            )

                except Exception as e:
                    errors.append(
                        f"Failed to fetch {metric_query['namespace']}/{metric_query['name']}: {str(e)}"
                    )

        except Exception as e:
            errors.append(f"Failed to access CloudWatch: {str(e)}")

        return {"metrics": metrics, "errors": errors}

    def _sync_ecs_services(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync ECS services (synchronous, runs in thread pool)."""
        services: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            session = self._get_boto3_session(credentials)
            ecs_client = session.client("ecs")

            # List all clusters
            clusters_response = ecs_client.list_clusters(maxResults=min(limit, 100))
            cluster_arns = clusters_response.get("clusterArns", [])

            for cluster_arn in cluster_arns:
                cluster_name = cluster_arn.split("/")[-1]

                try:
                    # List services in cluster
                    services_response = ecs_client.list_services(
                        cluster=cluster_arn,
                        maxResults=min(limit, 100),
                    )
                    service_arns = services_response.get("serviceArns", [])

                    if service_arns:
                        # Describe services (max 10 at a time)
                        for i in range(0, len(service_arns), 10):
                            batch = service_arns[i : i + 10]
                            describe_response = ecs_client.describe_services(
                                cluster=cluster_arn,
                                services=batch,
                            )

                            for svc in describe_response.get("services", []):
                                deployments = svc.get("deployments", [])
                                primary_deployment = next(
                                    (d for d in deployments if d.get("status") == "PRIMARY"),
                                    {},
                                )

                                services.append(
                                    {
                                        "service_arn": svc.get("serviceArn"),
                                        "service_name": svc.get("serviceName"),
                                        "cluster_name": cluster_name,
                                        "cluster_arn": cluster_arn,
                                        "status": svc.get("status"),
                                        "desired_count": svc.get("desiredCount"),
                                        "running_count": svc.get("runningCount"),
                                        "pending_count": svc.get("pendingCount"),
                                        "launch_type": svc.get("launchType"),
                                        "task_definition": svc.get("taskDefinition"),
                                        "deployment": {
                                            "id": primary_deployment.get("id"),
                                            "status": primary_deployment.get("status"),
                                            "running_count": primary_deployment.get(
                                                "runningCount"
                                            ),
                                            "desired_count": primary_deployment.get(
                                                "desiredCount"
                                            ),
                                            "created_at": primary_deployment.get(
                                                "createdAt", ""
                                            ).isoformat()
                                            if isinstance(
                                                primary_deployment.get("createdAt"), datetime
                                            )
                                            else str(primary_deployment.get("createdAt", "")),
                                            "rollout_state": primary_deployment.get(
                                                "rolloutState"
                                            ),
                                        }
                                        if primary_deployment
                                        else None,
                                        "created_at": svc.get("createdAt", "").isoformat()
                                        if isinstance(svc.get("createdAt"), datetime)
                                        else str(svc.get("createdAt", "")),
                                    }
                                )

                except Exception as e:
                    errors.append(
                        f"Failed to fetch services for cluster {cluster_name}: {str(e)}"
                    )

        except Exception as e:
            errors.append(f"Failed to access ECS: {str(e)}")

        return {"services": services, "errors": errors}

    def _sync_lambda_functions(
        self,
        credentials: IntegrationCredentials,
        limit: int,
    ) -> dict[str, Any]:
        """Sync Lambda functions (synchronous, runs in thread pool)."""
        functions: list[dict[str, Any]] = []
        errors: list[str] = []

        try:
            session = self._get_boto3_session(credentials)
            lambda_client = session.client("lambda")

            # List functions with pagination
            paginator = lambda_client.get_paginator("list_functions")
            page_iterator = paginator.paginate(
                PaginationConfig={"MaxItems": limit, "PageSize": 50}
            )

            for page in page_iterator:
                for func in page.get("Functions", []):
                    functions.append(
                        {
                            "function_name": func.get("FunctionName"),
                            "function_arn": func.get("FunctionArn"),
                            "description": func.get("Description"),
                            "runtime": func.get("Runtime"),
                            "handler": func.get("Handler"),
                            "code_size": func.get("CodeSize"),
                            "memory_size": func.get("MemorySize"),
                            "timeout": func.get("Timeout"),
                            "last_modified": func.get("LastModified"),
                            "version": func.get("Version"),
                            "state": func.get("State"),
                            "state_reason": func.get("StateReason"),
                            "architectures": func.get("Architectures", []),
                            "package_type": func.get("PackageType"),
                            "environment_variables": list(
                                func.get("Environment", {}).get("Variables", {}).keys()
                            ),  # Only keys, not values
                        }
                    )

        except Exception as e:
            errors.append(f"Failed to access Lambda: {str(e)}")

        return {"functions": functions, "errors": errors}

    async def get_ecs_service(
        self,
        credentials: IntegrationCredentials,
        cluster: str,
        service: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific ECS service.

        Args:
            credentials: AWS credentials.
            cluster: Cluster name or ARN.
            service: Service name or ARN.

        Returns:
            Service details or None if not found.
        """
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(
                    executor,
                    self._get_ecs_service_sync,
                    credentials,
                    cluster,
                    service,
                )
        except Exception:
            return None

    def _get_ecs_service_sync(
        self,
        credentials: IntegrationCredentials,
        cluster: str,
        service: str,
    ) -> Optional[dict[str, Any]]:
        """Synchronous ECS service fetch."""
        try:
            session = self._get_boto3_session(credentials)
            ecs_client = session.client("ecs")

            response = ecs_client.describe_services(
                cluster=cluster,
                services=[service],
            )

            services = response.get("services", [])
            return services[0] if services else None

        except Exception:
            return None

    async def get_lambda_function(
        self,
        credentials: IntegrationCredentials,
        function_name: str,
    ) -> Optional[dict[str, Any]]:
        """Get a specific Lambda function.

        Args:
            credentials: AWS credentials.
            function_name: Function name or ARN.

        Returns:
            Function details or None if not found.
        """
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(
                    executor,
                    self._get_lambda_function_sync,
                    credentials,
                    function_name,
                )
        except Exception:
            return None

    def _get_lambda_function_sync(
        self,
        credentials: IntegrationCredentials,
        function_name: str,
    ) -> Optional[dict[str, Any]]:
        """Synchronous Lambda function fetch."""
        try:
            session = self._get_boto3_session(credentials)
            lambda_client = session.client("lambda")

            response = lambda_client.get_function(FunctionName=function_name)
            return response.get("Configuration")

        except Exception:
            return None

    async def invoke_lambda(
        self,
        credentials: IntegrationCredentials,
        function_name: str,
        payload: Optional[dict[str, Any]] = None,
        invocation_type: str = "RequestResponse",
    ) -> Optional[dict[str, Any]]:
        """Invoke a Lambda function.

        Args:
            credentials: AWS credentials.
            function_name: Function name or ARN.
            payload: Optional JSON payload for the function.
            invocation_type: "RequestResponse" (sync) or "Event" (async).

        Returns:
            Invocation result or None on failure.
        """
        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                return await loop.run_in_executor(
                    executor,
                    self._invoke_lambda_sync,
                    credentials,
                    function_name,
                    payload,
                    invocation_type,
                )
        except Exception:
            return None

    def _invoke_lambda_sync(
        self,
        credentials: IntegrationCredentials,
        function_name: str,
        payload: Optional[dict[str, Any]],
        invocation_type: str,
    ) -> Optional[dict[str, Any]]:
        """Synchronous Lambda invocation."""
        import json

        try:
            session = self._get_boto3_session(credentials)
            lambda_client = session.client("lambda")

            invoke_kwargs = {
                "FunctionName": function_name,
                "InvocationType": invocation_type,
            }

            if payload:
                invoke_kwargs["Payload"] = json.dumps(payload)

            response = lambda_client.invoke(**invoke_kwargs)

            result = {
                "status_code": response.get("StatusCode"),
                "function_error": response.get("FunctionError"),
                "executed_version": response.get("ExecutedVersion"),
            }

            # Read response payload
            if "Payload" in response:
                payload_data = response["Payload"].read()
                try:
                    result["payload"] = json.loads(payload_data)
                except json.JSONDecodeError:
                    result["payload"] = payload_data.decode("utf-8")

            return result

        except Exception:
            return None
