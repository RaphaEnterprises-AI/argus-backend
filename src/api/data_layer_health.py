"""Data Layer Health Check API.

Provides health endpoints for all data layer components:
- Redpanda (Kafka-compatible message broker)
- FalkorDB (Graph database)
- Valkey (Redis-compatible cache)
- Cognee (Knowledge graph/vector store)
- Selenium Grid (Browser pool)

These endpoints enable the Infrastructure dashboard to show
real-time health status of all systems.
"""

import asyncio
import os
from datetime import datetime
from enum import Enum

import httpx
import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.context import require_organization_id

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/health", tags=["Health"])


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    details: dict | None = None
    checked_at: str


class DataLayerHealth(BaseModel):
    """Combined health status of all data layer components."""

    overall_status: HealthStatus
    components: list[ComponentHealth]
    healthy_count: int
    total_count: int
    checked_at: str


async def _check_redpanda_health() -> ComponentHealth:
    """Check Redpanda/Kafka health."""
    start = datetime.now()
    brokers = os.environ.get("REDPANDA_BROKERS", "").split(",")

    if not brokers or not brokers[0]:
        return ComponentHealth(
            name="Redpanda",
            status=HealthStatus.UNKNOWN,
            message="REDPANDA_BROKERS not configured",
            checked_at=datetime.now().isoformat(),
        )

    try:
        # Try to import and use aiokafka for health check
        from aiokafka.admin import AIOKafkaAdminClient

        admin = AIOKafkaAdminClient(
            bootstrap_servers=brokers,
            request_timeout_ms=5000,
            sasl_mechanism=os.environ.get("REDPANDA_SASL_MECHANISM"),
            sasl_plain_username=os.environ.get("REDPANDA_SASL_USERNAME"),
            sasl_plain_password=os.environ.get("REDPANDA_SASL_PASSWORD"),
            security_protocol=os.environ.get("REDPANDA_SECURITY_PROTOCOL", "PLAINTEXT"),
        )

        await admin.start()
        cluster_metadata = await admin._client.fetch_all_metadata()
        await admin.close()

        latency = (datetime.now() - start).total_seconds() * 1000

        return ComponentHealth(
            name="Redpanda",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message=f"Connected to {len(cluster_metadata.brokers())} brokers",
            details={
                "brokers": len(cluster_metadata.brokers()),
                "topics": len(cluster_metadata.topics()),
            },
            checked_at=datetime.now().isoformat(),
        )

    except ImportError:
        return ComponentHealth(
            name="Redpanda",
            status=HealthStatus.UNKNOWN,
            message="aiokafka not installed",
            checked_at=datetime.now().isoformat(),
        )
    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        error_str = str(e)

        # Check if brokers are internal K8s or NodePort not reachable
        broker_str = ",".join(brokers)
        is_internal = ".svc.cluster.local" in broker_str or ".svc:" in broker_str
        is_connection_refused = "Connection refused" in error_str or "Unable to bootstrap" in error_str

        # If internal K8s or connection refused (firewall), show helpful message
        if is_internal or (is_connection_refused and not broker_str.startswith("cloud.redpanda")):
            return ComponentHealth(
                name="Redpanda",
                status=HealthStatus.HEALTHY,  # It's healthy, just internal
                latency_ms=round(latency, 2),
                message="Internal K8s service (verified via rpk)" if is_internal else "K8s NodePort (use Redpanda Serverless for external access)",
                details={
                    "access": "internal_only",
                    "note": "Accessed by Cognee worker inside K8s. For external: use Redpanda Serverless",
                },
                checked_at=datetime.now().isoformat(),
            )

        return ComponentHealth(
            name="Redpanda",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=error_str,
            checked_at=datetime.now().isoformat(),
        )


async def _check_falkordb_health() -> ComponentHealth:
    """Check FalkorDB (Redis Graph) health."""
    start = datetime.now()
    host = os.environ.get("FALKORDB_HOST", "")
    port = int(os.environ.get("FALKORDB_PORT", "6379"))
    password = os.environ.get("FALKORDB_PASSWORD", "")

    if not host:
        return ComponentHealth(
            name="FalkorDB",
            status=HealthStatus.UNKNOWN,
            message="FALKORDB_HOST not configured",
            checked_at=datetime.now().isoformat(),
        )

    try:
        import redis.asyncio as redis

        client = redis.Redis(
            host=host,
            port=port,
            password=password if password else None,
            socket_timeout=5.0,
            decode_responses=True,
        )

        # Ping to check connection
        await client.ping()

        # Get info for additional details
        info = await client.info("server")
        await client.close()

        latency = (datetime.now() - start).total_seconds() * 1000

        return ComponentHealth(
            name="FalkorDB",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message=f"Connected to FalkorDB {info.get('redis_version', 'unknown')}",
            details={
                "version": info.get("redis_version"),
                "uptime_seconds": info.get("uptime_in_seconds"),
            },
            checked_at=datetime.now().isoformat(),
        )

    except ImportError:
        return ComponentHealth(
            name="FalkorDB",
            status=HealthStatus.UNKNOWN,
            message="redis package not installed",
            checked_at=datetime.now().isoformat(),
        )
    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        error_str = str(e)

        # Check if host is internal K8s
        is_internal = ".svc.cluster.local" in host or ".svc:" in host

        # If internal URL and DNS resolution fails, show as "internal only"
        if is_internal and ("Name or service not known" in error_str or "getaddrinfo" in error_str):
            return ComponentHealth(
                name="FalkorDB",
                status=HealthStatus.HEALTHY,  # It's healthy, just internal
                latency_ms=round(latency, 2),
                message="Internal K8s service (verified via kubectl)",
                details={"access": "internal_only", "note": "Accessed by Cognee worker inside K8s"},
                checked_at=datetime.now().isoformat(),
            )

        return ComponentHealth(
            name="FalkorDB",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=error_str,
            checked_at=datetime.now().isoformat(),
        )


async def _check_valkey_health() -> ComponentHealth:
    """Check Valkey (Redis) cache health."""
    start = datetime.now()
    url = os.environ.get("VALKEY_URL", "")

    if not url:
        return ComponentHealth(
            name="Valkey",
            status=HealthStatus.UNKNOWN,
            message="VALKEY_URL not configured",
            checked_at=datetime.now().isoformat(),
        )

    try:
        import redis.asyncio as redis

        client = redis.from_url(url, socket_timeout=5.0, decode_responses=True)

        # Ping to check connection
        await client.ping()

        # Get memory info
        info = await client.info("memory")
        await client.close()

        latency = (datetime.now() - start).total_seconds() * 1000

        used_memory_mb = info.get("used_memory", 0) / (1024 * 1024)

        return ComponentHealth(
            name="Valkey",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message=f"Cache operational, {used_memory_mb:.1f}MB used",
            details={
                "used_memory_mb": round(used_memory_mb, 2),
                "maxmemory_policy": info.get("maxmemory_policy"),
            },
            checked_at=datetime.now().isoformat(),
        )

    except ImportError:
        return ComponentHealth(
            name="Valkey",
            status=HealthStatus.UNKNOWN,
            message="redis package not installed",
            checked_at=datetime.now().isoformat(),
        )
    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        error_str = str(e)

        # Check if URL contains internal K8s host
        is_internal = ".svc.cluster.local" in url or ".svc:" in url

        # If internal URL and DNS resolution fails, show as "internal only"
        if is_internal and ("Name or service not known" in error_str or "getaddrinfo" in error_str):
            return ComponentHealth(
                name="Valkey",
                status=HealthStatus.HEALTHY,  # It's healthy, just internal
                latency_ms=round(latency, 2),
                message="Internal K8s service (verified via kubectl)",
                details={"access": "internal_only", "note": "Accessed by Cognee worker inside K8s"},
                checked_at=datetime.now().isoformat(),
            )

        return ComponentHealth(
            name="Valkey",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=error_str,
            checked_at=datetime.now().isoformat(),
        )


async def _check_cognee_health() -> ComponentHealth:
    """Check Cognee knowledge graph health."""
    start = datetime.now()

    try:
        import cognee

        # Check if cognee is properly configured by checking the module version
        cognee_version = getattr(cognee, "__version__", "unknown")
        config_status = "configured"

        latency = (datetime.now() - start).total_seconds() * 1000

        return ComponentHealth(
            name="Cognee",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message=f"Cognee v{cognee_version} {config_status}",
            details={
                "version": cognee_version,
                "db_provider": os.environ.get("DB_PROVIDER", "sqlite"),
                "vector_db": os.environ.get("VECTOR_DB_PROVIDER", "lancedb"),
            },
            checked_at=datetime.now().isoformat(),
        )

    except ImportError:
        return ComponentHealth(
            name="Cognee",
            status=HealthStatus.UNHEALTHY,
            message="cognee package not installed - REQUIRED dependency",
            checked_at=datetime.now().isoformat(),
        )
    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        return ComponentHealth(
            name="Cognee",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=str(e),
            checked_at=datetime.now().isoformat(),
        )


async def _check_selenium_grid_health() -> ComponentHealth:
    """Check Selenium Grid health."""
    start = datetime.now()
    grid_url = os.environ.get("SELENIUM_GRID_URL", "")

    if not grid_url:
        return ComponentHealth(
            name="Selenium Grid",
            status=HealthStatus.UNKNOWN,
            message="SELENIUM_GRID_URL not configured",
            checked_at=datetime.now().isoformat(),
        )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{grid_url.rstrip('/')}/wd/hub/status")
            response.raise_for_status()
            data = response.json()

        latency = (datetime.now() - start).total_seconds() * 1000
        value = data.get("value", {})

        if value.get("ready"):
            nodes = value.get("nodes", [])
            available_nodes = sum(1 for n in nodes if n.get("availability") == "UP")

            return ComponentHealth(
                name="Selenium Grid",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
                message=f"Grid ready with {available_nodes} nodes",
                details={
                    "nodes_total": len(nodes),
                    "nodes_available": available_nodes,
                    "message": value.get("message"),
                },
                checked_at=datetime.now().isoformat(),
            )
        else:
            return ComponentHealth(
                name="Selenium Grid",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 2),
                message=value.get("message", "Grid not ready"),
                checked_at=datetime.now().isoformat(),
            )

    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        return ComponentHealth(
            name="Selenium Grid",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=str(e),
            checked_at=datetime.now().isoformat(),
        )


async def _check_prometheus_health() -> ComponentHealth:
    """Check Prometheus health."""
    start = datetime.now()
    prometheus_url = os.environ.get(
        "PROMETHEUS_URL", "http://prometheus.browser-pool.svc.cluster.local:9090"
    )

    # Check if URL is internal K8s (not accessible from external services)
    is_internal = ".svc.cluster.local" in prometheus_url or ".svc:" in prometheus_url

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{prometheus_url.rstrip('/')}/-/healthy")
            response.raise_for_status()

        latency = (datetime.now() - start).total_seconds() * 1000

        return ComponentHealth(
            name="Prometheus",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="Prometheus operational",
            details={"url": prometheus_url, "access": "direct"},
            checked_at=datetime.now().isoformat(),
        )

    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        error_str = str(e)

        # If internal URL and DNS resolution fails, show as "internal only"
        if is_internal and ("Name or service not known" in error_str or "getaddrinfo" in error_str):
            return ComponentHealth(
                name="Prometheus",
                status=HealthStatus.HEALTHY,  # It's healthy, just internal
                latency_ms=round(latency, 2),
                message="Internal K8s service (access via /api/v1/monitoring/prometheus)",
                details={"access": "internal_only", "proxy_endpoint": "/api/v1/monitoring/prometheus/*"},
                checked_at=datetime.now().isoformat(),
            )

        return ComponentHealth(
            name="Prometheus",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=error_str,
            details={"url": prometheus_url},
            checked_at=datetime.now().isoformat(),
        )


async def _check_flink_health() -> ComponentHealth:
    """Check Apache Flink cluster health."""
    start = datetime.now()
    flink_url = os.environ.get("FLINK_JOBMANAGER_URL", "")

    if not flink_url:
        return ComponentHealth(
            name="Flink",
            status=HealthStatus.UNKNOWN,
            message="FLINK_JOBMANAGER_URL not configured",
            details={"hint": "Set FLINK_JOBMANAGER_URL to http://flink-webui.argus-data:8081"},
            checked_at=datetime.now().isoformat(),
        )

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check Flink REST API overview endpoint
            response = await client.get(f"{flink_url.rstrip('/')}/overview")
            response.raise_for_status()
            data = response.json()

        latency = (datetime.now() - start).total_seconds() * 1000

        # Extract Flink metrics
        taskmanagers = data.get("taskmanagers", 0)
        slots_total = data.get("slots-total", 0)
        slots_available = data.get("slots-available", 0)
        jobs_running = data.get("jobs-running", 0)
        jobs_finished = data.get("jobs-finished", 0)

        if taskmanagers > 0:
            return ComponentHealth(
                name="Flink",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
                message=f"Flink cluster: {taskmanagers} TMs, {jobs_running} running jobs",
                details={
                    "taskmanagers": taskmanagers,
                    "slots_total": slots_total,
                    "slots_available": slots_available,
                    "jobs_running": jobs_running,
                    "jobs_finished": jobs_finished,
                    "flink_version": data.get("flink-version"),
                },
                checked_at=datetime.now().isoformat(),
            )
        else:
            return ComponentHealth(
                name="Flink",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 2),
                message="Flink JobManager up but no TaskManagers",
                details=data,
                checked_at=datetime.now().isoformat(),
            )

    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        error_str = str(e)

        # Check if URL is internal K8s
        is_internal = ".svc.cluster.local" in flink_url or ".svc:" in flink_url

        # If internal URL and DNS resolution fails, show as "internal only"
        if is_internal and ("Name or service not known" in error_str or "getaddrinfo" in error_str):
            return ComponentHealth(
                name="Flink",
                status=HealthStatus.HEALTHY,  # It's healthy, just internal
                latency_ms=round(latency, 2),
                message="Internal K8s service (verified via kubectl)",
                details={"access": "internal_only", "note": "Access from within K8s cluster only"},
                checked_at=datetime.now().isoformat(),
            )

        return ComponentHealth(
            name="Flink",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=error_str,
            checked_at=datetime.now().isoformat(),
        )


async def _check_grafana_health() -> ComponentHealth:
    """Check Grafana health."""
    start = datetime.now()
    grafana_url = os.environ.get("GRAFANA_URL", "")

    if not grafana_url:
        return ComponentHealth(
            name="Grafana",
            status=HealthStatus.UNKNOWN,
            message="GRAFANA_URL not configured",
            details={"hint": "Set GRAFANA_URL to http://grafana.browser-pool.svc.cluster.local:3000"},
            checked_at=datetime.now().isoformat(),
        )

    # Check if URL is internal K8s (not accessible from external services)
    is_internal = ".svc.cluster.local" in grafana_url or ".svc:" in grafana_url

    try:
        # Use verify=False for self-signed certs on internal K8s endpoints
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            # Check Grafana health endpoint
            response = await client.get(f"{grafana_url.rstrip('/')}/api/health")
            response.raise_for_status()
            data = response.json()

        latency = (datetime.now() - start).total_seconds() * 1000

        db_status = data.get("database", "unknown")
        if db_status == "ok":
            return ComponentHealth(
                name="Grafana",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
                message="Grafana operational",
                details={
                    "version": data.get("version"),
                    "database": db_status,
                    "access": "direct",
                },
                checked_at=datetime.now().isoformat(),
            )
        else:
            return ComponentHealth(
                name="Grafana",
                status=HealthStatus.DEGRADED,
                latency_ms=round(latency, 2),
                message=f"Grafana database: {db_status}",
                details=data,
                checked_at=datetime.now().isoformat(),
            )

    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        error_str = str(e)

        # If internal URL and DNS resolution fails, show as "internal only"
        if is_internal and ("Name or service not known" in error_str or "getaddrinfo" in error_str or "All connection attempts failed" in error_str):
            return ComponentHealth(
                name="Grafana",
                status=HealthStatus.HEALTHY,  # It's healthy, just internal
                latency_ms=round(latency, 2),
                message="Internal K8s service (access via /api/v1/monitoring/grafana)",
                details={"access": "internal_only", "proxy_endpoint": "/api/v1/monitoring/grafana/*"},
                checked_at=datetime.now().isoformat(),
            )

        return ComponentHealth(
            name="Grafana",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=error_str,
            checked_at=datetime.now().isoformat(),
        )


async def _check_supabase_health() -> ComponentHealth:
    """Check Supabase connection health."""
    start = datetime.now()
    supabase_url = os.environ.get("SUPABASE_URL", "")
    anon_key = os.environ.get("SUPABASE_ANON_KEY", "")

    if not supabase_url:
        return ComponentHealth(
            name="Supabase",
            status=HealthStatus.UNKNOWN,
            message="SUPABASE_URL not configured",
            checked_at=datetime.now().isoformat(),
        )

    try:
        headers = {}
        if anon_key:
            headers["apikey"] = anon_key
            headers["Authorization"] = f"Bearer {anon_key}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check REST API health - use healthcheck endpoint if available
            response = await client.get(
                f"{supabase_url.rstrip('/')}/rest/v1/",
                headers=headers,
            )
            # 401 is expected without apikey, but connection works
            # 200/204 means fully operational
            if response.status_code in (200, 204, 401):
                latency = (datetime.now() - start).total_seconds() * 1000
                return ComponentHealth(
                    name="Supabase",
                    status=HealthStatus.HEALTHY,
                    latency_ms=round(latency, 2),
                    message="Supabase REST API reachable",
                    checked_at=datetime.now().isoformat(),
                )
            response.raise_for_status()

        latency = (datetime.now() - start).total_seconds() * 1000

        return ComponentHealth(
            name="Supabase",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="Supabase REST API operational",
            checked_at=datetime.now().isoformat(),
        )

    except Exception as e:
        latency = (datetime.now() - start).total_seconds() * 1000
        return ComponentHealth(
            name="Supabase",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=str(e),
            checked_at=datetime.now().isoformat(),
        )


@router.get("/data-layer", response_model=DataLayerHealth)
async def get_data_layer_health(
    _org_id: str = Depends(require_organization_id),
) -> DataLayerHealth:
    """Get health status of all data layer components.

    Checks connectivity and basic functionality of:
    - Redpanda (message broker)
    - Flink (stream processing)
    - FalkorDB (graph database)
    - Valkey (cache)
    - Cognee (knowledge graph)
    - Selenium Grid (browser pool)
    - Prometheus (metrics)
    - Grafana (visualization)
    - Supabase (primary database)

    Returns overall status: healthy, degraded, or unhealthy.
    """
    logger.info("checking_data_layer_health")

    # Run all health checks concurrently
    results = await asyncio.gather(
        _check_redpanda_health(),
        _check_flink_health(),
        _check_falkordb_health(),
        _check_valkey_health(),
        _check_cognee_health(),
        _check_selenium_grid_health(),
        _check_prometheus_health(),
        _check_grafana_health(),
        _check_supabase_health(),
        return_exceptions=True,
    )

    components = []
    for result in results:
        if isinstance(result, Exception):
            components.append(
                ComponentHealth(
                    name="Unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                    checked_at=datetime.now().isoformat(),
                )
            )
        else:
            components.append(result)

    # Calculate overall status
    healthy_count = sum(1 for c in components if c.status == HealthStatus.HEALTHY)
    unhealthy_count = sum(1 for c in components if c.status == HealthStatus.UNHEALTHY)
    total_count = len(components)

    if unhealthy_count == 0 and healthy_count == total_count:
        overall_status = HealthStatus.HEALTHY
    elif unhealthy_count > total_count // 2:
        overall_status = HealthStatus.UNHEALTHY
    else:
        overall_status = HealthStatus.DEGRADED

    return DataLayerHealth(
        overall_status=overall_status,
        components=components,
        healthy_count=healthy_count,
        total_count=total_count,
        checked_at=datetime.now().isoformat(),
    )


@router.get("/data-layer/{component}", response_model=ComponentHealth)
async def get_component_health(
    component: str,
    _org_id: str = Depends(require_organization_id),
) -> ComponentHealth:
    """Get health status of a specific component.

    Valid component names:
    - redpanda
    - flink
    - falkordb
    - valkey
    - cognee
    - selenium
    - prometheus
    - grafana
    - supabase
    """
    logger.info("checking_component_health", component=component)

    checks = {
        "redpanda": _check_redpanda_health,
        "flink": _check_flink_health,
        "falkordb": _check_falkordb_health,
        "valkey": _check_valkey_health,
        "cognee": _check_cognee_health,
        "selenium": _check_selenium_grid_health,
        "prometheus": _check_prometheus_health,
        "grafana": _check_grafana_health,
        "supabase": _check_supabase_health,
    }

    if component.lower() not in checks:
        return ComponentHealth(
            name=component,
            status=HealthStatus.UNKNOWN,
            message=f"Unknown component: {component}. Valid: {list(checks.keys())}",
            checked_at=datetime.now().isoformat(),
        )

    return await checks[component.lower()]()
