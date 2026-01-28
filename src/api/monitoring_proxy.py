"""Monitoring Proxy API.

Provides authenticated proxy endpoints for Grafana and Prometheus.
This ensures monitoring tools are only accessible through the dashboard
with proper authentication, not directly via public IPs.
"""

import os
from typing import Any

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from src.api.context import require_organization_id

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])

# Internal URLs (K8s ClusterIP services)
# These point to the kube-prometheus-stack deployed in the monitoring namespace
GRAFANA_INTERNAL_URL = os.environ.get(
    "GRAFANA_INTERNAL_URL",
    os.environ.get("GRAFANA_URL", "http://monitoring-grafana.monitoring.svc.cluster.local:80"),
)
PROMETHEUS_INTERNAL_URL = os.environ.get(
    "PROMETHEUS_INTERNAL_URL",
    os.environ.get("PROMETHEUS_URL", "http://monitoring-prometheus.monitoring.svc.cluster.local:9090"),
)
ALERTMANAGER_INTERNAL_URL = os.environ.get(
    "ALERTMANAGER_INTERNAL_URL",
    "http://monitoring-alertmanager.monitoring.svc.cluster.local:9093",
)

# Optional auth for Grafana (service account token)
GRAFANA_SERVICE_TOKEN = os.environ.get("GRAFANA_SERVICE_TOKEN", "")


async def _proxy_request(
    base_url: str,
    path: str,
    request: Request,
    auth_header: str | None = None,
) -> Response:
    """Proxy a request to an internal service."""
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    headers = {}
    if auth_header:
        headers["Authorization"] = auth_header

    # Forward safe headers
    for key in ["Accept", "Content-Type"]:
        if key in request.headers:
            headers[key] = request.headers[key]

    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            if request.method == "GET":
                response = await client.get(url, headers=headers, params=dict(request.query_params))
            elif request.method == "POST":
                body = await request.body()
                response = await client.post(url, headers=headers, content=body)
            else:
                response = await client.request(
                    request.method, url, headers=headers, content=await request.body()
                )

        # Return proxied response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers={
                "Content-Type": response.headers.get("Content-Type", "application/json"),
            },
        )

    except httpx.ConnectError as e:
        logger.warning("Failed to connect to monitoring service", url=url, error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"Monitoring service unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error("Proxy request failed", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Grafana Proxy Endpoints
# =============================================================================


@router.get("/grafana/health")
async def grafana_health(_org_id: str = Depends(require_organization_id)) -> dict[str, Any]:
    """Check Grafana health (authenticated)."""
    try:
        headers = {}
        if GRAFANA_SERVICE_TOKEN:
            headers["Authorization"] = f"Bearer {GRAFANA_SERVICE_TOKEN}"

        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{GRAFANA_INTERNAL_URL.rstrip('/')}/api/health",
                headers=headers,
            )
            response.raise_for_status()
            return {
                "status": "healthy",
                "grafana": response.json(),
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/grafana/dashboards")
async def list_grafana_dashboards(
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """List available Grafana dashboards."""
    try:
        headers = {}
        if GRAFANA_SERVICE_TOKEN:
            headers["Authorization"] = f"Bearer {GRAFANA_SERVICE_TOKEN}"

        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{GRAFANA_INTERNAL_URL.rstrip('/')}/api/search?type=dash-db",
                headers=headers,
            )
            response.raise_for_status()
            dashboards = response.json()

            return {
                "dashboards": [
                    {
                        "uid": d.get("uid"),
                        "title": d.get("title"),
                        "url": d.get("url"),
                        "tags": d.get("tags", []),
                    }
                    for d in dashboards
                ],
            }
    except Exception as e:
        logger.error("Failed to list Grafana dashboards", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/grafana/dashboard/{uid}")
async def get_grafana_dashboard(
    uid: str,
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get a specific Grafana dashboard by UID."""
    try:
        headers = {}
        if GRAFANA_SERVICE_TOKEN:
            headers["Authorization"] = f"Bearer {GRAFANA_SERVICE_TOKEN}"

        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{GRAFANA_INTERNAL_URL.rstrip('/')}/api/dashboards/uid/{uid}",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Dashboard not found")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Failed to get Grafana dashboard", uid=uid, error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/grafana/render/{path:path}")
async def proxy_grafana_render(
    path: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Proxy Grafana render requests for embedding panels."""
    auth = f"Bearer {GRAFANA_SERVICE_TOKEN}" if GRAFANA_SERVICE_TOKEN else None
    return await _proxy_request(GRAFANA_INTERNAL_URL, f"render/{path}", request, auth)


@router.api_route("/grafana/api/{path:path}", methods=["GET", "POST"])
async def proxy_grafana_api(
    path: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Proxy Grafana API requests."""
    auth = f"Bearer {GRAFANA_SERVICE_TOKEN}" if GRAFANA_SERVICE_TOKEN else None
    return await _proxy_request(GRAFANA_INTERNAL_URL, f"api/{path}", request, auth)


# =============================================================================
# Prometheus Proxy Endpoints
# =============================================================================


@router.get("/prometheus/health")
async def prometheus_health(_org_id: str = Depends(require_organization_id)) -> dict[str, Any]:
    """Check Prometheus health (authenticated)."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(f"{PROMETHEUS_INTERNAL_URL.rstrip('/')}/-/healthy")
            response.raise_for_status()
            return {
                "status": "healthy",
                "prometheus": "Prometheus is Healthy.",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/prometheus/query")
async def prometheus_query(
    query: str = Query(..., description="PromQL query"),
    time: str | None = Query(None, description="Evaluation timestamp"),
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Execute a Prometheus instant query."""
    try:
        params = {"query": query}
        if time:
            params["time"] = time

        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.get(
                f"{PROMETHEUS_INTERNAL_URL.rstrip('/')}/api/v1/query",
                params=params,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("Prometheus query failed", query=query, error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/prometheus/query_range")
async def prometheus_query_range(
    query: str = Query(..., description="PromQL query"),
    start: str = Query(..., description="Start timestamp"),
    end: str = Query(..., description="End timestamp"),
    step: str = Query("60s", description="Query resolution step"),
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Execute a Prometheus range query."""
    try:
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step,
        }

        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.get(
                f"{PROMETHEUS_INTERNAL_URL.rstrip('/')}/api/v1/query_range",
                params=params,
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("Prometheus range query failed", query=query, error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/prometheus/targets")
async def prometheus_targets(
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get Prometheus scrape targets status."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{PROMETHEUS_INTERNAL_URL.rstrip('/')}/api/v1/targets",
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("Failed to get Prometheus targets", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/prometheus/alerts")
async def prometheus_alerts(
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get active Prometheus alerts."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{PROMETHEUS_INTERNAL_URL.rstrip('/')}/api/v1/alerts",
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("Failed to get Prometheus alerts", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.api_route("/prometheus/api/v1/{path:path}", methods=["GET", "POST"])
async def proxy_prometheus_api(
    path: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Proxy Prometheus API requests."""
    return await _proxy_request(PROMETHEUS_INTERNAL_URL, f"api/v1/{path}", request)


# =============================================================================
# AlertManager Proxy Endpoints
# =============================================================================


@router.get("/alertmanager/health")
async def alertmanager_health(_org_id: str = Depends(require_organization_id)) -> dict[str, Any]:
    """Check AlertManager health (authenticated)."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(f"{ALERTMANAGER_INTERNAL_URL.rstrip('/')}/-/healthy")
            response.raise_for_status()
            return {
                "status": "healthy",
                "alertmanager": "AlertManager is Healthy.",
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/alertmanager/alerts")
async def alertmanager_alerts(
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get active alerts from AlertManager."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{ALERTMANAGER_INTERNAL_URL.rstrip('/')}/api/v2/alerts",
            )
            response.raise_for_status()
            return {"alerts": response.json()}
    except Exception as e:
        logger.error("Failed to get AlertManager alerts", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/alertmanager/silences")
async def alertmanager_silences(
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get active silences from AlertManager."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{ALERTMANAGER_INTERNAL_URL.rstrip('/')}/api/v2/silences",
            )
            response.raise_for_status()
            return {"silences": response.json()}
    except Exception as e:
        logger.error("Failed to get AlertManager silences", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/alertmanager/silences")
async def create_alertmanager_silence(
    request: Request,
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Create a new silence in AlertManager."""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.post(
                f"{ALERTMANAGER_INTERNAL_URL.rstrip('/')}/api/v2/silences",
                json=body,
            )
            response.raise_for_status()
            return {"silence": response.json()}
    except Exception as e:
        logger.error("Failed to create AlertManager silence", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.delete("/alertmanager/silence/{silence_id}")
async def delete_alertmanager_silence(
    silence_id: str,
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Delete a silence from AlertManager."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.delete(
                f"{ALERTMANAGER_INTERNAL_URL.rstrip('/')}/api/v2/silence/{silence_id}",
            )
            response.raise_for_status()
            return {"status": "deleted", "silence_id": silence_id}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="Silence not found")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("Failed to delete AlertManager silence", silence_id=silence_id, error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/alertmanager/receivers")
async def alertmanager_receivers(
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get configured receivers from AlertManager."""
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(
                f"{ALERTMANAGER_INTERNAL_URL.rstrip('/')}/api/v2/receivers",
            )
            response.raise_for_status()
            return {"receivers": response.json()}
    except Exception as e:
        logger.error("Failed to get AlertManager receivers", error=str(e))
        raise HTTPException(status_code=503, detail=str(e))


@router.api_route("/alertmanager/api/v2/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_alertmanager_api(
    path: str,
    request: Request,
    _org_id: str = Depends(require_organization_id),
):
    """Proxy AlertManager API requests."""
    return await _proxy_request(ALERTMANAGER_INTERNAL_URL, f"api/v2/{path}", request)
