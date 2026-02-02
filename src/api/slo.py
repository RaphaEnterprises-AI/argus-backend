"""SLO/SLI API Endpoints.

Provides REST API for viewing SLO definitions and SLI compliance.

RAP-329: Define SLO/SLI Framework

Endpoints:
- GET /api/v1/slo/definitions - List all SLO definitions
- GET /api/v1/slo/definitions/{service} - Get SLOs for a service
- GET /api/v1/slo/compliance - Get overall compliance status
- GET /api/v1/slo/compliance/{service} - Get compliance for a service
- GET /api/v1/slo/error-budget/{service} - Get error budget status
- GET /api/v1/slo/critical - Get critical SLO status
"""

from datetime import datetime
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.context import require_organization_id
from src.slo import (
    SLORegistry,
    SLITracker,
    get_slo_registry,
    get_sli_tracker,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/slo", tags=["SLO"])


# =============================================================================
# Response Models
# =============================================================================


class SLODefinitionResponse(BaseModel):
    """Response model for a single SLO definition."""
    name: str
    service: str
    category: str
    target: float
    unit: str
    description: str
    alerting_threshold: float | None
    error_budget_percent: float
    critical: bool
    tags: list[str]


class SLODefinitionsListResponse(BaseModel):
    """Response model for listing SLO definitions."""
    total_slos: int
    critical_slos: int
    services: list[str]
    slos: list[SLODefinitionResponse]


class SLIMetricResponse(BaseModel):
    """Response model for an SLI measurement."""
    slo_name: str
    service: str
    category: str
    target: float
    current_value: float | None
    unit: str
    status: str
    target_met: bool
    deviation_percent: float | None
    measured_at: str
    window_hours: int
    error_message: str | None = None


class ErrorBudgetResponse(BaseModel):
    """Response model for error budget status."""
    slo_name: str
    service: str
    total_budget_percent: float
    consumed_percent: float
    remaining_percent: float
    burn_rate: float
    time_to_exhaustion_hours: float | None
    status: str
    window_days: int
    calculated_at: str


class ComplianceResponse(BaseModel):
    """Response model for SLO compliance."""
    service: str
    overall_status: str
    compliance_percent: float
    critical_violations: int
    warning_count: int
    total_slos: int
    metrics: list[SLIMetricResponse]
    error_budgets: list[ErrorBudgetResponse]
    calculated_at: str


class AllComplianceResponse(BaseModel):
    """Response model for all services compliance."""
    services: dict[str, ComplianceResponse]
    overall_healthy: int
    overall_warning: int
    overall_critical: int
    calculated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/definitions", response_model=SLODefinitionsListResponse)
async def list_slo_definitions(
    service: str | None = Query(None, description="Filter by service name"),
    category: str | None = Query(None, description="Filter by category"),
    critical_only: bool = Query(False, description="Only return critical SLOs"),
    _org_id: str = Depends(require_organization_id),
) -> SLODefinitionsListResponse:
    """List all SLO definitions.

    Optionally filter by service, category, or critical status.
    """
    registry = get_slo_registry()

    # Start with all SLOs
    if critical_only:
        slos = registry.get_critical()
    elif service:
        slos = registry.get_by_service(service)
    else:
        slos = registry.get_all()

    # Apply category filter
    if category:
        from src.slo.definitions import SLOCategory
        try:
            cat = SLOCategory(category)
            slos = [s for s in slos if s.category == cat]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category: {category}. Valid: availability, latency, error_rate, throughput, saturation"
            )

    return SLODefinitionsListResponse(
        total_slos=len(slos),
        critical_slos=sum(1 for s in slos if s.critical),
        services=list(set(s.service for s in slos)),
        slos=[
            SLODefinitionResponse(
                name=s.name,
                service=s.service,
                category=s.category.value,
                target=s.target,
                unit=s.unit,
                description=s.description,
                alerting_threshold=s.alerting_threshold,
                error_budget_percent=s.error_budget_percent,
                critical=s.critical,
                tags=s.tags,
            )
            for s in slos
        ],
    )


@router.get("/definitions/{service}", response_model=SLODefinitionsListResponse)
async def get_service_slo_definitions(
    service: str,
    _org_id: str = Depends(require_organization_id),
) -> SLODefinitionsListResponse:
    """Get SLO definitions for a specific service."""
    registry = get_slo_registry()
    slos = registry.get_by_service(service)

    if not slos:
        raise HTTPException(
            status_code=404,
            detail=f"No SLOs defined for service: {service}. Available services: {registry.get_services()}"
        )

    return SLODefinitionsListResponse(
        total_slos=len(slos),
        critical_slos=sum(1 for s in slos if s.critical),
        services=[service],
        slos=[
            SLODefinitionResponse(
                name=s.name,
                service=s.service,
                category=s.category.value,
                target=s.target,
                unit=s.unit,
                description=s.description,
                alerting_threshold=s.alerting_threshold,
                error_budget_percent=s.error_budget_percent,
                critical=s.critical,
                tags=s.tags,
            )
            for s in slos
        ],
    )


@router.get("/compliance", response_model=AllComplianceResponse)
async def get_all_compliance(
    window_hours: int = Query(24, ge=1, le=720, description="Time window in hours"),
    _org_id: str = Depends(require_organization_id),
) -> AllComplianceResponse:
    """Get SLO compliance status for all services.

    Returns compliance metrics, error budgets, and overall status for each service.
    """
    tracker = get_sli_tracker()
    all_status = await tracker.get_all_slo_status(window_hours)

    services = {}
    overall_healthy = 0
    overall_warning = 0
    overall_critical = 0

    for service_name, compliance in all_status.items():
        comp_dict = compliance.to_dict()

        # Convert to response model
        services[service_name] = ComplianceResponse(
            service=comp_dict["service"],
            overall_status=comp_dict["overall_status"],
            compliance_percent=comp_dict["compliance_percent"],
            critical_violations=comp_dict["critical_violations"],
            warning_count=comp_dict["warning_count"],
            total_slos=comp_dict["total_slos"],
            metrics=[SLIMetricResponse(**m) for m in comp_dict["metrics"]],
            error_budgets=[ErrorBudgetResponse(**b) for b in comp_dict["error_budgets"]],
            calculated_at=comp_dict["calculated_at"],
        )

        # Count overall status
        if comp_dict["overall_status"] == "healthy":
            overall_healthy += 1
        elif comp_dict["overall_status"] == "warning":
            overall_warning += 1
        elif comp_dict["overall_status"] == "critical":
            overall_critical += 1

    return AllComplianceResponse(
        services=services,
        overall_healthy=overall_healthy,
        overall_warning=overall_warning,
        overall_critical=overall_critical,
    )


@router.get("/compliance/{service}", response_model=ComplianceResponse)
async def get_service_compliance(
    service: str,
    window_hours: int = Query(24, ge=1, le=720, description="Time window in hours"),
    _org_id: str = Depends(require_organization_id),
) -> ComplianceResponse:
    """Get SLO compliance status for a specific service.

    Returns all SLI metrics, error budgets, and compliance percentage.
    """
    registry = get_slo_registry()
    if service not in registry.get_services():
        raise HTTPException(
            status_code=404,
            detail=f"No SLOs defined for service: {service}. Available services: {registry.get_services()}"
        )

    tracker = get_sli_tracker()
    compliance = await tracker.calculate_compliance(service, window_hours)
    comp_dict = compliance.to_dict()

    return ComplianceResponse(
        service=comp_dict["service"],
        overall_status=comp_dict["overall_status"],
        compliance_percent=comp_dict["compliance_percent"],
        critical_violations=comp_dict["critical_violations"],
        warning_count=comp_dict["warning_count"],
        total_slos=comp_dict["total_slos"],
        metrics=[SLIMetricResponse(**m) for m in comp_dict["metrics"]],
        error_budgets=[ErrorBudgetResponse(**b) for b in comp_dict["error_budgets"]],
        calculated_at=comp_dict["calculated_at"],
    )


@router.get("/error-budget/{service}")
async def get_error_budgets(
    service: str,
    window_days: int = Query(30, ge=1, le=90, description="Window in days"),
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get error budget status for a service.

    Shows budget consumption, burn rate, and time to exhaustion for
    availability SLOs.
    """
    registry = get_slo_registry()
    slos = registry.get_by_service(service)

    if not slos:
        raise HTTPException(
            status_code=404,
            detail=f"No SLOs defined for service: {service}"
        )

    # Filter to availability SLOs only
    from src.slo.definitions import SLOCategory
    availability_slos = [s for s in slos if s.category == SLOCategory.AVAILABILITY]

    if not availability_slos:
        return {
            "service": service,
            "message": "No availability SLOs defined for this service",
            "error_budgets": [],
        }

    tracker = get_sli_tracker()
    budgets = []

    for slo in availability_slos:
        budget = await tracker.calculate_error_budget(slo, window_days)
        budgets.append(budget.to_dict())

    return {
        "service": service,
        "window_days": window_days,
        "error_budgets": budgets,
        "calculated_at": datetime.utcnow().isoformat(),
    }


@router.get("/critical")
async def get_critical_slo_status(
    window_hours: int = Query(24, ge=1, le=720, description="Time window in hours"),
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """Get status of all critical SLOs.

    Critical SLOs are those marked as critical=True and typically include:
    - API Availability
    - Health Endpoint Availability
    - Core data layer availability

    This endpoint is useful for quick status checks and incident response.
    """
    tracker = get_sli_tracker()
    metrics = await tracker.get_critical_slo_status(window_hours)

    # Group by status
    healthy = [m for m in metrics if m.status.value == "healthy"]
    warning = [m for m in metrics if m.status.value == "warning"]
    critical = [m for m in metrics if m.status.value == "critical"]
    unknown = [m for m in metrics if m.status.value == "unknown"]

    return {
        "summary": {
            "total": len(metrics),
            "healthy": len(healthy),
            "warning": len(warning),
            "critical": len(critical),
            "unknown": len(unknown),
        },
        "overall_status": (
            "critical" if critical else
            "warning" if warning else
            "healthy" if healthy else
            "unknown"
        ),
        "metrics": [m.to_dict() for m in metrics],
        "window_hours": window_hours,
        "calculated_at": datetime.utcnow().isoformat(),
    }


@router.get("/services")
async def list_services(
    _org_id: str = Depends(require_organization_id),
) -> dict[str, Any]:
    """List all services with SLO definitions."""
    registry = get_slo_registry()
    services = registry.get_services()

    service_info = []
    for service in services:
        slos = registry.get_by_service(service)
        service_info.append({
            "name": service,
            "slo_count": len(slos),
            "critical_count": sum(1 for s in slos if s.critical),
            "categories": list(set(s.category.value for s in slos)),
        })

    return {
        "services": service_info,
        "total_services": len(services),
    }
