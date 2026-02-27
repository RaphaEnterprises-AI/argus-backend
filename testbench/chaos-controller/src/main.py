import asyncio
import json
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.clients.argus import ArgusClient
from src.clients.chaos import ChaosClient
from src.config import settings
from src.models import (
    ResultsSummary,
    ScenarioInfo,
    ScenarioRunRequest,
    ScenarioRunStatus,
)

# In-memory storage for runs
runs: dict[str, ScenarioRunStatus] = {}
argus_client = ArgusClient()
chaos_client = ChaosClient()

# Paths that don't require authentication
PUBLIC_PATHS = {"/health", "/openapi.json", "/docs", "/redoc"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Argus Testbench Chaos Controller",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Require bearer token for all endpoints except health."""
    if not settings.secret or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})

    token = auth[7:]
    if not secrets.compare_digest(token, settings.secret):
        return JSONResponse(status_code=401, content={"detail": "Invalid bearer token"})

    return await call_next(request)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "chaos-controller",
        "total_runs": len(runs),
        "active_runs": sum(
            1 for r in runs.values() if r.status == "running"
        ),
    }


@app.get("/scenarios", response_model=list[ScenarioInfo])
async def list_scenarios():
    from src.scenarios.registry import SCENARIOS

    return [
        ScenarioInfo(
            name=name,
            description=cls.description,
            category=cls.category,
            steps_count=cls.steps_count,
        )
        for name, cls in SCENARIOS.items()
    ]


@app.post("/scenarios/run")
async def run_scenario(request: ScenarioRunRequest):
    from src.scenarios.registry import SCENARIOS

    if request.scenario not in SCENARIOS:
        raise HTTPException(
            status_code=404,
            detail=f"Scenario '{request.scenario}' not found. Available: {list(SCENARIOS.keys())}",
        )

    run_id = str(uuid.uuid4())
    run_status = ScenarioRunStatus(
        run_id=run_id,
        scenario=request.scenario,
        status="pending",
        started_at=datetime.utcnow(),
    )
    runs[run_id] = run_status

    # Launch scenario in background
    scenario_cls = SCENARIOS[request.scenario]
    scenario = scenario_cls(argus=argus_client, chaos=chaos_client)

    async def execute():
        run_status.status = "running"
        try:
            await scenario.execute(run_status)
            run_status.status = "passed"
        except Exception as e:
            run_status.status = "failed"
            run_status.error = str(e)
        finally:
            run_status.completed_at = datetime.utcnow()

    asyncio.create_task(execute())

    return {
        "run_id": run_id,
        "scenario": request.scenario,
        "status": "running",
    }


@app.get("/scenarios/{run_id}/status", response_model=ScenarioRunStatus)
async def scenario_status(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return runs[run_id]


@app.get("/scenarios/{run_id}/stream")
async def scenario_stream(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        run = runs[run_id]
        last_step_count = 0
        while run.status in ("pending", "running"):
            if len(run.steps) > last_step_count:
                for step in run.steps[last_step_count:]:
                    yield f"data: {json.dumps(step.model_dump(mode='json'))}\n\n"
                last_step_count = len(run.steps)
            await asyncio.sleep(1)
        # Send final status
        yield f"data: {json.dumps({'status': run.status, 'error': run.error})}\n\n"

    return StreamingResponse(
        event_generator(), media_type="text/event-stream"
    )


@app.post("/scenarios/{run_id}/cancel")
async def cancel_scenario(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    runs[run_id].status = "cancelled"
    return {"run_id": run_id, "status": "cancelled"}


# Direct chaos control (proxy to chaos app)
@app.post("/chaos/{app}/enable-bug/{bug_id}")
async def enable_bug(app: str, bug_id: str):
    client = _get_chaos_client(app)
    return await client.enable_bug(bug_id)


@app.post("/chaos/{app}/disable-bug/{bug_id}")
async def disable_bug(app: str, bug_id: str):
    client = _get_chaos_client(app)
    return await client.disable_bug(bug_id)


@app.post("/chaos/{app}/reset")
async def reset_bugs(app: str):
    client = _get_chaos_client(app)
    return await client.reset_all()


# Results
@app.get("/results")
async def list_results():
    return [
        r.model_dump(mode="json")
        for r in sorted(
            runs.values(),
            key=lambda x: x.started_at or datetime.min,
            reverse=True,
        )
    ]


@app.get("/results/summary", response_model=ResultsSummary)
async def results_summary():
    completed = [
        r for r in runs.values() if r.status in ("passed", "failed")
    ]
    passed = sum(1 for r in completed if r.status == "passed")
    return ResultsSummary(
        total_runs=len(completed),
        passed=passed,
        failed=len(completed) - passed,
        pass_rate=passed / len(completed) if completed else 0,
        last_run=max(
            (r.completed_at for r in completed), default=None
        ),
    )


@app.get("/results/{run_id}")
async def get_result(run_id: str):
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return runs[run_id].model_dump(mode="json")


def _get_chaos_client(app: str) -> ChaosClient:
    urls = {
        "chaos-app": settings.chaos_app_url,
    }
    if app not in urls:
        raise HTTPException(status_code=404, detail=f"Unknown app: {app}")
    return ChaosClient(base_url=urls[app])
