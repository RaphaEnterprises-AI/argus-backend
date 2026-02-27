from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ScenarioRunRequest(BaseModel):
    scenario: str
    params: dict | None = None


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScenarioStep(BaseModel):
    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None


class ScenarioRunStatus(BaseModel):
    run_id: str
    scenario: str
    status: str  # pending, running, passed, failed, cancelled
    steps: list[ScenarioStep] = []
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None


class ScenarioInfo(BaseModel):
    name: str
    description: str
    category: str
    steps_count: int


class ResultsSummary(BaseModel):
    total_runs: int
    passed: int
    failed: int
    pass_rate: float
    last_run: datetime | None = None
