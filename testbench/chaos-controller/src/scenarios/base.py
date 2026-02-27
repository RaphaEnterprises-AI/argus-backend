import time
from datetime import datetime

from src.clients.argus import ArgusClient
from src.clients.chaos import ChaosClient
from src.models import ScenarioRunStatus, ScenarioStep, StepStatus


class BaseScenario:
    name: str = ""
    description: str = ""
    category: str = ""
    steps_count: int = 0

    def __init__(self, argus: ArgusClient, chaos: ChaosClient):
        self.argus = argus
        self.chaos = chaos

    async def execute(self, run: ScenarioRunStatus) -> None:
        raise NotImplementedError

    async def step(
        self, run: ScenarioRunStatus, name: str, description: str, fn
    ) -> dict | None:
        step = ScenarioStep(
            name=name,
            description=description,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        run.steps.append(step)
        start = time.perf_counter()
        try:
            result = await fn()
            step.status = StepStatus.PASSED
            step.duration_ms = (time.perf_counter() - start) * 1000
            step.completed_at = datetime.utcnow()
            return result
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.duration_ms = (time.perf_counter() - start) * 1000
            step.completed_at = datetime.utcnow()
            raise
