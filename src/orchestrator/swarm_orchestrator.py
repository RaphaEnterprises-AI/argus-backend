# src/orchestrator/swarm_orchestrator.py
"""Swarm Orchestrator — scatter-gather pattern for concurrent agent execution.

Spawns N worker agents concurrently on a task, streams AG-UI events for
real-time dashboard visualization, and merges results via consensus.

Three swarm modes:
- full_crawl: 5-8 agents analyze entire app
- targeted_blitz: 3-5 agents test a specific flow
- pr_analysis: 4-6 agents analyze a PR diff

Uses:
- SwarmThrottler for resource management (3-layer)
- AGUIEmitter for real-time SSE streaming
- ConfidenceWeighted consensus for result merging
- Existing agents from src/agents/
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.streaming.agui_emitter import AGUIEmitter, create_emitter, remove_emitter
from src.streaming.agui_events import (
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StateDeltaEvent,
    StepFinishedEvent,
    StepStartedEvent,
)

from .swarm_throttler import get_swarm_throttler

logger = structlog.get_logger()


class SwarmMode(str, Enum):
    """Available swarm execution modes."""

    FULL_CRAWL = "full_crawl"
    TARGETED_BLITZ = "targeted_blitz"
    PR_ANALYSIS = "pr_analysis"


# Agent types for each swarm mode
SWARM_MODE_AGENTS: dict[SwarmMode, list[str]] = {
    SwarmMode.FULL_CRAWL: [
        "code_analyzer",
        "auto_discovery",
        "ui_tester",
        "api_tester",
        "security_scanner",
        "accessibility_checker",
        "performance_analyzer",
        "visual_ai",
    ],
    SwarmMode.TARGETED_BLITZ: [
        "code_analyzer",
        "ui_tester",
        "api_tester",
        "self_healer",
    ],
    SwarmMode.PR_ANALYSIS: [
        "code_analyzer",
        "test_impact_analyzer",
        "smart_test_selector",
        "security_scanner",
        "mr_analyzer",
        "flaky_detector",
    ],
}


@dataclass
class WorkerResult:
    """Result from a single swarm worker."""

    agent_id: str
    agent_type: str
    success: bool
    duration_ms: float
    cost_usd: float
    findings: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    error: str | None = None


@dataclass
class SwarmResult:
    """Aggregated result from a swarm execution."""

    swarm_id: str
    mode: SwarmMode
    success: bool
    total_duration_ms: float
    total_cost_usd: float
    worker_results: list[WorkerResult] = field(default_factory=list)
    consensus_score: float = 0.0
    summary: str = ""


@dataclass
class SwarmConfig:
    """Configuration for a swarm run."""

    mode: SwarmMode
    org_id: str
    project_id: str
    user_id: str
    target_url: str | None = None
    target_flow: str | None = None
    pr_number: int | None = None
    changed_files: list[dict] | None = None  # [{path, status, additions, deletions}]
    agent_types: list[str] | None = None  # Override default agents for mode
    codebase_path: str | None = None  # Local path for code_analyzer
    repository_url: str | None = None  # GitHub URL for code-aware analysis


class SwarmOrchestrator:
    """Orchestrates concurrent agent execution in a scatter-gather pattern."""

    def __init__(self):
        self._active_swarms: dict[str, asyncio.Task] = {}

    async def launch(self, config: SwarmConfig) -> tuple[str, AGUIEmitter]:
        """Launch a swarm. Returns (swarm_id, emitter) immediately.

        The swarm runs in the background. Use the emitter to stream events.
        """
        swarm_id = f"swarm_{uuid.uuid4().hex[:12]}"
        throttler = get_swarm_throttler()

        # Reserve swarm slot (Layer 2) — sync method, may raise RuntimeError
        try:
            throttler.reserve_swarm(config.org_id, swarm_id)
        except RuntimeError as e:
            raise SwarmQuotaExceeded(str(e))

        # Create AG-UI emitter
        emitter = create_emitter(swarm_id)

        # Determine agent types
        agent_types = config.agent_types or SWARM_MODE_AGENTS.get(
            config.mode, SWARM_MODE_AGENTS[SwarmMode.TARGETED_BLITZ]
        )

        # Launch background task
        task = asyncio.create_task(
            self._run_swarm(swarm_id, config, agent_types, emitter)
        )
        self._active_swarms[swarm_id] = task

        return swarm_id, emitter

    async def cancel(self, swarm_id: str) -> bool:
        """Cancel a running swarm."""
        task = self._active_swarms.get(swarm_id)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def _run_swarm(
        self,
        swarm_id: str,
        config: SwarmConfig,
        agent_types: list[str],
        emitter: AGUIEmitter,
    ) -> SwarmResult:
        """Main swarm execution loop."""
        throttler = get_swarm_throttler()
        start_time = time.time()

        try:
            # Emit RUN_STARTED
            await emitter.emit(
                RunStartedEvent(
                    run_id=swarm_id,
                    swarm_id=swarm_id,
                    mode=config.mode.value,
                    worker_count=len(agent_types),
                    agent_types=agent_types,
                )
            )

            # Scatter: spawn all workers concurrently
            tasks = []
            for agent_type in agent_types:
                agent_id = f"{agent_type}_{uuid.uuid4().hex[:8]}"
                task = asyncio.create_task(
                    self._run_worker(
                        swarm_id=swarm_id,
                        agent_id=agent_id,
                        agent_type=agent_type,
                        config=config,
                        emitter=emitter,
                        throttler=throttler,
                    )
                )
                tasks.append(task)

            # Gather: wait for all workers
            worker_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            results: list[WorkerResult] = []
            for r in worker_results:
                if isinstance(r, Exception):
                    results.append(
                        WorkerResult(
                            agent_id="unknown",
                            agent_type="unknown",
                            success=False,
                            duration_ms=0,
                            cost_usd=0,
                            error=str(r),
                        )
                    )
                else:
                    results.append(r)

            # Consensus
            consensus_score = self._compute_consensus(results)

            total_duration = (time.time() - start_time) * 1000
            total_cost = sum(r.cost_usd for r in results)
            workers_completed = sum(1 for r in results if r.success)
            workers_failed = len(results) - workers_completed

            summary = self._build_summary(config.mode, results, consensus_score)

            # Emit RUN_FINISHED
            await emitter.emit(
                RunFinishedEvent(
                    run_id=swarm_id,
                    swarm_id=swarm_id,
                    success=workers_failed == 0,
                    total_duration_ms=total_duration,
                    total_cost_usd=total_cost,
                    workers_completed=workers_completed,
                    workers_failed=workers_failed,
                    consensus_score=consensus_score,
                    summary=summary,
                )
            )

            result = SwarmResult(
                swarm_id=swarm_id,
                mode=config.mode,
                success=workers_failed == 0,
                total_duration_ms=total_duration,
                total_cost_usd=total_cost,
                worker_results=results,
                consensus_score=consensus_score,
                summary=summary,
            )

            return result

        except asyncio.CancelledError:
            await emitter.emit(
                RunErrorEvent(run_id=swarm_id, error="Swarm cancelled by user")
            )
            raise

        except Exception as e:
            logger.exception("Swarm execution failed", swarm_id=swarm_id)
            await emitter.emit(
                RunErrorEvent(run_id=swarm_id, error=str(e))
            )
            raise

        finally:
            await emitter.close()
            throttler.release_swarm(config.org_id, swarm_id)
            self._active_swarms.pop(swarm_id, None)
            remove_emitter(swarm_id)

    async def _run_worker(
        self,
        swarm_id: str,
        agent_id: str,
        agent_type: str,
        config: SwarmConfig,
        emitter: AGUIEmitter,
        throttler: "SwarmThrottler",
    ) -> WorkerResult:
        """Execute a single worker agent within the swarm."""
        start_time = time.time()

        # Emit STEP_STARTED
        await emitter.emit(
            StepStartedEvent(
                step_id=agent_id,
                agent_id=agent_id,
                agent_type=agent_type,
                task=f"{config.mode.value} analysis",
                swarm_id=swarm_id,
            )
        )

        try:
            # Check budget before execution (may raise RuntimeError)
            try:
                throttler.check_budget(config.org_id, swarm_id)
            except RuntimeError as e:
                raise BudgetExceededError(str(e))

            # Acquire worker slot (Layer 1 semaphore)
            async with throttler.acquire_worker(config.org_id, swarm_id):
                # Emit progress: executing
                await emitter.emit(
                    StateDeltaEvent(
                        agent_id=agent_id,
                        progress=10,
                        phase="executing",
                        message=f"{agent_type} starting analysis",
                    )
                )

                # Execute the actual agent
                result = await self._execute_agent(
                    agent_type=agent_type,
                    config=config,
                    emitter=emitter,
                    agent_id=agent_id,
                )

                # Emit progress: complete
                await emitter.emit(
                    StateDeltaEvent(
                        agent_id=agent_id,
                        progress=100,
                        phase="complete",
                        message=f"{agent_type} finished",
                    )
                )

            duration_ms = (time.time() - start_time) * 1000
            cost = result.get("cost_usd", 0.0)

            # Record cost
            throttler.record_cost(config.org_id, swarm_id, cost)

            worker_result = WorkerResult(
                agent_id=agent_id,
                agent_type=agent_type,
                success=True,
                duration_ms=duration_ms,
                cost_usd=cost,
                findings=result.get("findings", []),
                summary=result.get("summary", ""),
                confidence=result.get("confidence", 0.5),
            )

            # Emit STEP_FINISHED
            await emitter.emit(
                StepFinishedEvent(
                    step_id=agent_id,
                    agent_id=agent_id,
                    agent_type=agent_type,
                    success=True,
                    duration_ms=duration_ms,
                    cost_usd=cost,
                    findings_count=len(worker_result.findings),
                    result_summary=worker_result.summary,
                )
            )

            return worker_result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(
                "Worker failed",
                agent_type=agent_type,
                agent_id=agent_id,
                error=str(e),
            )

            await emitter.emit(
                StepFinishedEvent(
                    step_id=agent_id,
                    agent_id=agent_id,
                    agent_type=agent_type,
                    success=False,
                    duration_ms=duration_ms,
                    result_summary=f"Error: {e}",
                )
            )

            return WorkerResult(
                agent_id=agent_id,
                agent_type=agent_type,
                success=False,
                duration_ms=duration_ms,
                cost_usd=0,
                error=str(e),
            )

    async def _execute_agent(
        self,
        agent_type: str,
        config: SwarmConfig,
        emitter: AGUIEmitter,
        agent_id: str,
    ) -> dict[str, Any]:
        """Execute an agent and return its results as a dict.

        Routes to real agent implementations. Each branch instantiates the
        concrete agent, calls its execute/analyze method, and normalizes the
        result into {findings, summary, confidence, cost_usd}.

        Falls back to a stub result if the agent raises an exception, so the
        swarm never crashes because of a single agent failure.
        """
        # Emit progress updates during execution
        await emitter.emit(
            StateDeltaEvent(
                agent_id=agent_id,
                progress=30,
                phase="analyzing",
                message=f"{agent_type} analyzing target",
            )
        )

        # ── Dispatch to real agents ──────────────────────────────────

        # 1. Security Scanner (URL-only)
        if agent_type == "security_scanner" and config.target_url:
            return await self._run_security_scanner(config, emitter, agent_id)

        # 2. Accessibility Checker (URL-only)
        if agent_type == "accessibility_checker" and config.target_url:
            return await self._run_accessibility_checker(config, emitter, agent_id)

        # 3. Performance Analyzer (URL-only)
        if agent_type == "performance_analyzer" and config.target_url:
            return await self._run_performance_analyzer(config, emitter, agent_id)

        # 4. Auto Discovery (URL-only)
        if agent_type == "auto_discovery" and config.target_url:
            return await self._run_auto_discovery(config, emitter, agent_id)

        # 5. Code Analyzer (needs codebase_path or target_url)
        if agent_type == "code_analyzer":
            return await self._run_code_analyzer(config, emitter, agent_id)

        # 6. MR/PR Analyzer (needs changed_files)
        if agent_type == "mr_analyzer":
            return await self._run_mr_analyzer(config, emitter, agent_id)

        # 7. Test Impact Analyzer (needs changed_files)
        if agent_type == "test_impact_analyzer":
            return await self._run_test_impact_analyzer(config, emitter, agent_id)

        # 8. Smart Test Selector (needs changed_files)
        if agent_type == "smart_test_selector":
            return await self._run_smart_test_selector(config, emitter, agent_id)

        # ── Agents that still need upstream context (stubs) ──────────
        # TODO: Wire when TestSpec context is available
        if agent_type == "ui_tester":
            return self._stub_result(agent_type, "Needs TestSpec with steps/assertions")

        # TODO: Wire when TestSpec context is available
        if agent_type == "api_tester":
            return self._stub_result(agent_type, "Needs TestSpec with endpoint definitions")

        # TODO: Wire when prior failure details are available
        if agent_type == "self_healer":
            return self._stub_result(agent_type, "Needs prior failure context to heal")

        # TODO: Wire when test execution history is available
        if agent_type == "flaky_detector":
            return self._stub_result(agent_type, "Needs test execution history")

        # TODO: Wire when baseline screenshots are available
        if agent_type == "visual_ai":
            return self._stub_result(agent_type, "Needs baseline screenshots for comparison")

        # Fallback for any unknown agent type
        return self._stub_result(agent_type, "Unknown agent type")

    # ── Individual agent runners ─────────────────────────────────────

    async def _run_security_scanner(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        try:
            from src.agents import SecurityScannerAgent

            agent = SecurityScannerAgent()
            result = await agent.execute(url=config.target_url, scan_type="standard")

            await self._emit_progress(emitter, agent_id, 70, "reporting", "security_scanner compiling results")

            if result.success and result.data:
                scan = result.data
                return {
                    "findings": [v.to_dict() for v in scan.vulnerabilities],
                    "summary": scan.summary,
                    "confidence": max(0.0, 1.0 - (scan.risk_score / 100.0)),
                    "cost_usd": result.cost,
                }
            return {
                "findings": [],
                "summary": result.error or "Security scan returned no results",
                "confidence": 0.0,
                "cost_usd": result.cost,
            }
        except Exception as e:
            logger.warning("security_scanner failed, using stub", error=str(e))
            return self._stub_result("security_scanner", str(e))

    async def _run_accessibility_checker(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        try:
            from src.agents import AccessibilityCheckerAgent

            agent = AccessibilityCheckerAgent()
            result = await agent.execute(url=config.target_url, wcag_level="AA")

            await self._emit_progress(emitter, agent_id, 70, "reporting", "accessibility_checker compiling results")

            if result.success and result.data:
                check = result.data
                return {
                    "findings": [i.to_dict() for i in check.issues],
                    "summary": check.summary,
                    "confidence": check.score / 100.0,
                    "cost_usd": result.cost,
                }
            return {
                "findings": [],
                "summary": result.error or "Accessibility check returned no results",
                "confidence": 0.0,
                "cost_usd": result.cost,
            }
        except Exception as e:
            logger.warning("accessibility_checker failed, using stub", error=str(e))
            return self._stub_result("accessibility_checker", str(e))

    async def _run_performance_analyzer(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        try:
            from src.agents import PerformanceAnalyzerAgent

            agent = PerformanceAnalyzerAgent()
            result = await agent.execute(url=config.target_url, device="mobile")

            await self._emit_progress(emitter, agent_id, 70, "reporting", "performance_analyzer compiling results")

            if result.success and result.data:
                perf = result.data
                findings = [
                    {"type": issue.category, "severity": issue.severity, "title": issue.title, "description": issue.description}
                    for issue in perf.issues
                ]
                grade_confidence = {"excellent": 0.95, "good": 0.8, "needs_work": 0.5, "poor": 0.3}
                return {
                    "findings": findings,
                    "summary": perf.summary,
                    "confidence": grade_confidence.get(perf.overall_grade.value, 0.5),
                    "cost_usd": result.cost,
                }
            return {
                "findings": [],
                "summary": result.error or "Performance analysis returned no results",
                "confidence": 0.0,
                "cost_usd": result.cost,
            }
        except Exception as e:
            logger.warning("performance_analyzer failed, using stub", error=str(e))
            return self._stub_result("performance_analyzer", str(e))

    async def _run_auto_discovery(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        try:
            from src.agents import AutoDiscovery

            discovery = AutoDiscovery(app_url=config.target_url, max_pages=10)
            result = await discovery.discover()

            await self._emit_progress(emitter, agent_id, 70, "reporting", "auto_discovery compiling results")

            return {
                "findings": result.suggested_tests,
                "summary": (
                    f"Discovered {len(result.pages_discovered)} pages, "
                    f"{len(result.flows_discovered)} flows, "
                    f"{len(result.suggested_tests)} test suggestions"
                ),
                "confidence": 0.75,
                "cost_usd": 0.0,  # AutoDiscovery doesn't track cost via AgentResult
            }
        except Exception as e:
            logger.warning("auto_discovery failed, using stub", error=str(e))
            return self._stub_result("auto_discovery", str(e))

    async def _run_code_analyzer(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        try:
            from src.agents import CodeAnalyzerAgent

            agent = CodeAnalyzerAgent()
            codebase_path = config.codebase_path or "."
            app_url = config.target_url or "http://localhost:3000"

            result = await agent.execute(
                codebase_path=codebase_path,
                app_url=app_url,
                changed_files=[f.get("path", f) if isinstance(f, dict) else f for f in (config.changed_files or [])],
            )

            await self._emit_progress(emitter, agent_id, 70, "reporting", "code_analyzer compiling results")

            if result.success and result.data:
                analysis = result.data
                findings = []
                for surface in getattr(analysis, "testable_surfaces", []):
                    if isinstance(surface, dict):
                        findings.append(surface)
                    else:
                        findings.append({"type": "testable_surface", "data": str(surface)})
                return {
                    "findings": findings,
                    "summary": getattr(analysis, "summary", f"Code analysis complete for {codebase_path}"),
                    "confidence": 0.8,
                    "cost_usd": result.cost,
                }
            return {
                "findings": [],
                "summary": result.error or "Code analysis returned no results",
                "confidence": 0.0,
                "cost_usd": result.cost,
            }
        except Exception as e:
            logger.warning("code_analyzer failed, using stub", error=str(e))
            return self._stub_result("code_analyzer", str(e))

    async def _run_mr_analyzer(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        if not config.changed_files:
            return self._stub_result("mr_analyzer", "No changed_files provided")
        try:
            from src.agents import MRAnalyzerAgent

            agent = MRAnalyzerAgent()
            result = await agent.execute(
                changes=config.changed_files,
                project_id=config.project_id,
            )

            await self._emit_progress(emitter, agent_id, 70, "reporting", "mr_analyzer compiling results")

            if result.success and result.data:
                data = result.data
                suggestions = data.get("suggestions", [])
                analysis = data.get("analysis", {})
                return {
                    "findings": suggestions,
                    "summary": analysis.get("summary", f"Analyzed {len(config.changed_files)} changed files"),
                    "confidence": analysis.get("confidence", 0.7),
                    "cost_usd": result.cost,
                }
            return {
                "findings": [],
                "summary": result.error or "MR analysis returned no results",
                "confidence": 0.0,
                "cost_usd": result.cost,
            }
        except Exception as e:
            logger.warning("mr_analyzer failed, using stub", error=str(e))
            return self._stub_result("mr_analyzer", str(e))

    async def _run_test_impact_analyzer(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        if not config.changed_files:
            return self._stub_result("test_impact_analyzer", "No changed_files provided")
        try:
            from src.agents import TestImpactAnalyzer, CodeChange
            from datetime import datetime

            analyzer = TestImpactAnalyzer()
            change = CodeChange(
                id=f"swarm_{config.pr_number or 'local'}",
                files=config.changed_files,
                message="Swarm analysis",
                author=config.user_id,
                timestamp=datetime.utcnow(),
                branch="unknown",
            )
            impact = await analyzer.analyze_impact(change=change, all_tests=[])

            await self._emit_progress(emitter, agent_id, 70, "reporting", "test_impact_analyzer compiling results")

            return {
                "findings": [
                    {"type": "affected_test", "test_id": t}
                    for t in impact.affected_tests
                ],
                "summary": (
                    f"{len(impact.affected_tests)} tests affected, "
                    f"{len(impact.unaffected_tests)} safe to skip"
                ),
                "confidence": impact.confidence,
                "cost_usd": 0.0,
            }
        except Exception as e:
            logger.warning("test_impact_analyzer failed, using stub", error=str(e))
            return self._stub_result("test_impact_analyzer", str(e))

    async def _run_smart_test_selector(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        if not config.changed_files:
            return self._stub_result("smart_test_selector", "No changed_files provided")
        try:
            from src.agents import TestImpactAnalyzer, SmartTestSelector, CodeChange
            from datetime import datetime

            analyzer = TestImpactAnalyzer()
            selector = SmartTestSelector(impact_analyzer=analyzer)
            change = CodeChange(
                id=f"swarm_{config.pr_number or 'local'}",
                files=config.changed_files,
                message="Swarm analysis",
                author=config.user_id,
                timestamp=datetime.utcnow(),
                branch="unknown",
            )
            selection = await selector.select_tests(change=change, all_tests=[])

            await self._emit_progress(emitter, agent_id, 70, "reporting", "smart_test_selector compiling results")

            must_run = selection.get("must_run", [])
            should_run = selection.get("should_run", [])
            can_skip = selection.get("can_skip", [])
            return {
                "findings": [
                    {"type": "must_run", "tests": must_run},
                    {"type": "should_run", "tests": should_run},
                    {"type": "can_skip", "tests": can_skip},
                ],
                "summary": (
                    f"Selected {len(must_run)} must-run, {len(should_run)} should-run, "
                    f"{len(can_skip)} skippable tests"
                ),
                "confidence": selection.get("coverage_estimate", 0.7),
                "cost_usd": 0.0,
            }
        except Exception as e:
            logger.warning("smart_test_selector failed, using stub", error=str(e))
            return self._stub_result("smart_test_selector", str(e))

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _stub_result(agent_type: str, reason: str = "") -> dict[str, Any]:
        """Return a minimal stub result for agents that can't run yet."""
        note = f" ({reason})" if reason else ""
        return {
            "findings": [],
            "summary": f"{agent_type} not executed{note}",
            "confidence": 0.0,
            "cost_usd": 0.0,
        }

    @staticmethod
    async def _emit_progress(
        emitter: AGUIEmitter, agent_id: str, progress: int, phase: str, message: str,
    ) -> None:
        """Emit a progress update event."""
        await emitter.emit(
            StateDeltaEvent(agent_id=agent_id, progress=progress, phase=phase, message=message)
        )

    def _compute_consensus(self, results: list[WorkerResult]) -> float:
        """Compute consensus score from worker results.

        Uses confidence-weighted agreement. Higher score means more agents
        agree on findings.
        """
        successful = [r for r in results if r.success]
        if not successful:
            return 0.0

        total_confidence = sum(r.confidence for r in successful)
        if total_confidence == 0:
            return 0.0

        # Weighted average confidence
        weighted = sum(r.confidence * r.confidence for r in successful) / total_confidence
        return round(min(weighted, 1.0), 3)

    def _build_summary(
        self, mode: SwarmMode, results: list[WorkerResult], consensus: float
    ) -> str:
        """Build a human-readable swarm summary."""
        successful = sum(1 for r in results if r.success)
        total_findings = sum(len(r.findings) for r in results)
        return (
            f"{mode.value} swarm: {successful}/{len(results)} agents completed, "
            f"{total_findings} findings, {consensus:.0%} consensus"
        )


class SwarmQuotaExceeded(Exception):
    """Raised when org swarm quota is exceeded."""
    pass


class BudgetExceededError(Exception):
    """Raised when cost budget is exceeded."""
    pass


# Singleton orchestrator
_orchestrator: SwarmOrchestrator | None = None


def get_swarm_orchestrator() -> SwarmOrchestrator:
    """Get or create the global swarm orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SwarmOrchestrator()
    return _orchestrator
