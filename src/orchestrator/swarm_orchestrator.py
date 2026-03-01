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

from .intent_verifier import IntentVerdict, verify_intent
from .swarm_throttler import get_swarm_throttler

logger = structlog.get_logger()


class SwarmMode(str, Enum):
    """Available swarm execution modes."""

    FULL_CRAWL = "full_crawl"
    TARGETED_BLITZ = "targeted_blitz"
    PR_ANALYSIS = "pr_analysis"
    DISCOVERY_SWARM = "discovery_swarm"


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
    SwarmMode.DISCOVERY_SWARM: [
        "auto_discovery",  # Phase 1 only — Phase 2 agents are dynamic
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
    intent_verdict: "IntentVerdict | None" = None


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
    max_discovery_agents: int | None = None  # Cap for DISCOVERY_SWARM Phase 2


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

            # Discovery swarm: two-phase execution
            if config.mode == SwarmMode.DISCOVERY_SWARM and config.target_url:
                return await self._run_discovery_swarm(
                    swarm_id, config, emitter, throttler
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

            # Intent verification — lightweight post-check
            verdict = await self._verify_intent(config, results, total_cost)
            if verdict:
                summary += f" | Intent: {verdict.verdict} ({verdict.completion_score:.0%})"
                if verdict.gaps:
                    summary += f" — gaps: {', '.join(verdict.gaps[:3])}"

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
                intent_verdict=verdict,
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

        Falls back to an error result if the agent raises an exception, so the
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

        # 9. UI Tester (exploratory — needs target_url)
        if agent_type == "ui_tester" and config.target_url:
            return await self._run_ui_tester(config, emitter, agent_id)

        # 10. API Tester (exploratory endpoint probe — needs target_url)
        if agent_type == "api_tester" and config.target_url:
            return await self._run_api_tester(config, emitter, agent_id)

        # 11. Self Healer (queries DB for recent failures)
        if agent_type == "self_healer":
            return await self._run_self_healer(config, emitter, agent_id)

        # 12. Flaky Detector (statistical analysis from DB)
        if agent_type == "flaky_detector":
            return await self._run_flaky_detector(config, emitter, agent_id)

        # 13. Visual AI (single-screenshot analysis — needs target_url)
        if agent_type == "visual_ai" and config.target_url:
            return await self._run_visual_ai(config, emitter, agent_id)

        # Fallback for any unknown agent type
        return self._error_result(agent_type, "unrecognized agent type")

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
            logger.warning("security_scanner failed, returning error result", error=str(e))
            return self._error_result("security_scanner", str(e))

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
            logger.warning("accessibility_checker failed, returning error result", error=str(e))
            return self._error_result("accessibility_checker", str(e))

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
            logger.warning("performance_analyzer failed, returning error result", error=str(e))
            return self._error_result("performance_analyzer", str(e))

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
            logger.warning("auto_discovery failed, returning error result", error=str(e))
            return self._error_result("auto_discovery", str(e))

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
            logger.warning("code_analyzer failed, returning error result", error=str(e))
            return self._error_result("code_analyzer", str(e))

    async def _run_mr_analyzer(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        if not config.changed_files:
            return self._error_result("mr_analyzer", "No changed_files provided")
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
            logger.warning("mr_analyzer failed, returning error result", error=str(e))
            return self._error_result("mr_analyzer", str(e))

    async def _run_test_impact_analyzer(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        if not config.changed_files:
            return self._error_result("test_impact_analyzer", "No changed_files provided")
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
            logger.warning("test_impact_analyzer failed, returning error result", error=str(e))
            return self._error_result("test_impact_analyzer", str(e))

    async def _run_smart_test_selector(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        if not config.changed_files:
            return self._error_result("smart_test_selector", "No changed_files provided")
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
            logger.warning("smart_test_selector failed, returning error result", error=str(e))
            return self._error_result("smart_test_selector", str(e))

    # ── Newly wired agent runners ─────────────────────────────────────

    async def _run_ui_tester(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        """Exploratory UI test: navigate to target_url, screenshot, verify body."""
        try:
            from src.agents import UITesterAgent

            agent = UITesterAgent()
            test_spec = {
                "id": f"swarm_ui_{uuid.uuid4().hex[:8]}",
                "name": "Swarm exploratory UI check",
                "steps": [
                    {"action": "goto", "target": config.target_url},
                    {"action": "wait", "timeout": 3000},
                    {"action": "screenshot"},
                    {"action": "assert", "target": "body", "value": "visible"},
                ],
            }
            result = await agent.execute(
                test_spec=test_spec,
                app_url=config.target_url,
                capture_screenshots=True,
            )

            await self._emit_progress(emitter, agent_id, 70, "reporting", "ui_tester compiling results")

            if result.success and result.data:
                ui = result.data
                findings = [
                    {"type": "ui_step", "step": s.action if hasattr(s, "action") else str(s), "status": getattr(s, "status", "unknown")}
                    for s in getattr(ui, "steps", [])
                ]
                return {
                    "findings": findings,
                    "summary": f"UI check {'passed' if ui.status == 'passed' else 'failed'}: {config.target_url}",
                    "confidence": 0.8 if ui.status == "passed" else 0.4,
                    "cost_usd": result.cost,
                }
            return {
                "findings": [],
                "summary": result.error or "UI test returned no results",
                "confidence": 0.0,
                "cost_usd": result.cost,
            }
        except Exception as e:
            logger.warning("ui_tester failed, returning error result", error=str(e))
            return self._error_result("ui_tester", str(e))

    async def _run_api_tester(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        """Exploratory API probe: check common endpoints for reachability."""
        try:
            from src.agents import APITesterAgent

            agent = APITesterAgent()
            probe_paths = ["/", "/api", "/health", "/api/v1/health"]
            test_spec = {
                "id": f"swarm_api_{uuid.uuid4().hex[:8]}",
                "name": "Swarm exploratory API probe",
                "requests": [
                    {"method": "GET", "path": path, "expected_status": [200, 301, 302, 404]}
                    for path in probe_paths
                ],
            }
            result = await agent.execute(
                test_spec=test_spec,
                app_url=config.target_url,
            )

            await self._emit_progress(emitter, agent_id, 70, "reporting", "api_tester compiling results")

            if result.success and result.data:
                api = result.data
                findings = [
                    {
                        "type": "api_endpoint",
                        "path": getattr(r, "path", "unknown"),
                        "status_code": getattr(r, "status_code", 0),
                        "success": getattr(r, "success", False),
                    }
                    for r in getattr(api, "requests", [])
                ]
                return {
                    "findings": findings,
                    "summary": f"API probe: {sum(1 for f in findings if f.get('success'))}/{len(findings)} endpoints reachable",
                    "confidence": 0.7,
                    "cost_usd": result.cost,
                }
            return {
                "findings": [],
                "summary": result.error or "API test returned no results",
                "confidence": 0.0,
                "cost_usd": result.cost,
            }
        except Exception as e:
            logger.warning("api_tester failed, returning error result", error=str(e))
            return self._error_result("api_tester", str(e))

    async def _run_self_healer(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        """Query recent failures from DB and attempt healing."""
        try:
            from src.integrations.supabase import get_supabase

            supabase = await get_supabase()
            if not supabase:
                return self._error_result("self_healer", "Supabase not available")

            # Fetch recent failures for this project
            failures = await supabase.select(
                "test_results",
                filters={"project_id": f"eq.{config.project_id}", "status": "eq.failed"},
                limit=3,
                order_by="created_at",
                ascending=False,
            )

            await self._emit_progress(emitter, agent_id, 50, "analyzing", "self_healer reviewing failures")

            if not failures:
                return {
                    "findings": [],
                    "summary": "No recent failures to heal",
                    "confidence": 1.0,
                    "cost_usd": 0.0,
                }

            # Use SelfHealerAgent on the failures
            from src.agents import SelfHealerAgent

            healer = SelfHealerAgent(
                org_id=config.org_id,
                project_id=config.project_id,
            )

            findings = []
            total_cost = 0.0
            for failure in failures:
                try:
                    from src.orchestrator.state import TestingState
                    state = TestingState(
                        messages=[],
                        codebase_path=config.codebase_path or ".",
                        failures=[failure],
                        healing_attempts=0,
                    )
                    healed_state = await healer.execute(state=state)
                    for healed in getattr(healed_state, "healed_tests", []):
                        findings.append({
                            "type": "healing_suggestion",
                            "test_id": failure.get("test_id", "unknown"),
                            "suggestion": str(healed),
                        })
                except Exception as heal_err:
                    findings.append({
                        "type": "healing_failed",
                        "test_id": failure.get("test_id", "unknown"),
                        "error": str(heal_err),
                    })

            await self._emit_progress(emitter, agent_id, 70, "reporting", "self_healer compiling results")

            return {
                "findings": findings,
                "summary": f"Analyzed {len(failures)} failures, produced {len(findings)} healing suggestions",
                "confidence": 0.6,
                "cost_usd": total_cost,
            }
        except Exception as e:
            logger.warning("self_healer failed, returning error result", error=str(e))
            return self._error_result("self_healer", str(e))

    async def _run_flaky_detector(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        """Statistical flaky test analysis from DB — zero AI cost."""
        try:
            from src.agents import FlakyTestDetector
            from src.integrations.supabase import get_supabase

            supabase = await get_supabase()
            if not supabase:
                return self._error_result("flaky_detector", "Supabase not available")

            # Fetch recent test results for statistical analysis
            rows = await supabase.select(
                "test_results",
                filters={"project_id": f"eq.{config.project_id}"},
                limit=500,
                order_by="created_at",
                ascending=False,
            )

            await self._emit_progress(emitter, agent_id, 50, "analyzing", "flaky_detector crunching statistics")

            if not rows:
                return {
                    "findings": [],
                    "summary": "No test history available for flaky analysis",
                    "confidence": 1.0,
                    "cost_usd": 0.0,
                }

            detector = FlakyTestDetector()

            # Group runs by test_id and feed into detector
            from collections import defaultdict
            by_test: dict[str, list] = defaultdict(list)
            for row in rows:
                tid = row.get("test_id") or row.get("id", "unknown")
                by_test[tid].append(row)

            findings = []
            for test_id, runs in by_test.items():
                for run in runs:
                    from datetime import datetime as _dt

                    from src.agents.flaky_detector import TestRun
                    raw_ts = run.get("created_at", "")
                    try:
                        ts = _dt.fromisoformat(raw_ts.replace("Z", "+00:00")) if raw_ts else _dt.now()
                    except (ValueError, AttributeError):
                        ts = _dt.now()
                    detector.record_run(TestRun(
                        test_id=test_id,
                        passed=run.get("status") == "passed",
                        duration_ms=run.get("duration_ms", 0),
                        timestamp=ts,
                    ))

                report = detector.analyze_test(test_id=test_id)
                if report.flakiness_score > 0.1:
                    findings.append({
                        "type": "flaky_test",
                        "test_id": test_id,
                        "flakiness_score": round(report.flakiness_score, 3),
                        "level": report.flakiness_level.value,
                        "pass_rate": round(report.pass_rate, 3),
                        "total_runs": report.total_runs,
                        "likely_cause": report.likely_cause.value,
                        "should_quarantine": report.should_quarantine,
                    })

            await self._emit_progress(emitter, agent_id, 70, "reporting", "flaky_detector compiling results")

            findings.sort(key=lambda f: f["flakiness_score"], reverse=True)
            return {
                "findings": findings[:20],  # Top 20 flakiest
                "summary": f"Analyzed {len(by_test)} tests from {len(rows)} runs, found {len(findings)} flaky",
                "confidence": 0.85,
                "cost_usd": 0.0,  # Pure statistics, no AI cost
            }
        except Exception as e:
            logger.warning("flaky_detector failed, returning error result", error=str(e))
            return self._error_result("flaky_detector", str(e))

    async def _run_visual_ai(
        self, config: SwarmConfig, emitter: AGUIEmitter, agent_id: str,
    ) -> dict[str, Any]:
        """Single-screenshot analysis of target_url (no baseline needed)."""
        try:
            import tempfile
            from src.agents import VisualAI
            from src.browser.pool_client import BrowserPoolClient

            await self._emit_progress(emitter, agent_id, 40, "capturing", "visual_ai taking screenshot")

            # Take screenshot via Browser Worker
            screenshot_path = None
            try:
                async with BrowserPoolClient() as browser:
                    screenshot_bytes = await browser.screenshot(url=config.target_url)
                    if screenshot_bytes:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                            f.write(screenshot_bytes)
                            screenshot_path = f.name
            except Exception as browser_err:
                logger.warning("Browser screenshot failed for visual_ai", error=str(browser_err))
                return self._error_result("visual_ai", f"Browser unavailable: {browser_err}")

            if not screenshot_path:
                return self._error_result("visual_ai", "No screenshot captured")

            await self._emit_progress(emitter, agent_id, 60, "analyzing", "visual_ai analyzing screenshot")

            visual = VisualAI()
            analysis = await visual.analyze_single(
                screenshot=screenshot_path,
                context=f"Target URL: {config.target_url}",
            )

            await self._emit_progress(emitter, agent_id, 70, "reporting", "visual_ai compiling results")

            issues = analysis.get("issues", [])
            elements = analysis.get("main_elements", [])
            health = analysis.get("overall_health", "unknown")

            health_confidence = {"healthy": 0.9, "degraded": 0.5, "broken": 0.2}
            findings = [{"type": "visual_element", "element": e} for e in elements]
            if issues:
                findings.extend([{"type": "visual_issue", "issue": i} for i in issues])

            return {
                "findings": findings,
                "summary": f"Visual analysis: {health}, {len(elements)} elements, {len(issues)} issues",
                "confidence": health_confidence.get(health, 0.5),
                "cost_usd": 0.01,  # Approximate single-vision-call cost
            }
        except Exception as e:
            logger.warning("visual_ai failed, returning error result", error=str(e))
            return self._error_result("visual_ai", str(e))

    # ── Discovery Swarm ─────────────────────────────────────────────

    async def _run_discovery_swarm(
        self,
        swarm_id: str,
        config: SwarmConfig,
        emitter: AGUIEmitter,
        throttler: "SwarmThrottler",
    ) -> SwarmResult:
        """Two-phase discovery swarm: discover targets, then test each one.

        Phase 1: Run AutoDiscovery on target_url to find pages, flows, endpoints.
        Phase 2: Spawn one agent per discovered target (up to tier max).
        """
        from .swarm_throttler import TIER_CONFIGS, ThrottlerTier

        start_time = time.time()

        # Determine max agents from config or tier
        org_config = throttler._get_config_for_org(config.org_id)
        max_agents = config.max_discovery_agents or org_config.max_workers_per_swarm

        # ── Phase 1: Discovery ─────────────────────────
        await emitter.emit(
            StateDeltaEvent(
                agent_id="discovery_phase",
                progress=10,
                phase="discovering",
                message="Phase 1: Discovering application structure",
            )
        )

        try:
            from src.agents import AutoDiscovery

            discovery = AutoDiscovery(app_url=config.target_url, max_pages=max_agents)
            discovery_result = await discovery.discover()
        except Exception as e:
            logger.error("Discovery phase failed", error=str(e))
            total_duration = (time.time() - start_time) * 1000
            await emitter.emit(
                RunFinishedEvent(
                    run_id=swarm_id,
                    swarm_id=swarm_id,
                    success=False,
                    total_duration_ms=total_duration,
                    total_cost_usd=0.0,
                    workers_completed=0,
                    workers_failed=1,
                    consensus_score=0.0,
                    summary=f"Discovery phase failed: {e}",
                )
            )
            return SwarmResult(
                swarm_id=swarm_id,
                mode=config.mode,
                success=False,
                total_duration_ms=total_duration,
                total_cost_usd=0.0,
                summary=f"Discovery phase failed: {e}",
            )

        await emitter.emit(
            StateDeltaEvent(
                agent_id="discovery_phase",
                progress=40,
                phase="planning",
                message=(
                    f"Phase 1 complete: {len(discovery_result.pages_discovered)} pages, "
                    f"{len(discovery_result.flows_discovered)} flows"
                ),
            )
        )

        # ── Phase 2: Fan out agents per target ─────────
        targets = self._build_discovery_targets(discovery_result, max_agents)

        if not targets:
            total_duration = (time.time() - start_time) * 1000
            await emitter.emit(
                RunFinishedEvent(
                    run_id=swarm_id,
                    swarm_id=swarm_id,
                    success=True,
                    total_duration_ms=total_duration,
                    total_cost_usd=0.0,
                    workers_completed=0,
                    workers_failed=0,
                    consensus_score=1.0,
                    summary="Discovery found no testable targets",
                )
            )
            return SwarmResult(
                swarm_id=swarm_id,
                mode=config.mode,
                success=True,
                total_duration_ms=total_duration,
                total_cost_usd=0.0,
                consensus_score=1.0,
                summary="Discovery found no testable targets",
            )

        # Spawn one worker per target
        tasks = []
        for i, (agent_type, target_url) in enumerate(targets):
            agent_id = f"{agent_type}_disc_{i}_{uuid.uuid4().hex[:6]}"
            # Create a per-target config with the specific URL
            target_config = SwarmConfig(
                mode=config.mode,
                org_id=config.org_id,
                project_id=config.project_id,
                user_id=config.user_id,
                target_url=target_url,
                codebase_path=config.codebase_path,
                repository_url=config.repository_url,
            )
            task = asyncio.create_task(
                self._run_worker(
                    swarm_id=swarm_id,
                    agent_id=agent_id,
                    agent_type=agent_type,
                    config=target_config,
                    emitter=emitter,
                    throttler=throttler,
                )
            )
            tasks.append(task)

        # Gather all Phase 2 results
        worker_results_raw = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[WorkerResult] = []
        for r in worker_results_raw:
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

        consensus_score = self._compute_consensus(results)
        total_duration = (time.time() - start_time) * 1000
        total_cost = sum(r.cost_usd for r in results)
        workers_completed = sum(1 for r in results if r.success)
        workers_failed = len(results) - workers_completed

        summary = (
            f"Discovery swarm: discovered {len(discovery_result.pages_discovered)} pages, "
            f"spawned {len(targets)} agents, "
            f"{workers_completed}/{len(results)} completed, "
            f"{consensus_score:.0%} consensus"
        )

        # Intent verification
        verdict = await self._verify_intent(config, results, total_cost)
        if verdict:
            summary += f" | Intent: {verdict.verdict} ({verdict.completion_score:.0%})"
            if verdict.gaps:
                summary += f" — gaps: {', '.join(verdict.gaps[:3])}"

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

        return SwarmResult(
            swarm_id=swarm_id,
            mode=config.mode,
            success=workers_failed == 0,
            total_duration_ms=total_duration,
            total_cost_usd=total_cost,
            worker_results=results,
            consensus_score=consensus_score,
            summary=summary,
            intent_verdict=verdict,
        )

    @staticmethod
    def _select_agent_for_target(target_type: str) -> str:
        """Pick the right agent type based on what was discovered."""
        mapping = {
            "page": "ui_tester",
            "form": "ui_tester",
            "flow": "ui_tester",
            "api_endpoint": "api_tester",
            "endpoint": "api_tester",
        }
        return mapping.get(target_type, "ui_tester")

    @staticmethod
    def _build_discovery_targets(
        discovery_result: Any, max_agents: int,
    ) -> list[tuple[str, str]]:
        """Build (agent_type, url) pairs from discovery results, capped at max_agents."""
        targets: list[tuple[str, str]] = []

        # Pages → ui_tester
        for page in getattr(discovery_result, "pages_discovered", []):
            url = getattr(page, "url", None)
            if url:
                targets.append(("ui_tester", url))

        # Flows → ui_tester (use first step URL or app_url)
        for flow in getattr(discovery_result, "flows_discovered", []):
            steps = getattr(flow, "steps", [])
            if steps and isinstance(steps[0], dict) and steps[0].get("url"):
                targets.append(("ui_tester", steps[0]["url"]))

        # Suggested tests may reference API endpoints
        for suggestion in getattr(discovery_result, "suggested_tests", []):
            if isinstance(suggestion, dict):
                stype = suggestion.get("type", "")
                url = suggestion.get("url") or suggestion.get("endpoint", "")
                if "api" in stype.lower() and url:
                    targets.append(("api_tester", url))

        # Deduplicate by URL, cap at max_agents
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for agent_type, url in targets:
            if url not in seen:
                seen.add(url)
                unique.append((agent_type, url))
        return unique[:max_agents]

    # ── Intent verification ─────────────────────────────────────────

    @staticmethod
    async def _verify_intent(
        config: SwarmConfig, results: list[WorkerResult], total_cost: float,
    ) -> IntentVerdict | None:
        """Run lightweight intent check. Never blocks the pipeline on failure."""
        try:
            return await verify_intent(config, results, total_cost)
        except Exception as e:
            logger.warning("Intent verification failed entirely", error=str(e))
            return None

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _error_result(agent_type: str, reason: str = "") -> dict[str, Any]:
        """Return a minimal error/skip result when an agent cannot produce real output."""
        note = f": {reason}" if reason else ""
        return {
            "findings": [],
            "summary": f"{agent_type} error{note}",
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
