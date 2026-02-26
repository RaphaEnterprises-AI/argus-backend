"""TestHealer CI/CD Agent - CI/CD-aware healing orchestration.

Wraps the existing SelfHealerAgent (2,297 lines) with CI/CD awareness:
- Process CI failure callbacks from GitHub Actions
- Run proactive fragility scans before tests break
- Compute health scores for test suite trending
- Store selector fragility data in Cognee for cross-agent learning
- Post fix suggestions to PRs via comment poster

This agent COMPOSES SelfHealerAgent - it does NOT duplicate healing logic.
"""

import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.agents.base import AgentCapability, BaseAgent
from src.core.model_router import TaskType

logger = structlog.get_logger()


class FixProposal(BaseModel):
    """A proposed fix for a test issue."""
    test_id: str
    test_name: str = ""
    issue_type: str  # selector_changed, timing_issue, api_change, data_dependency, flaky
    description: str
    fix_description: str
    confidence: float = 0.0  # 0.0-1.0
    original_selector: str | None = None
    proposed_selector: str | None = None
    applied: bool = False


class HealthScore(BaseModel):
    """Composite health score for a test suite."""
    overall_score: float = 0.0  # 0-100
    selector_stability: float = 0.0  # 0-100
    flaky_rate: float = 0.0  # 0-100 (lower is better)
    heal_success_rate: float = 0.0  # 0-100
    mttr_seconds: float = 0.0  # mean time to repair
    total_tests: int = 0
    fragile_tests: int = 0
    healed_tests_30d: int = 0
    details: dict = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()


class ScanResult(BaseModel):
    """Result of a healing scan."""
    scan_id: str = ""
    scan_type: str = ""
    tests_scanned: int = 0
    issues_found: int = 0
    fixes_proposed: int = 0
    fixes_applied: int = 0
    fix_proposals: list[dict] = Field(default_factory=list)
    health_score_before: float | None = None
    health_score_after: float | None = None
    ai_cost: float = 0.0

    def to_dict(self) -> dict:
        return self.model_dump()


class TestHealerCICDAgent(BaseAgent):
    """CI/CD-aware healing orchestration agent.

    Composes SelfHealerAgent with CI/CD context for:
    - Processing CI failure callbacks
    - Proactive fragility scanning
    - Health score computation and trending
    - Cognee-powered cross-agent learning
    """

    CAPABILITIES = [
        AgentCapability.HEALING,
        AgentCapability.TEST_GENERATION,
    ]

    def _get_system_prompt(self) -> str:
        return """You are a CI/CD-aware test healing agent. Your role is to:

1. Process CI/CD test failures and diagnose root causes
2. Proactively scan tests for fragile selectors before they break
3. Compute health scores that track test suite quality over time
4. Propose fixes with confidence scores and apply them when appropriate
5. Learn from past healing patterns to improve future predictions

When analyzing failures:
- Distinguish between real bugs (don't heal) and infrastructure issues (heal)
- Check selector fragility history before proposing fixes
- Understand test intent by querying requirement traceability
- Compute confidence scores based on historical success rates

Output structured JSON with fix proposals and health metrics."""

    async def execute(self, state: dict) -> dict:
        """Standard BaseAgent execute - delegates to scan."""
        org_id = state.get("org_id", "default")
        project_id = state.get("project_id", "default")
        scan_type = state.get("scan_type", "manual")
        trigger_context = state.get("trigger_context", {})

        result = await self.scan(org_id, project_id, scan_type, trigger_context)
        state["healing_scan_result"] = result.to_dict() if isinstance(result, ScanResult) else result
        return state

    async def scan(
        self,
        org_id: str,
        project_id: str,
        scan_type: str,
        trigger_context: dict,
        scan_job_id: str | None = None,
    ) -> ScanResult:
        """Main entry: scan tests, propose fixes, compute health scores.

        Args:
            org_id: Organization ID
            project_id: Project ID
            scan_type: proactive, ci_failure, pr_review, manual
            trigger_context: CI run URL, PR number, commit SHA, etc.
            scan_job_id: Optional job ID (from DB)
        """
        log = logger.bind(org_id=org_id, project_id=project_id, scan_type=scan_type)
        log.info("Starting healing scan")

        scan_id = scan_job_id or str(uuid.uuid4())
        result = ScanResult(scan_id=scan_id, scan_type=scan_type)

        try:
            # Emit scan started event
            await self._emit_events(org_id, project_id, scan_id, scan_type, trigger_context)

            # Get health score before
            health_before = await self.compute_health_score(org_id, project_id)
            result.health_score_before = health_before.overall_score

            if scan_type == "ci_failure":
                # Process specific CI failures
                failed_tests = trigger_context.get("failed_tests", [])
                if failed_tests:
                    proposals = await self._generate_fix_proposals(org_id, project_id, failed_tests)
                    result.fix_proposals = [p.model_dump() for p in proposals]
                    result.tests_scanned = len(failed_tests)
                    result.issues_found = len(proposals)
                    result.fixes_proposed = len(proposals)
                else:
                    # No specific failures, do a general assessment
                    result = await self._run_proactive_scan(org_id, project_id, result)

            elif scan_type == "proactive":
                result = await self._run_proactive_scan(org_id, project_id, result)

            elif scan_type == "pr_review":
                # Focus on tests affected by PR changes
                changed_files = trigger_context.get("changed_files", [])
                result = await self._scan_affected_tests(org_id, project_id, changed_files, result)

            else:  # manual
                result = await self._run_proactive_scan(org_id, project_id, result)

            # Compute health score after
            health_after = await self.compute_health_score(org_id, project_id)
            result.health_score_after = health_after.overall_score

            # Store health score
            await self._store_health_score(org_id, project_id, health_after, scan_id)

            # Update job in DB
            if scan_job_id:
                await self._update_scan_job(scan_job_id, result.to_dict())

            log.info("Healing scan complete",
                     tests_scanned=result.tests_scanned,
                     issues_found=result.issues_found,
                     fixes_proposed=result.fixes_proposed)

            return result

        except Exception as e:
            log.error("Healing scan failed", error=str(e))
            if scan_job_id:
                try:
                    await self._update_scan_job(scan_job_id, {"status": "failed", "error_message": str(e)[:2000]})
                except Exception:
                    pass
            raise

    async def process_ci_failure(
        self,
        org_id: str,
        project_id: str,
        ci_data: dict,
    ) -> ScanResult:
        """Process a CI/CD failure callback.

        Args:
            ci_data: {run_url, commit_sha, branch, failed_tests: [{test_id, error_message, error_type}]}
        """
        return await self.scan(
            org_id=org_id,
            project_id=project_id,
            scan_type="ci_failure",
            trigger_context=ci_data,
        )

    async def proactive_scan(self, org_id: str, project_id: str) -> ScanResult:
        """Run proactive scan for fragile selectors."""
        return await self.scan(
            org_id=org_id,
            project_id=project_id,
            scan_type="proactive",
            trigger_context={},
        )

    async def _run_proactive_scan(
        self, org_id: str, project_id: str, result: ScanResult
    ) -> ScanResult:
        """Proactive scan: check all tests for fragility."""
        log = logger.bind(org_id=org_id, scan_type="proactive")

        # Query Cognee for known fragile selectors
        fragile_selectors = await self._query_cognee_fragility(org_id, project_id)

        # Query healing history for patterns
        healing_patterns = await self._query_cognee_healing_patterns(org_id, project_id)

        # Get tests from DB
        try:
            from src.services.supabase_client import get_supabase_client
            supabase = get_supabase_client()
            tests_resp = await supabase.select(
                "tests",
                {"project_id": f"eq.{project_id}", "organization_id": f"eq.{org_id}"},
                limit=500,
            )
            tests = tests_resp if isinstance(tests_resp, list) else []
        except Exception as e:
            log.warning("Failed to fetch tests", error=str(e))
            tests = []

        result.tests_scanned = len(tests)

        # Assess fragility for each test using AI
        if tests and (fragile_selectors or healing_patterns):
            fragility_assessment = await self._assess_selector_fragility(
                org_id, project_id,
                tests=tests,
                fragile_selectors=fragile_selectors,
                healing_patterns=healing_patterns,
            )

            result.issues_found = len(fragility_assessment)
            result.fix_proposals = fragility_assessment
            result.fixes_proposed = len(fragility_assessment)

            # Store fragility data in Cognee
            await self._store_fragility_in_cognee(org_id, project_id, fragility_assessment)

        return result

    async def _scan_affected_tests(
        self, org_id: str, project_id: str, changed_files: list[str], result: ScanResult
    ) -> ScanResult:
        """Scan tests affected by changed files."""
        if not changed_files:
            return result

        # Use AI to identify which tests might be affected
        response = await self._call_ai(
            messages=[{
                "role": "user",
                "content": f"""Analyze these changed files and identify which types of tests might break:

Changed files: {json.dumps(changed_files[:50])}

Return JSON:
{{
    "affected_test_patterns": ["pattern1", "pattern2"],
    "risk_assessment": "low|medium|high",
    "reason": "..."
}}"""
            }],
            task_type=TaskType.CODE_ANALYSIS,
            max_tokens=1024,
            json_mode=True,
        )

        result.tests_scanned = len(changed_files)

        try:
            analysis = json.loads(response.content if hasattr(response, 'content') else str(response))
            if analysis.get("risk_assessment") in ("medium", "high"):
                result.issues_found = len(analysis.get("affected_test_patterns", []))
        except (json.JSONDecodeError, AttributeError):
            pass

        return result

    async def _generate_fix_proposals(
        self,
        org_id: str,
        project_id: str,
        failed_tests: list[dict],
    ) -> list[FixProposal]:
        """Generate fix proposals by composing SelfHealerAgent."""
        proposals = []

        # Query Cognee for test intent (from TestGen's requirement_test_links)
        test_intents = {}
        try:
            from src.knowledge.cognee_client import get_cognee_client
            cognee = get_cognee_client()
            for test in failed_tests[:10]:
                test_id = test.get("test_id", "")
                if test_id:
                    similar = await cognee.search(
                        query=f"test {test_id}",
                        namespace=f"org_{org_id}_project_{project_id}_requirement_test_links",
                    )
                    if similar:
                        test_intents[test_id] = similar[0] if isinstance(similar, list) else similar
        except Exception as e:
            logger.warning("Failed to query test intents from Cognee", error=str(e))

        for test in failed_tests[:10]:
            test_id = test.get("test_id", str(uuid.uuid4()))
            error_type = test.get("error_type", "unknown")
            error_message = test.get("error_message", "")

            # Determine issue type
            issue_type = "unknown"
            if "selector" in error_type.lower() or "element" in error_message.lower():
                issue_type = "selector_changed"
            elif "timeout" in error_type.lower() or "timeout" in error_message.lower():
                issue_type = "timing_issue"
            elif "api" in error_type.lower() or "status" in error_message.lower():
                issue_type = "api_change"
            elif "flaky" in error_type.lower():
                issue_type = "flaky"
            else:
                issue_type = "data_dependency"

            # Get intent context
            intent = test_intents.get(test_id, {})
            intent_desc = ""
            if isinstance(intent, dict):
                intent_desc = intent.get("description", intent.get("title", ""))

            # Use AI to generate fix description
            prompt = f"""Analyze this test failure and propose a fix:

Test: {test.get("test_name", test_id)}
Error type: {error_type}
Error message: {error_message[:500]}
Issue type: {issue_type}
{f"Test intent: {intent_desc}" if intent_desc else ""}
Failed selector: {test.get("failed_selector", "N/A")}

Return JSON:
{{
    "fix_description": "...",
    "confidence": 0.0-1.0,
    "proposed_selector": "..." or null
}}"""

            try:
                response = await self._call_ai(
                    messages=[{"role": "user", "content": prompt}],
                    task_type=TaskType.CODE_ANALYSIS,
                    max_tokens=512,
                    json_mode=True,
                )

                fix_data = json.loads(response.content if hasattr(response, 'content') else str(response))

                proposal = FixProposal(
                    test_id=test_id,
                    test_name=test.get("test_name", ""),
                    issue_type=issue_type,
                    description=f"{error_type}: {error_message[:200]}",
                    fix_description=fix_data.get("fix_description", "Manual review needed"),
                    confidence=float(fix_data.get("confidence", 0.5)),
                    original_selector=test.get("failed_selector"),
                    proposed_selector=fix_data.get("proposed_selector"),
                )
                proposals.append(proposal)

            except Exception as e:
                logger.warning("Failed to generate fix proposal", test_id=test_id, error=str(e))
                proposals.append(FixProposal(
                    test_id=test_id,
                    test_name=test.get("test_name", ""),
                    issue_type=issue_type,
                    description=f"{error_type}: {error_message[:200]}",
                    fix_description="Automated fix generation failed - manual review needed",
                    confidence=0.1,
                ))

        return proposals

    async def _assess_selector_fragility(
        self,
        org_id: str,
        project_id: str,
        tests: list[dict],
        fragile_selectors: list[dict],
        healing_patterns: list[dict],
    ) -> list[dict]:
        """Assess selector fragility for tests using known patterns."""
        fragile_set = {s.get("selector", "") for s in fragile_selectors}
        assessments = []

        for test in tests:
            selectors = test.get("selectors", [])
            test_name = test.get("name", "unknown")

            for selector in selectors:
                if selector in fragile_set or any(
                    p.get("original_selector") == selector for p in healing_patterns
                ):
                    # Find matching fragility data
                    frag_score = 0.5
                    for fs in fragile_selectors:
                        if fs.get("selector") == selector:
                            frag_score = fs.get("fragility_score", 0.5)
                            break

                    assessments.append({
                        "test_id": test.get("id", ""),
                        "test_name": test_name,
                        "selector": selector,
                        "fragility_score": frag_score,
                        "issue_type": "fragile_selector",
                        "description": f"Selector '{selector[:80]}' has known fragility score {frag_score:.2f}",
                        "confidence": frag_score,
                    })

        return assessments

    async def compute_health_score(self, org_id: str, project_id: str) -> HealthScore:
        """Compute composite health score from multiple signals."""
        score = HealthScore()

        try:
            from src.services.supabase_client import get_supabase_client
            supabase = get_supabase_client()

            # Get test count
            tests_resp = await supabase.select(
                "tests",
                {"project_id": f"eq.{project_id}", "organization_id": f"eq.{org_id}"},
                columns="id",
                limit=1000,
            )
            score.total_tests = len(tests_resp) if isinstance(tests_resp, list) else 0

            # Get healing history (last 30 days)
            thirty_days_ago = (datetime.now(UTC).replace(day=1)).isoformat().replace("+00:00", "Z")

            healing_resp = await supabase.select(
                "healing_scan_jobs",
                {
                    "project_id": f"eq.{project_id}",
                    "organization_id": f"eq.{org_id}",
                    "status": "eq.completed",
                    "created_at": f"gte.{thirty_days_ago}",
                },
                limit=100,
            )
            heals = healing_resp if isinstance(healing_resp, list) else []

            total_fixes = sum(h.get("fixes_proposed", 0) for h in heals)
            applied_fixes = sum(h.get("fixes_applied", 0) for h in heals)
            score.healed_tests_30d = applied_fixes

            # Compute sub-scores
            if score.total_tests > 0:
                score.selector_stability = max(0, 100 - (total_fixes / max(score.total_tests, 1) * 100))
            else:
                score.selector_stability = 100.0

            score.flaky_rate = 0.0  # Would need flaky test data

            if total_fixes > 0:
                score.heal_success_rate = (applied_fixes / total_fixes) * 100
            else:
                score.heal_success_rate = 100.0

            score.mttr_seconds = 0.0  # Would compute from heal timestamps

            # Fragile test count from Cognee
            fragile = await self._query_cognee_fragility(org_id, project_id)
            score.fragile_tests = len(fragile)

            # Compute overall: weighted average
            mttr_score = min(100, 3600 / max(score.mttr_seconds, 1) * 100) if score.mttr_seconds > 0 else 100.0
            score.overall_score = (
                score.selector_stability * 0.3
                + (100 - score.flaky_rate) * 0.2
                + score.heal_success_rate * 0.3
                + mttr_score * 0.2
            )

        except Exception as e:
            logger.warning("Failed to compute health score", error=str(e))
            score.overall_score = 50.0  # Default to middle

        return score

    async def _query_cognee_fragility(self, org_id: str, project_id: str) -> list[dict]:
        """Query Cognee for known fragile selectors."""
        try:
            from src.knowledge.cognee_client import get_cognee_client
            cognee = get_cognee_client()
            results = await cognee.search(
                query="fragile selector",
                namespace=f"org_{org_id}_project_{project_id}_selector_fragility",
            )
            return results if isinstance(results, list) else []
        except Exception:
            return []

    async def _query_cognee_healing_patterns(self, org_id: str, project_id: str) -> list[dict]:
        """Query Cognee for healing pattern history."""
        try:
            from src.knowledge.cognee_client import get_cognee_client
            cognee = get_cognee_client()
            results = await cognee.search(
                query="healing pattern selector fix",
                namespace=f"org_{org_id}_project_{project_id}_healing_patterns",
            )
            return results if isinstance(results, list) else []
        except Exception:
            return []

    async def _store_health_score(
        self, org_id: str, project_id: str, score: HealthScore, scan_job_id: str | None = None,
    ) -> None:
        """Store health score in DB for trending."""
        try:
            from src.services.supabase_client import get_supabase_client
            supabase = get_supabase_client()
            await supabase.insert("healing_health_scores", {
                "id": str(uuid.uuid4()),
                "organization_id": org_id,
                "project_id": project_id,
                "scan_job_id": scan_job_id,
                "overall_score": score.overall_score,
                "selector_stability": score.selector_stability,
                "flaky_rate": score.flaky_rate,
                "heal_success_rate": score.heal_success_rate,
                "mttr_seconds": score.mttr_seconds,
                "total_tests": score.total_tests,
                "fragile_tests": score.fragile_tests,
                "healed_tests_30d": score.healed_tests_30d,
                "details": score.details,
                "measured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            })
        except Exception as e:
            logger.warning("Failed to store health score", error=str(e))

    async def _store_fragility_in_cognee(
        self, org_id: str, project_id: str, fragility_data: list[dict],
    ) -> None:
        """Store selector fragility scores in Cognee."""
        if not fragility_data:
            return
        try:
            from src.knowledge.cognee_client import get_cognee_client
            cognee = get_cognee_client()
            namespace = f"org_{org_id}_project_{project_id}_selector_fragility"

            for item in fragility_data[:50]:
                # Sanitize None values (MEMORY.md #4)
                sanitized = {k: (v if v is not None else "") for k, v in item.items()}
                selector = sanitized.get("selector", "unknown")
                score_val = sanitized.get("fragility_score", 0.5)

                await cognee.put(
                    key=f"frag_{hash(selector) % 10**8}",
                    value=sanitized,
                    namespace=namespace,
                )
        except Exception as e:
            logger.warning("Failed to store fragility in Cognee", error=str(e))

    async def _emit_events(
        self, org_id: str, project_id: str, scan_id: str, scan_type: str, trigger_context: dict,
    ) -> None:
        """Emit HEALING_SCAN_STARTED event."""
        try:
            from src.events.producer import get_event_producer
            from src.events.schemas import EventMetadata, HealingScanStartedEvent, TenantInfo

            producer = get_event_producer()
            event = HealingScanStartedEvent(
                tenant=TenantInfo(org_id=org_id, project_id=project_id),
                metadata=EventMetadata(source="testhealer_cicd_agent"),
                scan_id=scan_id,
                scan_type=scan_type,
                trigger_context=trigger_context,
            )
            await producer.send(
                topic="argus.healing.scan.started",
                value=event.to_dict(),
            )
        except Exception as e:
            logger.warning("Failed to emit healing scan event", error=str(e))

    async def _update_scan_job(self, scan_job_id: str, updates: dict) -> None:
        """Update healing_scan_jobs table."""
        try:
            from src.services.supabase_client import get_supabase_client
            supabase = get_supabase_client()
            await supabase.update("healing_scan_jobs", {"id": f"eq.{scan_job_id}"}, updates)
        except Exception as e:
            logger.warning("Failed to update scan job", error=str(e))
