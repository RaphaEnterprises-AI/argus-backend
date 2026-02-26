"""QA Assessment event handler for Cognee knowledge graph.

Consumes events:
- argus.coverage.analyzed -> Indexes coverage assessments for cross-session learning
- argus.coverage.gaps -> Indexes gap analyses for cross-project patterns

These enable:
- Incremental analysis (QA Engineer resumes from last state)
- Cross-project gap patterns ("auth is always under-tested")
- TestGen can query what gaps exist to auto-fill
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class QAAssessmentHandler:
    """Processes QA assessment events into Cognee knowledge graph."""

    HANDLED_EVENTS = ["coverage.analyzed", "coverage.gaps"]

    async def handle(self, event: dict) -> bool:
        """Route event to appropriate handler.

        Args:
            event: Deserialized Kafka event dict

        Returns:
            True if processed successfully, False otherwise
        """
        event_type = event.get("event_type", "")
        if event_type == "coverage.analyzed":
            return await self.handle_coverage_analyzed(event)
        elif event_type == "coverage.gaps":
            return await self.handle_coverage_gaps(event)
        else:
            logger.warning(
                f"Unknown event type for QAAssessmentHandler: {event_type}"
            )
            return False

    async def handle_coverage_analyzed(self, event: dict) -> bool:
        """Index coverage assessment for cross-session continuity."""
        import cognee

        tenant = event.get("tenant", {})
        org_id = tenant.get("org_id", "unknown")
        project_id = tenant.get("project_id", "unknown")

        job_id = event.get("job_id", "unknown")
        analysis_type = event.get("analysis_type", "unknown")
        quality_score = event.get("quality_score", 0.0)
        coverage_pct = event.get("test_coverage_pct", 0.0)
        components = event.get("components_analyzed", 0)
        flows = event.get("user_flows_discovered", 0)
        gaps = event.get("gap_count", 0)
        critical = event.get("critical_gaps", 0)
        commit_sha = event.get("commit_sha", "unknown")

        dataset_name = f"org_{org_id}_project_{project_id}_coverage_analysis"

        logger.info(
            f"Indexing coverage analysis into {dataset_name}",
            extra={"job_id": job_id, "quality_score": quality_score},
        )

        start = time.time()
        try:
            content = (
                f"Coverage analysis ({analysis_type}) for commit {commit_sha}: "
                f"quality score {quality_score:.1f}/100, "
                f"test coverage {coverage_pct:.1f}%, "
                f"{components} components analyzed, "
                f"{flows} user flows discovered, "
                f"{gaps} coverage gaps ({critical} critical)."
            )

            await cognee.add(
                data_type="text",
                data=[content],
                dataset_name=dataset_name,
            )
            await cognee.cognify(dataset_name=dataset_name)

            # Also store user flows for cross-session continuity
            if flows > 0:
                flows_dataset = f"org_{org_id}_project_{project_id}_user_flows"
                flows_content = (
                    f"Discovered {flows} user flows during {analysis_type} analysis "
                    f"at commit {commit_sha}."
                )
                await cognee.add(
                    data_type="text",
                    data=[flows_content],
                    dataset_name=flows_dataset,
                )
                await cognee.cognify(dataset_name=flows_dataset)

            duration = time.time() - start
            logger.info(f"Indexed coverage analysis in {duration:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to index coverage analysis: {e}")
            return False

    async def handle_coverage_gaps(self, event: dict) -> bool:
        """Index coverage gaps for cross-project pattern learning."""
        import cognee

        tenant = event.get("tenant", {})
        org_id = tenant.get("org_id", "unknown")
        project_id = tenant.get("project_id", "unknown")

        job_id = event.get("job_id", "unknown")
        total = event.get("total_gaps", 0)
        critical = event.get("critical_gaps", 0)
        high = event.get("high_gaps", 0)
        top_gaps = event.get("top_gaps", [])

        dataset_name = f"org_{org_id}_project_{project_id}_coverage_gaps"

        logger.info(
            f"Indexing {total} coverage gaps into {dataset_name}",
            extra={"critical": critical, "high": high},
        )

        start = time.time()
        try:
            # Build searchable content from top gaps
            gap_descriptions = []
            for gap in top_gaps[:10]:
                entity = gap.get("entity", "unknown")
                risk = gap.get("risk_level", "unknown")
                desc = gap.get("description", "")
                gap_descriptions.append(f"- {entity} ({risk}): {desc}")

            content = (
                f"Coverage gap analysis: {total} gaps found "
                f"({critical} critical, {high} high). "
                f"Top gaps:\n" + "\n".join(gap_descriptions)
            )

            await cognee.add(
                data_type="text",
                data=[content],
                dataset_name=dataset_name,
            )
            await cognee.cognify(dataset_name=dataset_name)

            duration = time.time() - start
            logger.info(f"Indexed coverage gaps in {duration:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to index coverage gaps: {e}")
            return False
