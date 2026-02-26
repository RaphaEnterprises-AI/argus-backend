"""TestGen event handler for Cognee knowledge graph.

Consumes events:
- argus.test.generated -> Indexes generated tests in Cognee
- argus.requirements.analyzed -> Indexes parsed requirements in Cognee

These enable cross-agent learning:
- TestHealer can query generated test patterns
- QA Engineer can find what's already been auto-generated
- Future test generation can reference past generations
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class TestGenHandler:
    """Processes test generation events into Cognee knowledge graph."""

    HANDLED_EVENTS = ["test.generated", "requirements.analyzed"]

    async def handle(self, event: dict) -> bool:
        """Route event to appropriate handler.

        Args:
            event: Deserialized Kafka event dict

        Returns:
            True if processed successfully, False otherwise
        """
        event_type = event.get("event_type", "")
        if event_type == "test.generated":
            return await self.handle_test_generated(event)
        elif event_type == "requirements.analyzed":
            return await self.handle_requirements_analyzed(event)
        else:
            logger.warning(f"Unknown event type for TestGenHandler: {event_type}")
            return False

    async def handle_test_generated(self, event: dict) -> bool:
        """Index generated tests in Cognee for cross-agent discovery."""
        import cognee

        tenant = event.get("tenant", {})
        org_id = tenant.get("org_id", "unknown")
        project_id = tenant.get("project_id", "unknown")

        job_id = event.get("job_id", "unknown")
        tests_generated = event.get("tests_generated", 0)
        framework = event.get("framework", "unknown")
        source_type = event.get("source_type", "unknown")
        quality_score = event.get("quality_score", 0.0)

        dataset_name = f"org_{org_id}_project_{project_id}_generated_tests"

        logger.info(
            f"Indexing {tests_generated} generated tests into {dataset_name}",
            extra={"job_id": job_id, "framework": framework},
        )

        start = time.time()
        try:
            # Build text content for Cognee ingestion
            content = (
                f"Test generation job {job_id} produced {tests_generated} tests "
                f"using {framework} framework from {source_type} source. "
                f"Quality score: {quality_score:.2f}."
            )

            # Add to Cognee dataset
            await cognee.add(
                data_type="text",
                data=[content],
                dataset_name=dataset_name,
            )

            # Run cognify pipeline to build knowledge graph
            await cognee.cognify(dataset_name=dataset_name)

            duration = time.time() - start
            logger.info(
                f"Indexed test generation event in {duration:.2f}s",
                extra={"dataset": dataset_name, "job_id": job_id},
            )
            return True

        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"Failed to index test generation event: {e}",
                extra={
                    "dataset": dataset_name,
                    "job_id": job_id,
                    "duration": duration,
                },
            )
            return False

    async def handle_requirements_analyzed(self, event: dict) -> bool:
        """Index analyzed requirements in Cognee for cross-agent discovery."""
        import cognee

        tenant = event.get("tenant", {})
        org_id = tenant.get("org_id", "unknown")
        project_id = tenant.get("project_id", "unknown")

        job_id = event.get("job_id", "unknown")
        requirements_count = event.get("requirements_count", 0)
        ambiguous_count = event.get("ambiguous_count", 0)
        avg_ambiguity = event.get("avg_ambiguity_score", 0.0)
        source_type = event.get("source_type", "unknown")

        dataset_name = f"org_{org_id}_project_{project_id}_requirements"

        logger.info(
            f"Indexing {requirements_count} requirements into {dataset_name}",
            extra={"job_id": job_id, "ambiguous": ambiguous_count},
        )

        start = time.time()
        try:
            content = (
                f"Requirements analysis job {job_id} parsed {requirements_count} requirements "
                f"from {source_type} source. "
                f"Found {ambiguous_count} ambiguous requirements "
                f"(avg ambiguity: {avg_ambiguity:.2f})."
            )

            # Also add ambiguity patterns for cross-project learning
            if ambiguous_count > 0:
                ambiguity_dataset = (
                    f"org_{org_id}_project_{project_id}_ambiguity_patterns"
                )
                ambiguity_content = (
                    f"Ambiguity analysis: {ambiguous_count}/{requirements_count} requirements "
                    f"had ambiguity issues from {source_type} source. "
                    f"Average ambiguity score: {avg_ambiguity:.2f}."
                )
                await cognee.add(
                    data_type="text",
                    data=[ambiguity_content],
                    dataset_name=ambiguity_dataset,
                )
                await cognee.cognify(dataset_name=ambiguity_dataset)

            await cognee.add(
                data_type="text",
                data=[content],
                dataset_name=dataset_name,
            )
            await cognee.cognify(dataset_name=dataset_name)

            duration = time.time() - start
            logger.info(
                f"Indexed requirements event in {duration:.2f}s",
                extra={"dataset": dataset_name, "job_id": job_id},
            )
            return True

        except Exception as e:
            duration = time.time() - start
            logger.error(
                f"Failed to index requirements event: {e}",
                extra={
                    "dataset": dataset_name,
                    "job_id": job_id,
                    "duration": duration,
                },
            )
            return False
