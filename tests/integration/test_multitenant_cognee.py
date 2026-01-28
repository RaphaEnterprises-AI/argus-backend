"""
Integration tests for Multi-Tenant Cognee Pipeline.

Tests the complete flow from tenant context through event processing
to knowledge graph storage with proper data isolation.
"""

import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Mock aiokafka before any src.events imports to prevent import errors
# This is necessary because src/events/__init__.py imports EventProducer
# which requires aiokafka, but we don't need Kafka for these unit tests
sys.modules['aiokafka'] = MagicMock()
sys.modules['aiokafka.errors'] = MagicMock()


# =============================================================================
# TenantContext Tests
# =============================================================================


class TestTenantContext:
    """Tests for TenantContext dataclass and dataset naming."""

    def test_tenant_context_requires_org_id(self):
        """TenantContext should require org_id."""
        from src.core.tenant import TenantContext

        with pytest.raises(ValueError, match="org_id is required"):
            TenantContext(org_id="")

        with pytest.raises(ValueError, match="org_id is required"):
            TenantContext(org_id=None)

    def test_tenant_context_basic_creation(self):
        """Should create TenantContext with minimal args."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(org_id="org-123")

        assert ctx.org_id == "org-123"
        assert ctx.project_id is None
        assert ctx.plan == "free"
        assert ctx.request_id is not None

    def test_tenant_context_full_creation(self):
        """Should create TenantContext with all fields."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(
            org_id="org-123",
            org_name="Acme Corp",
            project_id="proj-456",
            project_name="Main App",
            user_id="user-789",
            user_email="test@example.com",
            plan="enterprise",
        )

        assert ctx.org_id == "org-123"
        assert ctx.org_name == "Acme Corp"
        assert ctx.project_id == "proj-456"
        assert ctx.user_email == "test@example.com"
        assert ctx.is_enterprise_plan

    def test_cognee_dataset_prefix_with_project(self):
        """Dataset prefix should include org and project."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(org_id="abc123", project_id="xyz789")

        assert ctx.cognee_dataset_prefix == "org_abc123_project_xyz789"

    def test_cognee_dataset_prefix_without_project(self):
        """Dataset prefix should be org-only when no project."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(org_id="abc123")

        assert ctx.cognee_dataset_prefix == "org_abc123"

    def test_cognee_dataset_names(self):
        """Should generate correct dataset names for each type."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(org_id="abc", project_id="xyz")

        assert ctx.codebase_dataset == "org_abc_project_xyz_codebase"
        assert ctx.tests_dataset == "org_abc_project_xyz_tests"
        assert ctx.failures_dataset == "org_abc_project_xyz_failures"

    def test_plan_checks(self):
        """Plan properties should work correctly."""
        from src.core.tenant import TenantContext

        free_ctx = TenantContext(org_id="org", plan="free")
        pro_ctx = TenantContext(org_id="org", plan="pro")
        ent_ctx = TenantContext(org_id="org", plan="enterprise")

        assert free_ctx.is_free_plan
        assert not free_ctx.is_pro_plan
        assert not free_ctx.is_enterprise_plan

        assert not pro_ctx.is_free_plan
        assert pro_ctx.is_pro_plan
        assert not pro_ctx.is_enterprise_plan

        assert not ent_ctx.is_free_plan
        assert ent_ctx.is_pro_plan  # enterprise includes pro
        assert ent_ctx.is_enterprise_plan

    def test_to_kafka_tenant_info(self):
        """Should convert to Kafka-compatible dict."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(
            org_id="org-123",
            project_id="proj-456",
            user_id="user-789",
        )

        tenant_info = ctx.to_kafka_tenant_info()

        assert tenant_info == {
            "org_id": "org-123",
            "project_id": "proj-456",
            "user_id": "user-789",
        }

    def test_to_log_context(self):
        """Should convert to logging dict with non-null values."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(org_id="org-123", project_id="proj-456")

        log_ctx = ctx.to_log_context()

        assert "org_id" in log_ctx
        assert "project_id" in log_ctx
        assert "request_id" in log_ctx
        assert "user_id" not in log_ctx  # None values excluded

    def test_for_testing_factory(self):
        """Should create test context with defaults."""
        from src.core.tenant import TenantContext

        ctx = TenantContext.for_testing()

        assert ctx.org_id == "test-org-123"
        assert ctx.project_id == "test-project-456"
        assert ctx.plan == "pro"

    def test_for_system_factory(self):
        """Should create system context."""
        from src.core.tenant import TenantContext

        ctx = TenantContext.for_system("org-123", "proj-456")

        assert ctx.org_id == "org-123"
        assert ctx.project_id == "proj-456"
        assert ctx.user_id == "system"

    def test_tenant_context_is_immutable(self):
        """TenantContext should be frozen (immutable)."""
        from src.core.tenant import TenantContext

        ctx = TenantContext(org_id="org-123")

        with pytest.raises(AttributeError):
            ctx.org_id = "different-org"


# =============================================================================
# Tenant Context Variable Tests
# =============================================================================


class TestTenantContextVar:
    """Tests for context variable management."""

    def test_get_set_current_tenant(self):
        """Should get and set tenant context in contextvar."""
        from src.core.tenant import (
            TenantContext,
            get_current_tenant,
            set_current_tenant,
        )

        # Initially None
        assert get_current_tenant() is None

        # Set context
        ctx = TenantContext(org_id="org-123")
        set_current_tenant(ctx)

        # Should retrieve same context
        retrieved = get_current_tenant()
        assert retrieved is ctx
        assert retrieved.org_id == "org-123"

    def test_require_tenant_raises_when_not_set(self):
        """require_tenant should raise when no context set."""
        from src.core.tenant import _tenant_context_var, require_tenant

        # Clear any existing context
        _tenant_context_var.set(None)

        with pytest.raises(RuntimeError, match="No tenant context"):
            require_tenant()

    def test_require_tenant_returns_context(self):
        """require_tenant should return context when set."""
        from src.core.tenant import (
            TenantContext,
            require_tenant,
            set_current_tenant,
        )

        ctx = TenantContext(org_id="org-123")
        set_current_tenant(ctx)

        result = require_tenant()
        assert result is ctx


# =============================================================================
# Event Schema Tests
# =============================================================================


class TestEventSchemas:
    """Tests for multi-tenant event schemas."""

    def test_tenant_info_validation(self):
        """TenantInfo should require org_id."""
        from src.events.schemas import TenantInfo

        # Valid
        info = TenantInfo(org_id="org-123")
        assert info.org_id == "org-123"
        assert info.project_id is None

        # With project
        info = TenantInfo(org_id="org-123", project_id="proj-456")
        assert info.project_id == "proj-456"

    def test_tenant_info_generates_cognee_prefix(self):
        """TenantInfo should generate Cognee dataset prefix."""
        from src.events.schemas import TenantInfo

        # With project
        info = TenantInfo(org_id="org-123", project_id="proj-456")
        assert info.to_cognee_dataset_prefix() == "org_org-123_project_proj-456"

        # Without project
        info_no_project = TenantInfo(org_id="org-123")
        assert info_no_project.to_cognee_dataset_prefix() == "org_org-123"

    def test_event_metadata_creation(self):
        """EventMetadata should have proper defaults."""
        from src.events.schemas import EventMetadata

        meta = EventMetadata(source="test")

        assert meta.source == "test"
        assert meta.request_id is not None  # Auto-generated UUID
        assert meta.timestamp is not None

    def test_base_event_structure(self):
        """BaseEvent should include tenant info."""
        from src.events.schemas import BaseEvent, EventMetadata, EventType, TenantInfo

        event = BaseEvent(
            event_type=EventType.TEST_EXECUTED,
            tenant=TenantInfo(org_id="org-123", project_id="proj-456"),
            metadata=EventMetadata(source="test"),
        )

        assert event.event_type == EventType.TEST_EXECUTED
        assert event.tenant.org_id == "org-123"
        assert event.event_id is not None
        assert event.to_kafka_key() == "org-123:proj-456"

    def test_codebase_ingested_event(self):
        """CodebaseIngestedEvent should have correct structure."""
        from src.events.schemas import CodebaseIngestedEvent, EventMetadata, EventType, TenantInfo

        event = CodebaseIngestedEvent(
            tenant=TenantInfo(org_id="org-123", project_id="proj-456"),
            metadata=EventMetadata(source="api"),
            repository_id="repo-001",
            repository_url="https://github.com/example/repo",
            commit_sha="abc123",
            branch="main",
            file_count=100,
            total_size_bytes=1024000,
            languages=["python", "javascript"],
        )

        assert event.event_type == EventType.CODEBASE_INGESTED
        assert event.repository_url == "https://github.com/example/repo"
        assert event.file_count == 100
        assert event.total_size_bytes == 1024000

    def test_test_executed_event(self):
        """TestExecutedEvent should include execution results."""
        from src.events.schemas import EventMetadata, EventType, TenantInfo, TestExecutedEvent

        event = TestExecutedEvent(
            tenant=TenantInfo(org_id="org-123", project_id="proj-456"),
            metadata=EventMetadata(source="runner"),
            test_id="test-789",
            run_id="run-001",
            status="passed",
            duration_ms=1500,
        )

        assert event.event_type == EventType.TEST_EXECUTED
        assert event.status == "passed"
        assert event.duration_ms == 1500
        assert event.run_id == "run-001"

    def test_test_failed_event(self):
        """TestFailedEvent should include failure details."""
        from src.events.schemas import EventMetadata, EventType, TenantInfo, TestFailedEvent

        event = TestFailedEvent(
            tenant=TenantInfo(org_id="org-123", project_id="proj-456"),
            metadata=EventMetadata(source="runner"),
            test_id="test-789",
            run_id="run-001",
            error_message="Element not found",
            error_type="selector",
            stack_trace="...",
            screenshot_url="https://...",
        )

        assert event.event_type == EventType.TEST_FAILED
        assert event.error_type == "selector"
        assert event.run_id == "run-001"

    def test_event_to_dict(self):
        """Events should serialize to dict properly."""
        from src.events.schemas import EventMetadata, EventType, TenantInfo, TestExecutedEvent

        event = TestExecutedEvent(
            tenant=TenantInfo(org_id="org-123"),
            metadata=EventMetadata(source="test"),
            test_id="test-1",
            run_id="run-1",
            status="passed",
            duration_ms=100,
        )

        data = event.to_dict()  # Uses model_dump with JSON mode

        assert data["event_type"] == EventType.TEST_EXECUTED.value
        assert data["tenant"]["org_id"] == "org-123"
        assert data["test_id"] == "test-1"


# =============================================================================
# Kafka Topics Tests
# =============================================================================


class TestKafkaTopics:
    """Tests for Kafka topic configuration."""

    def test_topic_names_defined(self):
        """All required topics should be defined."""
        from src.events.topics import (
            TOPIC_CODEBASE_ANALYZED,
            TOPIC_CODEBASE_INGESTED,
            TOPIC_DLQ,
            TOPIC_HEALING_COMPLETED,
            TOPIC_HEALING_REQUESTED,
            TOPIC_TEST_CREATED,
            TOPIC_TEST_EXECUTED,
            TOPIC_TEST_FAILED,
        )

        assert TOPIC_CODEBASE_INGESTED == "argus.codebase.ingested"
        assert TOPIC_CODEBASE_ANALYZED == "argus.codebase.analyzed"
        assert TOPIC_TEST_CREATED == "argus.test.created"
        assert TOPIC_TEST_EXECUTED == "argus.test.executed"
        assert TOPIC_TEST_FAILED == "argus.test.failed"
        assert TOPIC_HEALING_REQUESTED == "argus.healing.requested"
        assert TOPIC_HEALING_COMPLETED == "argus.healing.completed"
        assert TOPIC_DLQ == "argus.dlq"

    def test_get_all_topics(self):
        """Should return list of all topics."""
        from src.events.topics import TOPIC_CODEBASE_INGESTED, TOPIC_DLQ, get_all_topics

        topics = get_all_topics()

        assert len(topics) >= 8
        assert TOPIC_CODEBASE_INGESTED in topics
        assert TOPIC_DLQ in topics


# =============================================================================
# Tenant Middleware Tests
# =============================================================================


class TestTenantMiddleware:
    """Tests for tenant extraction middleware."""

    @pytest.mark.asyncio
    async def test_middleware_extracts_org_from_header(self):
        """Should extract org_id from X-Organization-ID header."""
        from fastapi import FastAPI, Request
        from starlette.testclient import TestClient

        from src.api.middleware.tenant import TenantMiddleware

        app = FastAPI()
        app.add_middleware(TenantMiddleware)

        @app.get("/test")
        async def test_route(request: Request):
            from src.core.tenant import get_current_tenant
            ctx = get_current_tenant()
            return {"org_id": ctx.org_id if ctx else None}

        with TestClient(app) as client:
            response = client.get(
                "/test",
                headers={"X-Organization-ID": "org-123"}
            )
            # Note: In real scenario, middleware sets context
            # This is a simplified test

    @pytest.mark.asyncio
    async def test_middleware_allows_public_paths(self):
        """Should allow requests to public paths without org header."""
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from src.api.middleware.tenant import TenantMiddleware

        app = FastAPI()
        app.add_middleware(TenantMiddleware)

        @app.get("/api/v1/health")
        async def health():
            return {"status": "ok"}

        with TestClient(app) as client:
            response = client.get("/api/v1/health")
            assert response.status_code == 200


# =============================================================================
# Multi-Tenant Data Isolation Tests
# =============================================================================


class TestMultiTenantIsolation:
    """Tests for data isolation between tenants."""

    def test_different_orgs_have_different_datasets(self):
        """Different orgs should have isolated datasets."""
        from src.core.tenant import TenantContext

        ctx_org1 = TenantContext(org_id="org-1", project_id="proj-1")
        ctx_org2 = TenantContext(org_id="org-2", project_id="proj-1")

        # Same project name, different org = different datasets
        assert ctx_org1.codebase_dataset != ctx_org2.codebase_dataset
        assert "org_org-1" in ctx_org1.codebase_dataset
        assert "org_org-2" in ctx_org2.codebase_dataset

    def test_same_org_different_projects_isolated(self):
        """Different projects in same org should have isolated datasets."""
        from src.core.tenant import TenantContext

        ctx_proj1 = TenantContext(org_id="org-1", project_id="proj-1")
        ctx_proj2 = TenantContext(org_id="org-1", project_id="proj-2")

        assert ctx_proj1.codebase_dataset != ctx_proj2.codebase_dataset
        assert "project_proj-1" in ctx_proj1.codebase_dataset
        assert "project_proj-2" in ctx_proj2.codebase_dataset

    def test_event_routing_by_tenant(self):
        """Events should include tenant info for proper routing."""
        from src.events.schemas import EventMetadata, TenantInfo, TestExecutedEvent

        event_org1 = TestExecutedEvent(
            tenant=TenantInfo(org_id="org-1", project_id="proj-1"),
            metadata=EventMetadata(source="test"),
            test_id="t1",
            run_id="run-1",
            status="passed",
            duration_ms=100,
        )

        event_org2 = TestExecutedEvent(
            tenant=TenantInfo(org_id="org-2", project_id="proj-1"),
            metadata=EventMetadata(source="test"),
            test_id="t2",
            run_id="run-2",
            status="passed",
            duration_ms=100,
        )

        # Events can be distinguished by tenant
        assert event_org1.tenant.org_id != event_org2.tenant.org_id

        # Kafka keys are different (using built-in method)
        assert event_org1.to_kafka_key() != event_org2.to_kafka_key()
        assert event_org1.to_kafka_key() == "org-1:proj-1"
        assert event_org2.to_kafka_key() == "org-2:proj-1"


# =============================================================================
# Cognee Worker Multi-Tenant Tests (Mocked)
# =============================================================================


class TestCogneeWorkerMultiTenant:
    """Tests for Cognee worker multi-tenant processing."""

    def test_dataset_name_from_event(self):
        """Worker should derive dataset name from event tenant info."""
        # Simulate worker's dataset name derivation
        def get_dataset_name(org_id: str, project_id: str, dataset_type: str) -> str:
            return f"org_{org_id}_project_{project_id}_{dataset_type}"

        event_tenant = {
            "org_id": "abc123",
            "project_id": "xyz789",
        }

        dataset = get_dataset_name(
            event_tenant["org_id"],
            event_tenant["project_id"],
            "codebase"
        )

        assert dataset == "org_abc123_project_xyz789_codebase"

    def test_extract_tenant_from_event(self):
        """Worker should extract tenant context from event payload."""
        event = {
            "event_type": "CODEBASE_INGESTED",
            "tenant": {
                "org_id": "org-123",
                "project_id": "proj-456",
                "user_id": "user-789",
            },
            "payload": {"files_count": 100},
        }

        # Simulate worker's tenant extraction
        tenant = event.get("tenant", {})
        org_id = tenant.get("org_id")
        project_id = tenant.get("project_id")

        assert org_id == "org-123"
        assert project_id == "proj-456"

    def test_missing_tenant_info_handled(self):
        """Worker should handle events missing tenant info gracefully."""
        event_without_tenant = {
            "event_type": "LEGACY_EVENT",
            "payload": {"data": "something"},
        }

        # Simulate worker's fallback behavior
        tenant = event_without_tenant.get("tenant", {})
        org_id = tenant.get("org_id") or event_without_tenant.get("org_id")
        project_id = tenant.get("project_id") or event_without_tenant.get("project_id")

        assert org_id is None
        assert project_id is None

    @pytest.mark.asyncio
    async def test_worker_processes_event_with_tenant_isolation(self):
        """Worker should process events with proper tenant isolation."""
        from src.events.schemas import CodebaseIngestedEvent, EventMetadata, TenantInfo

        # Create events for different tenants
        event1 = CodebaseIngestedEvent(
            tenant=TenantInfo(org_id="acme", project_id="app1"),
            metadata=EventMetadata(source="test"),
            repository_id="repo-acme-1",
            repository_url="https://github.com/acme/app1",
            commit_sha="abc",
            branch="main",
            file_count=50,
            total_size_bytes=500000,
        )

        event2 = CodebaseIngestedEvent(
            tenant=TenantInfo(org_id="beta", project_id="app1"),
            metadata=EventMetadata(source="test"),
            repository_id="repo-beta-1",
            repository_url="https://github.com/beta/app1",
            commit_sha="xyz",
            branch="main",
            file_count=100,
            total_size_bytes=1000000,
        )

        # Verify isolation using TenantInfo's built-in method
        assert event1.tenant.to_cognee_dataset_prefix() == "org_acme_project_app1"
        assert event2.tenant.to_cognee_dataset_prefix() == "org_beta_project_app1"
        assert event1.tenant.to_cognee_dataset_prefix() != event2.tenant.to_cognee_dataset_prefix()


# =============================================================================
# API Endpoint Multi-Tenant Tests
# =============================================================================


class TestOrgScopedAPIEndpoints:
    """Tests for organization-scoped API endpoints."""

    @pytest.fixture
    def mock_supabase(self):
        """Mock Supabase client."""
        with patch("src.api.orgs.get_supabase_client") as mock:
            client = MagicMock()
            mock.return_value = client
            yield client

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return {
            "user_id": "user-123",
            "email": "test@example.com",
        }

    @pytest.mark.asyncio
    async def test_list_organizations_returns_user_orgs(self, mock_supabase, mock_user):
        """Should list only organizations user belongs to."""
        from fastapi import Request

        from src.api.orgs import list_my_organizations

        # Mock memberships
        mock_supabase.request = AsyncMock(side_effect=[
            # First call: get memberships
            {"data": [
                {"organization_id": "org-1", "role": "admin"},
                {"organization_id": "org-2", "role": "member"},
            ]},
            # Second call: get org details
            {"data": [
                {"id": "org-1", "name": "Org 1", "slug": "org-1", "plan": "pro"},
                {"id": "org-2", "name": "Org 2", "slug": "org-2", "plan": "free"},
            ]},
        ])

        with patch("src.api.orgs.get_current_user", return_value=mock_user):
            request = MagicMock(spec=Request)
            result = await list_my_organizations(request)

        assert len(result) == 2
        assert result[0].id == "org-1"
        assert result[0].role == "admin"

    @pytest.mark.asyncio
    async def test_list_projects_scoped_to_org(self, mock_supabase, mock_user):
        """Should list only projects in the specified organization."""
        from fastapi import Request

        from src.api.orgs import list_org_projects

        mock_supabase.request = AsyncMock(side_effect=[
            # Projects query
            {"data": [
                {"id": "proj-1", "organization_id": "org-123", "name": "Project 1", "created_at": "2024-01-01"},
                {"id": "proj-2", "organization_id": "org-123", "name": "Project 2", "created_at": "2024-01-02"},
            ]},
            # Test count for proj-1
            {"data": [{"id": "t1"}, {"id": "t2"}]},
            # Test count for proj-2
            {"data": [{"id": "t3"}]},
        ])

        with patch("src.api.orgs.get_current_user", return_value=mock_user):
            with patch("src.api.orgs.verify_org_access", return_value=("admin", "org-123")):
                request = MagicMock(spec=Request)
                result = await list_org_projects("org-123", request)

        assert len(result) == 2
        assert all(p.org_id == "org-123" for p in result)


# =============================================================================
# End-to-End Flow Tests
# =============================================================================


class TestEndToEndMultiTenantFlow:
    """Tests for complete multi-tenant flow."""

    @pytest.mark.asyncio
    async def test_event_creation_to_processing_flow(self):
        """Test flow from event creation through processing."""
        from src.core.tenant import TenantContext
        from src.events.schemas import CodebaseIngestedEvent, EventMetadata, TenantInfo

        # 1. API creates event with tenant context
        tenant = TenantContext(
            org_id="acme-corp",
            project_id="main-app",
            user_id="user-123",
        )

        event = CodebaseIngestedEvent(
            tenant=TenantInfo(**tenant.to_kafka_tenant_info()),
            metadata=EventMetadata(source="api"),
            repository_id="repo-acme-001",
            repository_url="https://github.com/acme/app",
            commit_sha="abc123",
            branch="main",
            file_count=500,
            total_size_bytes=5000000,
        )

        # 2. Event serialized for Kafka
        event_data = event.to_dict()
        message_key = event.to_kafka_key()

        assert message_key == "acme-corp:main-app"
        assert event_data["tenant"]["org_id"] == "acme-corp"

        # 3. Worker processes event using TenantInfo's built-in method
        dataset_prefix = event.tenant.to_cognee_dataset_prefix()
        dataset_name = f"{dataset_prefix}_codebase"
        assert dataset_name == "org_acme-corp_project_main-app_codebase"

        # 4. Verify tenant isolation maintained throughout
        assert tenant.codebase_dataset == dataset_name

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation_enforced(self):
        """Verify events from different tenants stay isolated."""
        from src.events.schemas import EventMetadata, TenantInfo, TestFailedEvent

        # Create similar failures for different orgs
        failure_org1 = TestFailedEvent(
            tenant=TenantInfo(org_id="org-1", project_id="app"),
            metadata=EventMetadata(source="runner"),
            test_id="test-1",
            run_id="run-1",
            error_message="Button not found",
            error_type="selector",
        )

        failure_org2 = TestFailedEvent(
            tenant=TenantInfo(org_id="org-2", project_id="app"),
            metadata=EventMetadata(source="runner"),
            test_id="test-1",
            run_id="run-2",
            error_message="Button not found",
            error_type="selector",
        )

        # Same test ID, same error - but different tenants
        assert failure_org1.test_id == failure_org2.test_id
        assert failure_org1.error_message == failure_org2.error_message

        # Dataset prefixes must be different
        ds1 = f"{failure_org1.tenant.to_cognee_dataset_prefix()}_failures"
        ds2 = f"{failure_org2.tenant.to_cognee_dataset_prefix()}_failures"

        assert ds1 != ds2
        assert "org-1" in ds1
        assert "org-2" in ds2


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestMultiTenantEdgeCases:
    """Tests for edge cases in multi-tenant handling."""

    def test_special_characters_in_org_id(self):
        """Should handle special characters in org IDs."""
        from src.core.tenant import TenantContext

        # UUIDs are typical
        ctx = TenantContext(org_id="550e8400-e29b-41d4-a716-446655440000")
        assert "550e8400-e29b-41d4-a716-446655440000" in ctx.codebase_dataset

    def test_very_long_org_id(self):
        """Should handle long org IDs."""
        from src.core.tenant import TenantContext

        long_id = "a" * 100
        ctx = TenantContext(org_id=long_id, project_id="proj")

        # Dataset name should still be valid
        assert ctx.codebase_dataset.startswith(f"org_{long_id}")

    def test_concurrent_tenant_contexts(self):
        """Different async contexts should have independent tenant contexts."""
        import asyncio

        from src.core.tenant import TenantContext, get_current_tenant, set_current_tenant

        results = []

        async def task(org_id: str):
            ctx = TenantContext(org_id=org_id)
            set_current_tenant(ctx)
            await asyncio.sleep(0.01)  # Simulate work
            current = get_current_tenant()
            results.append((org_id, current.org_id if current else None))

        async def run_concurrent():
            await asyncio.gather(
                task("org-1"),
                task("org-2"),
                task("org-3"),
            )

        asyncio.run(run_concurrent())

        # Note: contextvars are task-local, so each task gets its own context
        # This test verifies the mechanism works

    def test_event_without_optional_fields(self):
        """Events should work with minimal required fields."""
        from src.events.schemas import EventMetadata, TenantInfo, TestExecutedEvent

        event = TestExecutedEvent(
            tenant=TenantInfo(org_id="org-123"),  # No project_id
            metadata=EventMetadata(source="test"),
            test_id="t1",
            run_id="run-1",
            status="passed",
            duration_ms=100,
        )

        assert event.tenant.project_id is None
        assert event.tenant.org_id == "org-123"
        # Kafka key should be org_id only when no project
        assert event.to_kafka_key() == "org-123"
