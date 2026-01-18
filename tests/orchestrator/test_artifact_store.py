"""Tests for the artifact store module."""

from datetime import UTC, datetime


class TestArtifact:
    """Tests for the Artifact dataclass."""

    def test_artifact_creation(self, mock_env_vars):
        """Test Artifact creation with required fields."""
        from src.orchestrator.artifact_store import Artifact

        artifact = Artifact(
            id="screenshot_abc123",
            type="screenshot",
            content="base64_encoded_data",
        )

        assert artifact.id == "screenshot_abc123"
        assert artifact.type == "screenshot"
        assert artifact.content == "base64_encoded_data"
        assert artifact.metadata == {}
        assert artifact.created_at is not None

    def test_artifact_with_metadata(self, mock_env_vars):
        """Test Artifact creation with metadata."""
        from src.orchestrator.artifact_store import Artifact

        artifact = Artifact(
            id="video_xyz789",
            type="video",
            content="video_url",
            metadata={"duration": 30, "format": "mp4"},
        )

        assert artifact.metadata == {"duration": 30, "format": "mp4"}

    def test_artifact_to_reference(self, mock_env_vars):
        """Test Artifact.to_reference returns lightweight dict."""
        from src.orchestrator.artifact_store import Artifact

        artifact = Artifact(
            id="screenshot_test",
            type="screenshot",
            content="large_base64_data_here",
            metadata={"step": 1, "source": "test"},
            created_at="2024-01-15T10:00:00Z",
        )

        ref = artifact.to_reference()

        assert ref["artifact_id"] == "screenshot_test"
        assert ref["type"] == "screenshot"
        assert ref["created_at"] == "2024-01-15T10:00:00Z"
        assert ref["step"] == 1
        assert ref["source"] == "test"
        # Should NOT include content
        assert "content" not in ref

    def test_artifact_default_created_at(self, mock_env_vars):
        """Test Artifact created_at defaults to current time."""
        from src.orchestrator.artifact_store import Artifact

        before = datetime.now(UTC).isoformat()

        artifact = Artifact(
            id="test",
            type="screenshot",
            content="data",
        )

        after = datetime.now(UTC).isoformat()

        assert artifact.created_at >= before
        assert artifact.created_at <= after


class TestArtifactStoreInit:
    """Tests for ArtifactStore initialization."""

    def test_init_default_backend(self, mock_env_vars):
        """Test ArtifactStore initializes with default memory backend."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        assert store.backend == "memory"
        assert store._memory_store == {}

    def test_init_memory_backend(self, mock_env_vars):
        """Test ArtifactStore initializes with memory backend."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="memory")

        assert store.backend == "memory"

    def test_init_supabase_backend(self, mock_env_vars):
        """Test ArtifactStore initializes with supabase backend."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="supabase")

        assert store.backend == "supabase"

    def test_init_s3_backend(self, mock_env_vars):
        """Test ArtifactStore initializes with s3 backend."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="s3")

        assert store.backend == "s3"


class TestArtifactStoreStore:
    """Tests for ArtifactStore.store method."""

    def test_store_creates_artifact(self, mock_env_vars):
        """Test store creates and stores artifact."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        ref = store.store(
            content="test_content_data",
            artifact_type="json",
        )

        assert "artifact_id" in ref
        assert ref["type"] == "json"
        assert ref["artifact_id"].startswith("json_")

    def test_store_generates_unique_ids(self, mock_env_vars):
        """Test store generates unique IDs for different content."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        ref1 = store.store(content="content_1", artifact_type="text")
        ref2 = store.store(content="content_2", artifact_type="text")

        assert ref1["artifact_id"] != ref2["artifact_id"]

    def test_store_with_metadata(self, mock_env_vars):
        """Test store includes metadata in reference."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        ref = store.store(
            content="data",
            artifact_type="screenshot",
            metadata={"step": 5, "test_name": "login"},
        )

        assert ref["step"] == 5
        assert ref["test_name"] == "login"

    def test_store_memory_backend(self, mock_env_vars):
        """Test store uses memory backend correctly."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="memory")

        ref = store.store(content="test_data", artifact_type="html")

        assert ref["artifact_id"] in store._memory_store
        assert store._memory_store[ref["artifact_id"]].content == "test_data"

    def test_store_supabase_backend_fallback(self, mock_env_vars):
        """Test store with supabase backend falls back to memory."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="supabase")

        ref = store.store(content="test_data", artifact_type="json")

        # Should fall back to memory store (TODO in implementation)
        assert ref["artifact_id"] in store._memory_store

    def test_store_s3_backend_fallback(self, mock_env_vars):
        """Test store with s3 backend falls back to memory."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="s3")

        ref = store.store(content="test_data", artifact_type="json")

        # Should fall back to memory store (TODO in implementation)
        assert ref["artifact_id"] in store._memory_store


class TestArtifactStoreStoreScreenshot:
    """Tests for ArtifactStore.store_screenshot method."""

    def test_store_screenshot_basic(self, mock_env_vars):
        """Test store_screenshot stores screenshot artifact."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        ref = store.store_screenshot(base64_data)

        assert ref["type"] == "screenshot"
        assert ref["artifact_id"].startswith("screenshot_")

    def test_store_screenshot_with_metadata(self, mock_env_vars):
        """Test store_screenshot stores with metadata."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        ref = store.store_screenshot(
            base64_data="image_data",
            metadata={"step": 3, "action": "click"},
        )

        assert ref["step"] == 3
        assert ref["action"] == "click"


class TestArtifactStoreStoreTestResult:
    """Tests for ArtifactStore.store_test_result method."""

    def test_store_test_result_no_extraction(self, mock_env_vars):
        """Test store_test_result without artifact extraction."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "screenshot": "base64_data",
        }

        stored = store.store_test_result(result, extract_artifacts=False)

        assert stored == result

    def test_store_test_result_extracts_screenshot(self, mock_env_vars):
        """Test store_test_result extracts screenshot artifact."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "screenshot": "A" * 2000,  # Large base64 data
        }

        stored = store.store_test_result(result)

        # Screenshot should be replaced with artifact ID
        assert stored["screenshot"].startswith("screenshot_")
        assert "_artifact_refs" in stored
        assert len(stored["_artifact_refs"]) == 1

    def test_store_test_result_extracts_final_screenshot(self, mock_env_vars):
        """Test store_test_result extracts finalScreenshot artifact."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "finalScreenshot": "B" * 2000,
        }

        stored = store.store_test_result(result)

        assert stored["finalScreenshot"].startswith("screenshot_")
        assert len(stored["_artifact_refs"]) == 1

    def test_store_test_result_extracts_screenshots_array(self, mock_env_vars):
        """Test store_test_result extracts screenshots array."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "screenshots": [
                "C" * 2000,  # Large, will be extracted
                "small",  # Small, will remain
                "D" * 2000,  # Large, will be extracted
            ],
        }

        stored = store.store_test_result(result)

        assert stored["screenshots"][0].startswith("screenshot_")
        assert stored["screenshots"][1] == "small"
        assert stored["screenshots"][2].startswith("screenshot_")
        assert len(stored["_artifact_refs"]) == 2

    def test_store_test_result_extracts_step_screenshots(self, mock_env_vars):
        """Test store_test_result extracts step screenshots."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "steps": [
                {"instruction": "click button", "screenshot": "E" * 2000},
                {"instruction": "fill form", "screenshot": "short"},
                {"instruction": "submit", "screenshot": "F" * 2000},
            ],
        }

        stored = store.store_test_result(result)

        assert stored["steps"][0]["screenshot"].startswith("screenshot_")
        assert stored["steps"][1]["screenshot"] == "short"
        assert stored["steps"][2]["screenshot"].startswith("screenshot_")

    def test_store_test_result_creates_summary(self, mock_env_vars):
        """Test store_test_result creates summary."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "steps": [
                {"success": True},
                {"success": True},
                {"success": False},
            ],
        }

        stored = store.store_test_result(result)

        assert "_summary" in stored
        assert "Success: True" in stored["_summary"]
        assert "Steps: 2/3 passed" in stored["_summary"]

    def test_store_test_result_summary_with_error(self, mock_env_vars):
        """Test store_test_result summary includes error."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": False,
            "error": "Element not found: #submit-button",
        }

        stored = store.store_test_result(result)

        assert "Error: Element not found" in stored["_summary"]

    def test_store_test_result_summary_with_message(self, mock_env_vars):
        """Test store_test_result summary includes message."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "message": "Test completed successfully",
        }

        stored = store.store_test_result(result)

        assert "Message: Test completed" in stored["_summary"]

    def test_store_test_result_handles_empty_screenshot(self, mock_env_vars):
        """Test store_test_result handles empty screenshot."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "success": True,
            "screenshot": "",
        }

        stored = store.store_test_result(result)

        # Should not extract empty screenshot
        assert stored["screenshot"] == ""


class TestArtifactStoreCreateSummary:
    """Tests for ArtifactStore._create_summary method."""

    def test_create_summary_success(self, mock_env_vars):
        """Test _create_summary with success result."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {"success": True}
        summary = store._create_summary(result)

        assert "Success: True" in summary

    def test_create_summary_with_steps(self, mock_env_vars):
        """Test _create_summary with steps."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = {
            "steps": [
                {"success": True},
                {"success": False},
                {"success": True},
            ]
        }
        summary = store._create_summary(result)

        assert "Steps: 2/3 passed" in summary

    def test_create_summary_with_error(self, mock_env_vars):
        """Test _create_summary truncates long error."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        long_error = "A" * 300
        result = {"error": long_error}
        summary = store._create_summary(result)

        # Should be truncated to 200 chars
        assert len(summary) < len(long_error)
        assert "Error:" in summary

    def test_create_summary_empty_result(self, mock_env_vars):
        """Test _create_summary with empty result."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        summary = store._create_summary({})

        assert summary == "No summary available"


class TestArtifactStoreGet:
    """Tests for ArtifactStore.get method."""

    def test_get_found(self, mock_env_vars):
        """Test get returns artifact when found."""
        from src.orchestrator.artifact_store import Artifact, ArtifactStore

        store = ArtifactStore()

        artifact = Artifact(
            id="test_artifact",
            type="screenshot",
            content="data",
        )
        store._memory_store["test_artifact"] = artifact

        result = store.get("test_artifact")

        assert result == artifact

    def test_get_not_found(self, mock_env_vars):
        """Test get returns None when not found."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        result = store.get("nonexistent")

        assert result is None

    def test_get_memory_backend(self, mock_env_vars):
        """Test get uses memory backend."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="memory")

        ref = store.store("content", "json")
        artifact = store.get(ref["artifact_id"])

        assert artifact is not None
        assert artifact.content == "content"

    def test_get_supabase_backend(self, mock_env_vars):
        """Test get with supabase backend falls back to memory."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="supabase")

        ref = store.store("content", "json")
        artifact = store.get(ref["artifact_id"])

        # Falls back to memory store
        assert artifact is not None

    def test_get_s3_backend(self, mock_env_vars):
        """Test get with s3 backend falls back to memory."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore(backend="s3")

        ref = store.store("content", "json")
        artifact = store.get(ref["artifact_id"])

        # Falls back to memory store
        assert artifact is not None


class TestArtifactStoreGetContent:
    """Tests for ArtifactStore.get_content method."""

    def test_get_content_found(self, mock_env_vars):
        """Test get_content returns content when artifact found."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        ref = store.store("my_content_data", "text")
        content = store.get_content(ref["artifact_id"])

        assert content == "my_content_data"

    def test_get_content_not_found(self, mock_env_vars):
        """Test get_content returns None when artifact not found."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        content = store.get_content("nonexistent")

        assert content is None


class TestArtifactStoreCleanup:
    """Tests for ArtifactStore.cleanup_old method."""

    def test_cleanup_old_no_op(self, mock_env_vars):
        """Test cleanup_old is a no-op (TODO in implementation)."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        # Store some artifacts
        store.store("data1", "screenshot")
        store.store("data2", "screenshot")

        # Cleanup should not raise
        store.cleanup_old(max_age_hours=24)

        # Currently does nothing (TODO)
        assert len(store._memory_store) == 2


class TestGlobalArtifactStore:
    """Tests for global artifact store functions."""

    def test_get_artifact_store_creates_instance(self, mock_env_vars):
        """Test get_artifact_store creates new instance."""
        import src.orchestrator.artifact_store as module
        from src.orchestrator.artifact_store import get_artifact_store

        # Reset global instance
        module._artifact_store = None

        store = get_artifact_store()

        assert store is not None
        assert store.backend == "memory"

    def test_get_artifact_store_reuses_instance(self, mock_env_vars):
        """Test get_artifact_store reuses existing instance."""
        import src.orchestrator.artifact_store as module
        from src.orchestrator.artifact_store import get_artifact_store

        # Reset global instance
        module._artifact_store = None

        store1 = get_artifact_store()
        store2 = get_artifact_store()

        assert store1 is store2


class TestArtifactStoreIntegration:
    """Integration tests for ArtifactStore."""

    def test_full_workflow(self, mock_env_vars):
        """Test full store-retrieve workflow."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        # Simulate a test result with multiple artifacts
        test_result = {
            "success": True,
            "message": "Login test passed",
            "screenshot": "G" * 5000,  # Large final screenshot
            "steps": [
                {"instruction": "Navigate to login page", "success": True, "screenshot": "H" * 3000},
                {"instruction": "Enter credentials", "success": True, "screenshot": "I" * 3000},
                {"instruction": "Click submit", "success": True, "screenshot": "J" * 3000},
            ],
        }

        # Store with artifact extraction
        lightweight = store.store_test_result(test_result)

        # Verify lightweight result
        assert len(lightweight["screenshot"]) < 100  # Should be artifact ID, not full data
        assert "_artifact_refs" in lightweight
        assert "_summary" in lightweight

        # Verify all artifacts can be retrieved
        for ref in lightweight["_artifact_refs"]:
            artifact = store.get(ref["artifact_id"])
            assert artifact is not None
            assert len(artifact.content) >= 1000  # Original data preserved

    def test_deduplication_by_content(self, mock_env_vars):
        """Test that similar content generates IDs with same content hash prefix."""
        from src.orchestrator.artifact_store import ArtifactStore

        store = ArtifactStore()

        # Store same content twice
        ref1 = store.store("identical_content", "text")
        ref2 = store.store("identical_content", "text")

        # IDs have same type and content hash prefix, but different UUID suffix
        # Format is: {type}_{content_hash[:12]}_{uuid[:8]}
        # So first ~18 chars (text_ + 12 char hash + _) should match
        assert ref1["artifact_id"][:18] == ref2["artifact_id"][:18]
        # But full IDs are different due to UUID suffix
        assert ref1["artifact_id"] != ref2["artifact_id"]
