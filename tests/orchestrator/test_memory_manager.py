"""Tests for the Memory Manager module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMemoryManagerDataClasses:
    """Tests for Memory Manager data classes."""

    def test_episodic_memory_creation(self, mock_env_vars):
        """Test EpisodicMemory dataclass."""
        from src.orchestrator.memory_manager import EpisodicMemory, MemoryImportance

        memory = EpisodicMemory(
            memory_id="EP-001",
            content="User logged in successfully",
            context={"action": "login", "user_id": "123"},
            importance=MemoryImportance.HIGH,
            timestamp=datetime.now(timezone.utc),
            session_id="session_abc",
            linked_memories=["EP-000"],
        )

        assert memory.memory_id == "EP-001"
        assert memory.importance == MemoryImportance.HIGH
        assert len(memory.linked_memories) == 1

    def test_procedural_memory_creation(self, mock_env_vars):
        """Test ProceduralMemory dataclass."""
        from src.orchestrator.memory_manager import ProceduralMemory, MemoryImportance

        memory = ProceduralMemory(
            memory_id="PR-001",
            procedure_name="login_flow",
            steps=["Navigate to login", "Enter credentials", "Click submit"],
            preconditions=["User is logged out"],
            postconditions=["User is logged in"],
            success_rate=0.95,
            execution_count=100,
            importance=MemoryImportance.HIGH,
        )

        assert memory.memory_id == "PR-001"
        assert memory.procedure_name == "login_flow"
        assert len(memory.steps) == 3
        assert memory.success_rate == 0.95

    def test_memory_search_result_creation(self, mock_env_vars):
        """Test MemorySearchResult dataclass."""
        from src.orchestrator.memory_manager import MemorySearchResult, MemoryType

        result = MemorySearchResult(
            memory_id="EP-001",
            memory_type=MemoryType.EPISODIC,
            content="Login successful",
            relevance_score=0.92,
            context={"action": "login"},
        )

        assert result.memory_id == "EP-001"
        assert result.memory_type == MemoryType.EPISODIC
        assert result.relevance_score == 0.92


class TestMemoryManagerEnums:
    """Tests for Memory Manager enums."""

    def test_memory_type_values(self, mock_env_vars):
        """Test MemoryType enum."""
        from src.orchestrator.memory_manager import MemoryType

        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_memory_importance_values(self, mock_env_vars):
        """Test MemoryImportance enum."""
        from src.orchestrator.memory_manager import MemoryImportance

        assert MemoryImportance.CRITICAL.value == "critical"
        assert MemoryImportance.HIGH.value == "high"
        assert MemoryImportance.MEDIUM.value == "medium"
        assert MemoryImportance.LOW.value == "low"


class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_memory_manager_initialization(self, mock_env_vars):
        """Test MemoryManager initialization."""
        from src.orchestrator.memory_manager import MemoryManager

        manager = MemoryManager()

        assert manager is not None
        assert hasattr(manager, 'store_episode')
        assert hasattr(manager, 'store_procedure')
        assert hasattr(manager, 'search')
        assert hasattr(manager, 'link_memories')

    @pytest.mark.asyncio
    async def test_store_episode(self, mock_env_vars):
        """Test storing an episodic memory."""
        from src.orchestrator.memory_manager import MemoryManager, MemoryImportance

        manager = MemoryManager()

        with patch.object(manager, '_store_to_cognee', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = "EP-001"

            memory_id = await manager.store_episode(
                content="Test completed successfully",
                context={"test_id": "TEST-001", "result": "passed"},
                importance=MemoryImportance.MEDIUM,
                session_id="session_123",
            )

            assert memory_id is not None
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_procedure(self, mock_env_vars):
        """Test storing a procedural memory."""
        from src.orchestrator.memory_manager import MemoryManager, MemoryImportance

        manager = MemoryManager()

        with patch.object(manager, '_store_to_cognee', new_callable=AsyncMock) as mock_store:
            mock_store.return_value = "PR-001"

            memory_id = await manager.store_procedure(
                procedure_name="checkout_flow",
                steps=["Add to cart", "Enter shipping", "Pay", "Confirm"],
                preconditions=["User logged in", "Cart not empty"],
                postconditions=["Order created", "Confirmation email sent"],
                importance=MemoryImportance.HIGH,
            )

            assert memory_id is not None
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_memories(self, mock_env_vars):
        """Test searching memories."""
        from src.orchestrator.memory_manager import MemoryManager, MemoryType

        manager = MemoryManager()

        with patch.object(manager, '_search_cognee', new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [
                {
                    "memory_id": "EP-001",
                    "memory_type": "episodic",
                    "content": "Login test passed",
                    "relevance_score": 0.9,
                }
            ]

            results = await manager.search(
                query="login test",
                memory_types=[MemoryType.EPISODIC],
                top_k=5,
            )

            assert len(results) == 1
            assert results[0]["relevance_score"] == 0.9

    @pytest.mark.asyncio
    async def test_link_memories(self, mock_env_vars):
        """Test linking memories (Zettelkasten pattern)."""
        from src.orchestrator.memory_manager import MemoryManager

        manager = MemoryManager()

        with patch.object(manager, '_create_link', new_callable=AsyncMock) as mock_link:
            mock_link.return_value = True

            success = await manager.link_memories(
                source_id="EP-001",
                target_id="PR-001",
                link_type="follows_procedure",
            )

            assert success is True
            mock_link.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidate_memories(self, mock_env_vars):
        """Test memory consolidation."""
        from src.orchestrator.memory_manager import MemoryManager

        manager = MemoryManager()

        with patch.object(manager, '_get_recent_episodes', new_callable=AsyncMock) as mock_get:
            with patch.object(manager, '_consolidate_to_procedure', new_callable=AsyncMock) as mock_consolidate:
                mock_get.return_value = [
                    {"memory_id": "EP-001", "content": "Step 1"},
                    {"memory_id": "EP-002", "content": "Step 2"},
                    {"memory_id": "EP-003", "content": "Step 3"},
                ]
                mock_consolidate.return_value = "PR-001"

                result = await manager.consolidate_memories(
                    session_id="session_123",
                    min_episodes=3,
                )

                assert result is not None


class TestGetMemoryManager:
    """Tests for get_memory_manager singleton."""

    def test_get_memory_manager_returns_singleton(self, mock_env_vars):
        """Test that get_memory_manager returns a singleton."""
        from src.orchestrator.memory_manager import get_memory_manager

        manager1 = get_memory_manager()
        manager2 = get_memory_manager()

        assert manager1 is manager2


class TestMemoryManagerIntegration:
    """Integration tests for Memory Manager."""

    @pytest.mark.asyncio
    async def test_episode_to_procedure_workflow(self, mock_env_vars):
        """Test the workflow from episodes to procedure extraction."""
        from src.orchestrator.memory_manager import MemoryManager, MemoryImportance

        manager = MemoryManager()

        with patch.object(manager, '_store_to_cognee', new_callable=AsyncMock) as mock_store:
            with patch.object(manager, '_call_ai', new_callable=AsyncMock) as mock_ai:
                mock_store.side_effect = ["EP-001", "EP-002", "EP-003", "PR-001"]
                mock_ai.return_value = MagicMock(
                    content='''{
                        "procedure_name": "login_flow",
                        "steps": ["Navigate", "Enter creds", "Submit"],
                        "confidence": 0.9
                    }'''
                )

                # Store episodes
                for i in range(3):
                    await manager.store_episode(
                        content=f"Step {i + 1} of login",
                        context={"step": i + 1},
                        importance=MemoryImportance.MEDIUM,
                        session_id="session_123",
                    )

                # This simulates extracting a procedure
                assert mock_store.call_count == 3

    @pytest.mark.asyncio
    async def test_memory_linking_bidirectional(self, mock_env_vars):
        """Test bidirectional memory linking."""
        from src.orchestrator.memory_manager import MemoryManager

        manager = MemoryManager()

        with patch.object(manager, '_create_link', new_callable=AsyncMock) as mock_link:
            mock_link.return_value = True

            # Link A -> B
            await manager.link_memories("MEM-A", "MEM-B", "related_to")

            # Link B -> A (reverse)
            await manager.link_memories("MEM-B", "MEM-A", "related_to")

            assert mock_link.call_count == 2
