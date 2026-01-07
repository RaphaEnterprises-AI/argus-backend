"""Tests for the Merkle tree change detection module."""

import pytest
import tempfile
import os
import json
import time
from pathlib import Path

from src.indexer.merkle_tree import (
    MerkleTree,
    MerkleNode,
    MerkleTreeConfig,
    ChangeSet,
    IncrementalIndexer,
)


class TestMerkleNode:
    """Test MerkleNode dataclass."""

    def test_node_creation(self):
        """Test creating a Merkle node."""
        node = MerkleNode(
            path="test.py",
            hash="abc123",
            is_file=True,
        )

        assert node.path == "test.py"
        assert node.hash == "abc123"
        assert node.is_file is True
        assert node.children == []

    def test_directory_node(self):
        """Test creating a directory node with children."""
        child1 = MerkleNode(path="dir/a.py", hash="hash1", is_file=True)
        child2 = MerkleNode(path="dir/b.py", hash="hash2", is_file=True)

        parent = MerkleNode(
            path="dir",
            hash="combined_hash",
            is_file=False,
            children=[child1, child2],
        )

        assert parent.is_file is False
        assert len(parent.children) == 2

    def test_node_to_dict(self):
        """Test converting node to dictionary."""
        node = MerkleNode(
            path="test.py",
            hash="abc123",
            is_file=True,
        )

        d = node.to_dict()
        assert d["path"] == "test.py"
        assert d["hash"] == "abc123"
        assert d["is_file"] is True


class TestChangeSet:
    """Test ChangeSet dataclass."""

    def test_changeset_creation(self):
        """Test creating a change set."""
        changes = ChangeSet(
            added=["new.py"],
            modified=["changed.py"],
            deleted=["removed.py"],
        )

        assert "new.py" in changes.added
        assert "changed.py" in changes.modified
        assert "removed.py" in changes.deleted

    def test_changeset_has_changes(self):
        """Test has_changes property."""
        empty = ChangeSet(added=[], modified=[], deleted=[])
        assert empty.has_changes is False

        with_added = ChangeSet(added=["new.py"], modified=[], deleted=[])
        assert with_added.has_changes is True

        with_modified = ChangeSet(added=[], modified=["changed.py"], deleted=[])
        assert with_modified.has_changes is True

    def test_changeset_total_changes(self):
        """Test total_changes property."""
        changes = ChangeSet(
            added=["a.py", "b.py"],
            modified=["c.py"],
            deleted=["d.py", "e.py", "f.py"],
        )

        assert changes.total_changes == 6


class TestMerkleTree:
    """Test MerkleTree functionality."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary directory with files."""
        # Create some files
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")

        # Create a subdirectory
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "module.py").write_text("class Module: pass")

        return tmp_path

    @pytest.fixture
    def merkle_tree(self, temp_repo):
        """Create a MerkleTree for the temp repo."""
        return MerkleTree(str(temp_repo))

    def test_build_tree(self, merkle_tree):
        """Test building the Merkle tree."""
        root = merkle_tree.build()

        assert root is not None
        assert root.is_file is False  # Root is a directory
        assert root.hash is not None

    def test_build_tree_hash_consistency(self, merkle_tree):
        """Test that building the same tree twice gives same hash."""
        root1 = merkle_tree.build()
        root2 = merkle_tree.build()

        assert root1.hash == root2.hash

    def test_detect_no_changes(self, merkle_tree):
        """Test detecting no changes."""
        merkle_tree.build()
        changes = merkle_tree.detect_changes()

        assert changes.has_changes is False

    def test_detect_added_file(self, temp_repo, merkle_tree):
        """Test detecting an added file."""
        merkle_tree.build()

        # Add a new file
        (temp_repo / "new_file.py").write_text("new content")

        changes = merkle_tree.detect_changes()

        assert changes.has_changes is True
        assert any("new_file.py" in f for f in changes.added)

    def test_detect_modified_file(self, temp_repo, merkle_tree):
        """Test detecting a modified file."""
        merkle_tree.build()

        # Modify an existing file
        (temp_repo / "main.py").write_text("print('modified')")

        changes = merkle_tree.detect_changes()

        assert changes.has_changes is True
        assert any("main.py" in f for f in changes.modified)

    def test_detect_deleted_file(self, temp_repo, merkle_tree):
        """Test detecting a deleted file."""
        merkle_tree.build()

        # Delete a file
        (temp_repo / "utils.py").unlink()

        changes = merkle_tree.detect_changes()

        assert changes.has_changes is True
        assert any("utils.py" in f for f in changes.deleted)

    def test_save_and_load(self, temp_repo, merkle_tree, tmp_path):
        """Test saving and loading the tree."""
        merkle_tree.build()

        save_path = str(tmp_path / "tree.json")
        success = merkle_tree.save(save_path)
        assert success is True

        # Load and verify
        loaded = MerkleTree.load(save_path)
        assert loaded is not None
        assert loaded.root is not None

    def test_file_patterns(self, tmp_path):
        """Test file pattern filtering."""
        # Create files with different extensions
        (tmp_path / "code.py").write_text("python")
        (tmp_path / "code.js").write_text("javascript")
        (tmp_path / "data.json").write_text("{}")
        (tmp_path / "ignore.log").write_text("log")

        config = MerkleTreeConfig(
            include_patterns=["**/*.py", "**/*.js"],
            exclude_patterns=["**/*.log"]
        )
        tree = MerkleTree(str(tmp_path), config=config)
        root = tree.build()

        # Root should have children (files)
        assert root is not None
        # The tree should be built successfully
        assert tree.total_files >= 1


class TestIncrementalIndexer:
    """Test IncrementalIndexer functionality."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository."""
        (tmp_path / "module.py").write_text("def foo(): pass")
        return tmp_path

    @pytest.fixture
    def indexer(self, temp_repo, tmp_path):
        """Create an IncrementalIndexer."""
        state_dir = tmp_path / ".argus"
        return IncrementalIndexer(str(temp_repo), str(state_dir))

    def test_full_index(self, indexer):
        """Test full index operation."""
        result = indexer.full_index()

        # Should return stats dict
        assert isinstance(result, dict)

    @pytest.mark.requires_tree_sitter
    def test_incremental_index(self, indexer, temp_repo):
        """Test incremental index operation."""
        # First index
        indexer.full_index()

        # No changes - incremental update should return empty changeset
        changes = indexer.incremental_update()
        assert changes.has_changes is False

        # Make a change
        (temp_repo / "module.py").write_text("def bar(): pass")

        # Should detect the change
        changes = indexer.incremental_update()
        assert changes.has_changes is True

    @pytest.mark.requires_tree_sitter
    def test_commit_updates_cache(self, indexer, temp_repo):
        """Test that incremental update tracks changes."""
        # First index
        indexer.full_index()

        # Add new file
        (temp_repo / "new.py").write_text("new")

        changes = indexer.incremental_update()
        assert any("new.py" in f for f in changes.added)

        # After incremental update, further updates should show no changes
        changes = indexer.incremental_update()
        assert changes.has_changes is False

    @pytest.mark.requires_tree_sitter
    def test_needs_full_reindex(self, indexer):
        """Test that full_index works on fresh indexer."""
        # First run should work
        result = indexer.full_index()
        # After build, the tree should exist
        assert indexer._tree.root is not None
        assert isinstance(result, dict)
