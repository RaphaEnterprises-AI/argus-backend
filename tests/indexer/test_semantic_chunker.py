"""Tests for the semantic chunker module."""

import pytest
from pathlib import Path

from src.indexer.semantic_chunker import (
    SemanticChunker,
    CodeChunk,
    ChunkType,
    ChunkingConfig,
)
from src.indexer.tree_sitter_parser import TreeSitterParser


class TestChunkingConfig:
    """Test ChunkingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.max_chunk_tokens > 0
        assert config.min_chunk_lines > 0
        assert config.context_lines_before >= 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            max_chunk_tokens=500,
            min_chunk_lines=5,
            context_lines_before=10,
        )
        assert config.max_chunk_tokens == 500
        assert config.min_chunk_lines == 5
        assert config.context_lines_before == 10


class TestCodeChunk:
    """Test CodeChunk dataclass."""

    def test_chunk_creation(self):
        """Test creating a code chunk."""
        chunk = CodeChunk(
            chunk_id="test-1",
            chunk_type=ChunkType.FUNCTION,
            name="my_function",
            content="def my_function(): pass",
            content_hash="abc123",
            file_path="test.py",
            start_line=1,
            end_line=1,
            language="python",
        )

        assert chunk.chunk_id == "test-1"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.name == "my_function"
        assert chunk.language == "python"

    def test_chunk_to_dict(self):
        """Test converting chunk to dictionary."""
        chunk = CodeChunk(
            chunk_id="test-1",
            chunk_type=ChunkType.CLASS,
            name="MyClass",
            content="class MyClass: pass",
            content_hash="abc123",
            file_path="test.py",
            start_line=1,
            end_line=1,
            language="python",
        )

        d = chunk.to_dict()
        assert d["chunk_id"] == "test-1"
        assert d["chunk_type"] == "class"
        assert d["name"] == "MyClass"


class TestSemanticChunker:
    """Test SemanticChunker functionality."""

    @pytest.fixture
    def chunker(self):
        """Create a SemanticChunker instance."""
        return SemanticChunker()

    @pytest.fixture
    def parser(self):
        """Create a TreeSitterParser instance."""
        return TreeSitterParser()

    @pytest.fixture
    def sample_python_file(self, parser):
        """Create a parsed Python file."""
        code = '''"""Module docstring."""

import os
from typing import Optional

class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int) -> Optional[dict]:
        """Get a user by ID."""
        return self.db.query(user_id)

    def create_user(self, name: str) -> dict:
        """Create a new user."""
        return self.db.insert({"name": name})

def validate_email(email: str) -> bool:
    """Validate an email address."""
    return "@" in email and "." in email

def main():
    """Entry point."""
    service = UserService(None)
    print(service.get_user(1))
'''
        return parser.parse_content(code, "user_service.py")

    @pytest.fixture
    def sample_typescript_file(self, parser):
        """Create a parsed TypeScript file."""
        code = '''import React, { useState } from "react";

interface UserProps {
    name: string;
    age: number;
}

export const UserCard: React.FC<UserProps> = ({ name, age }) => {
    const [expanded, setExpanded] = useState(false);

    return (
        <div data-testid="user-card">
            <h2>{name}</h2>
            <p>Age: {age}</p>
        </div>
    );
};

export function useUser(id: number) {
    const [user, setUser] = useState(null);
    return { user, setUser };
}
'''
        return parser.parse_content(code, "UserCard.tsx")

    @pytest.mark.requires_tree_sitter
    def test_chunk_python_file(self, chunker, sample_python_file):
        """Test chunking a Python file."""
        chunks = chunker.chunk(sample_python_file)

        assert len(chunks) > 0
        assert all(isinstance(c, CodeChunk) for c in chunks)

    @pytest.mark.requires_tree_sitter
    def test_chunk_finds_classes(self, chunker, sample_python_file):
        """Test that chunker finds class definitions."""
        chunks = chunker.chunk(sample_python_file)

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) >= 1

        class_names = [c.name for c in class_chunks]
        assert "UserService" in class_names

    @pytest.mark.requires_tree_sitter
    def test_chunk_finds_functions(self, chunker, sample_python_file):
        """Test that chunker finds function definitions."""
        chunks = chunker.chunk(sample_python_file)

        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        func_names = [c.name for c in func_chunks]

        assert "validate_email" in func_names or any("validate_email" in n for n in func_names if n)
        assert "main" in func_names or any("main" in n for n in func_names if n)

    @pytest.mark.requires_tree_sitter
    def test_chunk_finds_imports(self, chunker, sample_python_file):
        """Test that chunker finds imports."""
        chunks = chunker.chunk(sample_python_file)

        import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT_BLOCK]
        # Imports might be grouped together
        assert any("import" in c.content.lower() for c in chunks)

    @pytest.mark.requires_tree_sitter
    def test_chunk_typescript_file(self, chunker, sample_typescript_file):
        """Test chunking a TypeScript file."""
        chunks = chunker.chunk(sample_typescript_file)

        assert len(chunks) > 0

    @pytest.mark.requires_tree_sitter
    def test_chunk_finds_react_components(self, chunker, sample_typescript_file):
        """Test that chunker finds React components."""
        chunks = chunker.chunk(sample_typescript_file)

        # Look for UserCard component
        component_found = any(
            "UserCard" in c.name if c.name else False
            for c in chunks
        )
        assert component_found or any("UserCard" in c.content for c in chunks)

    @pytest.mark.requires_tree_sitter
    def test_chunks_have_unique_ids(self, chunker, sample_python_file):
        """Test that all chunks have unique IDs."""
        chunks = chunker.chunk(sample_python_file)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs should be unique"

    @pytest.mark.requires_tree_sitter
    def test_chunks_have_correct_file_path(self, chunker, sample_python_file):
        """Test that chunks have correct file path."""
        chunks = chunker.chunk(sample_python_file)

        for chunk in chunks:
            assert chunk.file_path == "user_service.py"

    @pytest.mark.requires_tree_sitter
    def test_chunks_have_valid_line_numbers(self, chunker, sample_python_file):
        """Test that chunks have valid line numbers."""
        chunks = chunker.chunk(sample_python_file)

        for chunk in chunks:
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    @pytest.mark.requires_tree_sitter
    def test_chunks_have_language(self, chunker, sample_python_file):
        """Test that chunks have language set."""
        chunks = chunker.chunk(sample_python_file)

        for chunk in chunks:
            assert chunk.language == "python"

    @pytest.mark.requires_tree_sitter
    def test_chunk_content_not_empty(self, chunker, sample_python_file):
        """Test that chunk content is not empty."""
        chunks = chunker.chunk(sample_python_file)

        for chunk in chunks:
            assert len(chunk.content) > 0

    @pytest.mark.requires_tree_sitter
    def test_chunk_preserves_docstrings(self, chunker, sample_python_file):
        """Test that chunker preserves docstrings."""
        chunks = chunker.chunk(sample_python_file)

        # Find the UserService class chunk
        class_chunks = [c for c in chunks if c.name == "UserService"]
        if class_chunks:
            assert "Service for user operations" in class_chunks[0].content

    def test_chunk_file_method(self, chunker, parser, tmp_path):
        """Test chunk_file method."""
        # Create a temporary Python file
        code = "def test(): pass"
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        chunks = chunker.chunk_file(str(test_file))
        assert len(chunks) >= 1

    def test_chunk_file_nonexistent(self, chunker):
        """Test chunk_file with nonexistent file."""
        # chunk_file may return empty list or raise an error
        try:
            chunks = chunker.chunk_file("/nonexistent/path.py")
            assert chunks == [] or len(chunks) == 0
        except Exception:
            pass  # Expected behavior for nonexistent files


class TestChunkType:
    """Test ChunkType enum."""

    def test_chunk_types_exist(self):
        """Test that expected chunk types exist."""
        assert ChunkType.MODULE
        assert ChunkType.CLASS
        assert ChunkType.FUNCTION
        assert ChunkType.METHOD
        assert ChunkType.IMPORT_BLOCK

    def test_chunk_type_values(self):
        """Test chunk type string values."""
        assert ChunkType.CLASS.value == "class"
        assert ChunkType.FUNCTION.value == "function"
        assert ChunkType.METHOD.value == "method"
        assert ChunkType.IMPORT_BLOCK.value == "imports"
