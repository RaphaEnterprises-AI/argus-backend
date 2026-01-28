"""Argus Indexer - Local code indexing for privacy-preserving analysis.

This module provides:
- Tree-sitter parsing for fast AST generation
- Semantic chunking for meaningful code units
- Local embeddings (code never leaves)
- Merkle tree for efficient change detection
- Incremental indexing for 10x faster updates (RAP-233)
- Git integration for delta-aware indexing
"""

from .change_manifest import BlameInfo, ChangeManifest, FileChange, IndexUpdate
from .git_integration import CommitInfo, GitIntegration, get_git_integration
from .incremental_indexer import (
    IncrementalIndexer,
    IndexState,
    get_incremental_indexer,
)
from .local_embedder import EmbeddingResult, LocalEmbedder
from .merkle_tree import MerkleNode, MerkleTree
from .semantic_chunker import ChunkType, CodeChunk, SemanticChunker
from .tree_sitter_parser import ASTNode, ParsedFile, TreeSitterParser

__all__ = [
    # Tree-sitter parsing
    "TreeSitterParser",
    "ParsedFile",
    "ASTNode",
    # Semantic chunking
    "SemanticChunker",
    "CodeChunk",
    "ChunkType",
    # Local embeddings
    "LocalEmbedder",
    "EmbeddingResult",
    # Merkle tree change detection
    "MerkleTree",
    "MerkleNode",
    # Incremental indexing (RAP-233)
    "IncrementalIndexer",
    "IndexState",
    "get_incremental_indexer",
    # Git integration
    "GitIntegration",
    "CommitInfo",
    "get_git_integration",
    # Change manifest
    "ChangeManifest",
    "FileChange",
    "IndexUpdate",
    "BlameInfo",
]
