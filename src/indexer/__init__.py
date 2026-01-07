"""Argus Indexer - Local code indexing for privacy-preserving analysis.

This module provides:
- Tree-sitter parsing for fast AST generation
- Semantic chunking for meaningful code units
- Local embeddings (code never leaves)
- Merkle tree for efficient change detection
"""

from .tree_sitter_parser import TreeSitterParser, ParsedFile, ASTNode
from .semantic_chunker import SemanticChunker, CodeChunk, ChunkType
from .local_embedder import LocalEmbedder, EmbeddingResult
from .merkle_tree import MerkleTree, MerkleNode

__all__ = [
    "TreeSitterParser",
    "ParsedFile",
    "ASTNode",
    "SemanticChunker",
    "CodeChunk",
    "ChunkType",
    "LocalEmbedder",
    "EmbeddingResult",
    "MerkleTree",
    "MerkleNode",
]
