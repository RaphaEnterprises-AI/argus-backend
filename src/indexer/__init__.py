"""Argus Indexer - Local code indexing for privacy-preserving analysis.

This module provides:
- Tree-sitter parsing for fast AST generation
- Semantic chunking for meaningful code units
- Local embeddings (code never leaves)
- Merkle tree for efficient change detection
"""

from .local_embedder import EmbeddingResult, LocalEmbedder
from .merkle_tree import MerkleNode, MerkleTree
from .semantic_chunker import ChunkType, CodeChunk, SemanticChunker
from .tree_sitter_parser import ASTNode, ParsedFile, TreeSitterParser

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
