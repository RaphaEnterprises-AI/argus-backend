"""Local MCP Server - Privacy-Preserving Code Intelligence.

KEY INNOVATION: Code NEVER leaves the customer's machine.
This MCP server runs locally and exposes code analysis tools
that can be used by AI coding assistants (Claude, Cursor, Copilot, Windsurf).

What stays local:
- Source code
- AST analysis
- Semantic chunking
- All file operations

What goes to cloud (only if needed):
- Embeddings (vectors) - for semantic search
- Anonymized fingerprints - for deduplication

Usage:
    # Start the local MCP server
    python -m src.mcp.local_mcp_server

    # Or programmatically
    from src.mcp.local_mcp_server import start_local_mcp_server
    await start_local_mcp_server(repo_path="/path/to/your/code")

Configure in Claude Code / Cursor:
    Add to MCP settings:
    {
        "mcpServers": {
            "argus-local": {
                "command": "python",
                "args": ["-m", "src.mcp.local_mcp_server"],
                "cwd": "/path/to/argus",
                "env": {
                    "REPO_PATH": "/path/to/your/code"
                }
            }
        }
    }
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

# Try to import MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        CallToolResult,
        ListToolsResult,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed. Install with: pip install mcp")

# Import our indexer components
from src.indexer import (
    TreeSitterParser,
    SemanticChunker,
    LocalEmbedder,
    MerkleTree,
    ParsedFile,
    CodeChunk,
)


class LocalMCPServer:
    """Privacy-preserving local MCP server for code intelligence.

    This server exposes tools for:
    - Code analysis (functions, classes, imports)
    - Semantic search (find similar code)
    - Change detection (what changed since last index)
    - Dependency analysis
    - Test suggestion

    All analysis happens locally. Code never leaves.
    """

    def __init__(self, repo_path: str):
        """Initialize the local MCP server.

        Args:
            repo_path: Path to the repository to analyze
        """
        self.repo_path = Path(repo_path).resolve()
        self.parser = TreeSitterParser()
        self.chunker = SemanticChunker()
        self.embedder = LocalEmbedder()
        self.merkle_tree: Optional[MerkleTree] = None

        # Index cache
        self._chunks: list[CodeChunk] = []
        self._embeddings: dict[str, list[float]] = {}
        self._indexed = False

        self.log = logger.bind(component="local_mcp")

    async def initialize(self) -> None:
        """Initialize the indexer and build initial index."""
        self.log.info("Initializing local MCP server", repo_path=str(self.repo_path))

        # Build Merkle tree for change detection
        self.merkle_tree = MerkleTree(str(self.repo_path))
        self.merkle_tree.build()

        self.log.info(
            "Merkle tree built",
            files=self.merkle_tree.total_files,
            size_mb=self.merkle_tree.total_size / 1024 / 1024,
        )

        # Index files
        await self._index_all_files()

    async def _index_all_files(self) -> None:
        """Index all files in the repository."""
        if not self.merkle_tree:
            return

        self._chunks = []
        self._embeddings = {}

        for file_path in self.merkle_tree.get_all_files():
            full_path = self.repo_path / file_path

            try:
                # Parse and chunk the file
                parsed = self.parser.parse_file(str(full_path))
                chunks = self.chunker.chunk(parsed)

                for chunk in chunks:
                    self._chunks.append(chunk)

            except Exception as e:
                self.log.debug("Could not index file", file=file_path, error=str(e))

        self._indexed = True
        self.log.info("Indexing complete", total_chunks=len(self._chunks))

    def get_tools(self) -> list[dict]:
        """Get list of available tools.

        Returns:
            List of tool definitions for MCP protocol
        """
        return [
            {
                "name": "analyze_code",
                "description": "Analyze code structure in the repository. Returns functions, classes, and imports for a file or directory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File or directory path (relative to repo root)",
                        },
                        "include_content": {
                            "type": "boolean",
                            "description": "Include source code content (default: false)",
                            "default": False,
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "find_similar_code",
                "description": "Find code similar to a query or code snippet. Uses local embeddings - code never leaves your machine.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query or code snippet",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5)",
                            "default": 5,
                        },
                        "file_filter": {
                            "type": "string",
                            "description": "Filter by file extension (e.g., '.py', '.ts')",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "detect_changes",
                "description": "Detect what files have changed since last index. Uses Merkle tree for O(log n) detection.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "get_function_details",
                "description": "Get detailed information about a specific function or method.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "function_name": {
                            "type": "string",
                            "description": "Name of the function to find",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: specific file to search in",
                        },
                    },
                    "required": ["function_name"],
                },
            },
            {
                "name": "get_class_details",
                "description": "Get detailed information about a specific class.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "class_name": {
                            "type": "string",
                            "description": "Name of the class to find",
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Optional: specific file to search in",
                        },
                    },
                    "required": ["class_name"],
                },
            },
            {
                "name": "suggest_tests",
                "description": "Suggest test cases for a function or file based on code analysis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Function name or file path to generate tests for",
                        },
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "get_repo_stats",
                "description": "Get statistics about the repository (files, languages, size).",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "find_usages",
                "description": "Find all usages of a symbol (function, class, variable) in the codebase.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol name to find usages of",
                        },
                        "file_filter": {
                            "type": "string",
                            "description": "Filter by file extension",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        ]

    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool call.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result with content
        """
        handlers = {
            "analyze_code": self._analyze_code,
            "find_similar_code": self._find_similar_code,
            "detect_changes": self._detect_changes,
            "get_function_details": self._get_function_details,
            "get_class_details": self._get_class_details,
            "suggest_tests": self._suggest_tests,
            "get_repo_stats": self._get_repo_stats,
            "find_usages": self._find_usages,
        }

        handler = handlers.get(name)
        if not handler:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = await handler(arguments)
            return {"content": [{"type": "text", "text": result}]}
        except Exception as e:
            self.log.error("Tool execution failed", tool=name, error=str(e))
            return {"error": str(e)}

    async def _analyze_code(self, args: dict) -> str:
        """Analyze code structure."""
        path = args.get("path", "")
        include_content = args.get("include_content", False)

        full_path = self.repo_path / path

        if full_path.is_file():
            # Analyze single file
            parsed = self.parser.parse_file(str(full_path))
            return self._format_parsed_file(parsed, include_content)

        elif full_path.is_dir():
            # Analyze directory
            results = []
            for file_path in full_path.rglob("*"):
                if file_path.is_file() and file_path.suffix in self.parser.detect_language(str(file_path)).value:
                    try:
                        parsed = self.parser.parse_file(str(file_path))
                        rel_path = file_path.relative_to(self.repo_path)
                        results.append(f"## {rel_path}\n{self._format_parsed_file(parsed, include_content)}")
                    except Exception:
                        pass

            return "\n\n".join(results) if results else "No code files found"

        else:
            return f"Path not found: {path}"

    def _format_parsed_file(self, parsed: ParsedFile, include_content: bool) -> str:
        """Format parsed file as readable text."""
        lines = [
            f"**File:** {parsed.file_path}",
            f"**Language:** {parsed.language.value}",
            f"**Lines:** {parsed.line_count}",
        ]

        if parsed.has_errors:
            lines.append(f"**Parse Errors:** {', '.join(parsed.errors)}")

        # Functions
        functions = parsed.get_functions()
        if functions:
            lines.append(f"\n### Functions ({len(functions)})")
            for func in functions:
                name = self.chunker._extract_name(func, parsed.language) or "anonymous"
                lines.append(f"- `{name}` (lines {func.start_line}-{func.end_line})")
                if include_content:
                    lines.append(f"  ```\n  {func.text[:200]}{'...' if len(func.text) > 200 else ''}\n  ```")

        # Classes
        classes = parsed.get_classes()
        if classes:
            lines.append(f"\n### Classes ({len(classes)})")
            for cls in classes:
                name = self.chunker._extract_name(cls, parsed.language) or "anonymous"
                lines.append(f"- `{name}` (lines {cls.start_line}-{cls.end_line})")

        # Imports
        imports = parsed.get_imports()
        if imports:
            lines.append(f"\n### Imports ({len(imports)})")
            for imp in imports[:10]:  # Limit to first 10
                lines.append(f"- {imp.text.strip()}")
            if len(imports) > 10:
                lines.append(f"- ... and {len(imports) - 10} more")

        return "\n".join(lines)

    async def _find_similar_code(self, args: dict) -> str:
        """Find similar code using local embeddings."""
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        file_filter = args.get("file_filter")

        if not self._indexed:
            await self._index_all_files()

        # Filter chunks if needed
        chunks = self._chunks
        if file_filter:
            chunks = [c for c in chunks if c.file_path.endswith(file_filter)]

        if not chunks:
            return "No code chunks available for search"

        # Generate embedding for query
        query_result = self.embedder.embed(query)
        if not query_result:
            return "Could not generate embedding for query"

        # Build embeddings for chunks (cached)
        chunk_embeddings = []
        for chunk in chunks:
            if chunk.chunk_id not in self._embeddings:
                result = self.embedder.embed_chunk(chunk)
                if result:
                    self._embeddings[chunk.chunk_id] = result.embedding
            if chunk.chunk_id in self._embeddings:
                chunk_embeddings.append((chunk.chunk_id, self._embeddings[chunk.chunk_id]))

        # Find similar
        similar = self.embedder.find_similar(
            query_result.embedding,
            chunk_embeddings,
            top_k=top_k,
            threshold=0.3,
        )

        if not similar:
            return "No similar code found"

        # Format results
        lines = [f"## Similar Code (top {len(similar)} results)\n"]

        for chunk_id, score in similar:
            chunk = next((c for c in chunks if c.chunk_id == chunk_id), None)
            if chunk:
                lines.append(f"### {chunk.file_path}:{chunk.start_line}")
                lines.append(f"**Type:** {chunk.chunk_type.value}")
                lines.append(f"**Similarity:** {score:.2%}")
                if chunk.name:
                    lines.append(f"**Name:** `{chunk.name}`")
                lines.append(f"```\n{chunk.content[:500]}{'...' if len(chunk.content) > 500 else ''}\n```")
                lines.append("")

        return "\n".join(lines)

    async def _detect_changes(self, args: dict) -> str:
        """Detect changed files using Merkle tree."""
        if not self.merkle_tree:
            return "Index not initialized"

        changes = self.merkle_tree.detect_changes()

        lines = ["## Changed Files\n"]

        if changes.added:
            lines.append(f"### Added ({len(changes.added)})")
            for f in changes.added[:20]:
                lines.append(f"- `{f}`")
            if len(changes.added) > 20:
                lines.append(f"- ... and {len(changes.added) - 20} more")

        if changes.modified:
            lines.append(f"\n### Modified ({len(changes.modified)})")
            for f in changes.modified[:20]:
                lines.append(f"- `{f}`")
            if len(changes.modified) > 20:
                lines.append(f"- ... and {len(changes.modified) - 20} more")

        if changes.deleted:
            lines.append(f"\n### Deleted ({len(changes.deleted)})")
            for f in changes.deleted[:20]:
                lines.append(f"- `{f}`")
            if len(changes.deleted) > 20:
                lines.append(f"- ... and {len(changes.deleted) - 20} more")

        if not changes.has_changes:
            lines.append("No changes detected since last index.")

        return "\n".join(lines)

    async def _get_function_details(self, args: dict) -> str:
        """Get details about a specific function."""
        function_name = args.get("function_name", "")
        file_path = args.get("file_path")

        # Search in chunks
        if not self._indexed:
            await self._index_all_files()

        matching = []
        for chunk in self._chunks:
            if chunk.name and function_name.lower() in chunk.name.lower():
                if file_path is None or chunk.file_path.endswith(file_path):
                    matching.append(chunk)

        if not matching:
            return f"Function `{function_name}` not found"

        lines = [f"## Function: {function_name}\n"]

        for chunk in matching[:5]:
            lines.append(f"### {chunk.file_path}:{chunk.start_line}")
            lines.append(f"**Full Name:** `{chunk.full_name}`")
            lines.append(f"**Type:** {chunk.chunk_type.value}")
            lines.append(f"**Lines:** {chunk.start_line}-{chunk.end_line}")

            if chunk.signature:
                lines.append(f"**Signature:** `{chunk.signature}`")

            if chunk.docstring:
                lines.append(f"**Docstring:**\n> {chunk.docstring}")

            if chunk.calls_functions:
                lines.append(f"**Calls:** {', '.join(chunk.calls_functions[:10])}")

            lines.append(f"\n```{chunk.language}\n{chunk.content}\n```\n")

        return "\n".join(lines)

    async def _get_class_details(self, args: dict) -> str:
        """Get details about a specific class."""
        class_name = args.get("class_name", "")
        file_path = args.get("file_path")

        if not self._indexed:
            await self._index_all_files()

        matching = []
        for chunk in self._chunks:
            if chunk.chunk_type.value == "class" and chunk.name:
                if class_name.lower() in chunk.name.lower():
                    if file_path is None or chunk.file_path.endswith(file_path):
                        matching.append(chunk)

        if not matching:
            return f"Class `{class_name}` not found"

        lines = [f"## Class: {class_name}\n"]

        for chunk in matching[:3]:
            lines.append(f"### {chunk.file_path}:{chunk.start_line}")
            lines.append(f"**Lines:** {chunk.start_line}-{chunk.end_line}")

            if chunk.docstring:
                lines.append(f"**Docstring:**\n> {chunk.docstring}")

            lines.append(f"\n```{chunk.language}\n{chunk.content}\n```\n")

            # Find methods
            methods = [
                c for c in self._chunks
                if c.chunk_type.value == "method" and c.parent_name == chunk.name
            ]
            if methods:
                lines.append(f"**Methods ({len(methods)}):**")
                for m in methods[:10]:
                    lines.append(f"- `{m.name}` (line {m.start_line})")

        return "\n".join(lines)

    async def _suggest_tests(self, args: dict) -> str:
        """Suggest test cases for code."""
        target = args.get("target", "")

        # Find the target in chunks
        if not self._indexed:
            await self._index_all_files()

        target_chunks = [
            c for c in self._chunks
            if target.lower() in (c.name or "").lower()
            or target in c.file_path
        ]

        if not target_chunks:
            return f"Target `{target}` not found"

        chunk = target_chunks[0]

        lines = [f"## Test Suggestions for `{chunk.name or chunk.file_path}`\n"]

        # Analyze the code to suggest tests
        if chunk.chunk_type.value in ("function", "method"):
            lines.append("### Unit Tests")
            lines.append(f"1. Test normal execution of `{chunk.name}`")
            lines.append(f"2. Test edge cases (empty input, null values)")
            lines.append(f"3. Test error handling")

            if chunk.calls_functions:
                lines.append("\n### Mock Suggestions")
                for func in chunk.calls_functions[:5]:
                    lines.append(f"- Mock `{func}` for isolation")

            # If it has parameters, suggest parameter tests
            if "(" in (chunk.signature or ""):
                lines.append("\n### Parameter Tests")
                lines.append("- Test with valid parameters")
                lines.append("- Test with invalid parameters")
                lines.append("- Test boundary values")

        elif chunk.chunk_type.value == "class":
            lines.append("### Class Tests")
            lines.append(f"1. Test `{chunk.name}` initialization")
            lines.append("2. Test each public method")
            lines.append("3. Test state management")

        lines.append("\n### Test Template")
        lines.append(f"```python")
        lines.append(f"def test_{chunk.name or 'function'}():")
        lines.append(f"    # Arrange")
        lines.append(f"    # TODO: Set up test data")
        lines.append(f"    ")
        lines.append(f"    # Act")
        lines.append(f"    # TODO: Call the function")
        lines.append(f"    ")
        lines.append(f"    # Assert")
        lines.append(f"    # TODO: Verify results")
        lines.append(f"```")

        return "\n".join(lines)

    async def _get_repo_stats(self, args: dict) -> str:
        """Get repository statistics."""
        if not self.merkle_tree:
            self.merkle_tree = MerkleTree(str(self.repo_path))
            self.merkle_tree.build()

        stats = self.merkle_tree.get_stats()

        # Count by language
        language_counts: dict[str, int] = {}
        for file_path in self.merkle_tree.get_all_files():
            ext = Path(file_path).suffix
            language_counts[ext] = language_counts.get(ext, 0) + 1

        # Sort by count
        sorted_languages = sorted(
            language_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        lines = [
            "## Repository Statistics\n",
            f"**Root:** `{stats['root_path']}`",
            f"**Total Files:** {stats['total_files']:,}",
            f"**Total Size:** {stats['total_size_mb']:.2f} MB",
            f"**Build Time:** {stats['build_time_ms']:.1f}ms",
            f"**Root Hash:** `{stats['root_hash']}`",
            "",
            "### Files by Extension",
        ]

        for ext, count in sorted_languages[:15]:
            lines.append(f"- `{ext}`: {count:,} files")

        if len(sorted_languages) > 15:
            lines.append(f"- ... and {len(sorted_languages) - 15} more extensions")

        if self._indexed:
            lines.append(f"\n### Index Stats")
            lines.append(f"- Total Chunks: {len(self._chunks):,}")
            lines.append(f"- Cached Embeddings: {len(self._embeddings):,}")

        return "\n".join(lines)

    async def _find_usages(self, args: dict) -> str:
        """Find usages of a symbol."""
        symbol = args.get("symbol", "")
        file_filter = args.get("file_filter")

        if not self._indexed:
            await self._index_all_files()

        usages = []
        for chunk in self._chunks:
            if file_filter and not chunk.file_path.endswith(file_filter):
                continue

            if symbol in chunk.content:
                usages.append(chunk)

        if not usages:
            return f"No usages of `{symbol}` found"

        lines = [f"## Usages of `{symbol}` ({len(usages)} found)\n"]

        for chunk in usages[:20]:
            lines.append(f"### {chunk.file_path}:{chunk.start_line}")
            lines.append(f"**Type:** {chunk.chunk_type.value}")
            if chunk.name:
                lines.append(f"**In:** `{chunk.name}`")

            # Find the lines containing the symbol
            for i, line in enumerate(chunk.content.split("\n")):
                if symbol in line:
                    lines.append(f"  Line {chunk.start_line + i}: `{line.strip()[:80]}`")

            lines.append("")

        if len(usages) > 20:
            lines.append(f"... and {len(usages) - 20} more usages")

        return "\n".join(lines)


async def start_local_mcp_server(repo_path: Optional[str] = None) -> None:
    """Start the local MCP server.

    Args:
        repo_path: Path to repository (defaults to REPO_PATH env or cwd)
    """
    if not MCP_AVAILABLE:
        print("ERROR: MCP SDK not installed. Install with: pip install mcp")
        return

    repo_path = repo_path or os.environ.get("REPO_PATH") or os.getcwd()

    # Create server
    local_server = LocalMCPServer(repo_path)
    await local_server.initialize()

    # Create MCP server
    server = Server("argus-local")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        tools = local_server.get_tools()
        return [
            Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
            )
            for t in tools
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        result = await local_server.call_tool(name, arguments)
        if "error" in result:
            return [TextContent(type="text", text=f"Error: {result['error']}")]
        return [TextContent(type="text", text=result["content"][0]["text"])]

    # Run with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Main entry point."""
    asyncio.run(start_local_mcp_server())


if __name__ == "__main__":
    main()
