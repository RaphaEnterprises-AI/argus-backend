"""Semantic Chunker - Intelligent code splitting for RAG.

Unlike naive line-based chunking, semantic chunking:
- Preserves function/class boundaries
- Includes context (imports, docstrings, type hints)
- Creates overlapping chunks for better retrieval
- Handles different programming paradigms

This is critical for high-quality code embeddings.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .tree_sitter_parser import (
    ASTNode,
    ParsedFile,
    Language,
    get_parser,
)

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of semantic code chunks."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    IMPORT_BLOCK = "imports"
    CONSTANT = "constant"
    TYPE_DEFINITION = "type"
    TEST = "test"
    DOCSTRING = "docstring"
    COMMENT_BLOCK = "comment"
    CONFIG = "config"
    ROUTE = "route"
    COMPONENT = "component"
    HOOK = "hook"
    QUERY = "query"
    MIGRATION = "migration"


@dataclass
class CodeChunk:
    """A semantic unit of code for embedding."""

    # Core identification
    chunk_id: str
    chunk_type: ChunkType

    # Location
    file_path: str
    start_line: int
    end_line: int

    # Content
    content: str
    content_hash: str

    # Semantic info
    name: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None

    # Context for better retrieval
    parent_name: Optional[str] = None  # Class name for methods
    imports_used: list[str] = field(default_factory=list)
    calls_functions: list[str] = field(default_factory=list)

    # Language info
    language: str = "unknown"

    # For overlap/context
    context_before: Optional[str] = None  # Lines before chunk
    context_after: Optional[str] = None   # Lines after chunk

    # Metadata
    token_estimate: int = 0  # Estimated tokens for embedding
    complexity: int = 0      # Cyclomatic complexity estimate

    @property
    def line_count(self) -> int:
        """Number of lines in this chunk."""
        return self.end_line - self.start_line + 1

    @property
    def full_name(self) -> str:
        """Fully qualified name (e.g., ClassName.method_name)."""
        if self.parent_name and self.name:
            return f"{self.parent_name}.{self.name}"
        return self.name or self.chunk_id

    def to_embedding_text(self) -> str:
        """Generate text optimized for embedding.

        Includes structured context that helps semantic matching:
        - File path and location
        - Function/class signature
        - Docstring
        - Actual code
        """
        parts = []

        # Add structured header
        parts.append(f"# File: {self.file_path}")
        parts.append(f"# Type: {self.chunk_type.value}")

        if self.parent_name:
            parts.append(f"# Parent: {self.parent_name}")

        if self.signature:
            parts.append(f"# Signature: {self.signature}")

        if self.docstring:
            parts.append(f'"""')
            parts.append(self.docstring)
            parts.append(f'"""')

        # Add actual code
        parts.append("")
        parts.append(self.content)

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.value,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "name": self.name,
            "full_name": self.full_name,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent_name": self.parent_name,
            "language": self.language,
            "line_count": self.line_count,
            "token_estimate": self.token_estimate,
            "complexity": self.complexity,
            "content_hash": self.content_hash,
        }


@dataclass
class ChunkingConfig:
    """Configuration for semantic chunking."""

    # Size limits
    max_chunk_tokens: int = 1500      # Max tokens per chunk
    min_chunk_lines: int = 3          # Min lines to create chunk

    # Context
    context_lines_before: int = 5     # Lines of context before
    context_lines_after: int = 3      # Lines of context after

    # Behavior
    include_imports: bool = True      # Create import block chunk
    include_docstrings: bool = True   # Include docstrings in chunks
    split_large_functions: bool = True # Split functions > max_tokens
    include_tests: bool = True        # Include test functions

    # Special handling
    detect_components: bool = True    # Detect React/Vue components
    detect_routes: bool = True        # Detect API routes
    detect_queries: bool = True       # Detect SQL queries


class SemanticChunker:
    """Intelligent code chunker that preserves semantic boundaries.

    Usage:
        chunker = SemanticChunker()
        parsed = parser.parse_file("src/app.py")
        chunks = chunker.chunk(parsed)

        for chunk in chunks:
            embedding = embed(chunk.to_embedding_text())
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker with configuration."""
        self.config = config or ChunkingConfig()
        self._parser = get_parser()

    def chunk(self, parsed: ParsedFile) -> list[CodeChunk]:
        """Split parsed file into semantic chunks.

        Args:
            parsed: ParsedFile from tree-sitter parser

        Returns:
            List of CodeChunk objects
        """
        chunks = []

        if not parsed.root:
            # Fallback: create single file chunk
            return [self._create_file_chunk(parsed)]

        lines = parsed.content.split("\n")

        # Extract imports block
        if self.config.include_imports:
            import_chunk = self._extract_imports(parsed, lines)
            if import_chunk:
                chunks.append(import_chunk)

        # Extract classes (and their methods)
        for class_node in parsed.get_classes():
            class_chunks = self._process_class(parsed, class_node, lines)
            chunks.extend(class_chunks)

        # Extract standalone functions
        for func_node in parsed.get_functions():
            # Skip if this function is inside a class
            if self._is_method(func_node):
                continue

            chunk = self._process_function(parsed, func_node, lines)
            if chunk:
                chunks.append(chunk)

        # Extract special patterns based on language
        special_chunks = self._extract_special_patterns(parsed, lines)
        chunks.extend(special_chunks)

        # If no chunks created, create file-level chunk
        if not chunks:
            chunks.append(self._create_file_chunk(parsed))

        # Add context to all chunks
        for chunk in chunks:
            self._add_context(chunk, lines)

        return chunks

    def chunk_file(self, file_path: str) -> list[CodeChunk]:
        """Convenience method to parse and chunk a file.

        Args:
            file_path: Path to source file

        Returns:
            List of CodeChunk objects
        """
        parsed = self._parser.parse_file(file_path)
        return self.chunk(parsed)

    def _extract_imports(
        self,
        parsed: ParsedFile,
        lines: list[str]
    ) -> Optional[CodeChunk]:
        """Extract import statements as a single chunk."""
        imports = parsed.get_imports()

        if not imports:
            return None

        # Find import block boundaries
        start_line = min(i.start_line for i in imports)
        end_line = max(i.end_line for i in imports)

        content = "\n".join(lines[start_line - 1:end_line])

        if len(content.strip()) < 10:
            return None

        return CodeChunk(
            chunk_id=self._generate_id(parsed.file_path, "imports", 0),
            chunk_type=ChunkType.IMPORT_BLOCK,
            file_path=parsed.file_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            content_hash=self._hash_content(content),
            name="imports",
            language=parsed.language.value,
            token_estimate=self._estimate_tokens(content),
        )

    def _process_class(
        self,
        parsed: ParsedFile,
        class_node: ASTNode,
        lines: list[str]
    ) -> list[CodeChunk]:
        """Process a class definition into chunks."""
        chunks = []

        class_name = self._extract_name(class_node, parsed.language)
        class_content = class_node.text

        # Create class-level chunk (signature + docstring + attributes)
        class_header = self._extract_class_header(class_node, lines, parsed.language)

        if class_header:
            chunks.append(CodeChunk(
                chunk_id=self._generate_id(parsed.file_path, class_name, class_node.start_line),
                chunk_type=ChunkType.CLASS,
                file_path=parsed.file_path,
                start_line=class_node.start_line,
                end_line=class_node.start_line + class_header.count("\n"),
                content=class_header,
                content_hash=self._hash_content(class_header),
                name=class_name,
                signature=self._extract_class_signature(class_node, parsed.language),
                docstring=self._extract_docstring(class_node, parsed.language),
                language=parsed.language.value,
                token_estimate=self._estimate_tokens(class_header),
            ))

        # Process methods within the class
        method_types = self._get_method_node_types(parsed.language)

        for child in class_node.children:
            if child.type in method_types:
                method_chunk = self._process_method(
                    parsed, child, lines, class_name
                )
                if method_chunk:
                    chunks.append(method_chunk)

        return chunks

    def _process_function(
        self,
        parsed: ParsedFile,
        func_node: ASTNode,
        lines: list[str],
        parent_name: Optional[str] = None
    ) -> Optional[CodeChunk]:
        """Process a function into a chunk."""
        func_name = self._extract_name(func_node, parsed.language)

        if not func_name:
            return None

        content = func_node.text

        # Check if this is a test function
        chunk_type = ChunkType.FUNCTION
        if self._is_test_function(func_name, content):
            chunk_type = ChunkType.TEST
        elif self._is_hook(func_name, parsed.language):
            chunk_type = ChunkType.HOOK
        elif self._is_route_handler(func_node, parsed.language):
            chunk_type = ChunkType.ROUTE

        # Split if too large
        if self.config.split_large_functions:
            tokens = self._estimate_tokens(content)
            if tokens > self.config.max_chunk_tokens:
                return self._create_large_function_chunk(
                    parsed, func_node, lines, func_name, parent_name, chunk_type
                )

        return CodeChunk(
            chunk_id=self._generate_id(parsed.file_path, func_name, func_node.start_line),
            chunk_type=chunk_type,
            file_path=parsed.file_path,
            start_line=func_node.start_line,
            end_line=func_node.end_line,
            content=content,
            content_hash=self._hash_content(content),
            name=func_name,
            signature=self._extract_function_signature(func_node, parsed.language),
            docstring=self._extract_docstring(func_node, parsed.language),
            parent_name=parent_name,
            language=parsed.language.value,
            token_estimate=self._estimate_tokens(content),
            complexity=self._estimate_complexity(func_node),
            calls_functions=self._extract_function_calls(func_node, parsed.language),
        )

    def _process_method(
        self,
        parsed: ParsedFile,
        method_node: ASTNode,
        lines: list[str],
        class_name: str
    ) -> Optional[CodeChunk]:
        """Process a class method into a chunk."""
        chunk = self._process_function(
            parsed, method_node, lines, parent_name=class_name
        )

        if chunk:
            chunk.chunk_type = ChunkType.METHOD

        return chunk

    def _extract_special_patterns(
        self,
        parsed: ParsedFile,
        lines: list[str]
    ) -> list[CodeChunk]:
        """Extract special patterns based on language and file type."""
        chunks = []

        # React/Vue components
        if self.config.detect_components and parsed.language in (
            Language.TSX, Language.JSX, Language.VUE, Language.SVELTE
        ):
            chunks.extend(self._extract_components(parsed, lines))

        # SQL queries
        if self.config.detect_queries and parsed.language == Language.SQL:
            chunks.extend(self._extract_sql_queries(parsed, lines))

        # Python constants/config at module level
        if parsed.language == Language.PYTHON:
            chunks.extend(self._extract_python_constants(parsed, lines))

        # TypeScript type definitions
        if parsed.language in (Language.TYPESCRIPT, Language.TSX):
            chunks.extend(self._extract_type_definitions(parsed, lines))

        return chunks

    def _extract_components(
        self,
        parsed: ParsedFile,
        lines: list[str]
    ) -> list[CodeChunk]:
        """Extract React/Vue components."""
        chunks = []

        # Look for export default function/const patterns
        if not parsed.root:
            return chunks

        # Find arrow functions that return JSX
        for func in parsed.root.find_descendants("arrow_function"):
            if self._returns_jsx(func):
                name = self._get_component_name(func)
                if name:
                    chunks.append(CodeChunk(
                        chunk_id=self._generate_id(parsed.file_path, name, func.start_line),
                        chunk_type=ChunkType.COMPONENT,
                        file_path=parsed.file_path,
                        start_line=func.start_line,
                        end_line=func.end_line,
                        content=func.text,
                        content_hash=self._hash_content(func.text),
                        name=name,
                        language=parsed.language.value,
                        token_estimate=self._estimate_tokens(func.text),
                    ))

        return chunks

    def _extract_sql_queries(
        self,
        parsed: ParsedFile,
        lines: list[str]
    ) -> list[CodeChunk]:
        """Extract SQL queries as chunks."""
        chunks = []
        content = parsed.content

        # Split by semicolons (statement boundaries)
        statements = content.split(";")
        current_line = 1

        for stmt in statements:
            stmt = stmt.strip()
            if not stmt:
                continue

            # Detect query type
            upper = stmt.upper()
            if upper.startswith("SELECT"):
                chunk_type = ChunkType.QUERY
            elif upper.startswith(("CREATE TABLE", "ALTER TABLE")):
                chunk_type = ChunkType.MIGRATION
            elif upper.startswith(("INSERT", "UPDATE", "DELETE")):
                chunk_type = ChunkType.QUERY
            else:
                chunk_type = ChunkType.QUERY

            end_line = current_line + stmt.count("\n")

            chunks.append(CodeChunk(
                chunk_id=self._generate_id(parsed.file_path, "query", current_line),
                chunk_type=chunk_type,
                file_path=parsed.file_path,
                start_line=current_line,
                end_line=end_line,
                content=stmt,
                content_hash=self._hash_content(stmt),
                language="sql",
                token_estimate=self._estimate_tokens(stmt),
            ))

            current_line = end_line + 1

        return chunks

    def _extract_python_constants(
        self,
        parsed: ParsedFile,
        lines: list[str]
    ) -> list[CodeChunk]:
        """Extract Python module-level constants."""
        chunks = []

        if not parsed.root:
            return chunks

        # Look for uppercase assignments at module level
        for child in parsed.root.children:
            if child.type == "expression_statement":
                for assign in child.find_children("assignment"):
                    name = self._extract_assignment_name(assign)
                    if name and name.isupper():
                        chunks.append(CodeChunk(
                            chunk_id=self._generate_id(parsed.file_path, name, child.start_line),
                            chunk_type=ChunkType.CONSTANT,
                            file_path=parsed.file_path,
                            start_line=child.start_line,
                            end_line=child.end_line,
                            content=child.text,
                            content_hash=self._hash_content(child.text),
                            name=name,
                            language="python",
                            token_estimate=self._estimate_tokens(child.text),
                        ))

        return chunks

    def _extract_type_definitions(
        self,
        parsed: ParsedFile,
        lines: list[str]
    ) -> list[CodeChunk]:
        """Extract TypeScript type/interface definitions."""
        chunks = []

        if not parsed.root:
            return chunks

        # Find type aliases and interfaces
        for type_def in parsed.root.find_descendants("type_alias_declaration"):
            name = self._extract_name(type_def, parsed.language)
            if name:
                chunks.append(CodeChunk(
                    chunk_id=self._generate_id(parsed.file_path, name, type_def.start_line),
                    chunk_type=ChunkType.TYPE_DEFINITION,
                    file_path=parsed.file_path,
                    start_line=type_def.start_line,
                    end_line=type_def.end_line,
                    content=type_def.text,
                    content_hash=self._hash_content(type_def.text),
                    name=name,
                    language=parsed.language.value,
                    token_estimate=self._estimate_tokens(type_def.text),
                ))

        for interface in parsed.root.find_descendants("interface_declaration"):
            name = self._extract_name(interface, parsed.language)
            if name:
                chunks.append(CodeChunk(
                    chunk_id=self._generate_id(parsed.file_path, name, interface.start_line),
                    chunk_type=ChunkType.TYPE_DEFINITION,
                    file_path=parsed.file_path,
                    start_line=interface.start_line,
                    end_line=interface.end_line,
                    content=interface.text,
                    content_hash=self._hash_content(interface.text),
                    name=name,
                    language=parsed.language.value,
                    token_estimate=self._estimate_tokens(interface.text),
                ))

        return chunks

    def _create_file_chunk(self, parsed: ParsedFile) -> CodeChunk:
        """Create a single chunk for the entire file."""
        return CodeChunk(
            chunk_id=self._generate_id(parsed.file_path, "file", 1),
            chunk_type=ChunkType.MODULE,
            file_path=parsed.file_path,
            start_line=1,
            end_line=parsed.line_count,
            content=parsed.content,
            content_hash=parsed.content_hash,
            name=parsed.file_path.split("/")[-1],
            language=parsed.language.value,
            token_estimate=self._estimate_tokens(parsed.content),
        )

    def _create_large_function_chunk(
        self,
        parsed: ParsedFile,
        func_node: ASTNode,
        lines: list[str],
        func_name: str,
        parent_name: Optional[str],
        chunk_type: ChunkType
    ) -> CodeChunk:
        """Create chunk for large function with truncation warning."""
        content = func_node.text
        signature = self._extract_function_signature(func_node, parsed.language)
        docstring = self._extract_docstring(func_node, parsed.language)

        # Create a summary chunk instead
        summary_content = f"# Large function: {func_name}\n"
        if signature:
            summary_content += f"# Signature: {signature}\n"
        if docstring:
            summary_content += f'"""{docstring}"""\n'
        summary_content += f"# Lines: {func_node.start_line}-{func_node.end_line}\n"
        summary_content += f"# NOTE: Function too large ({self._estimate_tokens(content)} tokens)\n"
        summary_content += f"# See file for full implementation\n\n"

        # Include first portion
        max_lines = 50
        func_lines = content.split("\n")[:max_lines]
        summary_content += "\n".join(func_lines)
        if len(content.split("\n")) > max_lines:
            summary_content += "\n# ... (truncated)"

        return CodeChunk(
            chunk_id=self._generate_id(parsed.file_path, func_name, func_node.start_line),
            chunk_type=chunk_type,
            file_path=parsed.file_path,
            start_line=func_node.start_line,
            end_line=func_node.end_line,
            content=summary_content,
            content_hash=self._hash_content(content),  # Hash full content
            name=func_name,
            signature=signature,
            docstring=docstring,
            parent_name=parent_name,
            language=parsed.language.value,
            token_estimate=self._estimate_tokens(summary_content),
            complexity=self._estimate_complexity(func_node),
        )

    def _add_context(self, chunk: CodeChunk, lines: list[str]) -> None:
        """Add surrounding context to chunk."""
        if not self.config.context_lines_before and not self.config.context_lines_after:
            return

        # Add lines before
        if self.config.context_lines_before > 0:
            start = max(0, chunk.start_line - self.config.context_lines_before - 1)
            end = chunk.start_line - 1
            if start < end:
                chunk.context_before = "\n".join(lines[start:end])

        # Add lines after
        if self.config.context_lines_after > 0:
            start = chunk.end_line
            end = min(len(lines), chunk.end_line + self.config.context_lines_after)
            if start < end:
                chunk.context_after = "\n".join(lines[start:end])

    # Helper methods

    def _generate_id(self, file_path: str, name: str, line: int) -> str:
        """Generate unique chunk ID."""
        key = f"{file_path}:{name}:{line}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _hash_content(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count (rough: ~4 chars per token)."""
        return len(content) // 4

    def _estimate_complexity(self, node: ASTNode) -> int:
        """Estimate cyclomatic complexity."""
        complexity = 1

        # Count branching statements
        branch_types = {
            "if_statement", "elif_clause", "else_clause",
            "for_statement", "while_statement",
            "try_statement", "except_clause",
            "case_statement", "switch_statement",
            "conditional_expression", "ternary_expression",
        }

        for child in node.find_descendants("*"):
            if any(bt in child.type for bt in branch_types):
                complexity += 1

        return complexity

    def _extract_name(self, node: ASTNode, language: Language) -> Optional[str]:
        """Extract name from function/class/method node."""
        # Look for identifier child
        for child in node.children:
            if child.type == "identifier":
                return child.text
            if child.type == "name":
                return child.text
            if child.type == "property_identifier":
                return child.text

        return None

    def _extract_function_signature(
        self,
        node: ASTNode,
        language: Language
    ) -> Optional[str]:
        """Extract function signature."""
        # For Python: def name(params) -> return_type:
        # For JS/TS: function name(params): return_type

        text = node.text
        lines = text.split("\n")

        if language == Language.PYTHON:
            # Find the line with 'def' up to ':'
            for line in lines:
                if "def " in line:
                    return line.strip().rstrip(":")

        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX):
            # First line typically has signature
            first_line = lines[0].strip()
            if "function" in first_line or "=>" in first_line:
                return first_line.split("{")[0].strip()

        return None

    def _extract_class_signature(
        self,
        node: ASTNode,
        language: Language
    ) -> Optional[str]:
        """Extract class signature with inheritance."""
        text = node.text
        first_line = text.split("\n")[0].strip()

        if language == Language.PYTHON:
            # class Name(Parent):
            return first_line.rstrip(":")

        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX):
            # class Name extends Parent {
            return first_line.rstrip("{").strip()

        return None

    def _extract_class_header(
        self,
        node: ASTNode,
        lines: list[str],
        language: Language
    ) -> Optional[str]:
        """Extract class header (signature + docstring + attributes)."""
        header_lines = []
        in_class = False

        for line in node.text.split("\n"):
            stripped = line.strip()

            # Start of class
            if "class " in line and not in_class:
                header_lines.append(line)
                in_class = True
                continue

            if in_class:
                # Include docstring
                if '"""' in stripped or "'''" in stripped:
                    header_lines.append(line)
                    continue

                # Include class attributes (before methods)
                if stripped.startswith("def ") or stripped.startswith("async def "):
                    break

                # Include other attribute definitions
                if "=" in stripped and not stripped.startswith("#"):
                    header_lines.append(line)
                    continue

                # Include type annotations
                if ":" in stripped and "def" not in stripped:
                    header_lines.append(line)
                    continue

                # Stop at first method
                if stripped and not stripped.startswith("#"):
                    break

        return "\n".join(header_lines) if header_lines else None

    def _extract_docstring(
        self,
        node: ASTNode,
        language: Language
    ) -> Optional[str]:
        """Extract docstring from function/class."""
        if language == Language.PYTHON:
            # Look for string child at start of block
            for child in node.children:
                if child.type == "block":
                    for block_child in child.children:
                        if block_child.type == "expression_statement":
                            for expr in block_child.children:
                                if expr.type == "string":
                                    # Clean docstring
                                    doc = expr.text
                                    doc = doc.strip('"""').strip("'''").strip()
                                    return doc
                    break

        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX):
            # Look for JSDoc comment before function
            text = node.text
            if text.strip().startswith("/**"):
                end = text.find("*/")
                if end > 0:
                    doc = text[:end + 2]
                    # Clean JSDoc
                    doc = doc.replace("/**", "").replace("*/", "")
                    doc = "\n".join(
                        line.strip().lstrip("*").strip()
                        for line in doc.split("\n")
                    )
                    return doc.strip()

        return None

    def _extract_function_calls(
        self,
        node: ASTNode,
        language: Language
    ) -> list[str]:
        """Extract function calls within a node."""
        calls = []

        call_types = {"call_expression", "call", "function_call"}

        for child in node.find_descendants("call"):
            # Get function name
            for c in child.children:
                if c.type == "identifier":
                    calls.append(c.text)
                    break

        return list(set(calls))

    def _extract_assignment_name(self, node: ASTNode) -> Optional[str]:
        """Extract name from assignment."""
        for child in node.children:
            if child.type == "identifier":
                return child.text
        return None

    def _get_method_node_types(self, language: Language) -> list[str]:
        """Get AST node types for methods in given language."""
        if language == Language.PYTHON:
            return ["function_definition", "async_function_definition"]
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX):
            return ["method_definition", "function_declaration", "arrow_function"]
        elif language == Language.JAVA:
            return ["method_declaration", "constructor_declaration"]
        elif language == Language.GO:
            return ["method_declaration", "function_declaration"]
        return ["function", "method"]

    def _is_method(self, node: ASTNode) -> bool:
        """Check if function node is inside a class."""
        return node.parent_type in ("class_body", "class_definition", "class_declaration")

    def _is_test_function(self, name: str, content: str) -> bool:
        """Check if function is a test."""
        # Common test patterns
        if name.startswith("test_"):
            return True
        if name.startswith("Test"):
            return True
        if "@pytest" in content or "@test" in content.lower():
            return True
        if "describe(" in content or "it(" in content or "test(" in content:
            return True
        return False

    def _is_hook(self, name: str, language: Language) -> bool:
        """Check if function is a React hook."""
        if language not in (Language.TSX, Language.JSX, Language.TYPESCRIPT, Language.JAVASCRIPT):
            return False
        return name.startswith("use") and name[3:4].isupper()

    def _is_route_handler(self, node: ASTNode, language: Language) -> bool:
        """Check if function is an API route handler."""
        text = node.text

        # FastAPI/Flask decorators
        if "@app." in text or "@router." in text:
            return True

        # Express patterns
        if ".get(" in text or ".post(" in text or ".put(" in text:
            return True

        # Next.js API routes
        if "export default" in text and "req" in text and "res" in text:
            return True

        return False

    def _returns_jsx(self, node: ASTNode) -> bool:
        """Check if function returns JSX."""
        text = node.text
        return "<" in text and ">" in text and ("return" in text or "=>" in text)

    def _get_component_name(self, node: ASTNode) -> Optional[str]:
        """Get React component name from arrow function."""
        # Look for parent const declaration
        if node.parent_type == "variable_declarator":
            for sibling in node.children:
                if sibling.type == "identifier":
                    name = sibling.text
                    # Component names are PascalCase
                    if name and name[0].isupper():
                        return name
        return None


# Global instance
_chunker: Optional[SemanticChunker] = None


def get_chunker(config: Optional[ChunkingConfig] = None) -> SemanticChunker:
    """Get or create global chunker instance."""
    global _chunker
    if _chunker is None or config is not None:
        _chunker = SemanticChunker(config)
    return _chunker
