"""Tree-sitter Parser - Fast incremental parsing for 40+ languages.

Tree-sitter advantages over LSP:
- Milliseconds parsing (vs seconds for LSP)
- Incremental updates (only reparse changed parts)
- Error-tolerant (handles broken/incomplete code)
- Runs 100% locally (no server process needed)

This is the foundation of Argus's fast indexing.
"""

import hashlib
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    JSX = "jsx"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "c_sharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    HTML = "html"
    CSS = "css"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    MARKDOWN = "markdown"
    SQL = "sql"
    BASH = "bash"
    VUE = "vue"
    SVELTE = "svelte"
    UNKNOWN = "unknown"


# File extension to language mapping
EXTENSION_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".mts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    ".jsx": Language.JSX,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".css": Language.CSS,
    ".scss": Language.CSS,
    ".sass": Language.CSS,
    ".less": Language.CSS,
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".toml": Language.TOML,
    ".md": Language.MARKDOWN,
    ".mdx": Language.MARKDOWN,
    ".sql": Language.SQL,
    ".sh": Language.BASH,
    ".bash": Language.BASH,
    ".zsh": Language.BASH,
    ".vue": Language.VUE,
    ".svelte": Language.SVELTE,
}


@dataclass
class ASTNode:
    """Represents a node in the Abstract Syntax Tree."""
    type: str
    text: str
    start_line: int
    end_line: int
    start_column: int
    end_column: int
    start_byte: int
    end_byte: int
    children: list["ASTNode"] = field(default_factory=list)
    parent_type: str | None = None

    @property
    def is_named(self) -> bool:
        """Check if this is a named node (not anonymous)."""
        return not self.type.startswith("_")

    @property
    def line_count(self) -> int:
        """Number of lines this node spans."""
        return self.end_line - self.start_line + 1

    def find_children(self, node_type: str) -> list["ASTNode"]:
        """Find all children of a specific type."""
        return [c for c in self.children if c.type == node_type]

    def find_descendants(self, node_type: str) -> Generator["ASTNode", None, None]:
        """Find all descendants of a specific type (recursive)."""
        for child in self.children:
            if child.type == node_type:
                yield child
            yield from child.find_descendants(node_type)


@dataclass
class ParsedFile:
    """Result of parsing a source file."""
    file_path: str
    language: Language
    content: str
    content_hash: str
    root: ASTNode | None
    errors: list[str] = field(default_factory=list)
    parse_time_ms: float = 0.0

    @property
    def has_errors(self) -> bool:
        """Check if parsing had errors."""
        return len(self.errors) > 0

    @property
    def line_count(self) -> int:
        """Total lines in the file."""
        return self.content.count("\n") + 1

    def get_functions(self) -> list[ASTNode]:
        """Extract all function/method definitions."""
        if not self.root:
            return []

        function_types = {
            Language.PYTHON: ["function_definition", "async_function_definition"],
            Language.JAVASCRIPT: ["function_declaration", "function", "arrow_function", "method_definition"],
            Language.TYPESCRIPT: ["function_declaration", "function", "arrow_function", "method_definition"],
            Language.TSX: ["function_declaration", "function", "arrow_function", "method_definition"],
            Language.GO: ["function_declaration", "method_declaration"],
            Language.JAVA: ["method_declaration", "constructor_declaration"],
            Language.RUST: ["function_item"],
        }

        types_to_find = function_types.get(self.language, ["function"])
        functions = []

        for node_type in types_to_find:
            functions.extend(self.root.find_descendants(node_type))

        return functions

    def get_classes(self) -> list[ASTNode]:
        """Extract all class definitions."""
        if not self.root:
            return []

        class_types = {
            Language.PYTHON: ["class_definition"],
            Language.JAVASCRIPT: ["class_declaration", "class"],
            Language.TYPESCRIPT: ["class_declaration", "class"],
            Language.TSX: ["class_declaration", "class"],
            Language.JAVA: ["class_declaration", "interface_declaration"],
            Language.GO: ["type_declaration"],
            Language.RUST: ["struct_item", "impl_item"],
        }

        types_to_find = class_types.get(self.language, ["class"])
        classes = []

        for node_type in types_to_find:
            classes.extend(self.root.find_descendants(node_type))

        return classes

    def get_imports(self) -> list[ASTNode]:
        """Extract all import statements."""
        if not self.root:
            return []

        import_types = {
            Language.PYTHON: ["import_statement", "import_from_statement"],
            Language.JAVASCRIPT: ["import_statement", "import"],
            Language.TYPESCRIPT: ["import_statement", "import"],
            Language.TSX: ["import_statement", "import"],
            Language.GO: ["import_declaration"],
            Language.JAVA: ["import_declaration"],
            Language.RUST: ["use_declaration"],
        }

        types_to_find = import_types.get(self.language, ["import"])
        imports = []

        for node_type in types_to_find:
            imports.extend(self.root.find_descendants(node_type))

        return imports


class TreeSitterParser:
    """Fast incremental parser using tree-sitter.

    Usage:
        parser = TreeSitterParser()
        parsed = parser.parse_file("src/app.py")
        functions = parsed.get_functions()

    For incremental parsing (when file changes):
        new_parsed = parser.parse_incremental(old_parsed, new_content)
    """

    def __init__(self):
        """Initialize parser with tree-sitter bindings."""
        self._parsers: dict[Language, any] = {}
        self._tree_cache: dict[str, any] = {}  # file_path -> tree
        self._initialized = False

        # Try to import tree-sitter
        try:
            import tree_sitter
            self._ts = tree_sitter
            self._initialized = True
            logger.info("Tree-sitter initialized successfully")
        except ImportError:
            logger.warning(
                "tree-sitter not installed. Using fallback regex parser. "
                "Install with: pip install tree-sitter tree-sitter-languages"
            )
            self._ts = None

    def _get_parser(self, language: Language):
        """Get or create a parser for the given language."""
        if not self._initialized:
            return None

        if language in self._parsers:
            return self._parsers[language]

        try:
            import tree_sitter_languages
            parser = tree_sitter_languages.get_parser(language.value)
            self._parsers[language] = parser
            return parser
        except Exception as e:
            logger.warning(f"Could not load parser for {language.value}: {e}")
            return None

    def detect_language(self, file_path: str) -> Language:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        return EXTENSION_MAP.get(ext, Language.UNKNOWN)

    def parse_file(self, file_path: str) -> ParsedFile:
        """Parse a source file.

        Args:
            file_path: Path to source file

        Returns:
            ParsedFile with AST and metadata
        """
        import time
        start = time.perf_counter()

        path = Path(file_path)
        language = self.detect_language(file_path)

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return ParsedFile(
                file_path=str(file_path),
                language=language,
                content="",
                content_hash="",
                root=None,
                errors=[f"Could not read file: {e}"],
            )

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Try tree-sitter parsing
        root, errors = self._parse_with_tree_sitter(content, language)

        # Fallback to regex parsing if tree-sitter failed
        if root is None and not errors:
            root, errors = self._parse_with_regex(content, language)

        elapsed = (time.perf_counter() - start) * 1000

        return ParsedFile(
            file_path=str(file_path),
            language=language,
            content=content,
            content_hash=content_hash,
            root=root,
            errors=errors,
            parse_time_ms=elapsed,
        )

    def parse_content(
        self,
        content: str,
        file_path: str = "<string>",
    ) -> ParsedFile:
        """Parse content directly (without reading file).

        Args:
            content: Source code content
            file_path: File path used for language detection and context

        Returns:
            ParsedFile with AST
        """
        import time
        start = time.perf_counter()

        # Auto-detect language from file extension
        language = self.detect_language(file_path)

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        root, errors = self._parse_with_tree_sitter(content, language)

        if root is None and not errors:
            root, errors = self._parse_with_regex(content, language)

        elapsed = (time.perf_counter() - start) * 1000

        return ParsedFile(
            file_path=file_path,
            language=language,
            content=content,
            content_hash=content_hash,
            root=root,
            errors=errors,
            parse_time_ms=elapsed,
        )

    def _parse_with_tree_sitter(
        self,
        content: str,
        language: Language,
    ) -> tuple[ASTNode | None, list[str]]:
        """Parse using tree-sitter."""
        if not self._initialized:
            return None, []

        parser = self._get_parser(language)
        if parser is None:
            return None, []

        try:
            tree = parser.parse(content.encode("utf-8"))
            root = self._convert_tree(tree.root_node, content)

            # Check for syntax errors
            errors = []
            for error_node in self._find_errors(tree.root_node):
                line = error_node.start_point[0] + 1
                errors.append(f"Syntax error at line {line}")

            return root, errors

        except Exception as e:
            logger.error(f"Tree-sitter parsing failed: {e}")
            return None, [str(e)]

    def _convert_tree(self, node, content: str, parent_type: str = None) -> ASTNode:
        """Convert tree-sitter node to our ASTNode."""
        children = [
            self._convert_tree(child, content, node.type)
            for child in node.children
        ]

        return ASTNode(
            type=node.type,
            text=content[node.start_byte:node.end_byte],
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            start_column=node.start_point[1],
            end_column=node.end_point[1],
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            children=children,
            parent_type=parent_type,
        )

    def _find_errors(self, node) -> Generator:
        """Find all ERROR nodes in the tree."""
        if node.type == "ERROR":
            yield node
        for child in node.children:
            yield from self._find_errors(child)

    def _parse_with_regex(
        self,
        content: str,
        language: Language,
    ) -> tuple[ASTNode | None, list[str]]:
        """Fallback regex-based parsing when tree-sitter unavailable.

        This is less accurate but ensures we can always parse.
        """
        import re

        lines = content.split("\n")
        children = []
        errors = []

        # Language-specific patterns
        patterns = self._get_regex_patterns(language)

        for i, line in enumerate(lines):
            line_num = i + 1

            for pattern_type, pattern in patterns.items():
                match = re.match(pattern, line)
                if match:
                    children.append(ASTNode(
                        type=pattern_type,
                        text=line.strip(),
                        start_line=line_num,
                        end_line=line_num,
                        start_column=0,
                        end_column=len(line),
                        start_byte=sum(len(l) + 1 for l in lines[:i]),
                        end_byte=sum(len(l) + 1 for l in lines[:i+1]),
                    ))
                    break

        root = ASTNode(
            type="program",
            text=content,
            start_line=1,
            end_line=len(lines),
            start_column=0,
            end_column=len(lines[-1]) if lines else 0,
            start_byte=0,
            end_byte=len(content),
            children=children,
        )

        return root, errors

    def _get_regex_patterns(self, language: Language) -> dict[str, str]:
        """Get regex patterns for fallback parsing."""
        if language == Language.PYTHON:
            return {
                "function_definition": r"^\s*(?:async\s+)?def\s+(\w+)",
                "class_definition": r"^\s*class\s+(\w+)",
                "import_statement": r"^\s*(?:from\s+\S+\s+)?import\s+",
            }
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX, Language.JSX):
            return {
                "function_declaration": r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)",
                "arrow_function": r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(",
                "class_declaration": r"^\s*(?:export\s+)?class\s+(\w+)",
                "import_statement": r"^\s*import\s+",
            }
        elif language == Language.GO:
            return {
                "function_declaration": r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)",
                "type_declaration": r"^\s*type\s+(\w+)\s+(?:struct|interface)",
                "import_declaration": r"^\s*import\s+",
            }
        elif language == Language.JAVA:
            return {
                "method_declaration": r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)+(\w+)\s*\(",
                "class_declaration": r"^\s*(?:public\s+)?(?:abstract\s+)?class\s+(\w+)",
                "import_declaration": r"^\s*import\s+",
            }
        else:
            return {
                "function": r"^\s*(?:function|def|fn|func)\s+(\w+)",
                "class": r"^\s*(?:class|struct|type)\s+(\w+)",
                "import": r"^\s*(?:import|use|require|include)\s+",
            }

    def parse_incremental(
        self,
        old_parsed: ParsedFile,
        new_content: str,
    ) -> ParsedFile:
        """Incrementally parse updated content.

        This is more efficient than full reparse when only small
        parts of the file changed.

        Args:
            old_parsed: Previous parse result
            new_content: New file content

        Returns:
            Updated ParsedFile
        """
        # For now, just do full reparse
        # TODO: Implement true incremental parsing with tree-sitter edit API
        return self.parse_content(
            new_content,
            old_parsed.language,
            old_parsed.file_path,
        )


# Global instance (lazy initialized)
_parser: TreeSitterParser | None = None


def get_parser() -> TreeSitterParser:
    """Get or create the global parser instance."""
    global _parser
    if _parser is None:
        _parser = TreeSitterParser()
    return _parser
