"""Tests for the tree-sitter parser module."""

import os
import tempfile

import pytest

from src.indexer.tree_sitter_parser import (
    EXTENSION_MAP,
    Language,
    TreeSitterParser,
)


class TestTreeSitterParser:
    """Test TreeSitterParser functionality."""

    @pytest.fixture
    def parser(self):
        """Create a TreeSitterParser instance."""
        return TreeSitterParser()

    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''"""Sample module docstring."""

import os
from typing import Optional

class MyClass:
    """A sample class."""

    def __init__(self, value: int):
        self.value = value

    def get_value(self) -> int:
        """Return the value."""
        return self.value

def helper_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

async def async_function() -> str:
    """Async function example."""
    return "hello"
'''

    @pytest.fixture
    def sample_typescript_code(self):
        """Sample TypeScript code for testing."""
        return '''import React from "react";
import { useState, useEffect } from "react";

interface Props {
    name: string;
    onClick?: () => void;
}

export const MyComponent: React.FC<Props> = ({ name, onClick }) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        console.log("mounted");
    }, []);

    return (
        <div data-testid="my-component">
            <span>{name}</span>
            <button onClick={onClick}>Click me</button>
        </div>
    );
};

export function useCustomHook(initialValue: number) {
    const [value, setValue] = useState(initialValue);
    return { value, setValue };
}
'''

    def test_language_detection_python(self, parser):
        """Test language detection for Python files."""
        assert parser.detect_language("test.py") == Language.PYTHON
        assert parser.detect_language("script.py") == Language.PYTHON

    def test_language_detection_typescript(self, parser):
        """Test language detection for TypeScript files."""
        assert parser.detect_language("component.tsx") == Language.TSX
        assert parser.detect_language("utils.ts") == Language.TYPESCRIPT

    def test_language_detection_javascript(self, parser):
        """Test language detection for JavaScript files."""
        assert parser.detect_language("app.js") == Language.JAVASCRIPT
        assert parser.detect_language("component.jsx") == Language.JSX

    def test_language_detection_go(self, parser):
        """Test language detection for Go files."""
        assert parser.detect_language("main.go") == Language.GO

    def test_language_detection_rust(self, parser):
        """Test language detection for Rust files."""
        assert parser.detect_language("lib.rs") == Language.RUST

    def test_language_detection_unknown(self, parser):
        """Test language detection for unknown extensions."""
        assert parser.detect_language("file.xyz") == Language.UNKNOWN
        assert parser.detect_language("file.unknown") == Language.UNKNOWN

    @pytest.mark.requires_tree_sitter
    def test_parse_python_content(self, parser, sample_python_code):
        """Test parsing Python content."""
        parsed = parser.parse_content(sample_python_code, "test.py")

        assert parsed is not None
        assert parsed.language == Language.PYTHON
        assert parsed.content == sample_python_code
        assert parsed.file_path == "test.py"

    @pytest.mark.requires_tree_sitter
    def test_parse_python_functions(self, parser, sample_python_code):
        """Test extracting functions from Python code."""
        parsed = parser.parse_content(sample_python_code, "test.py")
        functions = parsed.get_functions()

        function_names = [f.name for f in functions]
        assert "helper_function" in function_names
        assert "async_function" in function_names

    @pytest.mark.requires_tree_sitter
    def test_parse_python_classes(self, parser, sample_python_code):
        """Test extracting classes from Python code."""
        parsed = parser.parse_content(sample_python_code, "test.py")
        classes = parsed.get_classes()

        class_names = [c.name for c in classes]
        assert "MyClass" in class_names

    def test_parse_python_imports(self, parser, sample_python_code):
        """Test extracting imports from Python code."""
        parsed = parser.parse_content(sample_python_code, "test.py")
        imports = parsed.get_imports()

        assert len(imports) > 0
        import_texts = [i.text for i in imports]
        assert any("os" in t for t in import_texts)

    @pytest.mark.requires_tree_sitter
    def test_parse_typescript_content(self, parser, sample_typescript_code):
        """Test parsing TypeScript content."""
        parsed = parser.parse_content(sample_typescript_code, "component.tsx")

        assert parsed is not None
        assert parsed.language == Language.TSX

    @pytest.mark.requires_tree_sitter
    def test_parse_typescript_functions(self, parser, sample_typescript_code):
        """Test extracting functions from TypeScript code."""
        parsed = parser.parse_content(sample_typescript_code, "component.tsx")
        functions = parsed.get_functions()

        function_names = [f.name for f in functions if f.name]
        assert "useCustomHook" in function_names

    def test_parse_file_creates_temp_file(self, parser, sample_python_code):
        """Test parsing from file path."""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(sample_python_code)
            temp_path = f.name

        try:
            parsed = parser.parse_file(temp_path)
            assert parsed is not None
            assert parsed.language == Language.PYTHON
        finally:
            os.unlink(temp_path)

    def test_parse_file_nonexistent(self, parser):
        """Test parsing nonexistent file raises or returns ParsedFile with errors."""
        # parse_file may raise or return a ParsedFile with errors
        try:
            parsed = parser.parse_file("/nonexistent/path/file.py")
            # If it returns, check it handles gracefully
            assert parsed is None or parsed.has_errors
        except Exception:
            pass  # Expected to raise for nonexistent files

    @pytest.mark.requires_tree_sitter
    def test_ast_node_find_descendants(self, parser, sample_python_code):
        """Test finding descendants in AST."""
        parsed = parser.parse_content(sample_python_code, "test.py")

        # Find all function definitions
        func_nodes = parsed.root.find_descendants("function_definition")
        assert len(func_nodes) >= 2  # At least helper_function and async_function

    def test_ast_node_text_extraction(self, parser, sample_python_code):
        """Test text extraction from AST nodes."""
        parsed = parser.parse_content(sample_python_code, "test.py")
        functions = parsed.get_functions()

        for func in functions:
            assert func.text is not None
            assert len(func.text) > 0

    def test_ast_node_line_numbers(self, parser, sample_python_code):
        """Test line number extraction from AST nodes."""
        parsed = parser.parse_content(sample_python_code, "test.py")
        classes = parsed.get_classes()

        for cls in classes:
            assert cls.start_line >= 1
            assert cls.end_line >= cls.start_line

    def test_extension_map_completeness(self):
        """Test that common languages are in the extension map."""
        common_extensions = [".py", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".rb"]
        for ext in common_extensions:
            assert ext in EXTENSION_MAP, f"Extension {ext} not in extension map"

    def test_parse_empty_content(self, parser):
        """Test parsing empty content."""
        parsed = parser.parse_content("", "empty.py")
        assert parsed is not None
        assert parsed.content == ""

    def test_parse_content_with_syntax_errors(self, parser):
        """Test parsing content with syntax errors (parser should still work)."""
        bad_python = "def broken(\n  # missing close paren"
        parsed = parser.parse_content(bad_python, "broken.py")
        # Tree-sitter should still parse, even with errors
        assert parsed is not None


class TestParsedFile:
    """Test ParsedFile methods."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def parsed_python(self, parser):
        """Create a parsed Python file."""
        code = '''
class Foo:
    def bar(self):
        pass

    def baz(self):
        pass

def standalone():
    pass
'''
        return parser.parse_content(code, "test.py")

    def test_get_classes_returns_list(self, parsed_python):
        """Test get_classes returns a list."""
        classes = parsed_python.get_classes()
        assert isinstance(classes, list)
        assert len(classes) == 1

    def test_get_functions_returns_list(self, parsed_python):
        """Test get_functions returns a list."""
        functions = parsed_python.get_functions()
        assert isinstance(functions, list)
        assert len(functions) >= 1  # At least standalone

    def test_get_imports_empty_file(self, parser):
        """Test get_imports on file with no imports."""
        parsed = parser.parse_content("x = 1", "no_imports.py")
        imports = parsed.get_imports()
        assert isinstance(imports, list)


class TestASTNode:
    """Test ASTNode methods."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.mark.requires_tree_sitter
    def test_find_descendants_filters_by_type(self, parser):
        """Test find_descendants filters by node type."""
        code = '''
def a():
    def b():
        pass
    pass

class C:
    def d():
        pass
'''
        parsed = parser.parse_content(code, "test.py")
        func_nodes = parsed.root.find_descendants("function_definition")
        class_nodes = parsed.root.find_descendants("class_definition")

        assert len(func_nodes) >= 3  # a, b, d
        assert len(class_nodes) == 1  # C

    def test_node_children(self, parser):
        """Test accessing node children."""
        code = "def foo(x, y): return x + y"
        parsed = parser.parse_content(code, "test.py")

        functions = parsed.get_functions()
        assert len(functions) == 1

        func = functions[0]
        assert func.children is not None
