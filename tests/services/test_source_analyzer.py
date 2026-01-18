"""Tests for the Source Code Analyzer service."""

import tempfile
from pathlib import Path

from src.services.source_analyzer import (
    ComponentInfo,
    ExtractedSelector,
    SelectorMapping,
    SourceAnalyzer,
    get_source_analyzer,
)


class TestDataClasses:
    """Tests for dataclass definitions."""

    def test_extracted_selector_creation(self):
        """Test ExtractedSelector dataclass creation."""
        selector = ExtractedSelector(
            selector="submit-button",
            selector_type="testid",
            file_path="src/components/Button.tsx",
            line_number=42,
            column=10,
            element_type="button",
            context='<button data-testid="submit-button">Submit</button>',
            semantic_name="submit button"
        )

        assert selector.selector == "submit-button"
        assert selector.selector_type == "testid"
        assert selector.file_path == "src/components/Button.tsx"
        assert selector.line_number == 42
        assert selector.column == 10
        assert selector.element_type == "button"
        assert selector.semantic_name == "submit button"

    def test_extracted_selector_defaults(self):
        """Test ExtractedSelector with default values."""
        selector = ExtractedSelector(
            selector="my-id",
            selector_type="id",
            file_path="test.tsx",
            line_number=1,
            column=1
        )

        assert selector.element_type is None
        assert selector.context == ""
        assert selector.semantic_name is None

    def test_component_info_creation(self):
        """Test ComponentInfo dataclass creation."""
        selectors = [
            ExtractedSelector("btn-1", "testid", "test.tsx", 1, 1),
            ExtractedSelector("btn-2", "testid", "test.tsx", 2, 1),
        ]
        component = ComponentInfo(
            name="MyButton",
            file_path="src/components/MyButton.tsx",
            selectors=selectors,
            exports=["MyButton", "default:MyButton"],
            imports=["react", "./utils"],
            props=["onClick", "disabled", "children"]
        )

        assert component.name == "MyButton"
        assert component.file_path == "src/components/MyButton.tsx"
        assert len(component.selectors) == 2
        assert "MyButton" in component.exports
        assert "react" in component.imports
        assert "onClick" in component.props

    def test_component_info_defaults(self):
        """Test ComponentInfo with default values."""
        component = ComponentInfo(
            name="Test",
            file_path="test.tsx"
        )

        assert component.selectors == []
        assert component.exports == []
        assert component.imports == []
        assert component.props == []

    def test_selector_mapping_creation(self):
        """Test SelectorMapping dataclass creation."""
        mapping = SelectorMapping(
            selector="login-button",
            semantic_purpose="login element",
            alternatives=["signin-btn", "auth-button"],
            stability_score=0.85,
            recommendation="Consider using data-testid"
        )

        assert mapping.selector == "login-button"
        assert mapping.semantic_purpose == "login element"
        assert len(mapping.alternatives) == 2
        assert mapping.stability_score == 0.85
        assert mapping.recommendation is not None

    def test_selector_mapping_defaults(self):
        """Test SelectorMapping with default values."""
        mapping = SelectorMapping(
            selector="test",
            semantic_purpose="test element"
        )

        assert mapping.alternatives == []
        assert mapping.stability_score == 0.5
        assert mapping.recommendation is None


class TestSourceAnalyzerInit:
    """Tests for SourceAnalyzer initialization."""

    def test_init_with_default_path(self):
        """Test initialization with default repo path."""
        analyzer = SourceAnalyzer()

        assert analyzer.repo_path == Path(".").resolve()
        assert analyzer._component_cache == {}
        assert analyzer._selector_cache == {}

    def test_init_with_custom_path(self):
        """Test initialization with custom repo path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SourceAnalyzer(repo_path=tmpdir)

            assert analyzer.repo_path == Path(tmpdir).resolve()

    def test_framework_extensions_defined(self):
        """Test that framework extensions are properly defined."""
        assert "react" in SourceAnalyzer.FRAMEWORK_EXTENSIONS
        assert "vue" in SourceAnalyzer.FRAMEWORK_EXTENSIONS
        assert "angular" in SourceAnalyzer.FRAMEWORK_EXTENSIONS
        assert "svelte" in SourceAnalyzer.FRAMEWORK_EXTENSIONS

        assert ".jsx" in SourceAnalyzer.FRAMEWORK_EXTENSIONS["react"]
        assert ".tsx" in SourceAnalyzer.FRAMEWORK_EXTENSIONS["react"]

    def test_selector_patterns_defined(self):
        """Test that selector patterns are properly defined."""
        assert "testid" in SourceAnalyzer.SELECTOR_PATTERNS
        assert "id" in SourceAnalyzer.SELECTOR_PATTERNS
        assert "class" in SourceAnalyzer.SELECTOR_PATTERNS
        assert "aria" in SourceAnalyzer.SELECTOR_PATTERNS
        assert "role" in SourceAnalyzer.SELECTOR_PATTERNS
        assert "text" in SourceAnalyzer.SELECTOR_PATTERNS

    def test_element_patterns_defined(self):
        """Test that element patterns are properly defined."""
        assert "button" in SourceAnalyzer.ELEMENT_PATTERNS
        assert "input" in SourceAnalyzer.ELEMENT_PATTERNS
        assert "link" in SourceAnalyzer.ELEMENT_PATTERNS
        assert "form" in SourceAnalyzer.ELEMENT_PATTERNS


class TestSourceAnalyzerAnalyzeFile:
    """Tests for analyze_file method."""

    def test_analyze_file_not_found(self):
        """Test analyzing a non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SourceAnalyzer(repo_path=tmpdir)

            component = analyzer.analyze_file("nonexistent.tsx")

            assert component.name == ""
            assert component.file_path == "nonexistent.tsx"
            assert component.selectors == []

    def test_analyze_react_component_with_testids(self):
        """Test analyzing React component with data-testid attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
import React from 'react';

export default function LoginForm() {
    return (
        <form data-testid="login-form">
            <input data-testid="email-input" type="email" />
            <input data-testid="password-input" type="password" />
            <button data-testid="submit-button">Login</button>
        </form>
    );
}
'''
            file_path = Path(tmpdir) / "LoginForm.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("LoginForm.tsx")

            assert component.name == "LoginForm"
            assert len(component.selectors) >= 4

            testids = [s.selector for s in component.selectors if s.selector_type == "testid"]
            assert "login-form" in testids
            assert "email-input" in testids
            assert "password-input" in testids
            assert "submit-button" in testids

    def test_analyze_component_with_id_selectors(self):
        """Test analyzing component with id attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return (
        <div id="main-container">
            <input id="search-box" />
        </div>
    );
}
'''
            file_path = Path(tmpdir) / "Component.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("Component.tsx")

            ids = [s.selector for s in component.selectors if s.selector_type == "id"]
            assert "main-container" in ids
            assert "search-box" in ids

    def test_analyze_component_with_class_selectors(self):
        """Test analyzing component with className attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return (
        <div className="container main-wrapper">
            <button className="btn btn-primary">Click</button>
        </div>
    );
}
'''
            file_path = Path(tmpdir) / "Component.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("Component.tsx")

            classes = [s.selector for s in component.selectors if s.selector_type == "class"]
            # Classes with spaces are split
            assert "container" in classes or "main-wrapper" in classes

    def test_analyze_component_with_aria_labels(self):
        """Test analyzing component with aria-label attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return (
        <button aria-label="Close dialog">X</button>
    );
}
'''
            file_path = Path(tmpdir) / "Component.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("Component.tsx")

            aria = [s.selector for s in component.selectors if s.selector_type == "aria"]
            assert "Close dialog" in aria

    def test_analyze_component_with_roles(self):
        """Test analyzing component with role attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return (
        <div role="dialog">
            <nav role="navigation">Menu</nav>
        </div>
    );
}
'''
            file_path = Path(tmpdir) / "Component.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("Component.tsx")

            roles = [s.selector for s in component.selectors if s.selector_type == "role"]
            assert "dialog" in roles
            assert "navigation" in roles

    def test_analyze_file_caches_result(self):
        """Test that analyze_file caches the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = 'function Test() { return <div />; }'
            file_path = Path(tmpdir) / "Test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)

            component1 = analyzer.analyze_file("Test.tsx")
            assert "Test.tsx" in analyzer._component_cache

            component2 = analyzer._component_cache["Test.tsx"]
            assert component1 is component2


class TestSourceAnalyzerExtractComponentName:
    """Tests for component name extraction."""

    def test_extract_export_default_function(self):
        """Test extracting name from export default function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = 'export default function MyComponent() { return null; }'
            file_path = Path(tmpdir) / "MyComponent.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("MyComponent.tsx")

            assert component.name == "MyComponent"

    def test_extract_function_component(self):
        """Test extracting name from function component."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = 'function TestComponent() { return null; }'
            file_path = Path(tmpdir) / "TestComponent.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("TestComponent.tsx")

            assert component.name == "TestComponent"

    def test_extract_const_arrow_function(self):
        """Test extracting name from const arrow function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = 'const ArrowComponent = () => { return null; }'
            file_path = Path(tmpdir) / "ArrowComponent.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("ArrowComponent.tsx")

            assert component.name == "ArrowComponent"

    def test_extract_class_component(self):
        """Test extracting name from class component."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
class ClassComponent extends React.Component {
    render() { return null; }
}
'''
            file_path = Path(tmpdir) / "ClassComponent.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("ClassComponent.tsx")

            assert component.name == "ClassComponent"

    def test_extract_vue_component_name(self):
        """Test extracting name from Vue SFC."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
<script>
export default {
    name: 'MyVueComponent',
    data() { return {}; }
}
</script>
'''
            file_path = Path(tmpdir) / "MyVueComponent.vue"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("MyVueComponent.vue")

            assert component.name == "MyVueComponent"

    def test_fallback_to_filename(self):
        """Test fallback to filename when name cannot be extracted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = 'const x = 1;'  # No component
            file_path = Path(tmpdir) / "unknown.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("unknown.tsx")

            assert component.name == "unknown"


class TestSourceAnalyzerDetectElementType:
    """Tests for element type detection."""

    def test_detect_button_element(self):
        """Test detecting button elements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return <button data-testid="my-button">Click</button>;
}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            button_selectors = [s for s in component.selectors if s.element_type == "button"]
            assert len(button_selectors) >= 1

    def test_detect_input_element(self):
        """Test detecting input elements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return <input data-testid="my-input" type="text" />;
}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            input_selectors = [s for s in component.selectors if s.element_type == "input"]
            assert len(input_selectors) >= 1

    def test_detect_link_element(self):
        """Test detecting link elements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return <a href="/home" data-testid="home-link">Home</a>;
}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            link_selectors = [s for s in component.selectors if s.element_type == "link"]
            assert len(link_selectors) >= 1


class TestSourceAnalyzerSemanticNaming:
    """Tests for semantic name inference."""

    def test_infer_submit_button(self):
        """Test inferring submit button semantic name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return <button data-testid="submit-form-btn">Submit</button>;
}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            submit_selectors = [s for s in component.selectors if "submit" in (s.semantic_name or "")]
            assert len(submit_selectors) >= 1

    def test_infer_email_input(self):
        """Test inferring email input semantic name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return <input data-testid="email-field" type="email" />;
}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            email_selectors = [s for s in component.selectors if "email" in (s.semantic_name or "")]
            assert len(email_selectors) >= 1

    def test_infer_login_element(self):
        """Test inferring login element semantic name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component() {
    return <button data-testid="login-btn">Login</button>;
}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            login_selectors = [s for s in component.selectors if "login" in (s.semantic_name or "")]
            assert len(login_selectors) >= 1


class TestSourceAnalyzerExtractImportsExports:
    """Tests for import/export extraction."""

    def test_extract_es6_imports(self):
        """Test extracting ES6 imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
import React from 'react';
import { useState, useEffect } from 'react';
import utils from './utils';
import styles from './styles.css';
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            assert "react" in component.imports
            assert "./utils" in component.imports
            assert "./styles.css" in component.imports

    def test_extract_require_imports(self):
        """Test extracting require() imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
const React = require('react');
const utils = require('./utils');
'''
            file_path = Path(tmpdir) / "test.js"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.js")

            assert "react" in component.imports
            assert "./utils" in component.imports

    def test_extract_named_exports(self):
        """Test extracting named exports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
export const MyComponent = () => null;
export function helper() {}
export class MyClass {}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            assert "MyComponent" in component.exports
            assert "helper" in component.exports
            assert "MyClass" in component.exports

    def test_extract_export_list(self):
        """Test extracting export list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
const a = 1;
const b = 2;
export { a, b, c as renamed };
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            assert "a" in component.exports
            assert "b" in component.exports
            assert "c" in component.exports

    def test_extract_default_export(self):
        """Test extracting default export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = 'export default function MainComponent() { return null; }'
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            assert "default:MainComponent" in component.exports


class TestSourceAnalyzerExtractProps:
    """Tests for props extraction."""

    def test_extract_typescript_interface_props(self):
        """Test extracting props from TypeScript interface."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
interface ButtonProps {
    onClick: () => void;
    disabled?: boolean;
    children: React.ReactNode;
}

function Button({ onClick, disabled, children }: ButtonProps) {
    return <button onClick={onClick} disabled={disabled}>{children}</button>;
}
'''
            file_path = Path(tmpdir) / "Button.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("Button.tsx")

            assert "onClick" in component.props
            assert "disabled" in component.props
            assert "children" in component.props

    def test_extract_destructured_props(self):
        """Test extracting destructured props."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
function Component({ name, age, email }) {
    return <div>{name} - {age} - {email}</div>;
}
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            component = analyzer.analyze_file("test.tsx")

            assert "name" in component.props
            assert "age" in component.props
            assert "email" in component.props


class TestSourceAnalyzerFindSelector:
    """Tests for find_selector method."""

    def test_find_selector_in_specific_file(self):
        """Test finding selector in a specific file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '<button data-testid="target-button">Click</button>'
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            results = analyzer.find_selector("target-button", file_path="test.tsx")

            assert len(results) == 1
            assert results[0].selector == "target-button"

    def test_find_selector_in_cache(self):
        """Test finding selector across all cached files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1_content = '<button data-testid="shared-id">A</button>'
            file2_content = '<div data-testid="shared-id">B</div>'

            (Path(tmpdir) / "file1.tsx").write_text(file1_content)
            (Path(tmpdir) / "file2.tsx").write_text(file2_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("file1.tsx")
            analyzer.analyze_file("file2.tsx")

            results = analyzer.find_selector("shared-id")

            assert len(results) == 2

    def test_find_selector_not_found(self):
        """Test finding non-existent selector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '<div data-testid="existing">Test</div>'
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            results = analyzer.find_selector("nonexistent")

            assert results == []


class TestSourceAnalyzerSimilarity:
    """Tests for similarity calculation and similar selector finding."""

    def test_calculate_similarity_exact_match(self):
        """Test similarity calculation for exact match."""
        analyzer = SourceAnalyzer()

        similarity = analyzer._calculate_similarity("test-button", "test-button")

        assert similarity == 1.0

    def test_calculate_similarity_substring(self):
        """Test similarity calculation for substring match."""
        analyzer = SourceAnalyzer()

        similarity = analyzer._calculate_similarity("button", "test-button")

        assert similarity >= 0.9

    def test_calculate_similarity_word_overlap(self):
        """Test similarity calculation for word overlap."""
        analyzer = SourceAnalyzer()

        similarity = analyzer._calculate_similarity("submit-button", "submit-btn")

        assert similarity > 0.5

    def test_calculate_similarity_no_match(self):
        """Test similarity calculation for completely different strings."""
        analyzer = SourceAnalyzer()

        similarity = analyzer._calculate_similarity("abc", "xyz")

        assert similarity < 0.5

    def test_find_similar_selectors(self):
        """Test finding similar selectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
<button data-testid="login-button">Login</button>
<button data-testid="login-btn">Login</button>
<button data-testid="signin-button">Sign In</button>
<button data-testid="logout-button">Logout</button>
'''
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            similar = analyzer.find_similar_selectors("login-button", threshold=0.5)

            assert len(similar) > 0
            # First result should be exact match
            assert similar[0][0].selector == "login-button"
            assert similar[0][1] == 1.0


class TestSourceAnalyzerSelectorMapping:
    """Tests for selector mapping functionality."""

    def test_get_selector_mapping_found(self):
        """Test getting selector mapping for existing selector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '<button data-testid="submit-button">Submit</button>'
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            mapping = analyzer.get_selector_mapping("submit-button")

            assert mapping is not None
            assert mapping.selector == "submit-button"
            assert "submit" in mapping.semantic_purpose

    def test_get_selector_mapping_not_found(self):
        """Test getting selector mapping for non-existent selector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SourceAnalyzer(repo_path=tmpdir)

            mapping = analyzer.get_selector_mapping("nonexistent")

            assert mapping is None

    def test_stability_score_testid(self):
        """Test that testid selectors have high stability score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '<button data-testid="stable-button">Click</button>'
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            mapping = analyzer.get_selector_mapping("stable-button")

            assert mapping is not None
            assert mapping.stability_score >= 0.9

    def test_stability_score_class(self):
        """Test that class selectors have lower stability score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '<button className="unstable-class">Click</button>'
            file_path = Path(tmpdir) / "test.tsx"
            file_path.write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            selectors = [s for s in analyzer.get_all_selectors() if s.selector == "unstable-class"]
            if selectors:
                score = analyzer._calculate_stability_score(selectors[0])
                assert score < 0.5

    def test_recommendation_for_class_selector(self):
        """Test recommendation is generated for class selectors."""
        selector = ExtractedSelector(
            selector="btn-primary",
            selector_type="class",
            file_path="test.tsx",
            line_number=1,
            column=1
        )

        analyzer = SourceAnalyzer()
        recommendation = analyzer._generate_recommendation(selector, 0.4)

        assert recommendation is not None
        assert "data-testid" in recommendation

    def test_recommendation_for_text_selector(self):
        """Test recommendation is generated for text selectors."""
        selector = ExtractedSelector(
            selector="Click Me",
            selector_type="text",
            file_path="test.tsx",
            line_number=1,
            column=1
        )

        analyzer = SourceAnalyzer()
        recommendation = analyzer._generate_recommendation(selector, 0.3)

        assert recommendation is not None
        assert "fragile" in recommendation.lower()


class TestSourceAnalyzerAnalyzeDirectory:
    """Tests for directory analysis."""

    def test_analyze_directory(self):
        """Test analyzing all files in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            (src_dir / "Component1.tsx").write_text('<div data-testid="c1" />')
            (src_dir / "Component2.tsx").write_text('<div data-testid="c2" />')
            (src_dir / "utils.ts").write_text('export const helper = () => {};')

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            results = analyzer.analyze_directory("src")

            assert len(results) >= 2

    def test_analyze_directory_skips_node_modules(self):
        """Test that node_modules is skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            node_dir = src_dir / "node_modules" / "package"
            node_dir.mkdir(parents=True)

            (src_dir / "Component.tsx").write_text('<div />')
            (node_dir / "index.tsx").write_text('<div />')

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            results = analyzer.analyze_directory("src")

            # Should only find the component, not node_modules
            assert all("node_modules" not in path for path in results.keys())

    def test_analyze_directory_skips_test_files(self):
        """Test that test files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()

            (src_dir / "Component.tsx").write_text('<div />')
            (src_dir / "Component.test.tsx").write_text('test()')
            (src_dir / "Component.spec.tsx").write_text('spec()')

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            results = analyzer.analyze_directory("src")

            assert all(".test." not in path and ".spec." not in path for path in results.keys())

    def test_analyze_directory_not_found(self):
        """Test analyzing non-existent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = SourceAnalyzer(repo_path=tmpdir)
            results = analyzer.analyze_directory("nonexistent")

            assert results == {}


class TestSourceAnalyzerHelperMethods:
    """Tests for helper methods."""

    def test_get_all_selectors(self):
        """Test getting all selectors from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.tsx").write_text('<div data-testid="a1" />')
            (Path(tmpdir) / "b.tsx").write_text('<div data-testid="b1" />')

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("a.tsx")
            analyzer.analyze_file("b.tsx")

            all_selectors = analyzer.get_all_selectors()

            assert len(all_selectors) >= 2

    def test_find_selectors_by_type(self):
        """Test finding selectors by type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
<div data-testid="test-id" id="html-id" className="css-class" />
'''
            (Path(tmpdir) / "test.tsx").write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            testids = analyzer.find_selectors_by_type("testid")
            ids = analyzer.find_selectors_by_type("id")

            assert len(testids) >= 1
            assert len(ids) >= 1

    def test_find_selectors_by_element(self):
        """Test finding selectors by element type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
<button data-testid="btn">Click</button>
<input data-testid="input" />
'''
            (Path(tmpdir) / "test.tsx").write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            buttons = analyzer.find_selectors_by_element("button")
            inputs = analyzer.find_selectors_by_element("input")

            assert len(buttons) >= 1
            assert len(inputs) >= 1

    def test_get_component_selector_map(self):
        """Test getting component to selector mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_content = '''
export default function LoginForm() {
    return (
        <form data-testid="login-form">
            <input data-testid="email" />
        </form>
    );
}
'''
            (Path(tmpdir) / "LoginForm.tsx").write_text(file_content)

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("LoginForm.tsx")

            selector_map = analyzer.get_component_selector_map()

            assert "LoginForm" in selector_map
            assert "login-form" in selector_map["LoginForm"]
            assert "email" in selector_map["LoginForm"]

    def test_clear_cache(self):
        """Test clearing the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.tsx").write_text('<div />')

            analyzer = SourceAnalyzer(repo_path=tmpdir)
            analyzer.analyze_file("test.tsx")

            assert len(analyzer._component_cache) > 0

            analyzer.clear_cache()

            assert len(analyzer._component_cache) == 0
            assert len(analyzer._selector_cache) == 0


class TestGetSourceAnalyzer:
    """Tests for get_source_analyzer factory function."""

    def test_get_source_analyzer_creates_instance(self):
        """Test that get_source_analyzer creates an instance."""
        import src.services.source_analyzer as module

        # Reset the global
        module._source_analyzer = None

        analyzer = get_source_analyzer()

        assert analyzer is not None
        assert isinstance(analyzer, SourceAnalyzer)

    def test_get_source_analyzer_returns_same_instance(self):
        """Test that get_source_analyzer returns the same instance."""
        import src.services.source_analyzer as module

        module._source_analyzer = None

        analyzer1 = get_source_analyzer()
        analyzer2 = get_source_analyzer()

        assert analyzer1 is analyzer2
