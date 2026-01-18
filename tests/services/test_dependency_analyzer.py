"""Tests for the dependency analyzer service."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock


class TestCodeModule:
    """Tests for CodeModule dataclass."""

    def test_code_module_creation(self):
        """Test creating a CodeModule instance."""
        from src.services.dependency_analyzer import CodeModule

        module = CodeModule(
            path="src/components/Button.tsx",
            name="Button",
            module_type="component",
            exports=["Button", "ButtonProps"],
            imports=["react", "./styles"],
            dependencies=["src/styles/button.css"],
            dependents=["src/pages/Home.tsx"],
        )

        assert module.path == "src/components/Button.tsx"
        assert module.name == "Button"
        assert module.module_type == "component"
        assert "Button" in module.exports
        assert "react" in module.imports
        assert len(module.dependencies) == 1
        assert len(module.dependents) == 1

    def test_code_module_default_lists(self):
        """Test CodeModule default empty lists."""
        from src.services.dependency_analyzer import CodeModule

        module = CodeModule(
            path="src/utils/helper.ts",
            name="helper",
            module_type="util",
        )

        assert module.exports == []
        assert module.imports == []
        assert module.dependencies == []
        assert module.dependents == []


class TestComponentNode:
    """Tests for ComponentNode dataclass."""

    def test_component_node_creation(self):
        """Test creating a ComponentNode instance."""
        from src.services.dependency_analyzer import ComponentNode

        node = ComponentNode(
            name="Dashboard",
            file_path="src/components/Dashboard.tsx",
            children=["Header", "Sidebar"],
            routes=["/dashboard", "/dashboard/analytics"],
            tests=["tests/Dashboard.spec.ts"],
        )

        assert node.name == "Dashboard"
        assert node.file_path == "src/components/Dashboard.tsx"
        assert "Header" in node.children
        assert "/dashboard" in node.routes
        assert len(node.tests) == 1

    def test_component_node_default_lists(self):
        """Test ComponentNode default empty lists."""
        from src.services.dependency_analyzer import ComponentNode

        node = ComponentNode(
            name="Button",
            file_path="src/components/Button.tsx",
        )

        assert node.children == []
        assert node.routes == []
        assert node.tests == []


class TestRouteNode:
    """Tests for RouteNode dataclass."""

    def test_route_node_creation(self):
        """Test creating a RouteNode instance."""
        from src.services.dependency_analyzer import RouteNode

        node = RouteNode(
            path="/users/[id]",
            file_path="app/users/[id]/page.tsx",
            component="UserProfile",
            nested_routes=["/users/[id]/settings"],
        )

        assert node.path == "/users/[id]"
        assert node.file_path == "app/users/[id]/page.tsx"
        assert node.component == "UserProfile"
        assert len(node.nested_routes) == 1


class TestImpactResult:
    """Tests for ImpactResult dataclass."""

    def test_impact_result_creation(self):
        """Test creating an ImpactResult instance."""
        from src.services.dependency_analyzer import ImpactResult

        result = ImpactResult(
            changed_files=["src/utils/format.ts"],
            affected_modules=["src/components/Price.tsx"],
            affected_components=["Price", "Cart"],
            affected_routes=["/cart", "/checkout"],
            affected_tests=["tests/cart.spec.ts"],
            skipped_tests=["tests/auth.spec.ts"],
            confidence=1.0,
            explanation="Price component imports format.ts",
        )

        assert len(result.changed_files) == 1
        assert len(result.affected_modules) == 1
        assert len(result.affected_components) == 2
        assert result.confidence == 1.0
        assert "Price" in result.explanation


class TestDependencyAnalyzerInit:
    """Tests for DependencyAnalyzer initialization."""

    def test_init_with_default_path(self):
        """Test initialization with default path."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer.repo_path == Path(".").resolve()
        assert analyzer.modules == {}
        assert analyzer.components == {}
        assert analyzer.routes == {}
        assert analyzer._built is False

    def test_init_with_custom_path(self):
        """Test initialization with custom path."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = DependencyAnalyzer(tmpdir)
            assert analyzer.repo_path == Path(tmpdir).resolve()


class TestDependencyAnalyzerImportPatterns:
    """Tests for import pattern extraction."""

    def test_extract_es6_imports(self):
        """Test extracting ES6 imports."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        content = '''
import React from 'react';
import { useState, useEffect } from 'react';
import * as Utils from './utils';
import type { Props } from './types';
'''

        imports = analyzer._extract_imports(content)

        assert "react" in imports
        assert "./utils" in imports
        assert "./types" in imports

    def test_extract_require_imports(self):
        """Test extracting require() imports."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        content = '''
const path = require('path');
const { readFile } = require("fs");
'''

        imports = analyzer._extract_imports(content)

        assert "path" in imports
        assert "fs" in imports

    def test_extract_dynamic_imports(self):
        """Test extracting dynamic imports."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        content = '''
const Component = lazy(() => import('./Component'));
const module = await import("./module");
'''

        imports = analyzer._extract_imports(content)

        assert "./Component" in imports
        assert "./module" in imports


class TestDependencyAnalyzerExportPatterns:
    """Tests for export pattern extraction."""

    def test_extract_named_exports(self):
        """Test extracting named exports."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        content = '''
export const Button = () => {};
export function helper() {}
export class Service {}
export let variable = 1;
'''

        exports = analyzer._extract_exports(content)

        assert "Button" in exports
        assert "helper" in exports
        assert "Service" in exports
        assert "variable" in exports

    def test_extract_export_list(self):
        """Test extracting export lists."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        content = '''
export { Button, helper, Service };
export { value as default };
'''

        exports = analyzer._extract_exports(content)

        assert "Button" in exports
        assert "helper" in exports
        assert "Service" in exports

    def test_extract_default_export(self):
        """Test extracting default exports."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        content = '''
export default function Component() {}
export default Button;
'''

        exports = analyzer._extract_exports(content)

        assert "Component" in exports
        assert "Button" in exports


class TestDependencyAnalyzerModuleTypeDetection:
    """Tests for module type detection."""

    def test_detect_test_module(self):
        """Test detecting test modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._detect_module_type("src/Button.test.tsx", "") == "test"
        assert analyzer._detect_module_type("src/Button.spec.ts", "") == "test"
        assert analyzer._detect_module_type("tests/utils.ts", "") == "test"
        assert analyzer._detect_module_type("src/__tests__/Button.tsx", "") == "test"

    def test_detect_component_module(self):
        """Test detecting component modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._detect_module_type("src/components/Button.tsx", "") == "component"
        assert analyzer._detect_module_type("components/Card.tsx", "") == "component"

    def test_detect_hook_module(self):
        """Test detecting hook modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._detect_module_type("src/hooks/useAuth.ts", "") == "hook"
        assert analyzer._detect_module_type("hooks/useData.ts", "") == "hook"
        # Note: files with 'use' in the stem are detected as hooks
        assert analyzer._detect_module_type("src/useStore.ts", "") == "hook"

    def test_detect_api_module(self):
        """Test detecting API modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._detect_module_type("src/api/users.ts", "") == "api"
        assert analyzer._detect_module_type("src/routes/auth.ts", "") == "api"

    def test_detect_util_module(self):
        """Test detecting utility modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._detect_module_type("src/utils/format.ts", "") == "util"
        assert analyzer._detect_module_type("src/lib/api.ts", "") == "util"
        assert analyzer._detect_module_type("src/helpers/parse.ts", "") == "util"

    def test_detect_route_module(self):
        """Test detecting route modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._detect_module_type("app/dashboard/page.tsx", "") == "route"
        assert analyzer._detect_module_type("pages/about.tsx", "") == "route"

    def test_detect_layout_module(self):
        """Test detecting layout modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._detect_module_type("app/layout.tsx", "") == "layout"
        assert analyzer._detect_module_type("app/dashboard/template.tsx", "") == "layout"

    def test_detect_component_from_content(self):
        """Test detecting component from JSX content."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        content = '''
export function MyComponent() {
    return (
        <div>Hello</div>
    );
}
'''

        # A file in utils folder but with JSX content
        module_type = analyzer._detect_module_type("src/misc/MyComponent.tsx", content)
        assert module_type == "component"


class TestDependencyAnalyzerPathToRoute:
    """Tests for path to route conversion."""

    def test_nextjs_app_router_page(self):
        """Test converting Next.js app router page paths."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._path_to_route("app/dashboard/page.tsx") == "/dashboard"
        assert analyzer._path_to_route("app/users/[id]/page.tsx") == "/users/[id]"

    def test_nextjs_app_router_root(self):
        """Test converting Next.js app router root page."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._path_to_route("app/page.tsx") == "/"

    def test_nextjs_pages_router(self):
        """Test converting Next.js pages router paths."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._path_to_route("pages/about.tsx") == "/about"
        assert analyzer._path_to_route("pages/users/profile.tsx") == "/users/profile"
        assert analyzer._path_to_route("pages/index.tsx") == "/"


class TestDependencyAnalyzerResolveImport:
    """Tests for import path resolution."""

    def test_skip_external_packages(self):
        """Test that external packages are skipped."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        assert analyzer._resolve_import("react", "src") is None
        assert analyzer._resolve_import("lodash", "src") is None
        assert analyzer._resolve_import("next/router", "src") is None

    def test_resolve_alias_import(self):
        """Test resolving alias imports (@/)."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = DependencyAnalyzer(tmpdir)

            # Create a module file
            src_dir = Path(tmpdir) / "src" / "components"
            src_dir.mkdir(parents=True)
            (src_dir / "Button.tsx").write_text("export const Button = () => {}")

            # Add the module to analyzer
            analyzer.modules["src/components/Button.tsx"] = MagicMock()

            result = analyzer._resolve_import("@/components/Button", "src/pages")
            assert result == "src/components/Button.tsx"


class TestDependencyAnalyzerGetDependents:
    """Tests for getting module dependents."""

    def test_get_direct_dependents(self):
        """Test getting direct dependents of a module."""
        from src.services.dependency_analyzer import CodeModule, DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        # Set up modules
        analyzer.modules["src/utils/format.ts"] = CodeModule(
            path="src/utils/format.ts",
            name="format",
            module_type="util",
            dependents=["src/components/Price.tsx"],
        )
        analyzer.modules["src/components/Price.tsx"] = CodeModule(
            path="src/components/Price.tsx",
            name="Price",
            module_type="component",
            dependencies=["src/utils/format.ts"],
            dependents=[],
        )

        dependents = analyzer.get_dependents("src/utils/format.ts", recursive=False)

        assert "src/components/Price.tsx" in dependents

    def test_get_recursive_dependents(self):
        """Test getting recursive dependents of a module."""
        from src.services.dependency_analyzer import CodeModule, DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        # Set up a chain: format -> Price -> Cart
        analyzer.modules["src/utils/format.ts"] = CodeModule(
            path="src/utils/format.ts",
            name="format",
            module_type="util",
            dependents=["src/components/Price.tsx"],
        )
        analyzer.modules["src/components/Price.tsx"] = CodeModule(
            path="src/components/Price.tsx",
            name="Price",
            module_type="component",
            dependents=["src/components/Cart.tsx"],
        )
        analyzer.modules["src/components/Cart.tsx"] = CodeModule(
            path="src/components/Cart.tsx",
            name="Cart",
            module_type="component",
            dependents=[],
        )

        dependents = analyzer.get_dependents("src/utils/format.ts", recursive=True)

        assert "src/components/Price.tsx" in dependents
        assert "src/components/Cart.tsx" in dependents

    def test_get_dependents_nonexistent_module(self):
        """Test getting dependents of nonexistent module."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        dependents = analyzer.get_dependents("nonexistent.ts")
        assert dependents == set()


class TestDependencyAnalyzerAnalyzeImpact:
    """Tests for impact analysis."""

    def test_analyze_impact_basic(self):
        """Test basic impact analysis."""
        from src.services.dependency_analyzer import CodeModule, DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        analyzer._built = True

        # Set up modules
        analyzer.modules["src/utils/format.ts"] = CodeModule(
            path="src/utils/format.ts",
            name="format",
            module_type="util",
            dependents=["src/components/Price.tsx"],
        )
        analyzer.modules["src/components/Price.tsx"] = CodeModule(
            path="src/components/Price.tsx",
            name="Price",
            module_type="component",
            dependencies=["src/utils/format.ts"],
            dependents=["tests/Price.spec.ts"],
        )
        analyzer.modules["tests/Price.spec.ts"] = CodeModule(
            path="tests/Price.spec.ts",
            name="Price.spec",
            module_type="test",
            dependencies=["src/components/Price.tsx"],
        )
        analyzer.modules["tests/auth.spec.ts"] = CodeModule(
            path="tests/auth.spec.ts",
            name="auth.spec",
            module_type="test",
        )

        result = analyzer.analyze_impact(["src/utils/format.ts"])

        assert "src/utils/format.ts" in result.affected_modules
        assert "src/components/Price.tsx" in result.affected_modules
        assert "tests/Price.spec.ts" in result.affected_tests
        assert "tests/auth.spec.ts" in result.skipped_tests
        assert result.confidence == 1.0

    def test_analyze_impact_with_components(self):
        """Test impact analysis populating affected components."""
        from src.services.dependency_analyzer import CodeModule, ComponentNode, DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        analyzer._built = True

        # Set up modules
        analyzer.modules["src/components/Button.tsx"] = CodeModule(
            path="src/components/Button.tsx",
            name="Button",
            module_type="component",
            dependents=[],
        )

        # Set up components
        analyzer.components["Button"] = ComponentNode(
            name="Button",
            file_path="src/components/Button.tsx",
            routes=["/form"],
            tests=["tests/Button.spec.ts"],
        )

        # Add the test module
        analyzer.modules["tests/Button.spec.ts"] = CodeModule(
            path="tests/Button.spec.ts",
            name="Button.spec",
            module_type="test",
        )

        result = analyzer.analyze_impact(["src/components/Button.tsx"])

        assert "Button" in result.affected_components
        assert "/form" in result.affected_routes
        assert "tests/Button.spec.ts" in result.affected_tests

    def test_analyze_impact_builds_graph_if_needed(self):
        """Test that analyze_impact builds graph if not built."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = DependencyAnalyzer(tmpdir)

            # Should build graph automatically
            analyzer.analyze_impact(["some/file.ts"])

            assert analyzer._built is True


class TestDependencyAnalyzerBuildImpactExplanation:
    """Tests for building impact explanations."""

    def test_build_explanation_with_changes(self):
        """Test building explanation with changes."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        explanation = analyzer._build_impact_explanation(
            changed_files=["src/utils/format.ts"],
            affected_components=["Price", "Cart"],
            affected_tests=["test1.spec.ts", "test2.spec.ts"],
        )

        assert "src/utils/format.ts" in explanation
        assert "Price" in explanation
        assert "Tests to run: 2" in explanation

    def test_build_explanation_no_tests(self):
        """Test building explanation with no affected tests."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        explanation = analyzer._build_impact_explanation(
            changed_files=["src/docs/readme.md"],
            affected_components=[],
            affected_tests=[],
        )

        assert "No tests affected" in explanation

    def test_build_explanation_truncates_long_lists(self):
        """Test that explanation truncates long lists."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        explanation = analyzer._build_impact_explanation(
            changed_files=[f"file{i}.ts" for i in range(10)],
            affected_components=[f"Component{i}" for i in range(10)],
            affected_tests=["test.spec.ts"],
        )

        assert "... and 5 more" in explanation


class TestDependencyAnalyzerComponentCoverage:
    """Tests for component coverage methods."""

    def test_get_component_coverage_existing(self):
        """Test getting coverage for existing component."""
        from src.services.dependency_analyzer import ComponentNode, DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        analyzer.components["Button"] = ComponentNode(
            name="Button",
            file_path="src/components/Button.tsx",
            children=["Icon"],
            routes=["/form"],
            tests=["tests/Button.spec.ts"],
        )

        coverage = analyzer.get_component_coverage("Button")

        assert coverage["covered"] is True
        assert "tests/Button.spec.ts" in coverage["tests"]
        assert "/form" in coverage["routes"]
        assert "Icon" in coverage["children"]

    def test_get_component_coverage_nonexistent(self):
        """Test getting coverage for nonexistent component."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        coverage = analyzer.get_component_coverage("NonexistentComponent")

        assert coverage["covered"] is False
        assert coverage["tests"] == []
        assert coverage["routes"] == []

    def test_get_untested_components(self):
        """Test getting list of untested components."""
        from src.services.dependency_analyzer import ComponentNode, DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        analyzer.components["Button"] = ComponentNode(
            name="Button",
            file_path="src/components/Button.tsx",
            tests=["tests/Button.spec.ts"],
        )
        analyzer.components["Card"] = ComponentNode(
            name="Card",
            file_path="src/components/Card.tsx",
            tests=[],  # No tests
        )
        analyzer.components["Icon"] = ComponentNode(
            name="Icon",
            file_path="src/components/Icon.tsx",
            tests=[],  # No tests
        )

        untested = analyzer.get_untested_components()

        assert "Card" in untested
        assert "Icon" in untested
        assert "Button" not in untested


class TestDependencyAnalyzerToDict:
    """Tests for serialization to dict."""

    def test_to_dict_empty(self):
        """Test serializing empty analyzer."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        analyzer = DependencyAnalyzer()

        result = analyzer.to_dict()

        assert "modules" in result
        assert "components" in result
        assert "routes" in result
        assert "stats" in result
        assert result["stats"]["total_modules"] == 0

    def test_to_dict_with_data(self):
        """Test serializing analyzer with data."""
        from src.services.dependency_analyzer import (
            CodeModule,
            ComponentNode,
            DependencyAnalyzer,
            RouteNode,
        )

        analyzer = DependencyAnalyzer()

        analyzer.modules["src/Button.tsx"] = CodeModule(
            path="src/Button.tsx",
            name="Button",
            module_type="component",
            exports=["Button"],
            dependencies=[],
            dependents=["src/pages/Home.tsx"],
        )

        analyzer.components["Button"] = ComponentNode(
            name="Button",
            file_path="src/Button.tsx",
            children=[],
            routes=["/home"],
            tests=[],
        )

        analyzer.routes["/home"] = RouteNode(
            path="/home",
            file_path="src/pages/Home.tsx",
            component="Button",
        )

        result = analyzer.to_dict()

        assert "src/Button.tsx" in result["modules"]
        assert "Button" in result["components"]
        assert "/home" in result["routes"]
        assert result["stats"]["total_modules"] == 1
        assert result["stats"]["total_components"] == 1
        assert result["stats"]["total_routes"] == 1
        assert result["stats"]["untested_components"] == 1


class TestDependencyAnalyzerBuildGraph:
    """Tests for building the dependency graph."""

    def test_build_graph_with_temp_dir(self):
        """Test building graph with temporary directory."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source structure
            src_dir = Path(tmpdir) / "src" / "components"
            src_dir.mkdir(parents=True)

            # Create a component file
            (src_dir / "Button.tsx").write_text('''
import React from 'react';
import './Button.css';

export const Button = () => {
    return <button>Click me</button>;
};
''')

            analyzer = DependencyAnalyzer(tmpdir)
            analyzer.build_graph(source_dirs=["src"], extensions=[".tsx"])

            assert analyzer._built is True
            assert len(analyzer.modules) >= 1

    def test_build_graph_skips_node_modules(self):
        """Test that build_graph skips node_modules."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source structure
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir(parents=True)
            (src_dir / "index.ts").write_text("export const x = 1;")

            # Create node_modules
            nm_dir = Path(tmpdir) / "src" / "node_modules" / "package"
            nm_dir.mkdir(parents=True)
            (nm_dir / "index.ts").write_text("export const y = 2;")

            analyzer = DependencyAnalyzer(tmpdir)
            analyzer.build_graph(source_dirs=["src"], extensions=[".ts"])

            # Should not include node_modules files
            assert all("node_modules" not in path for path in analyzer.modules.keys())


class TestGetDependencyAnalyzer:
    """Tests for get_dependency_analyzer factory function."""

    def test_get_dependency_analyzer_creates_singleton(self):
        """Test that get_dependency_analyzer creates singleton."""
        import src.services.dependency_analyzer as module
        from src.services.dependency_analyzer import get_dependency_analyzer

        # Reset singleton
        module._dependency_analyzer = None

        analyzer1 = get_dependency_analyzer("/tmp")
        analyzer2 = get_dependency_analyzer("/tmp")

        # Should return same instance
        assert analyzer1 is analyzer2

        # Cleanup
        module._dependency_analyzer = None

    def test_get_dependency_analyzer_default_path(self):
        """Test get_dependency_analyzer with default path."""
        import src.services.dependency_analyzer as module
        from src.services.dependency_analyzer import get_dependency_analyzer

        # Reset singleton
        module._dependency_analyzer = None

        analyzer = get_dependency_analyzer()

        assert analyzer is not None
        assert analyzer.repo_path is not None

        # Cleanup
        module._dependency_analyzer = None


class TestDependencyAnalyzerDiscoverModules:
    """Tests for module discovery."""

    def test_discover_modules_filters_extensions(self):
        """Test that module discovery filters by extension."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir(parents=True)

            # Create various files
            (src_dir / "index.ts").write_text("export const x = 1;")
            (src_dir / "style.css").write_text(".button { color: red; }")
            (src_dir / "config.json").write_text('{"key": "value"}')
            (src_dir / "script.js").write_text("const y = 2;")

            analyzer = DependencyAnalyzer(tmpdir)
            analyzer._discover_modules(src_dir, extensions=[".ts", ".js"])

            # Should only include .ts and .js files
            assert "src/index.ts" in analyzer.modules
            assert "src/script.js" in analyzer.modules
            assert "src/style.css" not in analyzer.modules
            assert "src/config.json" not in analyzer.modules


class TestDependencyAnalyzerParseModule:
    """Tests for module parsing."""

    def test_parse_module_handles_read_error(self):
        """Test that parse_module handles file read errors gracefully."""
        from src.services.dependency_analyzer import DependencyAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = DependencyAnalyzer(tmpdir)

            # Create a file path that doesn't exist
            file_path = Path(tmpdir) / "nonexistent.ts"

            # Should not raise, should return module with empty content
            module = analyzer._parse_module(file_path, "nonexistent.ts")

            assert module.path == "nonexistent.ts"
            assert module.exports == []
            assert module.imports == []
