"""Frontend Analyzer - React, Vue, Angular, Svelte component analysis.

Understands:
- Component structure and props
- Hooks and state management
- Test selectors (data-testid)
- Route definitions
- Import/export relationships
"""

import logging
import re
from pathlib import Path

from src.indexer import ASTNode, ParsedFile

from .base import (
    BaseAnalyzer,
    ComponentInfo,
    ComponentType,
    Issue,
    Severity,
)

logger = logging.getLogger(__name__)


class FrontendAnalyzer(BaseAnalyzer):
    """Analyzer for React, Vue, Angular, and Svelte frontends."""

    @property
    def analyzer_type(self) -> str:
        return "frontend"

    def get_file_patterns(self) -> list[str]:
        return [
            # React/Next.js
            "**/*.tsx",
            "**/*.jsx",
            # Vue
            "**/*.vue",
            # Angular
            "**/*.component.ts",
            # Svelte
            "**/*.svelte",
            # General JS/TS
            "**/components/**/*.ts",
            "**/hooks/**/*.ts",
            "**/pages/**/*.ts",
        ]

    def analyze_file(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze a frontend file for components, hooks, and patterns."""
        components: list[ComponentInfo] = []

        # Detect framework
        framework = self._detect_framework(parsed)

        if framework == "react":
            components.extend(self._analyze_react(parsed))
        elif framework == "vue":
            components.extend(self._analyze_vue(parsed))
        elif framework == "angular":
            components.extend(self._analyze_angular(parsed))
        elif framework == "svelte":
            components.extend(self._analyze_svelte(parsed))

        return components

    def _detect_framework(self, parsed: ParsedFile) -> str:
        """Detect which frontend framework is being used."""
        content = parsed.content

        # React indicators
        if "from 'react'" in content or 'from "react"' in content:
            return "react"
        if "React.Component" in content or "useState" in content:
            return "react"

        # Vue indicators
        if parsed.file_path.endswith(".vue"):
            return "vue"
        if "from 'vue'" in content or "defineComponent" in content:
            return "vue"

        # Angular indicators
        if "@Component" in content and "@angular" in content:
            return "angular"

        # Svelte indicators
        if parsed.file_path.endswith(".svelte"):
            return "svelte"

        # Default to React for .tsx/.jsx
        if parsed.file_path.endswith((".tsx", ".jsx")):
            return "react"

        return "unknown"

    def _analyze_react(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze React components and hooks."""
        components: list[ComponentInfo] = []
        content = parsed.content
        content.split("\n")

        # Extract imports
        imports = self.extract_imports(parsed)

        # Find function components
        for func in parsed.get_functions():
            name = func.name
            if not name:
                continue

            # Check if it's a component (PascalCase, returns JSX)
            if self._is_react_component(name, func, content):
                comp_type = self._get_component_type(name, parsed.file_path)

                component = ComponentInfo(
                    name=name,
                    component_type=comp_type,
                    file_path=parsed.file_path,
                    start_line=func.start_line,
                    end_line=func.end_line,
                    imports=imports,
                    exports=self.extract_exports(content),
                    selectors=self._extract_react_selectors(func.text),
                    props=self._extract_props(func.text),
                    methods=[],
                )

                # Add issues for missing best practices
                component.issues.extend(self._check_react_issues(func, name))
                components.append(component)

            # Check if it's a custom hook
            elif name.startswith("use") and name[3:4].isupper():
                hook = ComponentInfo(
                    name=name,
                    component_type=ComponentType.HOOK,
                    file_path=parsed.file_path,
                    start_line=func.start_line,
                    end_line=func.end_line,
                    imports=imports,
                    exports=self.extract_exports(content),
                )
                components.append(hook)

        # Find class components
        for cls in parsed.get_classes():
            name = cls.name
            if not name:
                continue

            # Check if it extends React.Component
            if "Component" in cls.text and "render" in cls.text:
                component = ComponentInfo(
                    name=name,
                    component_type=ComponentType.COMPONENT,
                    file_path=parsed.file_path,
                    start_line=cls.start_line,
                    end_line=cls.end_line,
                    imports=imports,
                    exports=self.extract_exports(content),
                    selectors=self._extract_react_selectors(cls.text),
                    methods=self._extract_class_methods(cls),
                )
                components.append(component)

        return components

    def _is_react_component(self, name: str, func: ASTNode, content: str) -> bool:
        """Check if a function is a React component."""
        # Must start with uppercase (PascalCase)
        if not name[0].isupper():
            return False

        # Check function body for JSX
        func_text = func.text
        jsx_indicators = ["<", "/>", "return (", "return <", "createElement"]
        return any(indicator in func_text for indicator in jsx_indicators)

    def _get_component_type(self, name: str, file_path: str) -> ComponentType:
        """Determine the type of React component."""
        path_lower = file_path.lower()

        if "/pages/" in path_lower or "page.tsx" in path_lower:
            return ComponentType.PAGE
        if "/layouts/" in path_lower or "layout.tsx" in path_lower:
            return ComponentType.LAYOUT
        if "/context/" in path_lower or name.endswith("Provider"):
            return ComponentType.CONTEXT
        if "/store/" in path_lower or "Store" in name:
            return ComponentType.STORE

        return ComponentType.COMPONENT

    def _extract_react_selectors(self, content: str) -> list[str]:
        """Extract test selectors from React JSX."""
        selectors = []

        # data-testid
        for match in re.finditer(r'data-testid=["\']{([^}]+)}["\']', content):
            selectors.append(f"[data-testid='${{{match.group(1)}}}']")
        for match in re.finditer(r'data-testid=["\']([^"\']+)["\']', content):
            selectors.append(f"[data-testid='{match.group(1)}']")

        # data-cy (Cypress)
        for match in re.finditer(r'data-cy=["\']([^"\']+)["\']', content):
            selectors.append(f"[data-cy='{match.group(1)}']")

        # role
        for match in re.finditer(r'role=["\']([^"\']+)["\']', content):
            selectors.append(f"[role='{match.group(1)}']")

        return selectors

    def _extract_props(self, func_text: str) -> list[str]:
        """Extract prop names from React component."""
        props = []

        # Destructured props: ({ name, age, onClick })
        match = re.search(r'\(\s*\{\s*([^}]+)\s*\}', func_text)
        if match:
            prop_str = match.group(1)
            # Parse prop names, handle defaults and types
            for prop in prop_str.split(","):
                prop = prop.strip()
                if not prop:
                    continue
                # Handle prop = default
                prop_name = prop.split("=")[0].strip()
                # Handle prop: Type
                prop_name = prop_name.split(":")[0].strip()
                if prop_name and not prop_name.startswith("..."):
                    props.append(prop_name)

        # Props interface/type
        for match in re.finditer(r'interface\s+\w*Props\s*\{([^}]+)\}', func_text):
            for line in match.group(1).split("\n"):
                line = line.strip()
                if line and not line.startswith("//"):
                    prop_name = line.split(":")[0].strip().rstrip("?")
                    if prop_name:
                        props.append(prop_name)

        return list(set(props))

    def _extract_class_methods(self, cls: ASTNode) -> list[str]:
        """Extract method names from a class component."""
        methods = []
        for method in cls.find_descendants("method_definition"):
            name = method.name
            if name and not name.startswith("_"):
                methods.append(name)
        return methods

    def _check_react_issues(self, func: ASTNode, name: str) -> list[Issue]:
        """Check for React best practice issues."""
        issues = []
        func_text = func.text

        # Missing key in map
        if ".map(" in func_text and "key=" not in func_text:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Component {name} may be missing 'key' prop in list rendering",
                file_path="",
                line_number=func.start_line,
                suggestion="Add unique 'key' prop to elements rendered in a loop",
                code="REACT_MISSING_KEY",
            ))

        # No error boundary wrapper for complex components
        if func_text.count("<") > 10 and "ErrorBoundary" not in func_text:
            issues.append(Issue(
                severity=Severity.INFO,
                message=f"Complex component {name} might benefit from an error boundary",
                file_path="",
                line_number=func.start_line,
                suggestion="Consider wrapping with an ErrorBoundary for better error handling",
                code="REACT_NO_ERROR_BOUNDARY",
            ))

        # Missing data-testid for interactive elements
        interactive = ["<button", "<input", "<select", "<a ", "onClick"]
        has_interactive = any(elem in func_text for elem in interactive)
        has_testid = "data-testid" in func_text or "data-cy" in func_text

        if has_interactive and not has_testid:
            issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Component {name} has interactive elements but no test selectors",
                file_path="",
                line_number=func.start_line,
                suggestion="Add data-testid attributes for reliable E2E testing",
                code="REACT_NO_TEST_SELECTOR",
            ))

        return issues

    def _analyze_vue(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Vue single-file components."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Extract component name from file
        name = Path(parsed.file_path).stem

        # Find script section
        script_match = re.search(
            r'<script[^>]*>(.*?)</script>',
            content,
            re.DOTALL
        )

        # Find template section for selectors
        template_match = re.search(
            r'<template[^>]*>(.*?)</template>',
            content,
            re.DOTALL
        )

        imports = []
        props = []
        methods = []
        selectors = []

        if script_match:
            script = script_match.group(1)

            # Extract imports
            for match in re.finditer(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", script):
                imports.append(match.group(1))

            # Extract props (Options API)
            props_match = re.search(r'props:\s*\{([^}]+)\}', script)
            if props_match:
                for line in props_match.group(1).split(","):
                    prop = line.strip().split(":")[0].strip()
                    if prop:
                        props.append(prop)

            # Extract props (Composition API)
            define_props = re.search(r'defineProps[<\(]([^>)\]]+)[>\)]', script)
            if define_props:
                for match in re.finditer(r'(\w+)\s*[?:]', define_props.group(1)):
                    props.append(match.group(1))

            # Extract methods
            methods_match = re.search(r'methods:\s*\{([^}]+)\}', script, re.DOTALL)
            if methods_match:
                for match in re.finditer(r'(\w+)\s*\(', methods_match.group(1)):
                    methods.append(match.group(1))

        if template_match:
            template = template_match.group(1)
            selectors = self.extract_selectors(template)

        component = ComponentInfo(
            name=name,
            component_type=ComponentType.COMPONENT,
            file_path=parsed.file_path,
            start_line=1,
            end_line=content.count("\n") + 1,
            imports=imports,
            props=props,
            methods=methods,
            selectors=selectors,
        )

        # Check for issues
        if template_match and not selectors:
            component.issues.append(Issue(
                severity=Severity.WARNING,
                message=f"Vue component {name} lacks test selectors",
                file_path=parsed.file_path,
                suggestion="Add data-testid attributes for E2E testing",
                code="VUE_NO_TEST_SELECTOR",
            ))

        components.append(component)
        return components

    def _analyze_angular(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Angular components."""
        components: list[ComponentInfo] = []
        content = parsed.content

        # Find @Component decorator
        component_match = re.search(
            r'@Component\s*\(\s*\{([^}]+)\}',
            content,
            re.DOTALL
        )

        if not component_match:
            return components

        # Find class definition
        class_match = re.search(
            r'export\s+class\s+(\w+)',
            content
        )

        if not class_match:
            return components

        name = class_match.group(1)

        # Extract selector from decorator
        selector_match = re.search(
            r"selector:\s*['\"]([^'\"]+)['\"]",
            component_match.group(1)
        )

        selectors = []
        if selector_match:
            selectors.append(selector_match.group(1))

        # Extract template content for additional selectors
        template_match = re.search(
            r"template:\s*`([^`]+)`",
            component_match.group(1)
        )
        if template_match:
            selectors.extend(self.extract_selectors(template_match.group(1)))

        # Extract methods
        methods = []
        for match in re.finditer(r'(?:public\s+|private\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{', content):
            method = match.group(1)
            if method not in ["constructor", "ngOnInit", "ngOnDestroy"]:
                methods.append(method)

        # Extract @Input() properties
        props = []
        for match in re.finditer(r'@Input\(\)\s+(\w+)', content):
            props.append(match.group(1))

        component = ComponentInfo(
            name=name,
            component_type=ComponentType.COMPONENT,
            file_path=parsed.file_path,
            start_line=1,
            end_line=content.count("\n") + 1,
            imports=self.extract_imports(parsed),
            props=props,
            methods=methods,
            selectors=selectors,
        )

        components.append(component)
        return components

    def _analyze_svelte(self, parsed: ParsedFile) -> list[ComponentInfo]:
        """Analyze Svelte components."""
        components: list[ComponentInfo] = []
        content = parsed.content
        name = Path(parsed.file_path).stem

        # Extract script section
        script_match = re.search(
            r'<script[^>]*>(.*?)</script>',
            content,
            re.DOTALL
        )

        imports = []
        props = []

        if script_match:
            script = script_match.group(1)

            # Extract imports
            for match in re.finditer(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", script):
                imports.append(match.group(1))

            # Extract props (export let)
            for match in re.finditer(r'export\s+let\s+(\w+)', script):
                props.append(match.group(1))

        # Extract selectors from template
        selectors = self.extract_selectors(content)

        component = ComponentInfo(
            name=name,
            component_type=ComponentType.COMPONENT,
            file_path=parsed.file_path,
            start_line=1,
            end_line=content.count("\n") + 1,
            imports=imports,
            props=props,
            selectors=selectors,
        )

        components.append(component)
        return components
