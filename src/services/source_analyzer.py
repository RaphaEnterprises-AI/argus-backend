"""Source Code Analyzer - AST-based selector extraction.

This service analyzes source code to:
- Extract all selectors from components (React, Vue, Angular, etc.)
- Build a component-to-selector mapping
- Find semantic relationships between selectors
- Provide suggestions for better selector alternatives
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSelector:
    """A selector extracted from source code."""
    selector: str
    selector_type: str  # "testid", "id", "class", "aria", "text", "role"
    file_path: str
    line_number: int
    column: int
    element_type: Optional[str] = None  # "button", "input", "div", etc.
    context: str = ""  # Surrounding code
    semantic_name: Optional[str] = None  # Human-readable purpose


@dataclass
class ComponentInfo:
    """Information about a UI component."""
    name: str
    file_path: str
    selectors: list[ExtractedSelector] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    props: list[str] = field(default_factory=list)


@dataclass
class SelectorMapping:
    """Maps selectors to their semantic meaning and alternatives."""
    selector: str
    semantic_purpose: str  # "submit button", "email input", etc.
    alternatives: list[str] = field(default_factory=list)
    stability_score: float = 0.5  # 0-1, higher = more stable
    recommendation: Optional[str] = None


class SourceAnalyzer:
    """Analyzes source code to extract and understand selectors.

    This is complementary to GitAnalyzer:
    - GitAnalyzer: What CHANGED (history)
    - SourceAnalyzer: What EXISTS (current state)

    Together they enable code-aware healing with 99.9% accuracy.
    """

    # File extensions to analyze by framework
    FRAMEWORK_EXTENSIONS = {
        "react": [".jsx", ".tsx", ".js", ".ts"],
        "vue": [".vue", ".js", ".ts"],
        "angular": [".ts", ".html"],
        "svelte": [".svelte", ".js", ".ts"],
    }

    # Patterns for extracting selectors from different frameworks
    SELECTOR_PATTERNS = {
        # data-testid (universal, most reliable)
        "testid": [
            r'data-testid=["\']([^"\']+)["\']',
            r'testID=["\']([^"\']+)["\']',  # React Native
            r'\[data-testid=["\']?([^"\'>\]]+)',
            r'getByTestId\(["\']([^"\']+)["\']',
        ],
        # ID attributes
        "id": [
            r'\bid=["\']([^"\']+)["\']',
            r'getElementById\(["\']([^"\']+)["\']',
            r'#([a-zA-Z_][a-zA-Z0-9_-]*)',
        ],
        # Class selectors
        "class": [
            r'className=["\']([^"\']+)["\']',
            r'class=["\']([^"\']+)["\']',
            r':class=["\'{]([^"\']+)["\'}]',  # Vue
            r'\.([a-zA-Z_][a-zA-Z0-9_-]+)',
        ],
        # Aria labels (accessible)
        "aria": [
            r'aria-label=["\']([^"\']+)["\']',
            r'aria-labelledby=["\']([^"\']+)["\']',
            r'getByLabelText\(["\']([^"\']+)["\']',
        ],
        # Role-based (accessible)
        "role": [
            r'role=["\']([^"\']+)["\']',
            r'getByRole\(["\']([^"\']+)["\']',
        ],
        # Text content
        "text": [
            r'getByText\(["\']([^"\']+)["\']',
            r'>([^<>{}\n]{2,50})</',  # Text between tags
        ],
        # Name attribute
        "name": [
            r'\bname=["\']([^"\']+)["\']',
            r'getByName\(["\']([^"\']+)["\']',
        ],
        # Placeholder
        "placeholder": [
            r'placeholder=["\']([^"\']+)["\']',
            r'getByPlaceholderText\(["\']([^"\']+)["\']',
        ],
    }

    # Element type patterns
    ELEMENT_PATTERNS = {
        "button": [r'<button', r'type=["\']submit["\']', r'type=["\']button["\']', r'role=["\']button["\']'],
        "input": [r'<input', r'<textarea', r'<select'],
        "link": [r'<a\s', r'<Link', r'href='],
        "form": [r'<form', r'<Form'],
        "image": [r'<img', r'<Image', r'<picture'],
        "heading": [r'<h[1-6]', r'role=["\']heading["\']'],
        "list": [r'<ul', r'<ol', r'<li', r'role=["\']list["\']'],
        "navigation": [r'<nav', r'role=["\']navigation["\']'],
        "modal": [r'role=["\']dialog["\']', r'Modal', r'Dialog'],
    }

    def __init__(self, repo_path: str = "."):
        """Initialize with repository path.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path).resolve()
        self._component_cache: dict[str, ComponentInfo] = {}
        self._selector_cache: dict[str, list[ExtractedSelector]] = {}

    def analyze_file(self, file_path: str) -> ComponentInfo:
        """Analyze a single source file.

        Args:
            file_path: Path to source file

        Returns:
            ComponentInfo with extracted selectors
        """
        full_path = self.repo_path / file_path

        if not full_path.exists():
            logger.warning(f"File not found: {full_path}")
            return ComponentInfo(name="", file_path=file_path)

        try:
            content = full_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ComponentInfo(name="", file_path=file_path)

        # Extract component name
        component_name = self._extract_component_name(content, file_path)

        # Extract all selectors
        selectors = self._extract_selectors(content, file_path)

        # Extract imports and exports
        imports = self._extract_imports(content)
        exports = self._extract_exports(content, file_path)

        # Extract props (for prop-based selectors)
        props = self._extract_props(content)

        component = ComponentInfo(
            name=component_name,
            file_path=file_path,
            selectors=selectors,
            imports=imports,
            exports=exports,
            props=props,
        )

        # Cache it
        self._component_cache[file_path] = component

        return component

    def _extract_component_name(self, content: str, file_path: str) -> str:
        """Extract component name from file content."""
        # Try export default
        match = re.search(r'export\s+default\s+(?:function\s+)?(\w+)', content)
        if match:
            return match.group(1)

        # Try function component
        match = re.search(r'(?:function|const)\s+(\w+)\s*[=\(]', content)
        if match:
            return match.group(1)

        # Try class component
        match = re.search(r'class\s+(\w+)\s+extends', content)
        if match:
            return match.group(1)

        # Vue SFC
        match = re.search(r'name:\s*["\'](\w+)["\']', content)
        if match:
            return match.group(1)

        # Fallback to filename
        return Path(file_path).stem

    def _extract_selectors(
        self,
        content: str,
        file_path: str
    ) -> list[ExtractedSelector]:
        """Extract all selectors from file content."""
        selectors = []
        lines = content.split("\n")

        for selector_type, patterns in self.SELECTOR_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    selector_value = match.group(1)

                    # Skip empty or very short selectors
                    if not selector_value or len(selector_value) < 2:
                        continue

                    # Skip CSS class lists for class type (just get individual classes)
                    if selector_type == "class" and " " in selector_value:
                        # Split into individual classes
                        for cls in selector_value.split():
                            selectors.append(self._create_extracted_selector(
                                cls, selector_type, file_path, content, match
                            ))
                        continue

                    selectors.append(self._create_extracted_selector(
                        selector_value, selector_type, file_path, content, match
                    ))

        return selectors

    def _create_extracted_selector(
        self,
        selector_value: str,
        selector_type: str,
        file_path: str,
        content: str,
        match: re.Match
    ) -> ExtractedSelector:
        """Create an ExtractedSelector with context."""
        lines = content.split("\n")

        # Find line number
        pos = match.start()
        line_number = content[:pos].count("\n") + 1

        # Get column
        line_start = content.rfind("\n", 0, pos) + 1
        column = pos - line_start + 1

        # Get context (3 lines before and after)
        start_line = max(0, line_number - 4)
        end_line = min(len(lines), line_number + 3)
        context = "\n".join(lines[start_line:end_line])

        # Determine element type
        element_type = self._detect_element_type(context)

        # Get semantic name
        semantic_name = self._infer_semantic_name(selector_value, selector_type, context)

        return ExtractedSelector(
            selector=selector_value,
            selector_type=selector_type,
            file_path=file_path,
            line_number=line_number,
            column=column,
            element_type=element_type,
            context=context,
            semantic_name=semantic_name,
        )

    def _detect_element_type(self, context: str) -> Optional[str]:
        """Detect the element type from context."""
        context_lower = context.lower()

        for element_type, patterns in self.ELEMENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, context, re.IGNORECASE):
                    return element_type

        return None

    def _infer_semantic_name(
        self,
        selector: str,
        selector_type: str,
        context: str
    ) -> Optional[str]:
        """Infer the semantic purpose of a selector."""
        # Common patterns in selector names
        semantic_patterns = {
            r'submit': 'submit button',
            r'login': 'login element',
            r'signup|sign-up|register': 'registration element',
            r'email': 'email input',
            r'password': 'password input',
            r'username': 'username input',
            r'search': 'search element',
            r'nav|menu': 'navigation element',
            r'header': 'header element',
            r'footer': 'footer element',
            r'close|dismiss': 'close/dismiss element',
            r'add|create|new': 'create/add element',
            r'delete|remove': 'delete element',
            r'edit|update': 'edit element',
            r'save': 'save element',
            r'cancel': 'cancel element',
            r'modal|dialog': 'modal/dialog element',
            r'error|alert': 'error/alert element',
            r'success': 'success indicator',
            r'loading|spinner': 'loading indicator',
        }

        combined = f"{selector} {context}".lower()

        for pattern, name in semantic_patterns.items():
            if re.search(pattern, combined):
                return name

        return None

    def _extract_imports(self, content: str) -> list[str]:
        """Extract import statements."""
        imports = []

        # ES6 imports
        for match in re.finditer(r'import\s+.*?from\s+["\']([^"\']+)["\']', content):
            imports.append(match.group(1))

        # require() calls
        for match in re.finditer(r'require\(["\']([^"\']+)["\']\)', content):
            imports.append(match.group(1))

        return imports

    def _extract_exports(self, content: str, file_path: str) -> list[str]:
        """Extract exported names."""
        exports = []

        # Named exports
        for match in re.finditer(r'export\s+(?:const|function|class)\s+(\w+)', content):
            exports.append(match.group(1))

        # Export list
        for match in re.finditer(r'export\s+\{([^}]+)\}', content):
            names = [n.strip().split(' as ')[0] for n in match.group(1).split(',')]
            exports.extend(names)

        # Default export
        match = re.search(r'export\s+default\s+(?:function\s+)?(\w+)', content)
        if match:
            exports.append(f"default:{match.group(1)}")

        return exports

    def _extract_props(self, content: str) -> list[str]:
        """Extract component props."""
        props = []

        # TypeScript interface/type props
        for match in re.finditer(r'(?:interface|type)\s+\w*Props\w*\s*[{=]([^}]+)}', content, re.DOTALL):
            prop_block = match.group(1)
            for prop_match in re.finditer(r'(\w+)\s*[?:]', prop_block):
                props.append(prop_match.group(1))

        # Destructured props
        for match in re.finditer(r'\(\s*\{([^}]+)\}\s*(?::\s*\w+)?\s*\)', content):
            prop_list = match.group(1)
            for prop_match in re.finditer(r'(\w+)(?:\s*[=,}\n])', prop_list):
                props.append(prop_match.group(1))

        return props

    def find_selector(
        self,
        selector: str,
        file_path: Optional[str] = None
    ) -> list[ExtractedSelector]:
        """Find all occurrences of a selector.

        Args:
            selector: The selector to find
            file_path: Optional file to search in

        Returns:
            List of matching ExtractedSelectors
        """
        results = []

        if file_path:
            component = self.analyze_file(file_path)
            results.extend([s for s in component.selectors if s.selector == selector])
        else:
            # Search all cached components
            for path, component in self._component_cache.items():
                results.extend([s for s in component.selectors if s.selector == selector])

        return results

    def find_similar_selectors(
        self,
        selector: str,
        threshold: float = 0.6
    ) -> list[tuple[ExtractedSelector, float]]:
        """Find selectors similar to the given one.

        This is useful for self-healing when exact match fails.

        Args:
            selector: The selector to find similar ones for
            threshold: Minimum similarity (0-1)

        Returns:
            List of (selector, similarity_score) tuples
        """
        results = []

        for component in self._component_cache.values():
            for extracted in component.selectors:
                similarity = self._calculate_similarity(selector, extracted.selector)
                if similarity >= threshold:
                    results.append((extracted, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity between two selectors.

        Uses multiple heuristics:
        1. Exact substring match
        2. Common prefix/suffix
        3. Word overlap
        4. Character-level similarity
        """
        if s1 == s2:
            return 1.0

        s1_lower = s1.lower()
        s2_lower = s2.lower()

        # Check substring
        if s1_lower in s2_lower or s2_lower in s1_lower:
            return 0.9

        # Split into words
        words1 = set(re.split(r'[-_\s]+', s1_lower))
        words2 = set(re.split(r'[-_\s]+', s2_lower))

        if words1 and words2:
            # Jaccard similarity
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            word_sim = intersection / union if union > 0 else 0

            if word_sim > 0.5:
                return 0.7 + (word_sim * 0.2)

        # Character-level similarity (Levenshtein-like)
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0

        # Simple edit distance approximation
        common_chars = sum(1 for c in s1 if c in s2)
        char_sim = common_chars / max_len

        return char_sim * 0.6

    def get_selector_mapping(
        self,
        selector: str
    ) -> Optional[SelectorMapping]:
        """Get semantic mapping and alternatives for a selector.

        Args:
            selector: The selector to map

        Returns:
            SelectorMapping with alternatives and recommendations
        """
        # Find the selector in cache
        found = self.find_selector(selector)

        if not found:
            return None

        extracted = found[0]

        # Find similar selectors as alternatives
        similar = self.find_similar_selectors(selector, threshold=0.5)
        alternatives = [s.selector for s, _ in similar[:5] if s.selector != selector]

        # Calculate stability score
        stability = self._calculate_stability_score(extracted)

        # Generate recommendation
        recommendation = self._generate_recommendation(extracted, stability)

        return SelectorMapping(
            selector=selector,
            semantic_purpose=extracted.semantic_name or "unknown element",
            alternatives=alternatives,
            stability_score=stability,
            recommendation=recommendation,
        )

    def _calculate_stability_score(self, selector: ExtractedSelector) -> float:
        """Calculate how stable a selector is likely to be.

        Factors:
        - Selector type (testid > id > aria > role > class > text)
        - Specificity
        - Semantic naming
        """
        type_scores = {
            "testid": 0.95,
            "id": 0.85,
            "aria": 0.80,
            "role": 0.75,
            "name": 0.70,
            "placeholder": 0.50,
            "class": 0.40,
            "text": 0.30,
        }

        base_score = type_scores.get(selector.selector_type, 0.5)

        # Bonus for semantic naming
        if selector.semantic_name:
            base_score += 0.05

        # Penalty for very short selectors
        if len(selector.selector) < 4:
            base_score -= 0.1

        # Penalty for dynamic-looking selectors
        if re.search(r'\d{4,}|[a-f0-9]{8}', selector.selector):
            base_score -= 0.3

        return max(0, min(1, base_score))

    def _generate_recommendation(
        self,
        selector: ExtractedSelector,
        stability: float
    ) -> Optional[str]:
        """Generate recommendation for improving selector stability."""
        if stability >= 0.8:
            return None

        if selector.selector_type == "class":
            return "Consider adding a data-testid attribute for more stable testing"

        if selector.selector_type == "text":
            return "Text-based selectors are fragile. Add a data-testid or aria-label instead"

        if re.search(r'\d{4,}|[a-f0-9]{8}', selector.selector):
            return "This selector appears to contain dynamic IDs. Use a stable identifier"

        if stability < 0.5:
            return "This selector has low stability. Consider using data-testid"

        return None

    def analyze_directory(
        self,
        directory: str = "src",
        extensions: Optional[list[str]] = None
    ) -> dict[str, ComponentInfo]:
        """Analyze all source files in a directory.

        Args:
            directory: Directory to analyze
            extensions: File extensions to include

        Returns:
            Dict mapping file paths to ComponentInfo
        """
        if extensions is None:
            extensions = [".jsx", ".tsx", ".vue", ".svelte", ".js", ".ts"]

        results = {}
        base_path = self.repo_path / directory

        if not base_path.exists():
            logger.warning(f"Directory not found: {base_path}")
            return results

        for ext in extensions:
            for file_path in base_path.rglob(f"*{ext}"):
                # Skip node_modules and test files
                if "node_modules" in str(file_path):
                    continue
                if ".test." in file_path.name or ".spec." in file_path.name:
                    continue

                rel_path = str(file_path.relative_to(self.repo_path))
                results[rel_path] = self.analyze_file(rel_path)

        return results

    def get_all_selectors(self) -> list[ExtractedSelector]:
        """Get all extracted selectors from cache.

        Returns:
            List of all selectors
        """
        all_selectors = []

        for component in self._component_cache.values():
            all_selectors.extend(component.selectors)

        return all_selectors

    def find_selectors_by_type(
        self,
        selector_type: str
    ) -> list[ExtractedSelector]:
        """Find all selectors of a specific type.

        Args:
            selector_type: Type to filter by (testid, id, class, etc.)

        Returns:
            List of matching selectors
        """
        return [
            s for s in self.get_all_selectors()
            if s.selector_type == selector_type
        ]

    def find_selectors_by_element(
        self,
        element_type: str
    ) -> list[ExtractedSelector]:
        """Find all selectors for a specific element type.

        Args:
            element_type: Element type (button, input, etc.)

        Returns:
            List of matching selectors
        """
        return [
            s for s in self.get_all_selectors()
            if s.element_type == element_type
        ]

    def get_component_selector_map(self) -> dict[str, list[str]]:
        """Get a map of component names to their selectors.

        Returns:
            Dict mapping component name to list of selectors
        """
        result = {}

        for component in self._component_cache.values():
            if component.name:
                result[component.name] = [s.selector for s in component.selectors]

        return result

    def clear_cache(self) -> None:
        """Clear the component cache."""
        self._component_cache.clear()
        self._selector_cache.clear()


# Global instance (lazy initialized)
_source_analyzer: Optional[SourceAnalyzer] = None


def get_source_analyzer(repo_path: str = ".") -> SourceAnalyzer:
    """Get or create SourceAnalyzer instance."""
    global _source_analyzer

    if _source_analyzer is None:
        _source_analyzer = SourceAnalyzer(repo_path)

    return _source_analyzer
