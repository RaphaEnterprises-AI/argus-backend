"""Visual AI module for intelligent visual regression testing.

This module provides AI-powered visual comparison capabilities that go beyond
pixel-by-pixel diffing to understand the semantic meaning of visual changes.
"""

# Accessibility analyzer
from .accessibility_analyzer import (
    AccessibilityAnalyzer,
    AccessibilityReport,
    ContrastViolation,
    ReadabilityIssue,
    TouchTargetViolation,
)

# Capture utilities
from .capture import EnhancedCapture, create_enhanced_capture
from .models import (
    ChangeCategory,
    ChangeIntent,
    Severity,
    VisualChange,
    VisualComparisonResult,
    VisualElement,
    VisualSnapshot,
)

# Responsive analyzer
from .responsive_analyzer import (
    BreakpointIssue,
    ResponsiveAnalyzer,
    ResponsiveDiff,
    ViewportConfig,
)

# Semantic analyzer
from .semantic_analyzer import (
    SemanticAnalysis,
    SemanticAnalyzer,
    create_semantic_analyzer,
)

# Structural analyzer
from .structural_analyzer import (
    DOMTreeParser,
    ElementBounds,
    LayoutRegion,
    LayoutShift,
    # New exports for VisualElement-based structural analysis
    StructuralAnalyzer,
    StructuralChange,
    StructuralChangeType,
    StructuralDiff,
    StructuralElement,
    VisualStructuralDiff,
)

# Optional imports - these analyzers may not be implemented yet
try:
    from .perceptual_analyzer import (
        ColorChange,
        PerceptualAnalyzer,
        TextRenderingDiff,
    )
    _HAS_PERCEPTUAL_ANALYZER = True
except ImportError:
    _HAS_PERCEPTUAL_ANALYZER = False

# Cross-browser analyzer
try:
    from .cross_browser_analyzer import (
        BrowserCompatibilityReport,
        BrowserConfig,
        BrowserDifference,
        CrossBrowserAnalyzer,
    )
    _HAS_CROSS_BROWSER_ANALYZER = True
except ImportError:
    _HAS_CROSS_BROWSER_ANALYZER = False

__all__ = [
    # Models
    "ChangeCategory",
    "ChangeIntent",
    "Severity",
    "VisualElement",
    "VisualChange",
    "VisualSnapshot",
    "VisualComparisonResult",
    # Structural analyzer
    "StructuralDiff",
    "StructuralChange",
    "StructuralChangeType",
    "StructuralElement",
    "ElementBounds",
    "LayoutRegion",
    "StructuralAnalyzer",
    "VisualStructuralDiff",
    "LayoutShift",
    "DOMTreeParser",
    # Semantic analyzer
    "SemanticAnalyzer",
    "SemanticAnalysis",
    "create_semantic_analyzer",
    # Accessibility analyzer
    "AccessibilityAnalyzer",
    "AccessibilityReport",
    "ContrastViolation",
    "TouchTargetViolation",
    "ReadabilityIssue",
    # Capture utilities
    "EnhancedCapture",
    "create_enhanced_capture",
    # Responsive analyzer
    "ViewportConfig",
    "BreakpointIssue",
    "ResponsiveDiff",
    "ResponsiveAnalyzer",
]

# Add perceptual analyzer exports if available
if _HAS_PERCEPTUAL_ANALYZER:
    __all__.extend([
        "PerceptualAnalyzer",
        "ColorChange",
        "TextRenderingDiff",
    ])

# Add cross-browser analyzer exports if available
if _HAS_CROSS_BROWSER_ANALYZER:
    __all__.extend([
        "CrossBrowserAnalyzer",
        "BrowserConfig",
        "BrowserDifference",
        "BrowserCompatibilityReport",
    ])
