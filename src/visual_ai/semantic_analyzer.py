"""Semantic analysis of visual changes using Claude Vision.

This module provides AI-powered understanding of visual changes, using
Claude's vision capabilities to analyze screenshots and determine:
- What visually changed
- Category of each change (layout, content, style, etc.)
- Severity assessment (0-4 scale)
- Whether changes are intentional or regressions
- Root cause analysis
- Recommendations for action

The SemanticAnalyzer works in conjunction with StructuralDiff data
to provide comprehensive visual regression analysis.
"""

import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic
import structlog

from ..config import get_settings
from ..core.model_registry import get_model_id
from .models import (
    ChangeCategory,
    ChangeIntent,
    Severity,
    VisualChange,
    VisualComparisonResult,
)
from .structural_analyzer import StructuralDiff

logger = structlog.get_logger()


@dataclass
class SemanticAnalysis:
    """Result of semantic analysis of visual changes."""

    # Identified changes
    changes: List[VisualChange] = field(default_factory=list)

    # Overall assessment
    overall_assessment: str = ""
    summary: str = ""

    # Auto-approval recommendation
    auto_approval_recommended: bool = False
    approval_confidence: float = 0.0  # 0-1

    # Blocking issues
    blocking_issues: List[str] = field(default_factory=list)

    # Suggestions for developers
    suggestions: List[str] = field(default_factory=list)

    # User impact prediction
    user_impact: Optional[Dict[str, Any]] = None

    # Metadata
    model_used: str = ""
    analysis_duration_ms: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changes": [c.to_dict() for c in self.changes],
            "overall_assessment": self.overall_assessment,
            "summary": self.summary,
            "auto_approval_recommended": self.auto_approval_recommended,
            "approval_confidence": self.approval_confidence,
            "blocking_issues": self.blocking_issues,
            "suggestions": self.suggestions,
            "user_impact": self.user_impact,
            "model_used": self.model_used,
            "analysis_duration_ms": self.analysis_duration_ms,
            "token_usage": self.token_usage,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
        }

    def has_regressions(self) -> bool:
        """Check if any changes are classified as regressions."""
        return any(c.is_regression() for c in self.changes)

    def has_blocking_issues(self) -> bool:
        """Check if there are any blocking issues."""
        return len(self.blocking_issues) > 0 or any(c.is_blocking() for c in self.changes)

    def get_highest_severity(self) -> Severity:
        """Get the highest severity among all changes."""
        if not self.changes:
            return Severity.SAFE
        return max(c.severity for c in self.changes)


class SemanticAnalyzer:
    """AI-powered semantic understanding of visual changes.

    Uses Claude Vision to analyze screenshots and understand visual
    changes at a semantic level, beyond simple pixel diffing.

    Features:
    - Visual change identification and categorization
    - Severity assessment using 0-4 scale
    - Intent detection (intentional vs regression)
    - Root cause analysis with code context
    - Auto-approval recommendations
    - User impact prediction

    Usage:
        analyzer = SemanticAnalyzer()

        analysis = await analyzer.analyze_changes(
            baseline_screenshot=baseline_bytes,
            current_screenshot=current_bytes,
            structural_diff=diff,
            git_diff="...",
            pr_description="Added new button styling"
        )

        if analysis.auto_approval_recommended:
            print("Safe to auto-approve")
        else:
            for issue in analysis.blocking_issues:
                print(f"Blocking: {issue}")
    """

    def __init__(self, model: str = "claude-sonnet-4-5"):
        """Initialize the semantic analyzer.

        Args:
            model: Claude model to use for vision analysis
        """
        self.settings = get_settings()
        api_key = self.settings.anthropic_api_key
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = get_model_id(model)
        self.log = logger.bind(component="semantic_analyzer", model=self.model)

    def _encode_image(self, image_bytes: bytes) -> tuple[str, str]:
        """Encode image bytes to base64 with media type."""
        data = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Detect image type from magic bytes
        if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            media_type = "image/png"
        elif image_bytes[:2] == b"\xff\xd8":
            media_type = "image/jpeg"
        elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
            media_type = "image/gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            media_type = "image/webp"
        else:
            media_type = "image/png"  # Default

        return data, media_type

    def _build_analysis_prompt(
        self,
        structural_diff: Optional[StructuralDiff] = None,
        context: Optional[str] = None,
        git_diff: Optional[str] = None,
        pr_description: Optional[str] = None,
    ) -> str:
        """Build the analysis prompt for Claude."""

        structural_context = ""
        if structural_diff:
            structural_context = f"""
STRUCTURAL ANALYSIS (pre-computed):
- Elements baseline: {structural_diff.total_elements_baseline}
- Elements current: {structural_diff.total_elements_current}
- Added: {structural_diff.elements_added}
- Removed: {structural_diff.elements_removed}
- Modified: {structural_diff.elements_modified}
- Pixel diff: {structural_diff.pixel_diff_percentage:.1f}%
- Layout shift score: {structural_diff.layout_shift_score:.3f}
"""

        git_context = ""
        if git_diff:
            # Truncate if too long
            truncated = git_diff[:2000] + "..." if len(git_diff) > 2000 else git_diff
            git_context = f"""
CODE CHANGES (git diff):
```diff
{truncated}
```
"""

        pr_context = ""
        if pr_description:
            pr_context = f"""
PR DESCRIPTION:
{pr_description}
"""

        page_context = ""
        if context:
            page_context = f"""
PAGE CONTEXT:
{context}
"""

        prompt = f"""You are a visual QA expert analyzing screenshots for visual regressions.

Compare IMAGE 1 (BASELINE - expected/before) with IMAGE 2 (CURRENT - actual/after).

{structural_context}
{page_context}
{git_context}
{pr_context}

ANALYSIS TASKS:

1. IDENTIFY CHANGES: What visually changed between the screenshots?

2. CATEGORIZE each change:
   - layout: Position, size, arrangement changes
   - content: Text, images, data changes
   - style: Colors, fonts, borders, shadows
   - structure: DOM hierarchy changes
   - responsive: Breakpoint/viewport changes
   - animation: Motion/transition changes
   - accessibility: Contrast, focus indicators
   - performance: Loading states, skeleton screens

3. ASSESS SEVERITY (0-4):
   - 0 (SAFE): Expected/insignificant (dynamic content, timestamps)
   - 1 (INFO): Minor difference worth noting
   - 2 (MINOR): Noticeable change worth reviewing
   - 3 (MAJOR): Significant change affecting UX
   - 4 (CRITICAL): Blocking issue, major regression

4. DETERMINE INTENT:
   - intentional: Matches code changes or PR description
   - regression: Unintended visual bug
   - dynamic: Expected dynamic content variation
   - environmental: Timing, fonts, rendering differences
   - unknown: Cannot determine from context

5. ROOT CAUSE: What likely caused each change?

6. RECOMMENDATIONS: What should be done about each change?

Respond with JSON in this exact format:
{{
    "changes": [
        {{
            "id": "change-1",
            "category": "layout|content|style|structure|responsive|animation|accessibility|performance",
            "description": "Clear description of what changed",
            "severity": 0-4,
            "intent": "intentional|regression|dynamic|environmental|unknown",
            "element": "Affected element (button, header, etc.)",
            "bounds_baseline": {{"x": 0, "y": 0, "width": 100, "height": 50}},
            "bounds_current": {{"x": 0, "y": 10, "width": 100, "height": 50}},
            "baseline_value": "What baseline shows",
            "current_value": "What current shows",
            "root_cause": "Likely cause (CSS change, content update, etc.)",
            "impact_assessment": "How this affects users",
            "recommendation": "What to do (accept, fix, investigate)",
            "confidence": 0.0-1.0,
            "related_files": ["path/to/file.css"]
        }}
    ],
    "overall_assessment": "Summary of all changes and their significance",
    "auto_approval_recommended": true/false,
    "approval_confidence": 0.0-1.0,
    "blocking_issues": ["List of issues that should block approval"],
    "suggestions": ["Actionable suggestions for developers"]
}}

IMPORTANT:
- Be precise about locations and affected elements
- Only mark as regression if clearly unintended
- Consider PR description when determining intent
- High approval_confidence only if all changes are clearly intentional
- Include blocking_issues for CRITICAL or MAJOR regressions
"""
        return prompt

    async def analyze_changes(
        self,
        baseline_screenshot: bytes,
        current_screenshot: bytes,
        structural_diff: Optional[StructuralDiff] = None,
        context: Optional[str] = None,
        git_diff: Optional[str] = None,
        pr_description: Optional[str] = None,
    ) -> SemanticAnalysis:
        """Analyze visual changes between screenshots using AI.

        This is the main entry point for semantic analysis. It sends both
        screenshots to Claude Vision along with structural and code context.

        Args:
            baseline_screenshot: Baseline (expected) screenshot bytes
            current_screenshot: Current (actual) screenshot bytes
            structural_diff: Pre-computed structural diff (optional)
            context: Description of the page/component being tested
            git_diff: Git diff of code changes (helps determine intent)
            pr_description: PR description (helps determine intent)

        Returns:
            SemanticAnalysis with categorized changes and recommendations
        """
        start_time = time.time()

        self.log.info(
            "Starting semantic analysis",
            baseline_size=len(baseline_screenshot),
            current_size=len(current_screenshot),
            has_structural_diff=structural_diff is not None,
            has_git_diff=git_diff is not None,
        )

        # Encode images
        baseline_data, baseline_type = self._encode_image(baseline_screenshot)
        current_data, current_type = self._encode_image(current_screenshot)

        # Build prompt
        prompt = self._build_analysis_prompt(
            structural_diff=structural_diff,
            context=context,
            git_diff=git_diff,
            pr_description=pr_description,
        )

        try:
            # Call Claude Vision
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "IMAGE 1 (BASELINE):"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": baseline_type,
                                    "data": baseline_data,
                                },
                            },
                            {"type": "text", "text": "IMAGE 2 (CURRENT):"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": current_type,
                                    "data": current_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            # Calculate cost (Sonnet pricing)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)

            # Parse response
            content = response.content[0].text
            result_data = self._parse_json_response(content)

            # Build analysis result
            analysis = self._build_analysis_from_response(
                result_data=result_data,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                start_time=start_time,
            )

            self.log.info(
                "Semantic analysis complete",
                changes=len(analysis.changes),
                auto_approve=analysis.auto_approval_recommended,
                confidence=analysis.approval_confidence,
                cost=f"${cost:.4f}",
            )

            return analysis

        except Exception as e:
            self.log.error("Semantic analysis failed", error=str(e))
            duration_ms = int((time.time() - start_time) * 1000)

            return SemanticAnalysis(
                overall_assessment=f"Analysis failed: {str(e)}",
                auto_approval_recommended=False,
                approval_confidence=0.0,
                blocking_issues=[f"Analysis error: {str(e)}"],
                analysis_duration_ms=duration_ms,
            )

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from Claude response, handling code blocks."""
        # Strip markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]

        return json.loads(content.strip())

    def _build_analysis_from_response(
        self,
        result_data: Dict[str, Any],
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        start_time: float,
    ) -> SemanticAnalysis:
        """Build SemanticAnalysis from parsed AI response."""
        duration_ms = int((time.time() - start_time) * 1000)

        # Parse changes
        changes = []
        for change_data in result_data.get("changes", []):
            change = self._parse_visual_change(change_data)
            changes.append(change)

        return SemanticAnalysis(
            changes=changes,
            overall_assessment=result_data.get("overall_assessment", ""),
            summary=result_data.get("overall_assessment", "")[:200],
            auto_approval_recommended=result_data.get("auto_approval_recommended", False),
            approval_confidence=result_data.get("approval_confidence", 0.0),
            blocking_issues=result_data.get("blocking_issues", []),
            suggestions=result_data.get("suggestions", []),
            model_used=model,
            analysis_duration_ms=duration_ms,
            token_usage={"input": input_tokens, "output": output_tokens},
            cost_usd=cost,
        )

    def _parse_visual_change(self, data: Dict[str, Any]) -> VisualChange:
        """Parse a single VisualChange from AI response data."""
        # Map category string to enum
        category_mapping = {
            "layout": ChangeCategory.LAYOUT,
            "content": ChangeCategory.CONTENT,
            "style": ChangeCategory.STYLE,
            "structure": ChangeCategory.STRUCTURE,
            "responsive": ChangeCategory.RESPONSIVE,
            "animation": ChangeCategory.ANIMATION,
            "accessibility": ChangeCategory.ACCESSIBILITY,
            "performance": ChangeCategory.PERFORMANCE,
        }

        # Map intent string to enum
        intent_mapping = {
            "intentional": ChangeIntent.INTENTIONAL,
            "regression": ChangeIntent.REGRESSION,
            "dynamic": ChangeIntent.DYNAMIC,
            "environmental": ChangeIntent.ENVIRONMENTAL,
            "unknown": ChangeIntent.UNKNOWN,
        }

        # Map severity int to enum
        severity_mapping = {
            0: Severity.SAFE,
            1: Severity.INFO,
            2: Severity.MINOR,
            3: Severity.MAJOR,
            4: Severity.CRITICAL,
        }

        category = category_mapping.get(
            data.get("category", "content"), ChangeCategory.CONTENT
        )
        intent = intent_mapping.get(
            data.get("intent", "unknown"), ChangeIntent.UNKNOWN
        )
        severity = severity_mapping.get(data.get("severity", 2), Severity.MINOR)

        return VisualChange(
            id=data.get("id", str(uuid.uuid4())[:8]),
            category=category,
            intent=intent,
            severity=severity,
            element=None,  # Could populate from bounds if needed
            bounds_baseline=data.get("bounds_baseline"),
            bounds_current=data.get("bounds_current"),
            property_name=None,
            baseline_value=data.get("baseline_value"),
            current_value=data.get("current_value"),
            description=data.get("description", ""),
            root_cause=data.get("root_cause"),
            impact_assessment=data.get("impact_assessment", ""),
            recommendation=data.get("recommendation", ""),
            confidence=data.get("confidence", 0.8),
            related_commit=None,
            related_files=data.get("related_files", []),
        )

    async def classify_change_intent(
        self,
        change: VisualChange,
        git_diff: Optional[str] = None,
        pr_description: Optional[str] = None,
    ) -> ChangeIntent:
        """Determine if a specific change is intentional based on code context.

        This method provides more focused intent classification for a single
        change when additional context is available.

        Args:
            change: The VisualChange to classify
            git_diff: Git diff for context
            pr_description: PR description for context

        Returns:
            ChangeIntent classification
        """
        if not git_diff and not pr_description:
            return change.intent  # Return existing classification

        prompt = f"""Determine if this visual change was intentional based on the code context.

VISUAL CHANGE:
- Description: {change.description}
- Category: {change.category.value}
- Element: {change.element.to_dict() if change.element else 'N/A'}
- Baseline value: {change.baseline_value}
- Current value: {change.current_value}

CODE CONTEXT:
{f"Git diff: {git_diff[:1500]}" if git_diff else "No git diff available"}

{f"PR description: {pr_description}" if pr_description else "No PR description"}

Respond with JSON:
{{
    "intent": "intentional|regression|dynamic|environmental|unknown",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            result = self._parse_json_response(content)

            intent_mapping = {
                "intentional": ChangeIntent.INTENTIONAL,
                "regression": ChangeIntent.REGRESSION,
                "dynamic": ChangeIntent.DYNAMIC,
                "environmental": ChangeIntent.ENVIRONMENTAL,
                "unknown": ChangeIntent.UNKNOWN,
            }

            return intent_mapping.get(result.get("intent", "unknown"), ChangeIntent.UNKNOWN)

        except Exception as e:
            self.log.warning("Intent classification failed", error=str(e))
            return change.intent

    async def generate_change_description(
        self,
        baseline: bytes,
        current: bytes,
        region: Optional[Dict[str, int]] = None,
    ) -> str:
        """Generate a human-readable description of what changed.

        Useful for generating PR comments or test reports.

        Args:
            baseline: Baseline screenshot bytes
            current: Current screenshot bytes
            region: Optional region to focus on {x, y, width, height}

        Returns:
            Human-readable description of the change
        """
        baseline_data, baseline_type = self._encode_image(baseline)
        current_data, current_type = self._encode_image(current)

        region_context = ""
        if region:
            region_context = f"""
Focus on this region: x={region['x']}, y={region['y']},
width={region['width']}, height={region['height']}
"""

        prompt = f"""Describe the visual difference between these two screenshots in 1-2 sentences.
Be specific and concise. Focus on what a user would notice.
{region_context}
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "BEFORE:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": baseline_type,
                                    "data": baseline_data,
                                },
                            },
                            {"type": "text", "text": "AFTER:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": current_type,
                                    "data": current_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            return response.content[0].text.strip()

        except Exception as e:
            self.log.warning("Description generation failed", error=str(e))
            return "Unable to generate description"

    async def predict_user_impact(
        self,
        changes: List[VisualChange],
    ) -> Dict[str, Any]:
        """Predict how visual changes affect user experience.

        Analyzes changes to predict impact on:
        - Usability and navigation
        - Accessibility
        - Conversion (CTAs, forms)
        - Brand consistency

        Args:
            changes: List of VisualChange objects

        Returns:
            Dict with impact predictions
        """
        if not changes:
            return {
                "overall_impact": "none",
                "usability_impact": "No changes detected",
                "accessibility_impact": "No changes detected",
                "conversion_impact": "No changes detected",
                "affected_user_flows": [],
                "risk_level": 0,
            }

        changes_summary = "\n".join(
            f"- {c.category.value}: {c.description} (severity: {c.severity.value})"
            for c in changes
        )

        prompt = f"""Analyze the user experience impact of these visual changes:

{changes_summary}

Respond with JSON:
{{
    "overall_impact": "none|low|medium|high|critical",
    "usability_impact": "How changes affect ease of use",
    "accessibility_impact": "Any accessibility concerns",
    "conversion_impact": "Impact on CTAs, forms, checkout",
    "affected_user_flows": ["List of affected user journeys"],
    "risk_level": 0-4,
    "recommendations": ["List of UX recommendations"]
}}
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text
            return self._parse_json_response(content)

        except Exception as e:
            self.log.warning("User impact prediction failed", error=str(e))
            return {
                "overall_impact": "unknown",
                "usability_impact": "Analysis failed",
                "risk_level": 2,
                "error": str(e),
            }


# Factory function for easy instantiation
def create_semantic_analyzer(model: str = "claude-sonnet-4-5") -> SemanticAnalyzer:
    """Create a SemanticAnalyzer instance.

    Args:
        model: Claude model to use for vision analysis

    Returns:
        Configured SemanticAnalyzer instance
    """
    return SemanticAnalyzer(model=model)
