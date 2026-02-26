"""Multi-source requirements parser with AI-powered ambiguity detection.

Parses requirements from multiple sources (plain text, Jira, GitHub PRs,
Figma annotations, codebase analysis) into a normalized ParsedRequirement
format.  Each requirement is enriched with an ambiguity score and
actionable suggestions for clarification.

The parser does NOT extend BaseAgent.  Instead it accepts an *ai_caller*
function (same signature as ``BaseAgent._call_ai``) so that the owning
agent (e.g. TestGenAgent) can inject its own method and retain unified
cost/token tracking.

RAP-350: TestGen Orchestrator Agent - Requirements Analysis Phase

Example usage::

    from src.agents.testgen import RequirementsParser

    parser = RequirementsParser(ai_caller=self._call_ai)
    reqs = await parser.parse("requirements_text", {"text": "User can log in ..."})
    reqs = await parser.analyze_ambiguity(reqs)
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog

from src.core.model_router import TaskType

logger = structlog.get_logger(__name__)

# Type alias for the AI caller function matching BaseAgent._call_ai signature.
# The caller returns an AIResponse (from src.agents.base) but we only depend on
# ``content: str`` attribute so we keep the type loose here to avoid circular
# imports.
AICaller = Callable[..., Coroutine[Any, Any, Any]]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RequirementSource(str, Enum):
    """Supported requirement sources."""

    TEXT = "requirements_text"
    JIRA = "jira"
    GITHUB_PR = "github_pr"
    FIGMA = "figma"
    CONFLUENCE = "confluence"
    CODEBASE = "codebase_analysis"


class AmbiguityType(str, Enum):
    """Categories of ambiguity detected in requirements."""

    VAGUE_TERM = "vague_term"
    MISSING_EDGE_CASE = "missing_edge_case"
    IMPLICIT_ASSUMPTION = "implicit_assumption"
    INCOMPLETE_SPEC = "incomplete_spec"
    CONTRADICTION = "contradiction"
    MISSING_BOUNDARY = "missing_boundary"
    UNCLEAR_PRIORITY = "unclear_priority"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AmbiguityDetail:
    """A single ambiguity finding for a requirement."""

    type: str  # AmbiguityType value
    description: str
    suggestion: str

    def to_dict(self) -> dict[str, str]:
        return {
            "type": self.type,
            "description": self.description,
            "suggestion": self.suggestion,
        }


@dataclass
class ParsedRequirement:
    """Normalized requirement regardless of source."""

    id: str
    title: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    ambiguity_score: float = 0.0  # 0.0 (clear) to 1.0 (very ambiguous)
    ambiguity_details: list[dict] = field(default_factory=list)
    priority: str = "medium"  # critical / high / medium / low
    tags: list[str] = field(default_factory=list)
    source_ref: str | None = None  # External ID (Jira key, PR number, etc.)
    source_type: str = RequirementSource.TEXT.value
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "ambiguity_score": self.ambiguity_score,
            "ambiguity_details": self.ambiguity_details,
            "priority": self.priority,
            "tags": self.tags,
            "source_ref": self.source_ref,
            "source_type": self.source_type,
            "raw_text": self.raw_text,
        }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PARSE_TEXT_PROMPT = """\
You are a requirements analyst. Parse the following free-form requirements \
text into discrete, testable requirements.

For EACH requirement output a JSON object with:
- "title": short summary (max 80 chars)
- "description": detailed description
- "acceptance_criteria": list of verifiable acceptance criteria strings
- "priority": one of "critical", "high", "medium", "low"
- "tags": list of category tags (e.g. "auth", "ui", "api", "performance")

Return a JSON array of requirement objects.  Only output valid JSON, no \
markdown fences.

---
REQUIREMENTS TEXT:
{text}
"""

_PARSE_JIRA_PROMPT = """\
You are a requirements analyst. Convert the following Jira issues into \
testable requirements.

For EACH issue output a JSON object with:
- "title": short summary (max 80 chars)
- "description": full description including context from the Jira issue
- "acceptance_criteria": list of verifiable acceptance criteria strings \
  (derive from the issue description and any explicit ACs)
- "priority": one of "critical", "high", "medium", "low"
- "tags": list of category tags
- "source_ref": the Jira issue key

Return a JSON array.  Only output valid JSON, no markdown fences.

---
JIRA ISSUES:
{issues_json}
"""

_PARSE_GITHUB_PR_PROMPT = """\
You are a requirements analyst. Derive testable requirements from this \
GitHub Pull Request.

Analyze:
1. The PR title and description for stated intent
2. The changed files list for scope
3. The diff for specific behavioral changes

For EACH derived requirement output a JSON object with:
- "title": short summary (max 80 chars)
- "description": what changed and why it needs testing
- "acceptance_criteria": verifiable ACs based on the code changes
- "priority": one of "critical", "high", "medium", "low"
- "tags": list of category tags
- "source_ref": the PR number as a string

Return a JSON array.  Only output valid JSON, no markdown fences.

---
PR TITLE: {title}

PR BODY:
{body}

CHANGED FILES:
{changed_files}

DIFF (truncated):
{diff}
"""

_PARSE_FIGMA_PROMPT = """\
You are a requirements analyst specializing in UI/UX. Convert Figma design \
annotations into testable UI requirements.

For EACH screen/annotation, derive requirements covering:
- Layout and component presence
- Interaction behaviors (clicks, hovers, transitions)
- Visual states (loading, error, empty, success)
- Responsive behavior if implied
- Accessibility (labels, contrast, focus order)

For EACH requirement output a JSON object with:
- "title": short summary (max 80 chars)
- "description": detailed UI requirement
- "acceptance_criteria": verifiable ACs
- "priority": one of "critical", "high", "medium", "low"
- "tags": list of category tags (always include "ui")

Return a JSON array.  Only output valid JSON, no markdown fences.

---
FIGMA SCREENS:
{screens_json}
"""

_PARSE_CODEBASE_PROMPT = """\
You are a requirements analyst. Derive testable requirements from a \
codebase analysis report.

Analyze the functions, API endpoints, and UI components to identify:
- Functional requirements (what each endpoint/component should do)
- Integration points that need testing
- Error handling paths
- Input validation rules
- Authentication/authorization requirements

For EACH requirement output a JSON object with:
- "title": short summary (max 80 chars)
- "description": detailed requirement derived from code
- "acceptance_criteria": verifiable ACs
- "priority": one of "critical", "high", "medium", "low"
- "tags": list of category tags

Return a JSON array.  Only output valid JSON, no markdown fences.

---
CODEBASE ANALYSIS:

Functions:
{functions_json}

Endpoints:
{endpoints_json}

Components:
{components_json}
"""

_AMBIGUITY_ANALYSIS_PROMPT = """\
You are a requirements quality auditor. Analyze the following requirements \
for ambiguity and completeness issues.

Detect these categories of problems:
1. **Vague terms**: Words like "should work", "properly", "fast", \
   "user-friendly", "intuitive", "seamless", "reasonable", "appropriate"
2. **Missing edge cases**: No error handling specified, no boundary \
   conditions, no timeout/retry behavior, no empty-state handling
3. **Implicit assumptions**: Assuming authentication exists, assuming \
   data is pre-populated, assuming specific browser/device, assuming \
   network connectivity
4. **Incomplete specifications**: Missing input formats, missing output \
   formats, missing error messages, missing state transitions
5. **Contradictions**: Requirements that conflict with each other \
   (analyze across the full set)
6. **Missing boundaries**: No limits on input size, no pagination, \
   no rate limits, no max retries

For EACH requirement, output a JSON object with:
- "requirement_id": the requirement ID
- "ambiguity_score": float 0.0 (perfectly clear) to 1.0 (extremely ambiguous)
- "issues": list of objects, each with:
  - "type": one of "vague_term", "missing_edge_case", \
    "implicit_assumption", "incomplete_spec", "contradiction", \
    "missing_boundary", "unclear_priority"
  - "description": what the problem is
  - "suggestion": actionable fix suggestion

Return a JSON array of analysis objects.  Only output valid JSON, no \
markdown fences.

---
REQUIREMENTS:
{requirements_json}
"""


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class RequirementsParser:
    """Multi-source requirement parser with AI-powered ambiguity detection.

    This class does NOT extend ``BaseAgent``.  It accepts an ``ai_caller``
    async function that has the same signature as ``BaseAgent._call_ai`` so
    the owning agent can pass in its own method and keep cost/token tracking
    unified.

    Args:
        ai_caller: Async function matching ``BaseAgent._call_ai`` signature.
            If *None*, the parser falls back to a simple heuristic-only mode
            (no AI enrichment).
    """

    def __init__(self, ai_caller: AICaller | None = None) -> None:
        self._call_ai = ai_caller
        self._log = logger.bind(component="RequirementsParser")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def parse(
        self,
        source_type: str,
        source_data: dict[str, Any],
    ) -> list[ParsedRequirement]:
        """Dispatch to the appropriate source-specific parser.

        Args:
            source_type: One of ``RequirementSource`` values (e.g.
                ``"requirements_text"``, ``"jira"``, ``"github_pr"``, etc.)
            source_data: Source-specific data dict.  See individual parse
                methods for expected keys.

        Returns:
            List of ``ParsedRequirement`` objects.

        Raises:
            ValueError: If *source_type* is not recognised.
        """
        self._log.info(
            "Parsing requirements",
            source_type=source_type,
            data_keys=list(source_data.keys()),
        )

        dispatch: dict[str, Callable[..., Coroutine[Any, Any, list[ParsedRequirement]]]] = {
            RequirementSource.TEXT.value: lambda d: self.parse_text(d.get("text", "")),
            RequirementSource.JIRA.value: lambda d: self.parse_jira(d),
            RequirementSource.GITHUB_PR.value: lambda d: self.parse_github_pr(d),
            RequirementSource.FIGMA.value: lambda d: self.parse_figma(d),
            RequirementSource.CONFLUENCE.value: lambda d: self.parse_text(
                d.get("content", d.get("text", ""))
            ),
            RequirementSource.CODEBASE.value: lambda d: self.parse_codebase(d),
        }

        handler = dispatch.get(source_type)
        if handler is None:
            raise ValueError(
                f"Unknown source_type '{source_type}'. "
                f"Supported: {[s.value for s in RequirementSource]}"
            )

        requirements = await handler(source_data)

        self._log.info(
            "Parsed requirements",
            source_type=source_type,
            count=len(requirements),
        )
        return requirements

    # ------------------------------------------------------------------
    # Source-specific parsers
    # ------------------------------------------------------------------

    async def parse_text(self, text: str) -> list[ParsedRequirement]:
        """Parse plain-text requirements.

        Uses AI to decompose free-form text into discrete testable
        requirements.  Falls back to line-splitting heuristic when no
        AI caller is available.

        Args:
            text: Free-form requirements text.

        Returns:
            List of ``ParsedRequirement``.
        """
        if not text or not text.strip():
            self._log.warning("Empty text provided to parse_text")
            return []

        if self._call_ai is None:
            return self._parse_text_heuristic(text)

        prompt = _PARSE_TEXT_PROMPT.format(text=text)
        raw_items = await self._ai_parse(
            prompt=prompt,
            task_type=TaskType.CODE_ANALYSIS,
            max_cost=0.04,
            context_label="parse_text",
        )

        return self._items_to_requirements(
            raw_items,
            source_type=RequirementSource.TEXT.value,
            raw_text=text,
        )

    async def parse_jira(self, jira_data: dict[str, Any]) -> list[ParsedRequirement]:
        """Parse Jira issues into requirements.

        Expected *jira_data* shape::

            {
                "issues": [
                    {
                        "key": "PROJ-123",
                        "summary": "...",
                        "description": "...",
                        "acceptance_criteria": "...",  # optional
                        "priority": "High",            # optional
                        "labels": ["auth", "backend"], # optional
                    },
                    ...
                ]
            }

        Args:
            jira_data: Dict containing a list of Jira issues.

        Returns:
            List of ``ParsedRequirement``.
        """
        issues = jira_data.get("issues", [])
        if not issues:
            self._log.warning("No Jira issues provided")
            return []

        if self._call_ai is None:
            return self._parse_jira_heuristic(issues)

        issues_json = json.dumps(issues, indent=2, default=str)
        prompt = _PARSE_JIRA_PROMPT.format(issues_json=issues_json)

        raw_items = await self._ai_parse(
            prompt=prompt,
            task_type=TaskType.CODE_ANALYSIS,
            max_cost=0.04,
            context_label="parse_jira",
        )

        requirements = self._items_to_requirements(
            raw_items,
            source_type=RequirementSource.JIRA.value,
            raw_text=issues_json,
        )

        # Ensure Jira keys are preserved as source_ref
        for req in requirements:
            if not req.source_ref:
                # Try matching from original issues by title similarity
                for issue in issues:
                    if issue.get("key") and issue.get("summary", "").lower() in req.title.lower():
                        req.source_ref = issue["key"]
                        break

        return requirements

    async def parse_github_pr(self, pr_data: dict[str, Any]) -> list[ParsedRequirement]:
        """Parse a GitHub Pull Request into testable requirements.

        Expected *pr_data* shape::

            {
                "title": "Add SSO login flow",
                "body": "## Description\\n...",
                "number": 42,                    # optional
                "changed_files": ["src/auth.py"], # optional
                "diff": "--- a/src/auth.py ...",  # optional
            }

        Args:
            pr_data: Dict describing a GitHub PR.

        Returns:
            List of ``ParsedRequirement``.
        """
        title = pr_data.get("title", "")
        body = pr_data.get("body", "")
        changed_files = pr_data.get("changed_files", [])
        diff = pr_data.get("diff", "")
        pr_number = str(pr_data.get("number", ""))

        if not title and not body:
            self._log.warning("Empty GitHub PR data")
            return []

        if self._call_ai is None:
            return self._parse_github_pr_heuristic(pr_data)

        # Truncate diff to avoid blowing up context
        max_diff_len = 8000
        if len(diff) > max_diff_len:
            diff = diff[:max_diff_len] + "\n... [truncated]"

        changed_files_str = "\n".join(f"- {f}" for f in changed_files) if changed_files else "(none)"

        prompt = _PARSE_GITHUB_PR_PROMPT.format(
            title=title,
            body=body,
            changed_files=changed_files_str,
            diff=diff,
        )

        raw_items = await self._ai_parse(
            prompt=prompt,
            task_type=TaskType.CODE_ANALYSIS,
            max_cost=0.05,
            context_label="parse_github_pr",
        )

        requirements = self._items_to_requirements(
            raw_items,
            source_type=RequirementSource.GITHUB_PR.value,
            raw_text=f"PR #{pr_number}: {title}\n\n{body}",
        )

        # Stamp source_ref with PR number if not already set
        for req in requirements:
            if not req.source_ref and pr_number:
                req.source_ref = pr_number

        return requirements

    async def parse_figma(self, figma_data: dict[str, Any]) -> list[ParsedRequirement]:
        """Parse Figma design annotations into UI requirements.

        Expected *figma_data* shape::

            {
                "screens": [
                    {
                        "name": "Login Screen",
                        "annotations": [
                            "Email field required",
                            "Password min 8 chars",
                            ...
                        ],
                    },
                    ...
                ]
            }

        Args:
            figma_data: Dict containing a list of Figma screens.

        Returns:
            List of ``ParsedRequirement``.
        """
        screens = figma_data.get("screens", [])
        if not screens:
            self._log.warning("No Figma screens provided")
            return []

        if self._call_ai is None:
            return self._parse_figma_heuristic(screens)

        screens_json = json.dumps(screens, indent=2, default=str)
        prompt = _PARSE_FIGMA_PROMPT.format(screens_json=screens_json)

        raw_items = await self._ai_parse(
            prompt=prompt,
            task_type=TaskType.CODE_ANALYSIS,
            max_cost=0.04,
            context_label="parse_figma",
        )

        return self._items_to_requirements(
            raw_items,
            source_type=RequirementSource.FIGMA.value,
            raw_text=screens_json,
        )

    async def parse_codebase(self, codebase_data: dict[str, Any]) -> list[ParsedRequirement]:
        """Derive requirements from codebase analysis.

        Expected *codebase_data* shape::

            {
                "functions": [...],   # List of function signatures/descriptions
                "endpoints": [...],   # List of API endpoint descriptions
                "components": [...],  # List of UI component descriptions
            }

        Args:
            codebase_data: Dict containing code analysis output.

        Returns:
            List of ``ParsedRequirement``.
        """
        functions = codebase_data.get("functions", [])
        endpoints = codebase_data.get("endpoints", [])
        components = codebase_data.get("components", [])

        if not functions and not endpoints and not components:
            self._log.warning("Empty codebase data provided")
            return []

        if self._call_ai is None:
            return self._parse_codebase_heuristic(codebase_data)

        prompt = _PARSE_CODEBASE_PROMPT.format(
            functions_json=json.dumps(functions, indent=2, default=str),
            endpoints_json=json.dumps(endpoints, indent=2, default=str),
            components_json=json.dumps(components, indent=2, default=str),
        )

        raw_items = await self._ai_parse(
            prompt=prompt,
            task_type=TaskType.CODE_ANALYSIS,
            max_cost=0.05,
            context_label="parse_codebase",
        )

        return self._items_to_requirements(
            raw_items,
            source_type=RequirementSource.CODEBASE.value,
            raw_text=prompt,
        )

    # ------------------------------------------------------------------
    # Ambiguity analysis
    # ------------------------------------------------------------------

    async def analyze_ambiguity(
        self,
        requirements: list[ParsedRequirement],
    ) -> list[ParsedRequirement]:
        """Enrich requirements with AI-powered ambiguity analysis.

        Each requirement will have its ``ambiguity_score`` and
        ``ambiguity_details`` fields populated after this call.

        When no AI caller is available the method falls back to a
        keyword-based heuristic.

        Args:
            requirements: List of parsed requirements to analyze.

        Returns:
            The same list, mutated in place with ambiguity information.
        """
        if not requirements:
            return requirements

        self._log.info(
            "Analyzing ambiguity",
            requirement_count=len(requirements),
        )

        if self._call_ai is None:
            return self._analyze_ambiguity_heuristic(requirements)

        # Build a compact representation for the prompt
        req_summaries = []
        for req in requirements:
            req_summaries.append({
                "requirement_id": req.id,
                "title": req.title,
                "description": req.description,
                "acceptance_criteria": req.acceptance_criteria,
            })

        prompt = _AMBIGUITY_ANALYSIS_PROMPT.format(
            requirements_json=json.dumps(req_summaries, indent=2)
        )

        try:
            response = await self._call_ai(
                messages=[{"role": "user", "content": prompt}],
                task_type=TaskType.SEMANTIC_UNDERSTANDING,
                max_cost=0.06,
                max_tokens=4096,
                temperature=0.0,
                json_mode=True,
            )

            analyses = self._safe_json_loads(response.content)
            if not isinstance(analyses, list):
                self._log.warning(
                    "Ambiguity analysis returned non-list",
                    type=type(analyses).__name__,
                )
                return self._analyze_ambiguity_heuristic(requirements)

            # Index analyses by requirement_id for efficient lookup
            analysis_map: dict[str, dict] = {}
            for item in analyses:
                if isinstance(item, dict) and "requirement_id" in item:
                    analysis_map[item["requirement_id"]] = item

            # Enrich each requirement
            for req in requirements:
                analysis = analysis_map.get(req.id)
                if analysis:
                    req.ambiguity_score = float(
                        min(1.0, max(0.0, analysis.get("ambiguity_score", 0.0)))
                    )
                    issues = analysis.get("issues", [])
                    req.ambiguity_details = [
                        AmbiguityDetail(
                            type=issue.get("type", AmbiguityType.VAGUE_TERM.value),
                            description=issue.get("description", ""),
                            suggestion=issue.get("suggestion", ""),
                        ).to_dict()
                        for issue in issues
                        if isinstance(issue, dict)
                    ]
                else:
                    # AI didn't return analysis for this req; use heuristic
                    self._apply_heuristic_ambiguity(req)

        except Exception:
            self._log.exception("AI ambiguity analysis failed, falling back to heuristic")
            return self._analyze_ambiguity_heuristic(requirements)

        self._log.info(
            "Ambiguity analysis complete",
            avg_score=round(
                sum(r.ambiguity_score for r in requirements) / len(requirements), 3
            ),
        )
        return requirements

    # ------------------------------------------------------------------
    # Internal helpers: AI interaction
    # ------------------------------------------------------------------

    async def _ai_parse(
        self,
        prompt: str,
        task_type: TaskType,
        max_cost: float,
        context_label: str,
    ) -> list[dict]:
        """Send a parse prompt to the AI and return a list of dicts.

        Args:
            prompt: The fully formatted prompt.
            task_type: Task type for model routing.
            max_cost: Budget ceiling for this call.
            context_label: Human label for logging.

        Returns:
            List of dicts parsed from the AI JSON response. Empty list if
            the call fails or returns unparseable output.
        """
        assert self._call_ai is not None, "_ai_parse called without ai_caller"

        try:
            response = await self._call_ai(
                messages=[{"role": "user", "content": prompt}],
                task_type=task_type,
                max_cost=max_cost,
                max_tokens=4096,
                temperature=0.0,
                json_mode=True,
            )

            parsed = self._safe_json_loads(response.content)

            if isinstance(parsed, list):
                self._log.debug(
                    "AI parse success",
                    context=context_label,
                    items=len(parsed),
                )
                return parsed

            # Sometimes the model wraps the array in an object
            if isinstance(parsed, dict):
                for key in ("requirements", "items", "results", "data"):
                    if isinstance(parsed.get(key), list):
                        self._log.debug(
                            "AI parse success (unwrapped)",
                            context=context_label,
                            wrapper_key=key,
                            items=len(parsed[key]),
                        )
                        return parsed[key]

            self._log.warning(
                "AI returned unexpected structure",
                context=context_label,
                type=type(parsed).__name__,
            )
            return []

        except Exception:
            self._log.exception("AI parse failed", context=context_label)
            return []

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _items_to_requirements(
        self,
        items: list[dict],
        source_type: str,
        raw_text: str,
    ) -> list[ParsedRequirement]:
        """Convert raw AI-output dicts into ``ParsedRequirement`` objects."""
        requirements: list[ParsedRequirement] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            title = str(item.get("title", "")).strip()
            if not title:
                continue

            description = str(item.get("description", title)).strip()

            ac_raw = item.get("acceptance_criteria", [])
            if isinstance(ac_raw, str):
                ac = [line.strip() for line in ac_raw.split("\n") if line.strip()]
            elif isinstance(ac_raw, list):
                ac = [str(a).strip() for a in ac_raw if str(a).strip()]
            else:
                ac = []

            priority = self._normalize_priority(item.get("priority", "medium"))

            tags_raw = item.get("tags", [])
            if isinstance(tags_raw, str):
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            elif isinstance(tags_raw, list):
                tags = [str(t).strip().lower() for t in tags_raw if str(t).strip()]
            else:
                tags = []

            source_ref = item.get("source_ref")
            if source_ref is not None:
                source_ref = str(source_ref)

            requirements.append(
                ParsedRequirement(
                    id=str(uuid.uuid4()),
                    title=title[:200],
                    description=description,
                    acceptance_criteria=ac,
                    priority=priority,
                    tags=tags,
                    source_ref=source_ref,
                    source_type=source_type,
                    raw_text=raw_text[:5000],
                )
            )

        return requirements

    @staticmethod
    def _normalize_priority(value: Any) -> str:
        """Normalize a priority value to one of the four canonical levels."""
        if not value:
            return "medium"
        normalized = str(value).strip().lower()
        mapping = {
            "critical": "critical",
            "blocker": "critical",
            "highest": "critical",
            "p0": "critical",
            "high": "high",
            "major": "high",
            "p1": "high",
            "medium": "medium",
            "normal": "medium",
            "minor": "medium",
            "p2": "medium",
            "low": "low",
            "trivial": "low",
            "p3": "low",
            "lowest": "low",
            "p4": "low",
        }
        return mapping.get(normalized, "medium")

    @staticmethod
    def _safe_json_loads(text: str) -> Any:
        """Parse JSON from AI response, stripping markdown fences if present."""
        if not text:
            return []

        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            # Remove opening fence (with optional language tag)
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find a JSON array or object within the text
            for pattern in (r"\[[\s\S]*\]", r"\{[\s\S]*\}"):
                match = re.search(pattern, cleaned)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        continue
            return []

    # ------------------------------------------------------------------
    # Heuristic fallbacks (no AI)
    # ------------------------------------------------------------------

    def _parse_text_heuristic(self, text: str) -> list[ParsedRequirement]:
        """Split text into requirements by bullet points or numbered items."""
        lines = text.strip().splitlines()
        requirements: list[ParsedRequirement] = []
        current_block: list[str] = []

        def flush_block() -> None:
            if not current_block:
                return
            block_text = "\n".join(current_block).strip()
            if not block_text:
                return
            # Use first line as title, rest as description
            parts = block_text.split("\n", 1)
            title = re.sub(r"^[\d\.\-\*\#\s]+", "", parts[0]).strip()
            description = parts[1].strip() if len(parts) > 1 else title
            if title:
                requirements.append(
                    ParsedRequirement(
                        id=str(uuid.uuid4()),
                        title=title[:200],
                        description=description,
                        source_type=RequirementSource.TEXT.value,
                        raw_text=block_text,
                    )
                )

        for line in lines:
            stripped = line.strip()
            # Detect new requirement boundaries
            is_boundary = bool(
                re.match(r"^(\d+[\.\)]\s|[-*]\s|#{1,3}\s)", stripped)
            )
            if is_boundary and current_block:
                flush_block()
                current_block = [stripped]
            elif stripped:
                current_block.append(stripped)
            else:
                # Blank line = potential boundary
                if current_block:
                    flush_block()
                    current_block = []

        flush_block()

        # If no structure detected, treat the whole text as one requirement
        if not requirements and text.strip():
            first_line = text.strip().split("\n")[0][:200]
            requirements.append(
                ParsedRequirement(
                    id=str(uuid.uuid4()),
                    title=first_line,
                    description=text.strip(),
                    source_type=RequirementSource.TEXT.value,
                    raw_text=text,
                )
            )

        return requirements

    def _parse_jira_heuristic(self, issues: list[dict]) -> list[ParsedRequirement]:
        """Convert Jira issues to requirements without AI."""
        requirements: list[ParsedRequirement] = []
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            key = issue.get("key", "")
            summary = issue.get("summary", "")
            description = issue.get("description", summary)
            ac_raw = issue.get("acceptance_criteria", "")
            if isinstance(ac_raw, str):
                ac = [line.strip() for line in ac_raw.split("\n") if line.strip()]
            elif isinstance(ac_raw, list):
                ac = [str(a) for a in ac_raw]
            else:
                ac = []
            priority = self._normalize_priority(issue.get("priority", "medium"))
            labels = issue.get("labels", [])
            if isinstance(labels, str):
                labels = [labels]

            requirements.append(
                ParsedRequirement(
                    id=str(uuid.uuid4()),
                    title=summary[:200] or f"Jira issue {key}",
                    description=str(description),
                    acceptance_criteria=ac,
                    priority=priority,
                    tags=[str(l).lower() for l in labels],
                    source_ref=key or None,
                    source_type=RequirementSource.JIRA.value,
                    raw_text=json.dumps(issue, default=str),
                )
            )
        return requirements

    def _parse_github_pr_heuristic(self, pr_data: dict) -> list[ParsedRequirement]:
        """Derive a single requirement from a GitHub PR without AI."""
        title = pr_data.get("title", "Untitled PR")
        body = pr_data.get("body", "")
        pr_number = str(pr_data.get("number", ""))
        changed_files = pr_data.get("changed_files", [])

        tags: list[str] = []
        for f in changed_files:
            f_str = str(f).lower()
            if "test" in f_str:
                tags.append("test")
            if "api" in f_str or "endpoint" in f_str:
                tags.append("api")
            if "auth" in f_str:
                tags.append("auth")
            if "ui" in f_str or "component" in f_str:
                tags.append("ui")
        tags = list(set(tags))

        return [
            ParsedRequirement(
                id=str(uuid.uuid4()),
                title=title[:200],
                description=body or title,
                tags=tags,
                source_ref=pr_number or None,
                source_type=RequirementSource.GITHUB_PR.value,
                raw_text=f"PR #{pr_number}: {title}\n\n{body}",
            )
        ]

    def _parse_figma_heuristic(self, screens: list[dict]) -> list[ParsedRequirement]:
        """Convert Figma screens to requirements without AI."""
        requirements: list[ParsedRequirement] = []
        for screen in screens:
            if not isinstance(screen, dict):
                continue
            name = screen.get("name", "Unnamed Screen")
            annotations = screen.get("annotations", [])
            if isinstance(annotations, str):
                annotations = [annotations]

            for annotation in annotations:
                annotation_str = str(annotation).strip()
                if not annotation_str:
                    continue
                requirements.append(
                    ParsedRequirement(
                        id=str(uuid.uuid4()),
                        title=f"{name}: {annotation_str[:150]}",
                        description=f"Screen '{name}' - {annotation_str}",
                        tags=["ui", "figma"],
                        source_type=RequirementSource.FIGMA.value,
                        raw_text=annotation_str,
                    )
                )

        return requirements

    def _parse_codebase_heuristic(self, codebase_data: dict) -> list[ParsedRequirement]:
        """Derive requirements from codebase analysis without AI."""
        requirements: list[ParsedRequirement] = []

        for endpoint in codebase_data.get("endpoints", []):
            ep_str = str(endpoint) if not isinstance(endpoint, dict) else ""
            if isinstance(endpoint, dict):
                method = endpoint.get("method", "GET")
                path = endpoint.get("path", endpoint.get("url", "unknown"))
                desc = endpoint.get("description", f"{method} {path}")
                ep_str = f"{method} {path}"
            else:
                desc = ep_str
                method = "GET"
                path = ep_str

            requirements.append(
                ParsedRequirement(
                    id=str(uuid.uuid4()),
                    title=f"API: {ep_str[:180]}",
                    description=str(desc),
                    tags=["api"],
                    source_type=RequirementSource.CODEBASE.value,
                    raw_text=str(endpoint),
                )
            )

        for component in codebase_data.get("components", []):
            if isinstance(component, dict):
                name = component.get("name", "UnknownComponent")
                desc = component.get("description", f"Component: {name}")
            else:
                name = str(component)
                desc = name

            requirements.append(
                ParsedRequirement(
                    id=str(uuid.uuid4()),
                    title=f"Component: {name[:180]}",
                    description=str(desc),
                    tags=["ui", "component"],
                    source_type=RequirementSource.CODEBASE.value,
                    raw_text=str(component),
                )
            )

        for func in codebase_data.get("functions", []):
            if isinstance(func, dict):
                name = func.get("name", "unknown_function")
                desc = func.get("description", f"Function: {name}")
            else:
                name = str(func)
                desc = name

            requirements.append(
                ParsedRequirement(
                    id=str(uuid.uuid4()),
                    title=f"Function: {name[:180]}",
                    description=str(desc),
                    tags=["function"],
                    source_type=RequirementSource.CODEBASE.value,
                    raw_text=str(func),
                )
            )

        return requirements

    # ------------------------------------------------------------------
    # Heuristic ambiguity detection
    # ------------------------------------------------------------------

    # Words / phrases that indicate vagueness
    _VAGUE_PATTERNS: list[tuple[str, str]] = [
        (r"\bshould work\b", "\"should work\" - specify concrete success criteria"),
        (r"\bproperly\b", "\"properly\" - define what proper behavior looks like"),
        (r"\bcorrectly\b", "\"correctly\" - specify exact expected output"),
        (r"\bfast\b", "\"fast\" - specify a concrete latency target (e.g. <200ms)"),
        (r"\bquickly\b", "\"quickly\" - specify a concrete time limit"),
        (r"\buser[- ]friendly\b", "\"user-friendly\" - list specific UX criteria"),
        (r"\bintuitive\b", "\"intuitive\" - describe expected user flow"),
        (r"\bseamless\b", "\"seamless\" - specify what seamless means in this context"),
        (r"\breasonable\b", "\"reasonable\" - define acceptable bounds"),
        (r"\bappropriate\b", "\"appropriate\" - specify exact expected behavior"),
        (r"\betc\.?\b", "\"etc.\" - enumerate all cases explicitly"),
        (r"\band so on\b", "\"and so on\" - enumerate all cases explicitly"),
        (r"\bsimple\b", "\"simple\" - define the specific constraints"),
        (r"\beasy\b", "\"easy\" - describe the concrete user flow"),
        (r"\bgood\b", "\"good\" - define measurable quality criteria"),
        (r"\bnice\b", "\"nice\" - specify the desired visual/behavioral outcome"),
    ]

    def _analyze_ambiguity_heuristic(
        self,
        requirements: list[ParsedRequirement],
    ) -> list[ParsedRequirement]:
        """Apply keyword-based ambiguity heuristic to all requirements."""
        for req in requirements:
            self._apply_heuristic_ambiguity(req)
        return requirements

    def _apply_heuristic_ambiguity(self, req: ParsedRequirement) -> None:
        """Apply heuristic ambiguity scoring to a single requirement."""
        full_text = f"{req.title} {req.description} {' '.join(req.acceptance_criteria)}"
        full_lower = full_text.lower()
        details: list[dict] = []

        # Check for vague terms
        for pattern, suggestion in self._VAGUE_PATTERNS:
            if re.search(pattern, full_lower):
                details.append(
                    AmbiguityDetail(
                        type=AmbiguityType.VAGUE_TERM.value,
                        description=f"Contains vague language: {suggestion.split(' - ')[0]}",
                        suggestion=suggestion,
                    ).to_dict()
                )

        # Check for missing edge cases
        error_keywords = ["error", "fail", "invalid", "empty", "null", "timeout", "retry"]
        has_error_handling = any(kw in full_lower for kw in error_keywords)
        if not has_error_handling:
            details.append(
                AmbiguityDetail(
                    type=AmbiguityType.MISSING_EDGE_CASE.value,
                    description="No error/failure scenarios described",
                    suggestion="Add acceptance criteria for error states, invalid inputs, and edge cases",
                ).to_dict()
            )

        # Check for missing acceptance criteria
        if not req.acceptance_criteria:
            details.append(
                AmbiguityDetail(
                    type=AmbiguityType.INCOMPLETE_SPEC.value,
                    description="No acceptance criteria defined",
                    suggestion="Add specific, verifiable acceptance criteria",
                ).to_dict()
            )

        # Check for implicit assumptions about auth
        auth_keywords = ["login", "auth", "session", "token", "permission", "role"]
        has_auth_mention = any(kw in full_lower for kw in auth_keywords)
        action_keywords = ["create", "update", "delete", "edit", "submit", "save", "add", "remove"]
        has_action = any(kw in full_lower for kw in action_keywords)
        if has_action and not has_auth_mention:
            details.append(
                AmbiguityDetail(
                    type=AmbiguityType.IMPLICIT_ASSUMPTION.value,
                    description="Describes data-mutating actions without mentioning authentication/authorization",
                    suggestion="Specify who can perform this action and what authentication is required",
                ).to_dict()
            )

        # Check for missing boundary conditions
        numeric_keywords = ["max", "min", "limit", "range", "size", "length", "count"]
        has_boundaries = any(kw in full_lower for kw in numeric_keywords)
        input_keywords = ["input", "field", "form", "enter", "type", "upload"]
        has_input = any(kw in full_lower for kw in input_keywords)
        if has_input and not has_boundaries:
            details.append(
                AmbiguityDetail(
                    type=AmbiguityType.MISSING_BOUNDARY.value,
                    description="Describes user input without specifying limits or validation rules",
                    suggestion="Define input constraints: max length, allowed characters, required format",
                ).to_dict()
            )

        # Compute score: 0.0 (no issues) to 1.0 (many issues)
        # Each issue contributes ~0.15, capped at 1.0
        score = min(1.0, len(details) * 0.15)
        req.ambiguity_score = round(score, 2)
        req.ambiguity_details = details
