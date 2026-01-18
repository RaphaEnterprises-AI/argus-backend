"""
Error-to-Code Correlator - Hybrid Approach

Correlates production errors with source code using:
1. ALGORITHMIC: File path matching, function lookup, git blame
2. LLM-POWERED: Semantic correlation, pattern discovery, root cause analysis

This is the intelligence layer that connects "what broke" to "why it broke".
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import structlog
from anthropic import AsyncAnthropic

from src.config import get_settings
from src.core.normalizer import NormalizedEvent, Severity

logger = structlog.get_logger()


class CorrelationType(str, Enum):
    """Types of correlations found."""
    DIRECT = "direct"           # Exact file:line match
    FUNCTION = "function"       # Same function, different line
    FILE = "file"               # Same file, different function
    MODULE = "module"           # Same module/directory
    SEMANTIC = "semantic"       # LLM-identified semantic relationship
    PATTERN = "pattern"         # Recurring error pattern
    TEMPORAL = "temporal"       # Time-based correlation (deployed together)


class ConfidenceLevel(str, Enum):
    """Confidence in correlation."""
    HIGH = "high"       # 90%+ certain
    MEDIUM = "medium"   # 60-90% certain
    LOW = "low"         # 30-60% certain
    SPECULATIVE = "speculative"  # <30%, needs verification


@dataclass
class CodeLocation:
    """A specific location in the codebase."""
    file_path: str
    function_name: str | None = None
    line_number: int | None = None
    line_end: int | None = None
    code_snippet: str | None = None

    # Git information
    last_modified: datetime | None = None
    last_author: str | None = None
    commit_sha: str | None = None

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "function_name": self.function_name,
            "line_number": self.line_number,
            "line_end": self.line_end,
            "code_snippet": self.code_snippet,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "last_author": self.last_author,
            "commit_sha": self.commit_sha,
        }


@dataclass
class Correlation:
    """A correlation between an error and code."""
    id: str
    event_id: str
    correlation_type: CorrelationType
    confidence: ConfidenceLevel
    confidence_score: float  # 0.0 - 1.0

    # What we found
    location: CodeLocation
    related_locations: list[CodeLocation] = field(default_factory=list)

    # Why we think this is related
    reason: str = ""
    evidence: list[str] = field(default_factory=list)

    # LLM-generated insights (if semantic)
    semantic_analysis: str | None = None
    suggested_fix: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "correlation_type": self.correlation_type.value,
            "confidence": self.confidence.value,
            "confidence_score": self.confidence_score,
            "location": self.location.to_dict(),
            "related_locations": [loc.to_dict() for loc in self.related_locations],
            "reason": self.reason,
            "evidence": self.evidence,
            "semantic_analysis": self.semantic_analysis,
            "suggested_fix": self.suggested_fix,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ErrorPattern:
    """A recurring error pattern across multiple events."""
    id: str
    name: str
    description: str

    # Pattern characteristics
    error_type: str | None = None
    affected_files: list[str] = field(default_factory=list)
    affected_components: list[str] = field(default_factory=list)

    # Occurrences
    event_ids: list[str] = field(default_factory=list)
    occurrence_count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    # Analysis
    root_cause: str | None = None
    recommended_fix: str | None = None
    severity: Severity = Severity.ERROR

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "error_type": self.error_type,
            "affected_files": self.affected_files,
            "affected_components": self.affected_components,
            "event_ids": self.event_ids,
            "occurrence_count": self.occurrence_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "root_cause": self.root_cause,
            "recommended_fix": self.recommended_fix,
            "severity": self.severity.value,
        }


class ErrorCorrelator:
    """
    Correlates errors to source code using hybrid algo + LLM approach.

    Algorithm handles:
    - Stack trace → file/line mapping
    - Function name matching
    - Module/directory grouping
    - Git blame lookups

    LLM handles:
    - Semantic correlation (understanding what code does)
    - Pattern discovery across errors
    - Root cause analysis
    - Fix suggestions
    """

    @property
    def SEMANTIC_MODEL(self) -> str:
        from src.core.model_registry import get_model_id
        return get_model_id("claude-sonnet-4-5")

    @property
    def FAST_MODEL(self) -> str:
        from src.core.model_registry import get_model_id
        return get_model_id("claude-haiku-4-5")

    def __init__(
        self,
        codebase_path: str | None = None,
        use_llm: bool = True,
    ):
        self.settings = get_settings()
        self.codebase_path = Path(codebase_path) if codebase_path else None
        self.use_llm = use_llm

        if use_llm:
            api_key = self.settings.anthropic_api_key
            if hasattr(api_key, 'get_secret_value'):
                api_key = api_key.get_secret_value()
            self.client = AsyncAnthropic(api_key=api_key)
        else:
            self.client = None

        self.log = logger.bind(component="error_correlator")

        # Cache for file contents and git info
        self._file_cache: dict[str, str] = {}
        self._git_cache: dict[str, dict] = {}

    # =========================================================================
    # ALGORITHMIC CORRELATION
    # =========================================================================

    async def correlate_event(
        self,
        event: NormalizedEvent,
        include_semantic: bool = True,
    ) -> list[Correlation]:
        """
        Find all correlations for an error event.

        Args:
            event: Normalized error event
            include_semantic: Whether to use LLM for semantic analysis

        Returns:
            List of correlations, sorted by confidence
        """
        correlations = []

        # Phase 1: Algorithmic correlation from stack trace
        if event.stack_frames:
            stack_correlations = await self._correlate_from_stack(event)
            correlations.extend(stack_correlations)

        # Phase 2: Algorithmic correlation from file path
        if event.file_path:
            file_correlation = await self._correlate_from_file(event)
            if file_correlation:
                correlations.append(file_correlation)

        # Phase 3: Component-based correlation
        if event.component:
            component_correlations = await self._correlate_from_component(event)
            correlations.extend(component_correlations)

        # Phase 4: LLM semantic correlation (if enabled)
        if include_semantic and self.use_llm and self.client:
            semantic_correlations = await self._correlate_semantic(event, correlations)
            correlations.extend(semantic_correlations)

        # Deduplicate and sort by confidence
        correlations = self._deduplicate_correlations(correlations)
        correlations.sort(key=lambda c: c.confidence_score, reverse=True)

        self.log.info(
            "Event correlated",
            event_id=event.id,
            correlation_count=len(correlations),
            top_confidence=correlations[0].confidence_score if correlations else 0,
        )

        return correlations

    async def _correlate_from_stack(self, event: NormalizedEvent) -> list[Correlation]:
        """Correlate using stack trace frames (ALGORITHMIC)."""
        import uuid
        correlations = []

        for i, frame in enumerate(event.stack_frames):
            if not frame.filename:
                continue

            # Skip non-app frames (node_modules, vendor, etc.)
            if not frame.in_app:
                continue
            if self._is_library_path(frame.filename):
                continue

            # Determine confidence based on position in stack
            # First frame (top of stack) is most likely the error location
            if i == 0:
                confidence = ConfidenceLevel.HIGH
                confidence_score = 0.95
                correlation_type = CorrelationType.DIRECT
            elif i < 3:
                confidence = ConfidenceLevel.MEDIUM
                confidence_score = 0.75 - (i * 0.1)
                correlation_type = CorrelationType.FUNCTION
            else:
                confidence = ConfidenceLevel.LOW
                confidence_score = 0.5 - (i * 0.05)
                correlation_type = CorrelationType.FILE

            # Try to get code context
            code_snippet = await self._get_code_snippet(
                frame.filename,
                frame.lineno,
                context_lines=3
            )

            # Try to get git info
            git_info = await self._get_git_blame(frame.filename, frame.lineno)

            location = CodeLocation(
                file_path=frame.filename,
                function_name=frame.function,
                line_number=frame.lineno,
                code_snippet=code_snippet,
                last_modified=git_info.get("date") if git_info else None,
                last_author=git_info.get("author") if git_info else None,
                commit_sha=git_info.get("commit") if git_info else None,
            )

            correlations.append(Correlation(
                id=str(uuid.uuid4()),
                event_id=event.id,
                correlation_type=correlation_type,
                confidence=confidence,
                confidence_score=confidence_score,
                location=location,
                reason=f"Stack frame #{i+1}: {frame.function or 'anonymous'} at line {frame.lineno}",
                evidence=[
                    f"Function: {frame.function}",
                    f"File: {frame.filename}:{frame.lineno}",
                    f"In-app code: {frame.in_app}",
                ],
            ))

        return correlations

    async def _correlate_from_file(self, event: NormalizedEvent) -> Correlation | None:
        """Correlate using explicit file path (ALGORITHMIC)."""
        import uuid

        if not event.file_path:
            return None

        code_snippet = await self._get_code_snippet(
            event.file_path,
            event.line_number,
            context_lines=5
        )

        git_info = await self._get_git_blame(event.file_path, event.line_number)

        location = CodeLocation(
            file_path=event.file_path,
            function_name=event.function_name,
            line_number=event.line_number,
            code_snippet=code_snippet,
            last_modified=git_info.get("date") if git_info else None,
            last_author=git_info.get("author") if git_info else None,
            commit_sha=git_info.get("commit") if git_info else None,
        )

        return Correlation(
            id=str(uuid.uuid4()),
            event_id=event.id,
            correlation_type=CorrelationType.DIRECT,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.98,
            location=location,
            reason=f"Direct file reference in error: {event.file_path}",
            evidence=[
                f"Error explicitly references {event.file_path}",
                f"Line number: {event.line_number}",
            ],
        )

    async def _correlate_from_component(self, event: NormalizedEvent) -> list[Correlation]:
        """Find files related to a UI component (ALGORITHMIC)."""
        import uuid
        correlations = []

        if not event.component or not self.codebase_path:
            return correlations

        component_name = event.component

        # Search patterns for common frameworks
        patterns = [
            f"**/{component_name}.tsx",
            f"**/{component_name}.jsx",
            f"**/{component_name}.vue",
            f"**/{component_name}.svelte",
            f"**/components/**/{component_name}.*",
            f"**/pages/**/{component_name}.*",
        ]

        found_files = []
        for pattern in patterns:
            matches = list(self.codebase_path.glob(pattern))
            found_files.extend(matches)

        for file_path in found_files[:5]:  # Limit to 5 matches
            relative_path = str(file_path.relative_to(self.codebase_path))

            correlations.append(Correlation(
                id=str(uuid.uuid4()),
                event_id=event.id,
                correlation_type=CorrelationType.MODULE,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.7,
                location=CodeLocation(file_path=relative_path),
                reason=f"Component '{component_name}' likely defined here",
                evidence=[
                    f"Component name: {component_name}",
                    "File matches component pattern",
                ],
            ))

        return correlations

    # =========================================================================
    # LLM-POWERED SEMANTIC CORRELATION
    # =========================================================================

    async def _correlate_semantic(
        self,
        event: NormalizedEvent,
        existing_correlations: list[Correlation],
    ) -> list[Correlation]:
        """
        Use LLM to find semantic correlations (LLM-POWERED).

        This goes beyond simple file matching to understand:
        - What the error actually means
        - What code is semantically related
        - Root cause analysis
        - Suggested fixes
        """
        import uuid

        if not self.client:
            return []

        # Gather context for LLM
        context_parts = [
            "ERROR DETAILS:",
            f"- Type: {event.error_type or 'Unknown'}",
            f"- Title: {event.title}",
            f"- Message: {event.message or 'No message'}",
            f"- Severity: {event.severity.value}",
            f"- URL: {event.url or 'Unknown'}",
            f"- Component: {event.component or 'Unknown'}",
        ]

        if event.raw_stack_trace:
            context_parts.append(f"\nSTACK TRACE:\n{event.raw_stack_trace[:2000]}")

        # Add code snippets from existing correlations
        if existing_correlations:
            context_parts.append("\nRELATED CODE LOCATIONS:")
            for corr in existing_correlations[:3]:
                if corr.location.code_snippet:
                    context_parts.append(
                        f"\n{corr.location.file_path}:{corr.location.line_number}\n"
                        f"```\n{corr.location.code_snippet}\n```"
                    )

        prompt = f"""{chr(10).join(context_parts)}

Analyze this error and provide:
1. ROOT CAUSE: What is the most likely root cause?
2. SEMANTIC CONNECTIONS: What other parts of the codebase might be involved?
3. FIX SUGGESTION: How should this be fixed?

Output as JSON:
{{
    "root_cause": "Brief explanation of root cause",
    "semantic_connections": [
        {{"area": "description", "reason": "why related", "confidence": 0.0-1.0}}
    ],
    "suggested_fix": "Specific fix recommendation",
    "additional_files_to_check": ["file patterns to investigate"]
}}"""

        try:
            response = await self.client.messages.create(
                model=self.FAST_MODEL,  # Use fast model for correlation
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start >= 0:
                analysis = json.loads(text[json_start:json_end])

                correlations = []

                # Create semantic correlations from LLM response
                for connection in analysis.get("semantic_connections", []):
                    correlations.append(Correlation(
                        id=str(uuid.uuid4()),
                        event_id=event.id,
                        correlation_type=CorrelationType.SEMANTIC,
                        confidence=ConfidenceLevel.MEDIUM if connection.get("confidence", 0.5) > 0.6 else ConfidenceLevel.LOW,
                        confidence_score=connection.get("confidence", 0.5),
                        location=CodeLocation(
                            file_path=connection.get("area", "Unknown"),
                        ),
                        reason=connection.get("reason", "Semantic relationship identified by AI"),
                        semantic_analysis=analysis.get("root_cause"),
                        suggested_fix=analysis.get("suggested_fix"),
                    ))

                return correlations

        except Exception as e:
            self.log.warning("Semantic correlation failed", error=str(e))

        return []

    # =========================================================================
    # PATTERN DETECTION (HYBRID)
    # =========================================================================

    async def detect_patterns(
        self,
        events: list[NormalizedEvent],
        min_occurrences: int = 3,
    ) -> list[ErrorPattern]:
        """
        Detect recurring error patterns across multiple events.

        Uses:
        - ALGORITHMIC: Fingerprint grouping, file clustering
        - LLM: Pattern naming, root cause synthesis
        """
        import uuid
        from collections import defaultdict

        # Group by fingerprint (algorithmic)
        fingerprint_groups: dict[str, list[NormalizedEvent]] = defaultdict(list)
        for event in events:
            fingerprint_groups[event.fingerprint].append(event)

        # Group by error type (algorithmic)
        type_groups: dict[str, list[NormalizedEvent]] = defaultdict(list)
        for event in events:
            if event.error_type:
                type_groups[event.error_type].append(event)

        # Group by file (algorithmic)
        file_groups: dict[str, list[NormalizedEvent]] = defaultdict(list)
        for event in events:
            if event.file_path:
                file_groups[event.file_path].append(event)

        patterns = []

        # Create patterns from fingerprint groups
        for fingerprint, group_events in fingerprint_groups.items():
            if len(group_events) >= min_occurrences:
                sample_event = group_events[0]

                pattern = ErrorPattern(
                    id=str(uuid.uuid4()),
                    name=f"{sample_event.error_type or 'Error'}: {sample_event.title[:50]}",
                    description=sample_event.message or sample_event.title,
                    error_type=sample_event.error_type,
                    affected_files=list(set(e.file_path for e in group_events if e.file_path)),
                    affected_components=list(set(e.component for e in group_events if e.component)),
                    event_ids=[e.id for e in group_events],
                    occurrence_count=len(group_events),
                    first_seen=min(e.created_at for e in group_events),
                    last_seen=max(e.created_at for e in group_events),
                    severity=max((e.severity for e in group_events), key=lambda s: ["info", "warning", "error", "fatal"].index(s.value)),
                )

                # Use LLM to enhance pattern analysis
                if self.use_llm and self.client:
                    pattern = await self._enhance_pattern_with_llm(pattern, group_events)

                patterns.append(pattern)

        # Sort by occurrence count
        patterns.sort(key=lambda p: p.occurrence_count, reverse=True)

        self.log.info(
            "Patterns detected",
            total_events=len(events),
            pattern_count=len(patterns),
        )

        return patterns

    async def _enhance_pattern_with_llm(
        self,
        pattern: ErrorPattern,
        events: list[NormalizedEvent],
    ) -> ErrorPattern:
        """Enhance pattern with LLM-generated insights."""
        if not self.client:
            return pattern

        # Sample events for context
        sample_events = events[:5]

        prompt = f"""Analyze this recurring error pattern:

PATTERN: {pattern.name}
OCCURRENCES: {pattern.occurrence_count}
AFFECTED FILES: {', '.join(pattern.affected_files[:5])}
AFFECTED COMPONENTS: {', '.join(pattern.affected_components[:5])}

SAMPLE ERROR MESSAGES:
{chr(10).join(f'- {e.message or e.title}' for e in sample_events)}

Provide:
1. A better name for this pattern (concise, descriptive)
2. Root cause analysis
3. Recommended fix

Output as JSON:
{{
    "name": "Better pattern name",
    "root_cause": "Root cause explanation",
    "recommended_fix": "How to fix this pattern"
}}"""

        try:
            response = await self.client.messages.create(
                model=self.FAST_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text
            json_start = text.find("{")
            json_end = text.rfind("}") + 1

            if json_start >= 0:
                analysis = json.loads(text[json_start:json_end])
                pattern.name = analysis.get("name", pattern.name)
                pattern.root_cause = analysis.get("root_cause")
                pattern.recommended_fix = analysis.get("recommended_fix")

        except Exception as e:
            self.log.warning("Pattern enhancement failed", error=str(e))

        return pattern

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _is_library_path(self, path: str) -> bool:
        """Check if path is a library/vendor path."""
        library_patterns = [
            "node_modules",
            "vendor",
            "site-packages",
            ".venv",
            "venv",
            "__pycache__",
            ".next",
            "dist",
            "build",
        ]
        return any(pattern in path for pattern in library_patterns)

    async def _get_code_snippet(
        self,
        file_path: str,
        line_number: int | None,
        context_lines: int = 3,
    ) -> str | None:
        """Get code snippet around a line number."""
        if not self.codebase_path or not line_number:
            return None

        full_path = self.codebase_path / file_path

        # Check cache
        cache_key = str(full_path)
        if cache_key not in self._file_cache:
            try:
                if full_path.exists():
                    self._file_cache[cache_key] = full_path.read_text()
                else:
                    return None
            except Exception:
                return None

        content = self._file_cache[cache_key]
        lines = content.split("\n")

        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)

        snippet_lines = []
        for i in range(start, end):
            marker = "→ " if i == line_number - 1 else "  "
            snippet_lines.append(f"{i+1:4d}{marker}{lines[i]}")

        return "\n".join(snippet_lines)

    async def _get_git_blame(
        self,
        file_path: str,
        line_number: int | None,
    ) -> dict | None:
        """Get git blame info for a file/line."""
        if not self.codebase_path or not line_number:
            return None

        cache_key = f"{file_path}:{line_number}"
        if cache_key in self._git_cache:
            return self._git_cache[cache_key]

        try:
            import subprocess

            result = subprocess.run(
                ["git", "blame", "-L", f"{line_number},{line_number}", "--porcelain", file_path],
                cwd=self.codebase_path,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                lines = result.stdout.split("\n")
                blame_info = {}

                for line in lines:
                    if line.startswith("author "):
                        blame_info["author"] = line[7:]
                    elif line.startswith("author-time "):
                        timestamp = int(line[12:])
                        blame_info["date"] = datetime.fromtimestamp(timestamp)
                    elif len(line) == 40 and all(c in "0123456789abcdef" for c in line):
                        blame_info["commit"] = line

                self._git_cache[cache_key] = blame_info
                return blame_info

        except Exception as e:
            self.log.debug("Git blame failed", file=file_path, error=str(e))

        return None

    def _deduplicate_correlations(self, correlations: list[Correlation]) -> list[Correlation]:
        """Remove duplicate correlations, keeping highest confidence."""
        seen = {}

        for corr in correlations:
            key = f"{corr.location.file_path}:{corr.location.line_number}"

            if key not in seen or corr.confidence_score > seen[key].confidence_score:
                seen[key] = corr

        return list(seen.values())
