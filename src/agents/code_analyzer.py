"""Code Analyzer Agent - Analyzes codebases to identify testable surfaces.

This agent scans codebases to identify:
- User-facing pages and routes
- API endpoints
- Authentication flows
- Database models
- Critical user journeys
"""

from dataclasses import dataclass
from pathlib import Path

from ..core.model_router import TaskType
from .base import AgentResult, BaseAgent
from .prompts import get_enhanced_prompt


@dataclass
class TestableSurface:
    """A surface in the application that can be tested."""

    type: str  # "ui", "api", "db"
    name: str
    path: str
    priority: str  # "critical", "high", "medium", "low"
    description: str
    test_scenarios: list[str]
    metadata: dict = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "name": self.name,
            "path": self.path,
            "priority": self.priority,
            "description": self.description,
            "test_scenarios": self.test_scenarios,
            "metadata": self.metadata or {},
        }


@dataclass
class CodeAnalysisResult:
    """Result from code analysis."""

    summary: str
    testable_surfaces: list[TestableSurface]
    framework_detected: str | None = None
    language: str | None = None
    recommendations: list[str] = None


class CodeAnalyzerAgent(BaseAgent):
    """Agent that analyzes codebases to identify testable surfaces.

    Uses multi-model AI to understand:
    - Application structure and routing
    - API endpoint definitions
    - Authentication/authorization patterns
    - Database models and relationships
    - Critical user flows

    Can use cheaper models (DeepSeek, GPT-4o) for code analysis tasks.
    """

    # Code analysis can use cost-effective models like DeepSeek
    DEFAULT_TASK_TYPE = TaskType.CODE_ANALYSIS

    def _get_system_prompt(self) -> str:
        """Get enhanced system prompt for code analysis."""
        # Use enhanced prompt from prompts.py for production-grade analysis
        enhanced = get_enhanced_prompt("code_analyzer")
        if enhanced:
            return enhanced

        # Fallback to basic prompt if enhanced not available
        return """You are an expert code analyzer specializing in identifying testable surfaces in web applications.

Your task is to analyze codebases and identify:
1. User-facing pages/routes that need UI testing
2. API endpoints that need integration testing
3. Database operations that need data validation
4. Authentication and authorization flows
5. Critical user journeys (signup, checkout, etc.)

For each testable surface, assess:
- Priority based on user impact and business criticality
- Specific test scenarios including happy path and edge cases
- Dependencies and preconditions

Be thorough but focused on what's actually testable via E2E tests.
Output must be valid JSON."""

    async def execute(
        self,
        codebase_path: str,
        app_url: str,
        changed_files: list[str] | None = None,
        file_contents: dict[str, str] | None = None,
    ) -> AgentResult[CodeAnalysisResult]:
        """Analyze a codebase to identify testable surfaces.

        Args:
            codebase_path: Path to the codebase root
            app_url: URL where the app is running
            changed_files: Optional list of changed files to prioritize
            file_contents: Optional dict of file paths to contents for analysis

        Returns:
            AgentResult containing CodeAnalysisResult
        """
        self.log.info(
            "Starting code analysis",
            codebase_path=codebase_path,
            app_url=app_url,
            changed_files_count=len(changed_files) if changed_files else 0,
        )

        if not self._check_cost_limit():
            return AgentResult(
                success=False,
                error="Cost limit exceeded before analysis",
            )

        # Build analysis prompt
        prompt = self._build_analysis_prompt(
            codebase_path, app_url, changed_files, file_contents
        )

        try:
            response = self._call_claude(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
            )

            # Parse response
            content = self._extract_text_response(response)
            result_data = self._parse_json_response(content)

            if not result_data:
                return AgentResult(
                    success=False,
                    error="Failed to parse analysis response",
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                )

            # Build result
            surfaces = []
            for surface_data in result_data.get("testable_surfaces", []):
                surfaces.append(
                    TestableSurface(
                        type=surface_data.get("type", "ui"),
                        name=surface_data.get("name", "Unknown"),
                        path=surface_data.get("path", "/"),
                        priority=surface_data.get("priority", "medium"),
                        description=surface_data.get("description", ""),
                        test_scenarios=surface_data.get("test_scenarios", []),
                        metadata=surface_data.get("metadata"),
                    )
                )

            analysis_result = CodeAnalysisResult(
                summary=result_data.get("summary", ""),
                testable_surfaces=surfaces,
                framework_detected=result_data.get("framework"),
                language=result_data.get("language"),
                recommendations=result_data.get("recommendations", []),
            )

            self.log.info(
                "Code analysis complete",
                surfaces_found=len(surfaces),
                framework=analysis_result.framework_detected,
            )

            return AgentResult(
                success=True,
                data=analysis_result,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cost=self._usage.total_cost,
            )

        except Exception as e:
            self.log.error("Code analysis failed", error=str(e))
            return AgentResult(
                success=False,
                error=f"Analysis failed: {str(e)}",
            )

    def _build_analysis_prompt(
        self,
        codebase_path: str,
        app_url: str,
        changed_files: list[str] | None,
        file_contents: dict[str, str] | None,
    ) -> str:
        """Build the analysis prompt."""
        prompt_parts = [
            "Analyze this codebase and identify all testable surfaces.",
            "",
            f"CODEBASE PATH: {codebase_path}",
            f"APP URL: {app_url}",
        ]

        if changed_files:
            prompt_parts.append("")
            prompt_parts.append("CHANGED FILES (prioritize testing these):")
            for f in changed_files[:20]:  # Limit to avoid token overflow
                prompt_parts.append(f"  - {f}")

        if file_contents:
            prompt_parts.append("")
            prompt_parts.append("FILE CONTENTS FOR ANALYSIS:")
            for path, content in list(file_contents.items())[:10]:
                # Truncate large files
                truncated = content[:2000] + "..." if len(content) > 2000 else content
                prompt_parts.append("")
                prompt_parts.append(f"=== {path} ===")
                prompt_parts.append(truncated)

        prompt_parts.extend([
            "",
            "Identify:",
            "1. User-facing pages/routes with their URLs",
            "2. API endpoints with methods and expected responses",
            "3. Authentication/authorization flows",
            "4. Critical user journeys (signup, login, checkout, etc.)",
            "5. Database operations that need validation",
            "",
            "Respond with JSON:",
            """{
    "summary": "Brief description of the application",
    "framework": "detected framework (React, Next.js, Django, etc.)",
    "language": "primary language",
    "testable_surfaces": [
        {
            "type": "ui|api|db",
            "name": "descriptive name",
            "path": "URL or endpoint path",
            "priority": "critical|high|medium|low",
            "description": "what this does",
            "test_scenarios": ["scenario 1", "scenario 2"],
            "metadata": {"optional": "extra info"}
        }
    ],
    "recommendations": ["testing recommendations"]
}""",
        ])

        return "\n".join(prompt_parts)

    async def analyze_with_file_access(
        self,
        codebase_path: str,
        app_url: str,
        patterns: list[str] | None = None,
    ) -> AgentResult[CodeAnalysisResult]:
        """Analyze codebase with direct file system access.

        This method reads relevant files from the codebase for deeper analysis.

        Args:
            codebase_path: Path to codebase root
            app_url: Application URL
            patterns: Glob patterns for files to analyze

        Returns:
            AgentResult with analysis
        """
        default_patterns = [
            "**/*.py",
            "**/*.ts",
            "**/*.tsx",
            "**/*.js",
            "**/*.jsx",
            "**/routes.*",
            "**/router.*",
            "**/api/**/*",
            "**/pages/**/*",
            "**/app/**/*",
        ]
        patterns = patterns or default_patterns

        codebase = Path(codebase_path)
        if not codebase.exists():
            return AgentResult(
                success=False,
                error=f"Codebase path does not exist: {codebase_path}",
            )

        # Collect relevant files
        file_contents = {}
        for pattern in patterns:
            for file_path in codebase.glob(pattern):
                if file_path.is_file() and not self._should_skip(file_path):
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        rel_path = str(file_path.relative_to(codebase))
                        file_contents[rel_path] = content
                    except Exception as e:
                        self.log.debug(f"Could not read {file_path}: {e}")

                # Limit total files to prevent token overflow
                if len(file_contents) >= 15:
                    break

        self.log.info(f"Collected {len(file_contents)} files for analysis")

        return await self.execute(
            codebase_path=codebase_path,
            app_url=app_url,
            file_contents=file_contents,
        )

    def _should_skip(self, path: Path) -> bool:
        """Check if a file should be skipped."""
        skip_dirs = {
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
        }
        skip_extensions = {".min.js", ".map", ".lock", ".svg", ".png", ".jpg"}

        for part in path.parts:
            if part in skip_dirs:
                return True

        for ext in skip_extensions:
            if path.name.endswith(ext):
                return True

        return False
