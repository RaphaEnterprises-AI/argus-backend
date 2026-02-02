"""
Agent-as-Judge Pattern for Output Evaluation.

Implements the Agent-as-Judge pattern from research showing it outperforms
LLM-as-Judge by 15-30% through structured evaluation and multi-agent debate.

RAP-255: Agent-as-Judge + MetaJudge Implementation

Key Features:
- Structured output evaluation with criteria scoring
- Execution-based verification (actually run code)
- Multi-agent debate (Scorer → Critic → Commander)
- Ground truth comparison from historical data
- Learning from evaluations via Cognee

References:
- Agent-as-Judge (2024): Improved evaluation via agent architecture
- Multi-Agent Debate (2024): Consensus through structured disagreement
- Constitutional AI: Critique and revision patterns
"""

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from .base import AgentCapability, AgentResult, AICapability, BaseAgent
from ..core.model_router import TaskType

logger = structlog.get_logger()


class JudgeCapability:
    """Judge-specific capabilities for A2A discovery."""
    OUTPUT_EVALUATION = "output_evaluation"
    GROUND_TRUTH_VERIFICATION = "ground_truth_verification"
    TEST_VALIDATION = "test_validation"
    CODE_VERIFICATION = "code_verification"
    EXECUTION_BASED_EVAL = "execution_based_eval"
    MULTI_AGENT_DEBATE = "multi_agent_debate"
    INTERMEDIATE_FEEDBACK = "intermediate_feedback"


class IssueSeverity(str, Enum):
    """Severity levels for evaluation issues."""
    CRITICAL = "critical"  # Must fix, blocks acceptance
    MAJOR = "major"        # Should fix, significant problem
    MINOR = "minor"        # Nice to fix, minor issue
    INFO = "info"          # Informational, suggestion


class EvaluationCategory(str, Enum):
    """Categories of evaluation."""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    MAINTAINABILITY = "maintainability"
    EDGE_CASES = "edge_cases"
    TEST_COVERAGE = "test_coverage"
    HALLUCINATION = "hallucination"  # RAP-330: Added for hallucination detection


class VerificationMethod(str, Enum):
    """Methods used for verification."""
    LLM_ANALYSIS = "llm_analysis"
    EXECUTION = "execution"
    GROUND_TRUTH = "ground_truth"
    HYBRID = "hybrid"
    SELFCHECK_GPT = "selfcheck_gpt"  # RAP-330: SelfCheckGPT pattern


@dataclass
class EvaluationIssue:
    """A single issue identified during evaluation."""
    id: str
    severity: IssueSeverity
    category: EvaluationCategory
    description: str
    location: str | None = None  # Where in the output
    suggested_fix: str | None = None
    evidence: str | None = None
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "location": self.location,
            "suggested_fix": self.suggested_fix,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class AgentEvaluation:
    """Complete evaluation of an agent's output."""
    id: str
    agent_type: str
    overall_score: float  # 0.0-1.0
    criteria_scores: dict[str, float]
    issues: list[EvaluationIssue]
    suggestions: list[str]
    verification_method: VerificationMethod
    execution_result: dict[str, Any] | None = None
    ground_truth_match: float | None = None
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_critical_issues(self) -> bool:
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)

    @property
    def has_major_issues(self) -> bool:
        return any(i.severity == IssueSeverity.MAJOR for i in self.issues)

    @property
    def is_acceptable(self) -> bool:
        return not self.has_critical_issues and self.overall_score >= 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_type": self.agent_type,
            "overall_score": self.overall_score,
            "criteria_scores": self.criteria_scores,
            "issues": [i.to_dict() for i in self.issues],
            "suggestions": self.suggestions,
            "verification_method": self.verification_method.value,
            "execution_result": self.execution_result,
            "ground_truth_match": self.ground_truth_match,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "has_critical_issues": self.has_critical_issues,
            "is_acceptable": self.is_acceptable,
        }


@dataclass
class TestValidation:
    """Validation result for a generated test specification."""
    is_valid: bool
    syntactic_validity: float
    semantic_validity: float
    edge_case_coverage: float
    flakiness_risk: float
    issues: list[EvaluationIssue]
    suggested_improvements: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "syntactic_validity": self.syntactic_validity,
            "semantic_validity": self.semantic_validity,
            "edge_case_coverage": self.edge_case_coverage,
            "flakiness_risk": self.flakiness_risk,
            "issues": [i.to_dict() for i in self.issues],
            "suggested_improvements": self.suggested_improvements,
        }


@dataclass
class HealingValidation:
    """Validation result for a self-healing suggestion."""
    is_valid: bool
    addresses_root_cause: bool
    preserves_test_intent: bool
    has_side_effects: bool
    confidence: float
    alternative_suggestions: list[dict[str, Any]]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "addresses_root_cause": self.addresses_root_cause,
            "preserves_test_intent": self.preserves_test_intent,
            "has_side_effects": self.has_side_effects,
            "confidence": self.confidence,
            "alternative_suggestions": self.alternative_suggestions,
            "reasoning": self.reasoning,
        }


@dataclass
class ExecutionEvaluation:
    """Result of execution-based evaluation."""
    executed: bool
    success: bool
    stdout: str
    stderr: str
    return_code: int
    behavior_matches_expected: bool
    discrepancies: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "success": self.success,
            "stdout": self.stdout[:1000] if self.stdout else "",
            "stderr": self.stderr[:1000] if self.stderr else "",
            "return_code": self.return_code,
            "behavior_matches_expected": self.behavior_matches_expected,
            "discrepancies": self.discrepancies,
        }


@dataclass
class DebateRound:
    """A single round of multi-agent debate."""
    round_number: int
    scorer_evaluation: dict[str, Any]
    critic_challenges: list[str]
    resolution: str
    score_adjustment: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "scorer_evaluation": self.scorer_evaluation,
            "critic_challenges": self.critic_challenges,
            "resolution": self.resolution,
            "score_adjustment": self.score_adjustment,
        }


@dataclass
class MetaEvaluation:
    """Complete meta-evaluation with debate history."""
    original_evaluation: AgentEvaluation
    debate_rounds: list[DebateRound]
    final_score: float
    consensus_reached: bool
    key_insights: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_evaluation": self.original_evaluation.to_dict(),
            "debate_rounds": [r.to_dict() for r in self.debate_rounds],
            "final_score": self.final_score,
            "consensus_reached": self.consensus_reached,
            "key_insights": self.key_insights,
        }


class AgentAsJudge(BaseAgent):
    """
    Agent-as-Judge for evaluating other agents' outputs.

    Provides structured evaluation with:
    - Multi-criteria scoring
    - Issue identification and categorization
    - Execution-based verification
    - Ground truth comparison
    - Learning from historical evaluations

    Usage:
        ```python
        judge = AgentAsJudge()

        # Evaluate a self-healer's output
        evaluation = await judge.evaluate_agent_output(
            agent_type="SelfHealerAgent",
            input_context={"selector": "#old-btn", "error": "Not found"},
            output={"new_selector": "#new-btn", "confidence": 0.9},
        )

        if evaluation.is_acceptable:
            # Apply the fix
            pass
        ```
    """

    CAPABILITIES = [
        JudgeCapability.OUTPUT_EVALUATION,
        JudgeCapability.TEST_VALIDATION,
        JudgeCapability.CODE_VERIFICATION,
        JudgeCapability.EXECUTION_BASED_EVAL,
        AgentCapability.CODE_ANALYSIS,
    ]

    DEFAULT_TASK_TYPE = TaskType.ERROR_CLASSIFICATION

    # Default evaluation criteria by agent type
    DEFAULT_CRITERIA = {
        "SelfHealerAgent": ["correctness", "root_cause_addressed", "side_effects", "confidence"],
        "CodeAnalyzerAgent": ["accuracy", "completeness", "relevance", "actionability"],
        "TestPlannerAgent": ["coverage", "prioritization", "feasibility", "clarity"],
        "UITesterAgent": ["correctness", "reliability", "edge_cases", "assertions"],
        "APITesterAgent": ["correctness", "coverage", "schema_validation", "error_handling"],
        "default": ["correctness", "completeness", "clarity", "practicality"],
    }

    def __init__(
        self,
        enable_execution: bool = True,
        enable_ground_truth: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_execution = enable_execution
        self.enable_ground_truth = enable_ground_truth
        self.log = logger.bind(agent="AgentAsJudge")

    def _get_system_prompt(self) -> str:
        return """You are an expert evaluator (Agent-as-Judge) responsible for critically assessing the outputs of other AI agents.

Your evaluation must be:
1. RIGOROUS: Apply high standards consistently
2. SPECIFIC: Point to exact issues with evidence
3. ACTIONABLE: Provide clear suggestions for improvement
4. FAIR: Acknowledge strengths alongside weaknesses

When evaluating, consider:
- Correctness: Is the output factually accurate?
- Completeness: Does it fully address the task?
- Clarity: Is it well-structured and understandable?
- Safety: Are there any risks or side effects?
- Practicality: Can it be implemented as-is?

Be critical but constructive. Your goal is to improve agent outputs, not just reject them."""

    async def execute(self, **kwargs) -> AgentResult:
        """Execute evaluation based on provided context."""
        task_type = kwargs.get("task_type", "evaluate")

        if task_type == "evaluate":
            result = await self.evaluate_agent_output(
                agent_type=kwargs.get("agent_type", "unknown"),
                input_context=kwargs.get("input_context", {}),
                output=kwargs.get("output"),
                criteria=kwargs.get("criteria"),
            )
            return AgentResult(success=True, data=result.to_dict())

        elif task_type == "validate_test":
            result = await self.validate_generated_test(
                test_spec=kwargs.get("test_spec", {}),
                target_app_context=kwargs.get("target_app_context", {}),
            )
            return AgentResult(success=True, data=result.to_dict())

        elif task_type == "validate_healing":
            result = await self.verify_healing_suggestion(
                original_error=kwargs.get("original_error", ""),
                healing_suggestion=kwargs.get("healing_suggestion", {}),
                test_context=kwargs.get("test_context", {}),
            )
            return AgentResult(success=True, data=result.to_dict())

        return AgentResult(success=False, error=f"Unknown task type: {task_type}")

    # =========================================================================
    # Main Evaluation Methods
    # =========================================================================

    async def evaluate_agent_output(
        self,
        agent_type: str,
        input_context: dict[str, Any],
        output: Any,
        criteria: list[str] | None = None,
        ground_truth: Any = None,
    ) -> AgentEvaluation:
        """Evaluate an agent's output comprehensively.

        Args:
            agent_type: Type of agent that produced the output
            input_context: The input/task given to the agent
            output: The agent's output to evaluate
            criteria: Specific criteria to evaluate (optional)
            ground_truth: Known correct answer for comparison (optional)

        Returns:
            AgentEvaluation with scores, issues, and suggestions
        """
        eval_id = f"eval_{uuid4().hex[:12]}"
        criteria = criteria or self.DEFAULT_CRITERIA.get(agent_type, self.DEFAULT_CRITERIA["default"])

        self.log.info(
            "Evaluating agent output",
            eval_id=eval_id,
            agent_type=agent_type,
            criteria=criteria,
        )

        # Step 1: LLM-based evaluation
        llm_eval = await self._llm_evaluation(agent_type, input_context, output, criteria)

        # Step 2: Execution-based verification (if applicable)
        execution_result = None
        if self.enable_execution and self._is_executable_output(output):
            execution_result = await self._execution_verification(output)

        # Step 3: Ground truth comparison
        gt_match = None
        if self.enable_ground_truth:
            if ground_truth:
                gt_match = self._compare_to_ground_truth(output, ground_truth)
            else:
                # Try to find similar past evaluations
                gt_match = await self._query_historical_ground_truth(agent_type, input_context)

        # Step 4: Combine results
        verification_method = VerificationMethod.LLM_ANALYSIS
        if execution_result and gt_match:
            verification_method = VerificationMethod.HYBRID
        elif execution_result:
            verification_method = VerificationMethod.EXECUTION
        elif gt_match:
            verification_method = VerificationMethod.GROUND_TRUTH

        # Adjust score based on verification
        final_score = llm_eval["overall_score"]
        if execution_result and not execution_result.success:
            final_score = min(final_score, 0.5)
        if gt_match is not None:
            final_score = final_score * 0.7 + gt_match * 0.3

        evaluation = AgentEvaluation(
            id=eval_id,
            agent_type=agent_type,
            overall_score=final_score,
            criteria_scores=llm_eval["criteria_scores"],
            issues=llm_eval["issues"],
            suggestions=llm_eval["suggestions"],
            verification_method=verification_method,
            execution_result=execution_result.to_dict() if execution_result else None,
            ground_truth_match=gt_match,
            reasoning=llm_eval["reasoning"],
        )

        # Store for future learning
        await self._store_evaluation(evaluation, input_context, output)

        return evaluation

    async def validate_generated_test(
        self,
        test_spec: dict[str, Any],
        target_app_context: dict[str, Any],
    ) -> TestValidation:
        """Validate a generated test specification.

        Args:
            test_spec: The test specification to validate
            target_app_context: Context about the target application

        Returns:
            TestValidation with validity scores and issues
        """
        validation_prompt = f"""Validate this test specification for quality and correctness.

Test Specification:
{self._format_dict(test_spec)}

Target Application Context:
{self._format_dict(target_app_context)}

Evaluate:
1. Syntactic validity (0.0-1.0): Is the test well-formed?
2. Semantic validity (0.0-1.0): Does it test what it claims?
3. Edge case coverage (0.0-1.0): Are edge cases covered?
4. Flakiness risk (0.0-1.0): How likely is this test to be flaky?

Identify any issues with severity (critical/major/minor/info).

Return as JSON:
{{
    "is_valid": true/false,
    "syntactic_validity": 0.0-1.0,
    "semantic_validity": 0.0-1.0,
    "edge_case_coverage": 0.0-1.0,
    "flakiness_risk": 0.0-1.0,
    "issues": [
        {{"severity": "...", "category": "...", "description": "..."}}
    ],
    "suggested_improvements": ["..."]
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": validation_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=2000,
        )

        result = self._parse_json_response(response.content, {
            "is_valid": False,
            "syntactic_validity": 0.5,
            "semantic_validity": 0.5,
            "edge_case_coverage": 0.5,
            "flakiness_risk": 0.5,
            "issues": [],
            "suggested_improvements": [],
        })

        issues = [
            EvaluationIssue(
                id=f"issue_{i}",
                severity=IssueSeverity(issue.get("severity", "minor")),
                category=EvaluationCategory(issue.get("category", "correctness")),
                description=issue.get("description", ""),
            )
            for i, issue in enumerate(result.get("issues", []))
        ]

        return TestValidation(
            is_valid=result["is_valid"],
            syntactic_validity=result["syntactic_validity"],
            semantic_validity=result["semantic_validity"],
            edge_case_coverage=result["edge_case_coverage"],
            flakiness_risk=result["flakiness_risk"],
            issues=issues,
            suggested_improvements=result.get("suggested_improvements", []),
        )

    async def verify_healing_suggestion(
        self,
        original_error: str,
        healing_suggestion: dict[str, Any],
        test_context: dict[str, Any],
    ) -> HealingValidation:
        """Verify a self-healing suggestion is valid.

        Args:
            original_error: The original error that triggered healing
            healing_suggestion: The proposed fix
            test_context: Context about the test

        Returns:
            HealingValidation with validity assessment
        """
        verification_prompt = f"""Verify this self-healing suggestion.

Original Error:
{original_error}

Healing Suggestion:
{self._format_dict(healing_suggestion)}

Test Context:
{self._format_dict(test_context)}

Verify:
1. Does it address the root cause (not just symptoms)?
2. Does it preserve the original test intent?
3. Are there any side effects or risks?

Return as JSON:
{{
    "is_valid": true/false,
    "addresses_root_cause": true/false,
    "preserves_test_intent": true/false,
    "has_side_effects": true/false,
    "confidence": 0.0-1.0,
    "alternative_suggestions": [],
    "reasoning": "..."
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": verification_prompt}],
            task_type=TaskType.ROOT_CAUSE_ANALYSIS,
            json_mode=True,
            max_tokens=1500,
        )

        result = self._parse_json_response(response.content, {
            "is_valid": False,
            "addresses_root_cause": False,
            "preserves_test_intent": True,
            "has_side_effects": False,
            "confidence": 0.5,
            "alternative_suggestions": [],
            "reasoning": "Unable to parse verification",
        })

        return HealingValidation(
            is_valid=result["is_valid"],
            addresses_root_cause=result["addresses_root_cause"],
            preserves_test_intent=result["preserves_test_intent"],
            has_side_effects=result["has_side_effects"],
            confidence=result["confidence"],
            alternative_suggestions=result.get("alternative_suggestions", []),
            reasoning=result["reasoning"],
        )

    async def check_hallucination(
        self,
        response: str,
        query: str,
        context: str | None = None,
        num_samples: int = 3,
    ) -> dict[str, Any]:
        """Check a response for hallucinations using SelfCheckGPT.

        RAP-330: Integrates the HallucinationDetectorAgent for comprehensive
        hallucination detection.

        Args:
            response: The response to check
            query: The original query
            context: Optional grounding context
            num_samples: Number of samples for consistency check

        Returns:
            Dict with hallucination detection results

        Example:
            ```python
            judge = AgentAsJudge()
            result = await judge.check_hallucination(
                response="The API uses 256-bit encryption...",
                query="How does the API secure data?",
                context=api_docs,
            )

            if not result["is_reliable"]:
                print(f"Hallucinated: {result['hallucinated_sentences']}")
            ```
        """
        from .hallucination_detector import HallucinationDetectorAgent

        detector = HallucinationDetectorAgent(
            num_samples=num_samples,
            enable_grounding=context is not None,
        )

        halluc_result = await detector.detect(
            response=response,
            query=query,
            context=context,
        )

        return halluc_result.to_dict()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _llm_evaluation(
        self,
        agent_type: str,
        input_context: dict[str, Any],
        output: Any,
        criteria: list[str],
    ) -> dict[str, Any]:
        """Perform LLM-based evaluation."""
        eval_prompt = f"""Evaluate this {agent_type} output.

Input Context:
{self._format_dict(input_context)}

Output to Evaluate:
{self._format_output(output)}

Criteria to Evaluate: {', '.join(criteria)}

For each criterion, provide a score (0.0-1.0) and identify any issues.

Return as JSON:
{{
    "overall_score": 0.0-1.0,
    "criteria_scores": {{
        "{criteria[0]}": 0.0-1.0,
        ...
    }},
    "issues": [
        {{
            "severity": "critical|major|minor|info",
            "category": "correctness|completeness|clarity|efficiency|safety",
            "description": "...",
            "location": "...",
            "suggested_fix": "...",
            "evidence": "..."
        }}
    ],
    "suggestions": ["..."],
    "reasoning": "..."
}}"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": eval_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            required_capabilities=[AICapability.REASONING],
            json_mode=True,
            max_tokens=2500,
        )

        result = self._parse_json_response(response.content, {
            "overall_score": 0.5,
            "criteria_scores": {c: 0.5 for c in criteria},
            "issues": [],
            "suggestions": [],
            "reasoning": "Unable to parse evaluation",
        })

        # Convert issues to EvaluationIssue objects
        issues = []
        for i, issue_data in enumerate(result.get("issues", [])):
            try:
                issues.append(EvaluationIssue(
                    id=f"issue_{i}",
                    severity=IssueSeverity(issue_data.get("severity", "minor")),
                    category=EvaluationCategory(issue_data.get("category", "correctness")),
                    description=issue_data.get("description", ""),
                    location=issue_data.get("location"),
                    suggested_fix=issue_data.get("suggested_fix"),
                    evidence=issue_data.get("evidence"),
                ))
            except (ValueError, KeyError):
                continue

        result["issues"] = issues
        return result

    async def _execution_verification(self, output: Any) -> ExecutionEvaluation:
        """Execute code output and verify results."""
        code = self._extract_code(output)
        if not code:
            return ExecutionEvaluation(
                executed=False,
                success=False,
                stdout="",
                stderr="No executable code found",
                return_code=-1,
                behavior_matches_expected=False,
                discrepancies=["Could not extract executable code"],
            )

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=30,
            )

            return ExecutionEvaluation(
                executed=True,
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                behavior_matches_expected=result.returncode == 0,
                discrepancies=[] if result.returncode == 0 else [result.stderr],
            )

        except subprocess.TimeoutExpired:
            return ExecutionEvaluation(
                executed=True,
                success=False,
                stdout="",
                stderr="Execution timed out",
                return_code=-1,
                behavior_matches_expected=False,
                discrepancies=["Code execution timed out"],
            )
        except Exception as e:
            return ExecutionEvaluation(
                executed=False,
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                behavior_matches_expected=False,
                discrepancies=[str(e)],
            )

    def _compare_to_ground_truth(self, output: Any, ground_truth: Any) -> float:
        """Compare output to known ground truth."""
        if output == ground_truth:
            return 1.0

        # For dicts, compare key-value overlap
        if isinstance(output, dict) and isinstance(ground_truth, dict):
            if not ground_truth:
                return 0.0
            matching_keys = set(output.keys()) & set(ground_truth.keys())
            matching_values = sum(
                1 for k in matching_keys
                if output.get(k) == ground_truth.get(k)
            )
            return matching_values / len(ground_truth)

        # For strings, use simple similarity
        if isinstance(output, str) and isinstance(ground_truth, str):
            words_out = set(output.lower().split())
            words_gt = set(ground_truth.lower().split())
            if not words_gt:
                return 0.0
            return len(words_out & words_gt) / len(words_gt)

        return 0.0

    async def _query_historical_ground_truth(
        self,
        agent_type: str,
        input_context: dict[str, Any],
    ) -> float | None:
        """Query historical evaluations for ground truth."""
        try:
            from ..orchestrator.memory_manager import get_memory_manager
            memory = get_memory_manager()

            query = f"{agent_type} {str(input_context)[:100]}"
            results = await memory.search_episodes(
                query=query,
                event_types=["agent_evaluation"],
                outcome="success",
                limit=3,
            )

            if results:
                # Return average score from similar successful evaluations
                scores = [r.memory.details.get("score", 0.7) for r in results]
                return sum(scores) / len(scores)

            return None

        except Exception:
            return None

    async def _store_evaluation(
        self,
        evaluation: AgentEvaluation,
        input_context: dict[str, Any],
        output: Any,
    ) -> None:
        """Store evaluation for future learning."""
        try:
            from ..orchestrator.memory_manager import get_memory_manager, MemoryImportance
            memory = get_memory_manager()

            await memory.store_episode(
                event_type="agent_evaluation",
                summary=f"Evaluated {evaluation.agent_type} output: score {evaluation.overall_score:.2f}",
                details={
                    "eval_id": evaluation.id,
                    "agent_type": evaluation.agent_type,
                    "score": evaluation.overall_score,
                    "input_context_summary": str(input_context)[:500],
                    "output_summary": str(output)[:500],
                    "issues_count": len(evaluation.issues),
                },
                outcome="success" if evaluation.is_acceptable else "failure",
                importance=MemoryImportance.MEDIUM,
            )

        except Exception as e:
            self.log.warning("Failed to store evaluation", error=str(e))

    def _is_executable_output(self, output: Any) -> bool:
        """Check if output contains executable code."""
        if isinstance(output, str):
            return "def " in output or "import " in output or "class " in output
        if isinstance(output, dict):
            return "code" in output or "script" in output
        return False

    def _extract_code(self, output: Any) -> str | None:
        """Extract executable code from output."""
        if isinstance(output, str):
            # Check for code blocks
            if "```python" in output:
                parts = output.split("```python")
                if len(parts) > 1:
                    code = parts[1].split("```")[0]
                    return code.strip()
            return output if "def " in output or "import " in output else None

        if isinstance(output, dict):
            return output.get("code") or output.get("script")

        return None

    def _format_dict(self, d: dict[str, Any]) -> str:
        """Format dict for prompt."""
        import json
        return json.dumps(d, indent=2, default=str)[:2000]

    def _format_output(self, output: Any) -> str:
        """Format output for prompt."""
        if isinstance(output, dict):
            return self._format_dict(output)
        return str(output)[:2000]


class MetaJudge:
    """
    Meta-evaluator using multi-agent debate pattern.

    Implements structured debate between:
    - Scorer: Proposes initial evaluation
    - Critic: Challenges weaknesses
    - Commander: Resolves disputes

    Usage:
        ```python
        meta_judge = MetaJudge()
        meta_eval = await meta_judge.evaluate_with_debate(
            original_evaluation=evaluation,
            debate_rounds=2,
        )
        ```
    """

    def __init__(self, base_agent: BaseAgent | None = None):
        self.agent = base_agent or AgentAsJudge()
        self.log = logger.bind(component="MetaJudge")

    async def evaluate_with_debate(
        self,
        original_evaluation: AgentEvaluation,
        debate_rounds: int = 2,
    ) -> MetaEvaluation:
        """Run multi-agent debate to refine evaluation.

        Args:
            original_evaluation: Initial evaluation to debate
            debate_rounds: Number of debate rounds

        Returns:
            MetaEvaluation with debate history and final score
        """
        rounds: list[DebateRound] = []
        current_score = original_evaluation.overall_score
        key_insights: list[str] = []

        for round_num in range(1, debate_rounds + 1):
            self.log.debug(f"Starting debate round {round_num}")

            # Scorer presents evaluation
            scorer_eval = {
                "score": current_score,
                "reasoning": original_evaluation.reasoning,
                "issues": len(original_evaluation.issues),
            }

            # Critic challenges
            challenges = await self._critic_challenge(original_evaluation, current_score)

            # Commander resolves
            resolution, score_adjustment = await self._commander_resolve(
                original_evaluation,
                scorer_eval,
                challenges,
            )

            current_score = max(0.0, min(1.0, current_score + score_adjustment))

            rounds.append(DebateRound(
                round_number=round_num,
                scorer_evaluation=scorer_eval,
                critic_challenges=challenges,
                resolution=resolution,
                score_adjustment=score_adjustment,
            ))

            if resolution:
                key_insights.append(f"Round {round_num}: {resolution[:100]}")

        return MetaEvaluation(
            original_evaluation=original_evaluation,
            debate_rounds=rounds,
            final_score=current_score,
            consensus_reached=abs(rounds[-1].score_adjustment) < 0.05 if rounds else True,
            key_insights=key_insights,
        )

    async def _critic_challenge(
        self,
        evaluation: AgentEvaluation,
        current_score: float,
    ) -> list[str]:
        """Generate critic challenges to the evaluation."""
        critic_prompt = f"""As a devil's advocate, challenge this evaluation:

Score: {current_score:.2f}
Issues Found: {len(evaluation.issues)}
Reasoning: {evaluation.reasoning}

Identify potential problems:
1. Are there missed issues?
2. Is the scoring too lenient or harsh?
3. What counterarguments exist?

Return as JSON:
{{
    "challenges": ["challenge1", "challenge2", ...]
}}"""

        response = await self.agent._call_ai(
            messages=[{"role": "user", "content": critic_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=1000,
        )

        result = self.agent._parse_json_response(response.content, {"challenges": []})
        return result.get("challenges", [])

    async def _commander_resolve(
        self,
        evaluation: AgentEvaluation,
        scorer_eval: dict[str, Any],
        challenges: list[str],
    ) -> tuple[str, float]:
        """Commander resolves debate and adjusts score."""
        resolve_prompt = f"""As the commander, resolve this debate:

Scorer's Position:
Score: {scorer_eval['score']:.2f}
Reasoning: {scorer_eval['reasoning'][:200]}

Critic's Challenges:
{chr(10).join(f'- {c}' for c in challenges)}

Original Issues Found: {len(evaluation.issues)}

Determine:
1. Which challenges are valid?
2. Should the score be adjusted?
3. What is the final resolution?

Return as JSON:
{{
    "resolution": "Brief resolution statement",
    "score_adjustment": -0.1 to 0.1 (negative = harsher, positive = more lenient)
}}"""

        response = await self.agent._call_ai(
            messages=[{"role": "user", "content": resolve_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=800,
        )

        result = self.agent._parse_json_response(response.content, {
            "resolution": "No resolution reached",
            "score_adjustment": 0.0,
        })

        return result.get("resolution", ""), result.get("score_adjustment", 0.0)


# =========================================================================
# Factory Functions
# =========================================================================

def create_agent_judge(**kwargs) -> AgentAsJudge:
    """Factory function to create an AgentAsJudge."""
    return AgentAsJudge(**kwargs)


def create_meta_judge(base_agent: BaseAgent | None = None) -> MetaJudge:
    """Factory function to create a MetaJudge."""
    return MetaJudge(base_agent)


async def evaluate_agent_output_simple(
    agent_type: str,
    input_context: dict[str, Any],
    output: Any,
) -> AgentEvaluation:
    """Simple one-shot evaluation without persistent judge."""
    judge = AgentAsJudge(enable_execution=False, enable_ground_truth=False)
    return await judge.evaluate_agent_output(agent_type, input_context, output)
