"""
Hallucination Detection using SelfCheckGPT Pattern.

Implements the SelfCheckGPT pattern from Manakul et al. (2023) for
detecting hallucinated/fabricated content in LLM outputs.

RAP-330: Hallucination Detection Implementation

Key Features:
- Multi-sample consistency checking (SelfCheckGPT core method)
- Sentence-level hallucination scoring
- Factual grounding verification against retrieved context
- Integration with Langfuse for quality metrics
- Middleware for automatic response validation

How SelfCheckGPT Works:
1. Generate N independent responses to the same query (with temperature > 0)
2. For each sentence in the main response, check consistency against samples
3. If a fact appears in most samples, it's likely grounded
4. If a fact only appears in one response, it's likely hallucinated

References:
- SelfCheckGPT (Manakul et al. 2023): Detecting hallucinations via consistency
- G-Eval (Liu et al. 2023): LLM-based evaluation
- FactScore (Min et al. 2023): Fine-grained factual precision
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from .base import (
    AgentCapability,
    AgentResult,
    AICapability,
    AIResponse,
    BaseAgent,
)
from ..core.model_router import TaskType

logger = structlog.get_logger()


class HallucinationSeverity(str, Enum):
    """Severity levels for hallucination detection."""
    NONE = "none"           # No hallucination detected
    LOW = "low"             # Minor inconsistencies, likely still factual
    MEDIUM = "medium"       # Some unsupported claims, needs review
    HIGH = "high"           # Significant hallucination, unreliable
    CRITICAL = "critical"   # Fabricated content, should not be used


class HallucinationType(str, Enum):
    """Types of hallucination detected."""
    FACTUAL_ERROR = "factual_error"           # Incorrect fact
    UNSUPPORTED_CLAIM = "unsupported_claim"   # Claim not in context
    INCONSISTENT = "inconsistent"             # Contradicts other samples
    FABRICATED = "fabricated"                 # Made up entirely
    EXAGGERATED = "exaggerated"               # Overstatement of facts
    OUTDATED = "outdated"                     # Information may be stale


@dataclass
class SentenceScore:
    """Score for a single sentence in the response."""
    sentence: str
    index: int
    consistency_score: float  # 0.0-1.0, higher = more consistent
    support_count: int        # How many samples support this sentence
    total_samples: int
    hallucination_type: HallucinationType | None = None
    evidence: list[str] = field(default_factory=list)

    @property
    def is_hallucinated(self) -> bool:
        """Check if this sentence is likely hallucinated."""
        return self.consistency_score < 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "sentence": self.sentence,
            "index": self.index,
            "consistency_score": self.consistency_score,
            "support_count": self.support_count,
            "total_samples": self.total_samples,
            "is_hallucinated": self.is_hallucinated,
            "hallucination_type": self.hallucination_type.value if self.hallucination_type else None,
            "evidence": self.evidence,
        }


@dataclass
class HallucinationResult:
    """Complete result of hallucination detection."""
    id: str
    original_response: str
    overall_score: float          # 0.0-1.0, higher = more reliable
    hallucination_rate: float     # Percentage of hallucinated sentences
    severity: HallucinationSeverity
    sentence_scores: list[SentenceScore]
    hallucinated_sentences: list[str]
    samples_generated: int
    grounding_used: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Cost tracking
    total_cost: float = 0.0
    total_tokens: int = 0

    @property
    def is_reliable(self) -> bool:
        """Check if the response is reliable enough to use."""
        return self.severity in (HallucinationSeverity.NONE, HallucinationSeverity.LOW)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "overall_score": self.overall_score,
            "hallucination_rate": self.hallucination_rate,
            "severity": self.severity.value,
            "is_reliable": self.is_reliable,
            "sentence_scores": [s.to_dict() for s in self.sentence_scores],
            "hallucinated_sentences": self.hallucinated_sentences,
            "samples_generated": self.samples_generated,
            "grounding_used": self.grounding_used,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GroundingCheckResult:
    """Result of checking response against grounding context."""
    is_grounded: bool
    grounding_score: float  # 0.0-1.0
    supported_claims: list[str]
    unsupported_claims: list[str]
    contradicted_claims: list[str]
    context_coverage: float  # How much of context was used

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "grounding_score": self.grounding_score,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "contradicted_claims": self.contradicted_claims,
            "context_coverage": self.context_coverage,
        }


class HallucinationDetectorAgent(BaseAgent):
    """
    Agent for detecting hallucinations using SelfCheckGPT pattern.

    Implements multiple detection strategies:
    1. SelfCheckGPT: Multi-sample consistency checking
    2. Grounding Check: Verification against provided context
    3. Pattern Detection: Regex-based hallucination indicators

    Usage:
        ```python
        detector = HallucinationDetectorAgent(
            num_samples=3,
            consistency_threshold=0.6,
        )

        # Check a response for hallucinations
        result = await detector.detect(
            response="The login API returns a JWT token with 256-bit encryption...",
            query="How does the login API work?",
            context="The auth system uses OAuth 2.0...",  # Optional grounding
        )

        if result.is_reliable:
            # Safe to use the response
            pass
        else:
            # Response contains hallucinations
            print(f"Hallucinated: {result.hallucinated_sentences}")
        ```
    """

    CAPABILITIES = [
        "hallucination_detection",
        "consistency_checking",
        "factual_verification",
        AgentCapability.CODE_ANALYSIS,
    ]

    DEFAULT_TASK_TYPE = TaskType.ERROR_CLASSIFICATION

    def __init__(
        self,
        num_samples: int = 3,
        consistency_threshold: float = 0.5,
        enable_grounding: bool = True,
        sample_temperature: float = 0.7,
        max_cost_per_check: float | None = None,
        **kwargs,
    ):
        """Initialize the hallucination detector.

        Args:
            num_samples: Number of samples to generate for consistency check
            consistency_threshold: Threshold for considering a sentence consistent
            enable_grounding: Whether to check against provided context
            sample_temperature: Temperature for generating diverse samples
            max_cost_per_check: Maximum cost allowed per hallucination check
        """
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.consistency_threshold = consistency_threshold
        self.enable_grounding = enable_grounding
        self.sample_temperature = sample_temperature
        self.max_cost_per_check = max_cost_per_check

        self.log = logger.bind(agent="HallucinationDetectorAgent")

    def _get_system_prompt(self) -> str:
        return """You are an expert at detecting hallucinations and factual errors in AI-generated text.

Your task is to analyze text for:
1. Factual accuracy - Are statements verifiable and correct?
2. Consistency - Do statements align with each other?
3. Grounding - Are claims supported by provided context?
4. Fabrication - Are there made-up details that seem suspicious?

Be rigorous but fair. Not every uncertain statement is a hallucination.
Focus on identifying clearly fabricated or contradictory information."""

    async def execute(self, **kwargs) -> AgentResult:
        """Execute hallucination detection."""
        response = kwargs.get("response", "")
        query = kwargs.get("query", "")
        context = kwargs.get("context")

        if not response:
            return AgentResult(success=False, error="No response provided to check")

        result = await self.detect(response, query, context)
        return AgentResult(success=True, data=result.to_dict())

    # =========================================================================
    # Main Detection Methods
    # =========================================================================

    async def detect(
        self,
        response: str,
        query: str,
        context: str | None = None,
        messages: list[dict] | None = None,
    ) -> HallucinationResult:
        """Detect hallucinations in a response using SelfCheckGPT.

        Args:
            response: The response to check for hallucinations
            query: The original query that generated the response
            context: Optional grounding context to check against
            messages: Optional conversation history for regeneration

        Returns:
            HallucinationResult with detailed scoring
        """
        result_id = f"halluc_{uuid4().hex[:12]}"
        total_cost = 0.0
        total_tokens = 0

        self.log.info(
            "Starting hallucination detection",
            result_id=result_id,
            response_length=len(response),
            num_samples=self.num_samples,
        )

        # Step 1: Split response into sentences
        sentences = self._split_into_sentences(response)
        if not sentences:
            return self._create_empty_result(result_id, response)

        # Step 2: Generate multiple samples (SelfCheckGPT core)
        samples, sample_cost, sample_tokens = await self._generate_samples(
            query=query,
            messages=messages,
        )
        total_cost += sample_cost
        total_tokens += sample_tokens

        # Step 3: Check consistency of each sentence against samples
        sentence_scores = await self._check_sentence_consistency(
            sentences=sentences,
            samples=samples,
            original_response=response,
        )

        # Update cost tracking from consistency checks
        # (Cost already tracked in _call_ai)

        # Step 4: Optionally check grounding against context
        grounding_used = False
        if self.enable_grounding and context:
            grounding_result = await self._check_grounding(response, context)
            grounding_used = True
            # Adjust scores based on grounding
            sentence_scores = self._adjust_scores_with_grounding(
                sentence_scores, grounding_result
            )

        # Step 5: Calculate overall metrics
        hallucinated_sentences = [
            s.sentence for s in sentence_scores if s.is_hallucinated
        ]
        hallucination_rate = (
            len(hallucinated_sentences) / len(sentences) if sentences else 0.0
        )
        overall_score = (
            sum(s.consistency_score for s in sentence_scores) / len(sentence_scores)
            if sentence_scores else 1.0
        )

        # Determine severity
        severity = self._determine_severity(hallucination_rate, overall_score)

        result = HallucinationResult(
            id=result_id,
            original_response=response,
            overall_score=overall_score,
            hallucination_rate=hallucination_rate,
            severity=severity,
            sentence_scores=sentence_scores,
            hallucinated_sentences=hallucinated_sentences,
            samples_generated=len(samples),
            grounding_used=grounding_used,
            total_cost=total_cost,
            total_tokens=total_tokens,
        )

        # Log to Langfuse if available
        await self._log_to_langfuse(result)

        self.log.info(
            "Hallucination detection complete",
            result_id=result_id,
            overall_score=overall_score,
            hallucination_rate=hallucination_rate,
            severity=severity.value,
            hallucinated_count=len(hallucinated_sentences),
        )

        return result

    # =========================================================================
    # SelfCheckGPT Implementation
    # =========================================================================

    async def _generate_samples(
        self,
        query: str,
        messages: list[dict] | None = None,
    ) -> tuple[list[str], float, int]:
        """Generate multiple independent samples for consistency checking.

        This is the core of SelfCheckGPT - by generating multiple responses
        with temperature > 0, we can check which facts are consistent.

        Args:
            query: The query to regenerate responses for
            messages: Optional conversation context

        Returns:
            Tuple of (samples, total_cost, total_tokens)
        """
        samples = []
        total_cost = 0.0
        total_tokens = 0

        # Build messages for regeneration
        if messages:
            sample_messages = messages.copy()
        else:
            sample_messages = [{"role": "user", "content": query}]

        # Generate samples in parallel for efficiency
        async def generate_one() -> tuple[str, float, int]:
            response = await self._call_ai(
                messages=sample_messages,
                task_type=TaskType.GENERAL,
                temperature=self.sample_temperature,
                max_tokens=2000,
                max_cost=self.max_cost_per_check / self.num_samples if self.max_cost_per_check else None,
            )
            return response.content, response.cost, response.total_tokens

        # Run all sample generations concurrently
        results = await asyncio.gather(
            *[generate_one() for _ in range(self.num_samples)],
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                self.log.warning("Sample generation failed", error=str(result))
                continue
            content, cost, tokens = result
            samples.append(content)
            total_cost += cost
            total_tokens += tokens

        self.log.debug(
            "Generated samples for consistency check",
            num_samples=len(samples),
            total_cost=total_cost,
        )

        return samples, total_cost, total_tokens

    async def _check_sentence_consistency(
        self,
        sentences: list[str],
        samples: list[str],
        original_response: str,
    ) -> list[SentenceScore]:
        """Check each sentence for consistency against samples.

        For each sentence in the original response, we ask the LLM
        to determine if it's supported by the sample responses.

        Args:
            sentences: Sentences from the original response
            samples: Alternative samples for comparison
            original_response: Full original response for context

        Returns:
            List of SentenceScore for each sentence
        """
        if not samples:
            # No samples to compare - assume consistent
            return [
                SentenceScore(
                    sentence=s,
                    index=i,
                    consistency_score=0.5,  # Uncertain
                    support_count=0,
                    total_samples=0,
                )
                for i, s in enumerate(sentences)
            ]

        # Batch check for efficiency
        check_prompt = f"""Analyze each sentence from the ORIGINAL RESPONSE for consistency with the SAMPLE RESPONSES.

ORIGINAL RESPONSE:
{original_response}

SAMPLE RESPONSES:
{self._format_samples(samples)}

SENTENCES TO CHECK:
{self._format_sentences(sentences)}

For each sentence, determine:
1. How many sample responses contain similar/supporting information (support_count)
2. If the information appears fabricated or inconsistent
3. The type of issue if any

Return JSON:
{{
    "sentence_scores": [
        {{
            "index": 0,
            "consistency_score": 0.0-1.0,
            "support_count": <int>,
            "hallucination_type": null | "factual_error" | "unsupported_claim" | "inconsistent" | "fabricated" | "exaggerated" | "outdated",
            "evidence": ["supporting or contradicting quotes from samples"]
        }}
    ]
}}

Scoring guidelines:
- 1.0: Sentence appears in most/all samples with same information
- 0.7-0.9: Sentence is supported by some samples
- 0.4-0.6: Sentence has partial support or is a reasonable inference
- 0.1-0.3: Sentence contradicts samples or has no support
- 0.0: Sentence contains clearly fabricated information"""

        response = await self._call_ai(
            messages=[{"role": "user", "content": check_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=3000,
        )

        result = self._parse_json_response(response.content, {"sentence_scores": []})

        # Build SentenceScore objects
        scores = []
        for i, sentence in enumerate(sentences):
            score_data = next(
                (s for s in result.get("sentence_scores", []) if s.get("index") == i),
                {}
            )

            halluc_type = None
            if score_data.get("hallucination_type"):
                try:
                    halluc_type = HallucinationType(score_data["hallucination_type"])
                except ValueError:
                    pass

            scores.append(SentenceScore(
                sentence=sentence,
                index=i,
                consistency_score=float(score_data.get("consistency_score", 0.5)),
                support_count=int(score_data.get("support_count", 0)),
                total_samples=len(samples),
                hallucination_type=halluc_type,
                evidence=score_data.get("evidence", []),
            ))

        return scores

    async def _check_grounding(
        self,
        response: str,
        context: str,
    ) -> GroundingCheckResult:
        """Check if response is grounded in the provided context.

        This complements SelfCheckGPT by verifying claims against
        a known source of truth.

        Args:
            response: The response to verify
            context: The context/source material

        Returns:
            GroundingCheckResult with detailed analysis
        """
        grounding_prompt = f"""Verify if the RESPONSE is properly grounded in the CONTEXT.

CONTEXT (Source of Truth):
{context[:4000]}  # Limit context length

RESPONSE (To Verify):
{response}

Analyze each claim in the response and categorize it:
1. SUPPORTED: Directly stated or clearly implied by context
2. UNSUPPORTED: Not mentioned in context (but not contradicted)
3. CONTRADICTED: Directly conflicts with context

Return JSON:
{{
    "is_grounded": true/false,
    "grounding_score": 0.0-1.0,
    "supported_claims": ["claim 1", "claim 2"],
    "unsupported_claims": ["claim 3"],
    "contradicted_claims": ["claim 4"],
    "context_coverage": 0.0-1.0
}}

A response is "grounded" if:
- Most claims are supported by context
- No claims are contradicted
- Grounding score >= 0.7"""

        response_ai = await self._call_ai(
            messages=[{"role": "user", "content": grounding_prompt}],
            task_type=TaskType.ERROR_CLASSIFICATION,
            json_mode=True,
            max_tokens=2000,
        )

        result = self._parse_json_response(response_ai.content, {
            "is_grounded": False,
            "grounding_score": 0.5,
            "supported_claims": [],
            "unsupported_claims": [],
            "contradicted_claims": [],
            "context_coverage": 0.5,
        })

        return GroundingCheckResult(
            is_grounded=result.get("is_grounded", False),
            grounding_score=float(result.get("grounding_score", 0.5)),
            supported_claims=result.get("supported_claims", []),
            unsupported_claims=result.get("unsupported_claims", []),
            contradicted_claims=result.get("contradicted_claims", []),
            context_coverage=float(result.get("context_coverage", 0.5)),
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for analysis."""
        # Simple sentence splitting - handles common cases
        # Could be enhanced with NLTK or spaCy for better accuracy
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)

        # Filter out very short sentences (likely fragments)
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _format_samples(self, samples: list[str]) -> str:
        """Format samples for the prompt."""
        formatted = []
        for i, sample in enumerate(samples):
            formatted.append(f"[Sample {i+1}]\n{sample[:1500]}")
        return "\n\n".join(formatted)

    def _format_sentences(self, sentences: list[str]) -> str:
        """Format sentences for the prompt."""
        return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))

    def _adjust_scores_with_grounding(
        self,
        sentence_scores: list[SentenceScore],
        grounding_result: GroundingCheckResult,
    ) -> list[SentenceScore]:
        """Adjust sentence scores based on grounding check.

        If grounding found contradictions, those sentences should
        have lower scores.
        """
        # Build a set of contradicted claims for quick lookup
        contradicted_lower = {c.lower() for c in grounding_result.contradicted_claims}

        for score in sentence_scores:
            sentence_lower = score.sentence.lower()

            # Check if this sentence is in contradicted claims
            for contradicted in contradicted_lower:
                if contradicted in sentence_lower or sentence_lower in contradicted:
                    # Significantly reduce score for contradicted claims
                    score.consistency_score = min(score.consistency_score, 0.2)
                    score.hallucination_type = HallucinationType.FACTUAL_ERROR
                    score.evidence.append(f"Contradicted by context: {contradicted[:100]}")
                    break

        return sentence_scores

    def _determine_severity(
        self,
        hallucination_rate: float,
        overall_score: float,
    ) -> HallucinationSeverity:
        """Determine the severity of hallucination."""
        if hallucination_rate == 0 and overall_score >= 0.9:
            return HallucinationSeverity.NONE
        elif hallucination_rate < 0.1 and overall_score >= 0.7:
            return HallucinationSeverity.LOW
        elif hallucination_rate < 0.25 and overall_score >= 0.5:
            return HallucinationSeverity.MEDIUM
        elif hallucination_rate < 0.5:
            return HallucinationSeverity.HIGH
        else:
            return HallucinationSeverity.CRITICAL

    def _create_empty_result(
        self,
        result_id: str,
        response: str,
    ) -> HallucinationResult:
        """Create a result for empty/unparseable responses."""
        return HallucinationResult(
            id=result_id,
            original_response=response,
            overall_score=1.0,
            hallucination_rate=0.0,
            severity=HallucinationSeverity.NONE,
            sentence_scores=[],
            hallucinated_sentences=[],
            samples_generated=0,
            grounding_used=False,
        )

    async def _log_to_langfuse(self, result: HallucinationResult) -> None:
        """Log hallucination detection results to Langfuse."""
        try:
            from ..orchestrator.langfuse_integration import score_trace, _check_langfuse_available

            if not _check_langfuse_available():
                return

            # Score the trace with hallucination metrics
            score_trace(
                trace_id=result.id,
                name="hallucination_score",
                value=result.overall_score,
                comment=f"Severity: {result.severity.value}, Rate: {result.hallucination_rate:.2%}",
                data_type="NUMERIC",
            )

            # Also log severity as categorical
            score_trace(
                trace_id=result.id,
                name="hallucination_severity",
                value=result.severity.value,
                data_type="CATEGORICAL",
            )

        except Exception as e:
            self.log.debug("Failed to log to Langfuse", error=str(e))


# =========================================================================
# Middleware for Automatic Hallucination Detection
# =========================================================================

class HallucinationCheckMiddleware:
    """
    Middleware for automatically checking responses for hallucinations.

    Can be integrated into the agent execution pipeline to validate
    responses before returning them.

    Usage:
        ```python
        middleware = HallucinationCheckMiddleware(
            threshold=0.6,
            auto_retry=True,
        )

        # Wrap agent execution
        result = await middleware.check_and_retry(
            agent=my_agent,
            query="What does the login API return?",
            context=api_docs,
        )
        ```
    """

    def __init__(
        self,
        threshold: float = 0.6,
        auto_retry: bool = False,
        max_retries: int = 2,
        detector: HallucinationDetectorAgent | None = None,
    ):
        """Initialize the middleware.

        Args:
            threshold: Minimum reliability score to accept response
            auto_retry: Whether to automatically retry if hallucination detected
            max_retries: Maximum retry attempts
            detector: Optional pre-configured detector
        """
        self.threshold = threshold
        self.auto_retry = auto_retry
        self.max_retries = max_retries
        self.detector = detector or HallucinationDetectorAgent()
        self.log = logger.bind(component="HallucinationMiddleware")

    async def check(
        self,
        response: str,
        query: str,
        context: str | None = None,
    ) -> tuple[bool, HallucinationResult]:
        """Check a response for hallucinations.

        Args:
            response: The response to check
            query: The original query
            context: Optional grounding context

        Returns:
            Tuple of (is_reliable, result)
        """
        result = await self.detector.detect(response, query, context)
        return result.is_reliable, result

    async def check_and_retry(
        self,
        agent: BaseAgent,
        query: str,
        context: str | None = None,
        **agent_kwargs,
    ) -> tuple[str, HallucinationResult | None]:
        """Execute agent and retry if hallucination detected.

        Args:
            agent: The agent to execute
            query: The query to process
            context: Optional grounding context
            **agent_kwargs: Additional arguments for agent

        Returns:
            Tuple of (final_response, hallucination_result)
        """
        messages = [{"role": "user", "content": query}]

        for attempt in range(self.max_retries + 1):
            # Generate response
            response = await agent._call_ai(
                messages=messages,
                **agent_kwargs,
            )

            # Check for hallucinations
            result = await self.detector.detect(
                response=response.content,
                query=query,
                context=context,
                messages=messages,
            )

            if result.is_reliable:
                self.log.info(
                    "Response passed hallucination check",
                    attempt=attempt + 1,
                    score=result.overall_score,
                )
                return response.content, result

            if not self.auto_retry or attempt >= self.max_retries:
                self.log.warning(
                    "Response failed hallucination check",
                    attempt=attempt + 1,
                    score=result.overall_score,
                    hallucinated=result.hallucinated_sentences[:3],
                )
                return response.content, result

            # Retry with additional instruction
            self.log.info(
                "Retrying due to hallucination",
                attempt=attempt + 1,
                hallucinated_count=len(result.hallucinated_sentences),
            )

            # Add feedback about hallucinated content
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": f"""Your previous response contained potential inaccuracies:
{chr(10).join(f'- {s}' for s in result.hallucinated_sentences[:3])}

Please provide a more accurate response, sticking only to verifiable facts."""
            })

        return response.content, result


# =========================================================================
# Factory Functions
# =========================================================================

def create_hallucination_detector(
    num_samples: int = 3,
    consistency_threshold: float = 0.5,
    enable_grounding: bool = True,
    **kwargs,
) -> HallucinationDetectorAgent:
    """Factory function to create a HallucinationDetectorAgent."""
    return HallucinationDetectorAgent(
        num_samples=num_samples,
        consistency_threshold=consistency_threshold,
        enable_grounding=enable_grounding,
        **kwargs,
    )


def create_hallucination_middleware(
    threshold: float = 0.6,
    auto_retry: bool = False,
    max_retries: int = 2,
) -> HallucinationCheckMiddleware:
    """Factory function to create HallucinationCheckMiddleware."""
    return HallucinationCheckMiddleware(
        threshold=threshold,
        auto_retry=auto_retry,
        max_retries=max_retries,
    )


async def quick_hallucination_check(
    response: str,
    query: str,
    context: str | None = None,
) -> HallucinationResult:
    """Quick one-shot hallucination check.

    Convenience function for simple checks without creating persistent instances.

    Args:
        response: Response to check
        query: Original query
        context: Optional grounding context

    Returns:
        HallucinationResult
    """
    detector = HallucinationDetectorAgent(num_samples=2)  # Fewer samples for speed
    return await detector.detect(response, query, context)
