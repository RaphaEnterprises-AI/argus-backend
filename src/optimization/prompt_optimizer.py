"""DSPy-style Prompt Optimization for Agent Prompts.

This module implements automatic prompt optimization inspired by DSPy's approach
of treating prompts as optimizable "weights". Instead of manually crafting prompts,
you define the input/output behavior (signature) and let optimizers find the best
prompt automatically.

Key Concepts:
- PromptSignature: Declarative definition of what a prompt should do
- Optimizers: Algorithms to find the best prompt (COPRO, MIPRO, SIMBA, GEPA)
- AgentPromptManager: Version control, A/B testing, and deployment

References:
- DSPy: https://github.com/stanfordnlp/dspy
- COPRO: Coordinate Ascent Prompt Optimization
- MIPROv2: Bayesian Multi-prompt Instruction Proposal Optimizer
- SIMBA: Stochastic Mini-Batch Analysis
- GEPA: Reflective Prompt Evolution
"""

import hashlib
import json
import random
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Core Types: PromptSignature and Related
# =============================================================================


@dataclass
class SignatureField:
    """Defines a single input or output field for a PromptSignature.

    Instead of writing prompt strings, you define the expected behavior:
    what inputs the prompt receives and what outputs it should produce.

    Attributes:
        name: Field name (e.g., "code", "analysis", "test_cases")
        description: Human-readable description of the field
        type: Expected type ("str", "list", "dict", "code", "json", etc.)
        required: Whether this field is required
        default: Default value if not provided
        constraints: Additional constraints on the field

    Example:
        ```python
        # Input field for source code
        code_input = SignatureField(
            name="code",
            description="Python source code to analyze",
            type="code",
            required=True,
        )

        # Output field for test cases
        tests_output = SignatureField(
            name="test_cases",
            description="Generated test cases",
            type="list",
            constraints=["Each test should have name, steps, assertions"],
        )
        ```
    """

    name: str
    description: str
    type: str = "str"
    required: bool = True
    default: Any = None
    constraints: list[str] = field(default_factory=list)

    def to_prompt_fragment(self) -> str:
        """Convert field to a prompt fragment for instruction generation."""
        type_hint = f" ({self.type})" if self.type != "str" else ""
        required_hint = " [required]" if self.required else " [optional]"
        constraints_hint = ""
        if self.constraints:
            constraints_hint = f"\n  Constraints: {', '.join(self.constraints)}"

        return f"- {self.name}{type_hint}{required_hint}: {self.description}{constraints_hint}"


@dataclass
class PromptSignature:
    """Declarative definition of prompt input/output behavior.

    Instead of writing raw prompt strings, define what the prompt should do.
    The optimizer will find the best prompt that satisfies this signature.

    Attributes:
        name: Unique identifier for this signature
        description: What this prompt accomplishes
        inputs: List of input fields the prompt receives
        outputs: List of output fields the prompt should produce
        constraints: Global constraints/rules to follow
        examples: Few-shot examples (input -> output)
        demos_k: Number of demos to include in compiled prompt

    Example:
        ```python
        signature = PromptSignature(
            name="code_analyzer",
            description="Analyze code for testable surfaces",
            inputs=[
                SignatureField(name="code", description="Source code", type="code"),
                SignatureField(name="framework", description="Framework name", type="str"),
            ],
            outputs=[
                SignatureField(name="surfaces", description="Testable surfaces", type="list"),
                SignatureField(name="priority", description="Priority ranking", type="list"),
            ],
            constraints=[
                "Be thorough but concise",
                "Focus on user-facing functionality",
                "Consider edge cases",
            ],
            examples=[
                {
                    "input": {"code": "def login(user, pwd): ...", "framework": "FastAPI"},
                    "output": {"surfaces": ["login endpoint"], "priority": ["high"]}
                }
            ],
        )
        ```
    """

    name: str
    description: str
    inputs: list[SignatureField]
    outputs: list[SignatureField]
    constraints: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    demos_k: int = 3

    def __post_init__(self):
        """Validate the signature."""
        if not self.inputs:
            raise ValueError("Signature must have at least one input field")
        if not self.outputs:
            raise ValueError("Signature must have at least one output field")

    def get_signature_hash(self) -> str:
        """Get a hash of the signature for versioning."""
        content = json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "inputs": [f.name for f in self.inputs],
                "outputs": [f.name for f in self.outputs],
                "constraints": self.constraints,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_base_prompt(self) -> str:
        """Generate a base prompt from the signature."""
        input_section = "\n".join(f.to_prompt_fragment() for f in self.inputs)
        output_section = "\n".join(f.to_prompt_fragment() for f in self.outputs)

        constraints_section = ""
        if self.constraints:
            constraints_section = "\n\n# Constraints\n" + "\n".join(
                f"- {c}" for c in self.constraints
            )

        return f"""# Task: {self.description}

# Inputs
{input_section}

# Expected Outputs
{output_section}
{constraints_section}

Analyze the inputs carefully and produce the expected outputs."""


@dataclass
class OptimizedPrompt:
    """Result of prompt optimization.

    Contains the optimized prompt along with metadata about
    the optimization process.

    Attributes:
        prompt: The optimized prompt string
        signature_hash: Hash of the source signature
        optimizer: Name of the optimizer used
        score: Final metric score
        iterations: Number of optimization iterations
        demos: Selected few-shot demonstrations
        metadata: Additional optimization metadata
    """

    prompt: str
    signature_hash: str
    optimizer: str
    score: float
    iterations: int
    demos: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "prompt": self.prompt,
            "signature_hash": self.signature_hash,
            "optimizer": self.optimizer,
            "score": self.score,
            "iterations": self.iterations,
            "demos": self.demos,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class OptimizationResult:
    """Complete result of an optimization run.

    Includes the optimized prompt, training history, and validation metrics.
    """

    optimized_prompt: OptimizedPrompt
    training_history: list[dict]
    validation_score: float
    best_iteration: int
    total_time_seconds: float
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "optimized_prompt": self.optimized_prompt.to_dict(),
            "training_history": self.training_history,
            "validation_score": self.validation_score,
            "best_iteration": self.best_iteration,
            "total_time_seconds": self.total_time_seconds,
            "total_cost_usd": self.total_cost_usd,
        }


@dataclass
class ABTestResult:
    """Result of an A/B test between prompt variants.

    Compares performance of two prompt variants on the same test cases.
    """

    variant_a: str
    variant_b: str
    variant_a_score: float
    variant_b_score: float
    winner: str
    confidence: float
    test_cases: int
    p_value: float
    details: dict = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running an agent on a benchmark.

    Tracks performance metrics for agent evaluation.
    """

    agent_type: str
    prompt_variant: str
    benchmark_name: str
    score: float
    accuracy: float
    latency_ms: float
    cost_usd: float
    test_cases: int
    passed: int
    failed: int
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Optimizers: COPRO, MIPRO, SIMBA, GEPA
# =============================================================================


class BaseOptimizer(ABC):
    """Base class for prompt optimizers.

    All optimizers follow the same interface:
    1. Take a signature and training data
    2. Iteratively improve the prompt
    3. Return the best prompt found

    Subclasses implement different optimization strategies.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        max_iterations: int = 20,
        patience: int = 5,
        temperature: float = 0.7,
    ):
        """Initialize optimizer.

        Args:
            model: Model to use for optimization
            max_iterations: Maximum optimization iterations
            patience: Early stopping patience
            temperature: Sampling temperature for generation
        """
        self.model = model
        self.max_iterations = max_iterations
        self.patience = patience
        self.temperature = temperature
        self.log = logger.bind(optimizer=self.__class__.__name__)

    @abstractmethod
    async def optimize(
        self,
        signature: PromptSignature,
        training_data: list[tuple[dict, dict]],
        metric: Callable[[dict, dict], float],
        validation_data: list[tuple[dict, dict]] | None = None,
    ) -> OptimizationResult:
        """Run optimization to find the best prompt.

        Args:
            signature: The prompt signature to optimize
            training_data: List of (input, expected_output) tuples
            metric: Scoring function (actual, expected) -> score
            validation_data: Optional separate validation set

        Returns:
            OptimizationResult with the best prompt found
        """
        pass

    async def _evaluate_prompt(
        self,
        prompt: str,
        examples: list[tuple[dict, dict]],
        metric: Callable[[dict, dict], float],
    ) -> float:
        """Evaluate a prompt on a set of examples.

        Returns average score across all examples.
        """
        scores = []
        for input_data, expected_output in examples:
            try:
                actual_output = await self._run_prompt(prompt, input_data)
                score = metric(actual_output, expected_output)
                scores.append(score)
            except Exception as e:
                self.log.warning("Prompt evaluation failed", error=str(e))
                scores.append(0.0)

        return statistics.mean(scores) if scores else 0.0

    async def _run_prompt(self, prompt: str, input_data: dict) -> dict:
        """Run a prompt with input data and return the output.

        Uses the BaseAgent's _call_ai() for consistency.
        """
        # Import here to avoid circular imports
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        # Format the prompt with input data
        formatted_prompt = prompt
        for key, value in input_data.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))

        # Create a minimal agent for inference
        class _InferenceAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return ""

            async def execute(self, **kwargs):
                pass

        agent = _InferenceAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": formatted_prompt}],
            task_type=TaskType.GENERAL,
            max_tokens=2048,
            temperature=0.0,
            json_mode=True,
        )

        # Parse JSON response
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code block
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())


class COPROOptimizer(BaseOptimizer):
    """Coordinate Ascent Prompt Optimization.

    COPRO uses hill-climbing to iteratively improve the prompt:
    1. Generate candidate instruction variations
    2. Evaluate each on training data
    3. Keep the best and repeat

    Simple but effective for many tasks. Good starting point.

    Reference: DSPy COPRO optimizer
    """

    def __init__(
        self,
        num_candidates: int = 5,
        **kwargs,
    ):
        """Initialize COPRO optimizer.

        Args:
            num_candidates: Number of candidate prompts per iteration
            **kwargs: Passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.num_candidates = num_candidates

    async def optimize(
        self,
        signature: PromptSignature,
        training_data: list[tuple[dict, dict]],
        metric: Callable[[dict, dict], float],
        validation_data: list[tuple[dict, dict]] | None = None,
    ) -> OptimizationResult:
        """Run COPRO optimization."""
        start_time = time.time()
        history = []

        # Start with base prompt
        current_prompt = signature.to_base_prompt()
        current_score = await self._evaluate_prompt(current_prompt, training_data, metric)

        best_prompt = current_prompt
        best_score = current_score
        best_iteration = 0
        no_improvement_count = 0

        history.append(
            {
                "iteration": 0,
                "score": current_score,
                "prompt_preview": current_prompt[:200],
            }
        )

        self.log.info(
            "COPRO starting",
            initial_score=current_score,
            training_examples=len(training_data),
        )

        for iteration in range(1, self.max_iterations + 1):
            # Generate candidate variations
            candidates = await self._generate_candidates(current_prompt, signature, iteration)

            # Evaluate all candidates
            candidate_scores = []
            for candidate in candidates:
                score = await self._evaluate_prompt(candidate, training_data, metric)
                candidate_scores.append((candidate, score))

            # Find best candidate
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            top_prompt, top_score = candidate_scores[0]

            self.log.debug(
                "COPRO iteration complete",
                iteration=iteration,
                best_candidate_score=top_score,
                current_best=best_score,
            )

            # Update if improved
            if top_score > current_score:
                current_prompt = top_prompt
                current_score = top_score
                no_improvement_count = 0

                if top_score > best_score:
                    best_prompt = top_prompt
                    best_score = top_score
                    best_iteration = iteration
            else:
                no_improvement_count += 1

            history.append(
                {
                    "iteration": iteration,
                    "score": top_score,
                    "best_score": best_score,
                    "improved": top_score > current_score,
                }
            )

            # Early stopping
            if no_improvement_count >= self.patience:
                self.log.info(
                    "COPRO early stopping",
                    iteration=iteration,
                    best_score=best_score,
                )
                break

        # Validate on held-out data
        validation_score = best_score
        if validation_data:
            validation_score = await self._evaluate_prompt(best_prompt, validation_data, metric)

        total_time = time.time() - start_time

        return OptimizationResult(
            optimized_prompt=OptimizedPrompt(
                prompt=best_prompt,
                signature_hash=signature.get_signature_hash(),
                optimizer="copro",
                score=best_score,
                iterations=len(history),
            ),
            training_history=history,
            validation_score=validation_score,
            best_iteration=best_iteration,
            total_time_seconds=total_time,
        )

    async def _generate_candidates(
        self,
        current_prompt: str,
        signature: PromptSignature,
        iteration: int,
    ) -> list[str]:
        """Generate candidate prompt variations."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        focus_area = (
            "clarity and structure"
            if iteration <= 5
            else "edge cases and robustness" if iteration <= 10 else "refinement and polish"
        )

        generation_prompt = f"""You are an expert prompt engineer. Your task is to generate improved variations of a prompt.

Current prompt:
```
{current_prompt}
```

Task description: {signature.description}

Required outputs: {', '.join(f.name for f in signature.outputs)}

Generate {self.num_candidates} improved variations of this prompt. Each variation should:
1. Be clearer and more specific
2. Include better examples or structure
3. Address potential edge cases
4. Maintain the core task requirements

Iteration {iteration}: Focus on {focus_area}.

Return as JSON array of strings:
```json
["variation1", "variation2", ...]
```"""

        class _GeneratorAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert prompt engineer."

            async def execute(self, **kwargs):
                pass

        agent = _GeneratorAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": generation_prompt}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=4096,
            temperature=self.temperature,
        )

        # Parse candidates
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            candidates = json.loads(content.strip())
            if isinstance(candidates, list):
                return candidates[: self.num_candidates]
        except (json.JSONDecodeError, IndexError):
            self.log.warning("Failed to parse candidates, using current prompt")

        return [current_prompt]


class MIPROOptimizer(BaseOptimizer):
    """Mixed Integer Programming for Prompt Optimization.

    MIPROv2 uses Bayesian optimization to efficiently search the prompt space:
    1. Build a surrogate model of prompt quality
    2. Use acquisition function to select promising candidates
    3. Update model with observed scores

    More sample-efficient than COPRO for complex tasks.

    Reference: DSPy MIPROv2 optimizer
    """

    def __init__(
        self,
        num_candidates: int = 10,
        exploration_weight: float = 0.3,
        demo_selection: str = "random",  # "random", "diverse", "hard"
        **kwargs,
    ):
        """Initialize MIPRO optimizer.

        Args:
            num_candidates: Candidates per iteration
            exploration_weight: Balance exploration vs exploitation
            demo_selection: Strategy for selecting demonstrations
            **kwargs: Passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.num_candidates = num_candidates
        self.exploration_weight = exploration_weight
        self.demo_selection = demo_selection

        # Surrogate model state
        self._observations: list[tuple[str, float]] = []

    async def optimize(
        self,
        signature: PromptSignature,
        training_data: list[tuple[dict, dict]],
        metric: Callable[[dict, dict], float],
        validation_data: list[tuple[dict, dict]] | None = None,
    ) -> OptimizationResult:
        """Run MIPRO optimization with Bayesian approach."""
        start_time = time.time()
        history = []
        self._observations = []

        # Initialize with base prompt and random variations
        base_prompt = signature.to_base_prompt()
        initial_candidates = [base_prompt]
        initial_candidates.extend(await self._generate_initial_candidates(signature, 4))

        best_prompt = base_prompt
        best_score = 0.0
        best_iteration = 0

        # Evaluate initial candidates
        for i, prompt in enumerate(initial_candidates):
            score = await self._evaluate_prompt(prompt, training_data, metric)
            self._observations.append((prompt, score))

            if score > best_score:
                best_prompt = prompt
                best_score = score
                best_iteration = 0

            history.append(
                {
                    "iteration": 0,
                    "candidate": i,
                    "score": score,
                    "best_score": best_score,
                }
            )

        self.log.info(
            "MIPRO initialization complete",
            initial_best=best_score,
            num_initial=len(initial_candidates),
        )

        no_improvement_count = 0

        for iteration in range(1, self.max_iterations + 1):
            # Select demonstrations using specified strategy
            demos = self._select_demonstrations(training_data, signature.demos_k)

            # Generate candidates using acquisition function
            candidates = await self._generate_bayesian_candidates(signature, demos, iteration)

            # Evaluate candidates
            iteration_best_score = 0.0
            for prompt in candidates:
                score = await self._evaluate_prompt(prompt, training_data, metric)
                self._observations.append((prompt, score))
                iteration_best_score = max(iteration_best_score, score)

                if score > best_score:
                    best_prompt = prompt
                    best_score = score
                    best_iteration = iteration
                    no_improvement_count = 0

            if iteration_best_score <= best_score:
                no_improvement_count += 1

            history.append(
                {
                    "iteration": iteration,
                    "best_candidate_score": iteration_best_score,
                    "best_overall": best_score,
                    "observations": len(self._observations),
                }
            )

            self.log.debug(
                "MIPRO iteration complete",
                iteration=iteration,
                best_score=best_score,
                observations=len(self._observations),
            )

            if no_improvement_count >= self.patience:
                self.log.info("MIPRO early stopping", iteration=iteration)
                break

        # Final validation
        validation_score = best_score
        if validation_data:
            validation_score = await self._evaluate_prompt(best_prompt, validation_data, metric)

        return OptimizationResult(
            optimized_prompt=OptimizedPrompt(
                prompt=best_prompt,
                signature_hash=signature.get_signature_hash(),
                optimizer="mipro",
                score=best_score,
                iterations=len(history),
                demos=demos,
                metadata={"observations": len(self._observations)},
            ),
            training_history=history,
            validation_score=validation_score,
            best_iteration=best_iteration,
            total_time_seconds=time.time() - start_time,
        )

    def _select_demonstrations(
        self,
        training_data: list[tuple[dict, dict]],
        k: int,
    ) -> list[dict]:
        """Select k demonstrations using the specified strategy."""
        if len(training_data) <= k:
            return [{"input": inp, "output": out} for inp, out in training_data]

        if self.demo_selection == "random":
            selected = random.sample(training_data, k)
        elif self.demo_selection == "diverse":
            # Simple diversity: spread evenly across data
            step = len(training_data) // k
            selected = [training_data[i * step] for i in range(k)]
        else:  # "hard" - would need difficulty scores
            selected = random.sample(training_data, k)

        return [{"input": inp, "output": out} for inp, out in selected]

    async def _generate_initial_candidates(
        self,
        signature: PromptSignature,
        n: int,
    ) -> list[str]:
        """Generate diverse initial candidates."""
        styles = [
            "concise and direct",
            "detailed with examples",
            "structured with clear sections",
            "conversational and friendly",
        ]

        candidates = []
        for style in styles[:n]:
            prompt = await self._generate_styled_prompt(signature, style)
            candidates.append(prompt)

        return candidates

    async def _generate_styled_prompt(
        self,
        signature: PromptSignature,
        style: str,
    ) -> str:
        """Generate a prompt in a specific style."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        request = f"""Create a prompt for the following task in a {style} style:

Task: {signature.description}

Inputs:
{chr(10).join(f.to_prompt_fragment() for f in signature.inputs)}

Expected outputs:
{chr(10).join(f.to_prompt_fragment() for f in signature.outputs)}

Constraints:
{chr(10).join(f'- {c}' for c in signature.constraints)}

Generate just the prompt text, no explanations."""

        class _StyleAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert prompt engineer."

            async def execute(self, **kwargs):
                pass

        agent = _StyleAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.8,
        )

        return response.content

    async def _generate_bayesian_candidates(
        self,
        signature: PromptSignature,
        demos: list[dict],
        iteration: int,
    ) -> list[str]:
        """Generate candidates using pseudo-Bayesian approach."""
        # Get top-k performing prompts from observations
        sorted_obs = sorted(self._observations, key=lambda x: x[1], reverse=True)
        top_prompts = [p for p, _ in sorted_obs[:3]] if sorted_obs else []

        # Balance exploration and exploitation
        exploit_count = int(self.num_candidates * (1 - self.exploration_weight))
        explore_count = self.num_candidates - exploit_count

        candidates = []

        # Exploitation: refine top performers
        if top_prompts:
            for prompt in top_prompts[:exploit_count]:
                refined = await self._refine_prompt(prompt, signature, demos)
                candidates.append(refined)

        # Exploration: generate novel variations
        for _ in range(explore_count):
            novel = await self._generate_novel_prompt(signature, demos, iteration)
            candidates.append(novel)

        return candidates

    async def _refine_prompt(
        self,
        prompt: str,
        signature: PromptSignature,
        demos: list[dict],
    ) -> str:
        """Refine an existing high-performing prompt."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        demos_text = "\n".join(
            f"Example {i + 1}:\nInput: {d['input']}\nOutput: {d['output']}"
            for i, d in enumerate(demos[:2])
        )

        request = f"""Improve this prompt to be more effective:

Current prompt:
```
{prompt}
```

Task: {signature.description}

Example input/outputs for reference:
{demos_text}

Generate an improved version that:
1. Is clearer and more specific
2. Better handles edge cases
3. Produces more consistent outputs

Return only the improved prompt text."""

        class _RefineAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert prompt engineer focused on refinement."

            async def execute(self, **kwargs):
                pass

        agent = _RefineAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.5,
        )

        return response.content

    async def _generate_novel_prompt(
        self,
        signature: PromptSignature,
        demos: list[dict],
        iteration: int,
    ) -> str:
        """Generate a novel prompt for exploration."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        approaches = [
            "step-by-step reasoning",
            "examples-first explanation",
            "constraint-focused structure",
            "output-driven backwards planning",
            "analogical reasoning",
        ]
        approach = random.choice(approaches)

        demos_text = "\n".join(f"Example: {d['input']} -> {d['output']}" for d in demos[:2])

        request = f"""Create a prompt using a {approach} approach for:

Task: {signature.description}

Inputs: {', '.join(f.name for f in signature.inputs)}
Outputs: {', '.join(f.name for f in signature.outputs)}

Examples:
{demos_text}

Create a creative, effective prompt using the {approach} approach.
Return only the prompt text."""

        class _NovelAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are a creative prompt engineer exploring novel approaches."

            async def execute(self, **kwargs):
                pass

        agent = _NovelAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.9,
        )

        return response.content


class SIMBAOptimizer(BaseOptimizer):
    """Stochastic Mini-Batch Analysis optimizer.

    SIMBA identifies challenging examples and uses self-reflection
    to generate improvement rules:
    1. Find high-variability examples (potential weaknesses)
    2. Analyze patterns in failures
    3. Generate self-reflective improvement rules
    4. Apply rules to improve prompt

    Best for tasks with clear failure patterns.

    Reference: DSPy SIMBA optimizer concept
    """

    def __init__(
        self,
        batch_size: int = 5,
        reflection_depth: int = 2,
        **kwargs,
    ):
        """Initialize SIMBA optimizer.

        Args:
            batch_size: Mini-batch size for analysis
            reflection_depth: Depth of self-reflection
            **kwargs: Passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.reflection_depth = reflection_depth

    async def optimize(
        self,
        signature: PromptSignature,
        training_data: list[tuple[dict, dict]],
        metric: Callable[[dict, dict], float],
        validation_data: list[tuple[dict, dict]] | None = None,
    ) -> OptimizationResult:
        """Run SIMBA optimization with self-reflection."""
        start_time = time.time()
        history = []

        current_prompt = signature.to_base_prompt()
        current_score = await self._evaluate_prompt(current_prompt, training_data, metric)

        best_prompt = current_prompt
        best_score = current_score
        best_iteration = 0

        # Track per-example scores for variability analysis
        example_scores: dict[int, list[float]] = {i: [] for i in range(len(training_data))}

        history.append(
            {
                "iteration": 0,
                "score": current_score,
                "method": "baseline",
            }
        )

        self.log.info(
            "SIMBA starting",
            initial_score=current_score,
            training_examples=len(training_data),
        )

        for iteration in range(1, self.max_iterations + 1):
            # Evaluate current prompt on all examples
            for i, (inp, exp) in enumerate(training_data):
                try:
                    out = await self._run_prompt(current_prompt, inp)
                    score = metric(out, exp)
                    example_scores[i].append(score)
                except Exception:
                    example_scores[i].append(0.0)

            # Find high-variability examples
            challenging_indices = self._find_challenging_examples(example_scores)
            challenging_batch = [training_data[i] for i in challenging_indices]

            # Generate self-reflective improvement rules
            rules = await self._generate_improvement_rules(
                current_prompt,
                signature,
                challenging_batch,
            )

            # Apply rules to improve prompt
            improved_prompt = await self._apply_rules(current_prompt, rules, signature)

            # Evaluate improved prompt
            improved_score = await self._evaluate_prompt(improved_prompt, training_data, metric)

            self.log.debug(
                "SIMBA iteration",
                iteration=iteration,
                current_score=current_score,
                improved_score=improved_score,
                rules_generated=len(rules),
            )

            # Update if improved
            if improved_score > current_score:
                current_prompt = improved_prompt
                current_score = improved_score

                if improved_score > best_score:
                    best_prompt = improved_prompt
                    best_score = improved_score
                    best_iteration = iteration

            history.append(
                {
                    "iteration": iteration,
                    "score": improved_score,
                    "best_score": best_score,
                    "rules": rules,
                    "challenging_examples": len(challenging_indices),
                }
            )

        validation_score = best_score
        if validation_data:
            validation_score = await self._evaluate_prompt(best_prompt, validation_data, metric)

        return OptimizationResult(
            optimized_prompt=OptimizedPrompt(
                prompt=best_prompt,
                signature_hash=signature.get_signature_hash(),
                optimizer="simba",
                score=best_score,
                iterations=len(history),
            ),
            training_history=history,
            validation_score=validation_score,
            best_iteration=best_iteration,
            total_time_seconds=time.time() - start_time,
        )

    def _find_challenging_examples(
        self,
        example_scores: dict[int, list[float]],
    ) -> list[int]:
        """Find examples with high variability or low scores."""
        challenges = []

        for idx, scores in example_scores.items():
            if not scores:
                continue

            avg_score = statistics.mean(scores)
            variability = statistics.stdev(scores) if len(scores) > 1 else 0

            # Low average OR high variability = challenging
            if avg_score < 0.5 or variability > 0.2:
                challenges.append((idx, avg_score, variability))

        # Sort by challenge (low score + high variability)
        challenges.sort(key=lambda x: x[1] - x[2])

        return [idx for idx, _, _ in challenges[: self.batch_size]]

    async def _generate_improvement_rules(
        self,
        prompt: str,
        signature: PromptSignature,
        challenging_examples: list[tuple[dict, dict]],
    ) -> list[str]:
        """Generate self-reflective improvement rules."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        examples_text = "\n".join(
            f"Input: {inp}\nExpected: {exp}" for inp, exp in challenging_examples[:3]
        )

        request = f"""Analyze this prompt and its challenging examples to identify improvement rules.

Prompt:
```
{prompt}
```

Challenging examples (the prompt struggles with these):
{examples_text}

Task: {signature.description}

Generate 3-5 specific rules to improve this prompt. Each rule should:
1. Address a specific weakness
2. Be actionable and clear
3. Focus on the challenging examples

Return as JSON array:
```json
["Rule 1: ...", "Rule 2: ...", ...]
```"""

        class _RulesAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert at analyzing prompt weaknesses."

            async def execute(self, **kwargs):
                pass

        agent = _RulesAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.ROOT_CAUSE_ANALYSIS,
            max_tokens=1024,
            temperature=0.5,
        )

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())
        except (json.JSONDecodeError, IndexError):
            return []

    async def _apply_rules(
        self,
        prompt: str,
        rules: list[str],
        signature: PromptSignature,
    ) -> str:
        """Apply improvement rules to generate better prompt."""
        if not rules:
            return prompt

        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        rules_text = "\n".join(f"- {rule}" for rule in rules)

        request = f"""Apply these improvement rules to the prompt:

Original prompt:
```
{prompt}
```

Rules to apply:
{rules_text}

Task: {signature.description}

Generate an improved prompt that incorporates all the rules.
Return only the improved prompt text."""

        class _ApplyAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert at improving prompts."

            async def execute(self, **kwargs):
                pass

        agent = _ApplyAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.3,
        )

        return response.content


class GEPAOptimizer(BaseOptimizer):
    """Reflective Prompt Evolution optimizer.

    GEPA uses evolutionary strategies with reflection:
    1. Maintain a population of prompts
    2. Reflect on program trajectory (what worked, what didn't)
    3. Use reflection to guide mutation and crossover
    4. Select fittest prompts for next generation

    Good for complex tasks requiring creative solutions.

    Reference: Evolutionary prompt optimization with reflection
    """

    def __init__(
        self,
        population_size: int = 6,
        elite_count: int = 2,
        mutation_rate: float = 0.3,
        **kwargs,
    ):
        """Initialize GEPA optimizer.

        Args:
            population_size: Number of prompts in population
            elite_count: Top prompts to preserve each generation
            mutation_rate: Probability of mutation
            **kwargs: Passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.population_size = population_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate

    async def optimize(
        self,
        signature: PromptSignature,
        training_data: list[tuple[dict, dict]],
        metric: Callable[[dict, dict], float],
        validation_data: list[tuple[dict, dict]] | None = None,
    ) -> OptimizationResult:
        """Run GEPA evolutionary optimization."""
        start_time = time.time()
        history = []

        # Initialize population
        population = await self._initialize_population(signature)

        # Evaluate initial population
        fitness = []
        for prompt in population:
            score = await self._evaluate_prompt(prompt, training_data, metric)
            fitness.append((prompt, score))

        fitness.sort(key=lambda x: x[1], reverse=True)
        best_prompt, best_score = fitness[0]
        best_iteration = 0

        history.append(
            {
                "iteration": 0,
                "best_score": best_score,
                "avg_score": statistics.mean(s for _, s in fitness),
                "population_size": len(population),
            }
        )

        self.log.info(
            "GEPA population initialized",
            best_score=best_score,
            population_size=len(population),
        )

        trajectory: list[tuple[str, float, int]] = []  # Track (prompt, score, iteration)

        for generation in range(1, self.max_iterations + 1):
            # Reflect on trajectory
            trajectory.extend([(p, s, generation - 1) for p, s in fitness])
            reflection = await self._reflect_on_trajectory(trajectory, signature)

            # Select elite
            elite = [p for p, _ in fitness[: self.elite_count]]

            # Generate new population through crossover and mutation
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_select(fitness)
                parent2 = self._tournament_select(fitness)

                # Crossover
                child = await self._crossover(parent1, parent2, signature, reflection)

                # Mutation
                if random.random() < self.mutation_rate:
                    child = await self._mutate(child, signature, reflection)

                new_population.append(child)

            population = new_population

            # Evaluate new population
            fitness = []
            for prompt in population:
                score = await self._evaluate_prompt(prompt, training_data, metric)
                fitness.append((prompt, score))

            fitness.sort(key=lambda x: x[1], reverse=True)

            if fitness[0][1] > best_score:
                best_prompt, best_score = fitness[0]
                best_iteration = generation

            history.append(
                {
                    "iteration": generation,
                    "best_score": fitness[0][1],
                    "overall_best": best_score,
                    "avg_score": statistics.mean(s for _, s in fitness),
                    "reflection_summary": reflection[:100],
                }
            )

            self.log.debug(
                "GEPA generation complete",
                generation=generation,
                best_score=best_score,
                avg_score=statistics.mean(s for _, s in fitness),
            )

        validation_score = best_score
        if validation_data:
            validation_score = await self._evaluate_prompt(best_prompt, validation_data, metric)

        return OptimizationResult(
            optimized_prompt=OptimizedPrompt(
                prompt=best_prompt,
                signature_hash=signature.get_signature_hash(),
                optimizer="gepa",
                score=best_score,
                iterations=len(history),
                metadata={"generations": len(history)},
            ),
            training_history=history,
            validation_score=validation_score,
            best_iteration=best_iteration,
            total_time_seconds=time.time() - start_time,
        )

    async def _initialize_population(
        self,
        signature: PromptSignature,
    ) -> list[str]:
        """Initialize diverse population of prompts."""
        population = [signature.to_base_prompt()]

        approaches = [
            "analytical",
            "creative",
            "structured",
            "examples-driven",
            "minimalist",
        ]

        for approach in approaches[: self.population_size - 1]:
            prompt = await self._generate_approach_prompt(signature, approach)
            population.append(prompt)

        return population

    async def _generate_approach_prompt(
        self,
        signature: PromptSignature,
        approach: str,
    ) -> str:
        """Generate prompt with specific approach."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        request = f"""Create a {approach} prompt for this task:

Task: {signature.description}
Inputs: {', '.join(f.name for f in signature.inputs)}
Outputs: {', '.join(f.name for f in signature.outputs)}

Use a {approach} approach. Return only the prompt."""

        class _ApproachAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return f"You are an expert at creating {approach} prompts."

            async def execute(self, **kwargs):
                pass

        agent = _ApproachAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.8,
        )

        return response.content

    async def _reflect_on_trajectory(
        self,
        trajectory: list[tuple[str, float, int]],
        signature: PromptSignature,
    ) -> str:
        """Reflect on optimization trajectory to guide evolution."""
        if len(trajectory) < 3:
            return "Early stage - explore diverse approaches."

        # Analyze trajectory
        sorted_traj = sorted(trajectory, key=lambda x: x[1], reverse=True)
        top_prompts = [p for p, _, _ in sorted_traj[:3]]
        bottom_prompts = [p for p, _, _ in sorted_traj[-3:]]

        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        request = f"""Analyze this optimization trajectory and provide guidance.

Task: {signature.description}

Top performing prompts (excerpts):
{chr(10).join(p[:200] + '...' for p in top_prompts)}

Low performing prompts (excerpts):
{chr(10).join(p[:200] + '...' for p in bottom_prompts)}

What patterns distinguish good from bad prompts? Provide 2-3 key insights for improving future prompts."""

        class _ReflectAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert at analyzing prompt effectiveness."

            async def execute(self, **kwargs):
                pass

        agent = _ReflectAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.ROOT_CAUSE_ANALYSIS,
            max_tokens=512,
            temperature=0.3,
        )

        return response.content

    def _tournament_select(
        self,
        fitness: list[tuple[str, float]],
        k: int = 3,
    ) -> str:
        """Tournament selection for parent."""
        candidates = random.sample(fitness, min(k, len(fitness)))
        return max(candidates, key=lambda x: x[1])[0]

    async def _crossover(
        self,
        parent1: str,
        parent2: str,
        signature: PromptSignature,
        reflection: str,
    ) -> str:
        """Crossover two parent prompts."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        request = f"""Combine the best elements of these two prompts:

Parent 1:
```
{parent1[:500]}
```

Parent 2:
```
{parent2[:500]}
```

Task: {signature.description}

Guidance from reflection:
{reflection[:200]}

Create a child prompt that combines the strengths of both parents.
Return only the prompt."""

        class _CrossoverAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert at combining prompt elements."

            async def execute(self, **kwargs):
                pass

        agent = _CrossoverAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.6,
        )

        return response.content

    async def _mutate(
        self,
        prompt: str,
        signature: PromptSignature,
        reflection: str,
    ) -> str:
        """Mutate a prompt based on reflection."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        mutations = [
            "add more specific examples",
            "restructure for clarity",
            "add edge case handling",
            "simplify and focus",
            "add step-by-step guidance",
        ]
        mutation_type = random.choice(mutations)

        request = f"""Mutate this prompt by applying: {mutation_type}

Original:
```
{prompt[:600]}
```

Task: {signature.description}

Reflection guidance:
{reflection[:150]}

Apply the mutation. Return only the mutated prompt."""

        class _MutateAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert at prompt mutation."

            async def execute(self, **kwargs):
                pass

        agent = _MutateAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.7,
        )

        return response.content


# =============================================================================
# PromptOptimizer: Main Entry Point
# =============================================================================


class PromptOptimizer:
    """High-level prompt optimization interface.

    Provides a simple API for optimizing prompts using various strategies.

    Example:
        ```python
        optimizer = PromptOptimizer()
        result = await optimizer.optimize(
            signature=my_signature,
            training_data=examples,
            metric=accuracy_metric,
            optimizer="mipro",
        )
        print(f"Best prompt (score={result.validation_score}):")
        print(result.optimized_prompt.prompt)
        ```
    """

    OPTIMIZERS = {
        "copro": COPROOptimizer,
        "mipro": MIPROOptimizer,
        "simba": SIMBAOptimizer,
        "gepa": GEPAOptimizer,
    }

    def __init__(self):
        """Initialize the optimizer."""
        self.log = logger.bind(component="PromptOptimizer")

    async def optimize(
        self,
        signature: PromptSignature,
        training_data: list[tuple[dict, dict]],
        metric: Callable[[dict, dict], float],
        optimizer: str = "mipro",
        validation_split: float = 0.2,
        max_iterations: int = 20,
        **kwargs,
    ) -> OptimizationResult:
        """Optimize a prompt using the specified optimizer.

        Args:
            signature: PromptSignature defining the task
            training_data: List of (input, expected_output) tuples
            metric: Scoring function (actual, expected) -> float [0, 1]
            optimizer: Optimizer name ("copro", "mipro", "simba", "gepa")
            validation_split: Fraction of data to use for validation
            max_iterations: Maximum optimization iterations
            **kwargs: Passed to optimizer constructor

        Returns:
            OptimizationResult with the best prompt found
        """
        # Validate optimizer
        if optimizer not in self.OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer: {optimizer}. " f"Available: {list(self.OPTIMIZERS.keys())}"
            )

        # Split data
        random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:] if validation_split > 0 else None

        self.log.info(
            "Starting optimization",
            optimizer=optimizer,
            signature=signature.name,
            train_examples=len(train_data),
            val_examples=len(val_data) if val_data else 0,
        )

        # Create and run optimizer
        optimizer_class = self.OPTIMIZERS[optimizer]
        opt = optimizer_class(max_iterations=max_iterations, **kwargs)

        result = await opt.optimize(
            signature=signature,
            training_data=train_data,
            metric=metric,
            validation_data=val_data,
        )

        self.log.info(
            "Optimization complete",
            optimizer=optimizer,
            best_score=result.optimized_prompt.score,
            validation_score=result.validation_score,
            iterations=result.optimized_prompt.iterations,
            time_seconds=result.total_time_seconds,
        )

        return result

    async def compile_signature(
        self,
        signature: PromptSignature,
        style: str = "default",
    ) -> str:
        """Compile a signature into a prompt string without optimization.

        Useful for generating a starting prompt from a signature.

        Args:
            signature: The signature to compile
            style: Prompt style ("default", "detailed", "minimal", "structured")

        Returns:
            Compiled prompt string
        """
        if style == "default":
            return signature.to_base_prompt()

        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        style_instructions = {
            "detailed": "Create a comprehensive, detailed prompt with examples",
            "minimal": "Create a concise, minimal prompt",
            "structured": "Create a well-structured prompt with clear sections",
        }

        instruction = style_instructions.get(style, style_instructions["default"])

        request = f"""{instruction}

Task: {signature.description}

Inputs:
{chr(10).join(f.to_prompt_fragment() for f in signature.inputs)}

Outputs:
{chr(10).join(f.to_prompt_fragment() for f in signature.outputs)}

Constraints:
{chr(10).join(f'- {c}' for c in signature.constraints)}

Generate the prompt."""

        class _CompileAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are an expert prompt engineer."

            async def execute(self, **kwargs):
                pass

        agent = _CompileAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": request}],
            task_type=TaskType.TEST_GENERATION,
            max_tokens=2048,
            temperature=0.5,
        )

        return response.content


# =============================================================================
# AgentPromptManager: Version Control and A/B Testing
# =============================================================================


class AgentPromptManager:
    """Manages prompts for agents with version control and A/B testing.

    Features:
    - Version control for prompts (with Cognee storage)
    - A/B testing capability
    - Automatic optimization scheduling
    - Rollback on regression

    Example:
        ```python
        manager = AgentPromptManager()

        # Get current prompt for an agent
        prompt = await manager.get_prompt("code_analyzer")

        # Deploy optimized prompt
        await manager.deploy_prompt(
            "code_analyzer",
            optimized_result.optimized_prompt,
            variant="v2_optimized",
        )

        # Run A/B test
        result = await manager.run_ab_test(
            "code_analyzer",
            variant_a="default",
            variant_b="v2_optimized",
            test_cases=test_data,
        )
        ```
    """

    def __init__(self):
        """Initialize the prompt manager."""
        self.log = logger.bind(component="AgentPromptManager")
        self._cache: dict[str, dict[str, str]] = {}  # agent -> variant -> prompt
        self._default_variants: dict[str, str] = {}  # agent -> default variant

    async def get_prompt(
        self,
        agent_type: str,
        variant: str = "default",
    ) -> str:
        """Get the current prompt for an agent.

        Args:
            agent_type: Agent class name (e.g., "CodeAnalyzerAgent")
            variant: Prompt variant name

        Returns:
            The prompt string
        """
        # Check cache first
        if agent_type in self._cache and variant in self._cache[agent_type]:
            return self._cache[agent_type][variant]

        # Try to load from Cognee
        try:
            from ..knowledge.cognee_client import get_cognee_client

            cognee = get_cognee_client()

            result = await cognee.get(
                key=f"prompt_{agent_type}_{variant}",
                namespace="prompt_versions",
            )

            if result:
                prompt = result.get("prompt", "")
                if agent_type not in self._cache:
                    self._cache[agent_type] = {}
                self._cache[agent_type][variant] = prompt
                return prompt
        except Exception as e:
            self.log.warning(
                "Failed to load prompt from Cognee",
                agent_type=agent_type,
                variant=variant,
                error=str(e),
            )

        # Fall back to static prompts
        from ..agents.prompts import ENHANCED_PROMPTS

        # Map agent type to prompt key
        prompt_key = agent_type.lower().replace("agent", "").replace("_", "")

        # Try various key formats
        for key in [prompt_key, f"{prompt_key}_", prompt_key.replace("analyzer", "")]:
            if key in ENHANCED_PROMPTS:
                return ENHANCED_PROMPTS[key]

        self.log.warning(
            "No prompt found for agent",
            agent_type=agent_type,
            variant=variant,
        )
        return ""

    async def deploy_prompt(
        self,
        agent_type: str,
        prompt: OptimizedPrompt | str,
        variant: str = "default",
        set_as_default: bool = False,
    ) -> None:
        """Deploy a prompt for an agent.

        Args:
            agent_type: Agent class name
            prompt: OptimizedPrompt or prompt string
            variant: Variant name for the deployment
            set_as_default: Whether to make this the default variant
        """
        prompt_str = prompt.prompt if isinstance(prompt, OptimizedPrompt) else prompt

        # Update cache
        if agent_type not in self._cache:
            self._cache[agent_type] = {}
        self._cache[agent_type][variant] = prompt_str

        if set_as_default:
            self._default_variants[agent_type] = variant

        # Store in Cognee
        try:
            from ..knowledge.cognee_client import get_cognee_client

            cognee = get_cognee_client()

            metadata = {}
            if isinstance(prompt, OptimizedPrompt):
                metadata = prompt.to_dict()
                metadata.pop("prompt", None)  # Don't duplicate

            await cognee.put(
                key=f"prompt_{agent_type}_{variant}",
                value={
                    "prompt": prompt_str,
                    "agent_type": agent_type,
                    "variant": variant,
                    "deployed_at": datetime.now(timezone.utc).isoformat(),
                    **metadata,
                },
                namespace="prompt_versions",
                embeddings=False,
            )

            self.log.info(
                "Prompt deployed",
                agent_type=agent_type,
                variant=variant,
                set_as_default=set_as_default,
            )
        except Exception as e:
            self.log.error(
                "Failed to store prompt in Cognee",
                error=str(e),
            )

    async def run_ab_test(
        self,
        agent_type: str,
        variant_a: str,
        variant_b: str,
        test_cases: list[dict],
        metric: Callable[[dict, dict], float] | None = None,
    ) -> ABTestResult:
        """Run an A/B test between two prompt variants.

        Args:
            agent_type: Agent class name
            variant_a: First variant name
            variant_b: Second variant name
            test_cases: List of test inputs with expected outputs
            metric: Scoring function (uses default if None)

        Returns:
            ABTestResult with winner and confidence
        """
        prompt_a = await self.get_prompt(agent_type, variant_a)
        prompt_b = await self.get_prompt(agent_type, variant_b)

        if not prompt_a or not prompt_b:
            raise ValueError(f"Could not load prompts for variants {variant_a}, {variant_b}")

        if metric is None:
            metric = self._default_metric

        # Run tests for both variants
        scores_a = []
        scores_b = []

        for case in test_cases:
            input_data = case.get("input", case)
            expected = case.get("expected", case.get("output", {}))

            # Run variant A
            try:
                output_a = await self._run_prompt(prompt_a, input_data)
                score_a = metric(output_a, expected)
                scores_a.append(score_a)
            except Exception:
                scores_a.append(0.0)

            # Run variant B
            try:
                output_b = await self._run_prompt(prompt_b, input_data)
                score_b = metric(output_b, expected)
                scores_b.append(score_b)
            except Exception:
                scores_b.append(0.0)

        # Calculate statistics
        mean_a = statistics.mean(scores_a)
        mean_b = statistics.mean(scores_b)

        # Simple statistical test (paired t-test approximation)
        differences = [a - b for a, b in zip(scores_a, scores_b)]
        if len(differences) > 1:
            mean_diff = statistics.mean(differences)
            std_diff = statistics.stdev(differences)
            t_stat = mean_diff / (std_diff / (len(differences) ** 0.5)) if std_diff > 0 else 0
            # Approximate p-value (simplified)
            p_value = max(0.001, min(1.0, 2 * (1 - min(abs(t_stat) / 3, 1))))
        else:
            p_value = 1.0

        # Determine winner
        if mean_a > mean_b and p_value < 0.05:
            winner = variant_a
            confidence = 1 - p_value
        elif mean_b > mean_a and p_value < 0.05:
            winner = variant_b
            confidence = 1 - p_value
        else:
            winner = "tie"
            confidence = 0.5

        return ABTestResult(
            variant_a=variant_a,
            variant_b=variant_b,
            variant_a_score=mean_a,
            variant_b_score=mean_b,
            winner=winner,
            confidence=confidence,
            test_cases=len(test_cases),
            p_value=p_value,
            details={
                "scores_a": scores_a,
                "scores_b": scores_b,
            },
        )

    async def optimize_agent_prompts(
        self,
        agent_type: str,
        training_data: list[tuple[dict, dict]],
        optimizer: str = "mipro",
        signature: PromptSignature | None = None,
    ) -> OptimizationResult:
        """Optimize prompts for a specific agent.

        Args:
            agent_type: Agent class name
            training_data: Training examples
            optimizer: Optimizer to use
            signature: Optional signature (inferred if not provided)

        Returns:
            OptimizationResult
        """
        # Get or infer signature
        if signature is None:
            signature = await self._infer_signature(agent_type, training_data)

        # Run optimization
        opt = PromptOptimizer()
        result = await opt.optimize(
            signature=signature,
            training_data=training_data,
            metric=self._default_metric,
            optimizer=optimizer,
        )

        return result

    async def _infer_signature(
        self,
        agent_type: str,
        training_data: list[tuple[dict, dict]],
    ) -> PromptSignature:
        """Infer a signature from agent type and training data."""
        # Analyze training data structure
        if training_data:
            sample_input, sample_output = training_data[0]
            input_fields = [
                SignatureField(name=k, description=f"Input {k}", type=type(v).__name__)
                for k, v in sample_input.items()
            ]
            output_fields = [
                SignatureField(name=k, description=f"Output {k}", type=type(v).__name__)
                for k, v in sample_output.items()
            ]
        else:
            input_fields = [SignatureField(name="input", description="Input data")]
            output_fields = [SignatureField(name="output", description="Output data")]

        return PromptSignature(
            name=agent_type,
            description=f"Task performed by {agent_type}",
            inputs=input_fields,
            outputs=output_fields,
        )

    def _default_metric(self, actual: dict, expected: dict) -> float:
        """Default metric: key overlap and value similarity."""
        if not actual or not expected:
            return 0.0

        # Key overlap
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        key_overlap = len(actual_keys & expected_keys) / len(expected_keys) if expected_keys else 0

        # Value similarity for matching keys
        value_scores = []
        for key in actual_keys & expected_keys:
            actual_val = str(actual[key])
            expected_val = str(expected[key])

            # Simple string similarity
            if actual_val == expected_val:
                value_scores.append(1.0)
            elif expected_val in actual_val or actual_val in expected_val:
                value_scores.append(0.7)
            else:
                # Jaccard similarity on words
                actual_words = set(actual_val.lower().split())
                expected_words = set(expected_val.lower().split())
                if actual_words or expected_words:
                    jaccard = len(actual_words & expected_words) / len(actual_words | expected_words)
                    value_scores.append(jaccard)
                else:
                    value_scores.append(0.0)

        value_similarity = statistics.mean(value_scores) if value_scores else 0.0

        return 0.3 * key_overlap + 0.7 * value_similarity

    async def _run_prompt(self, prompt: str, input_data: dict) -> dict:
        """Run a prompt with input data."""
        from ..agents.base import BaseAgent
        from ..core.model_router import TaskType

        # Format prompt
        formatted = prompt
        for key, value in input_data.items():
            formatted = formatted.replace(f"{{{key}}}", str(value))

        class _RunAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return ""

            async def execute(self, **kwargs):
                pass

        agent = _RunAgent()
        response = await agent._call_ai(
            messages=[{"role": "user", "content": formatted}],
            task_type=TaskType.GENERAL,
            max_tokens=2048,
            temperature=0.0,
            json_mode=True,
        )

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            return json.loads(content.strip())


# =============================================================================
# AgentBenchmarkRunner: Performance Tracking
# =============================================================================


class AgentBenchmarkRunner:
    """Runs agents on benchmarks to track performance.

    Enables:
    - Continuous optimization tracking
    - Regression detection
    - Performance comparison across variants

    Example:
        ```python
        runner = AgentBenchmarkRunner()

        # Run benchmark
        result = await runner.run_benchmark(
            agent_type="CodeAnalyzerAgent",
            benchmark_name="code_analysis_v1",
            test_cases=benchmark_cases,
        )

        # Compare variants
        comparison = await runner.compare_variants(
            agent_type="CodeAnalyzerAgent",
            variants=["default", "optimized_v1", "optimized_v2"],
            benchmark_name="code_analysis_v1",
        )
        ```
    """

    def __init__(self):
        """Initialize the benchmark runner."""
        self.log = logger.bind(component="AgentBenchmarkRunner")
        self._results_history: list[BenchmarkResult] = []
        self._prompt_manager = AgentPromptManager()

    async def run_benchmark(
        self,
        agent_type: str,
        benchmark_name: str,
        test_cases: list[dict],
        prompt_variant: str = "default",
        metric: Callable[[dict, dict], float] | None = None,
    ) -> BenchmarkResult:
        """Run a benchmark on an agent.

        Args:
            agent_type: Agent class name
            benchmark_name: Name of the benchmark
            test_cases: Test cases with inputs and expected outputs
            prompt_variant: Which prompt variant to test
            metric: Scoring function

        Returns:
            BenchmarkResult with detailed metrics
        """
        start_time = time.time()

        prompt = await self._prompt_manager.get_prompt(agent_type, prompt_variant)
        if not prompt:
            raise ValueError(f"No prompt found for {agent_type}:{prompt_variant}")

        if metric is None:
            metric = self._prompt_manager._default_metric

        scores = []
        passed = 0
        failed = 0
        errors = []
        total_cost = 0.0

        for case in test_cases:
            input_data = case.get("input", case)
            expected = case.get("expected", case.get("output", {}))

            try:
                output = await self._prompt_manager._run_prompt(prompt, input_data)
                score = metric(output, expected)
                scores.append(score)

                if score >= 0.7:  # Threshold for "pass"
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                scores.append(0.0)
                failed += 1
                errors.append(str(e))

        latency_ms = (time.time() - start_time) * 1000 / len(test_cases)

        result = BenchmarkResult(
            agent_type=agent_type,
            prompt_variant=prompt_variant,
            benchmark_name=benchmark_name,
            score=statistics.mean(scores) if scores else 0.0,
            accuracy=passed / len(test_cases) if test_cases else 0.0,
            latency_ms=latency_ms,
            cost_usd=total_cost,
            test_cases=len(test_cases),
            passed=passed,
            failed=failed,
            errors=errors[:10],  # Limit stored errors
        )

        self._results_history.append(result)

        self.log.info(
            "Benchmark complete",
            agent_type=agent_type,
            benchmark=benchmark_name,
            variant=prompt_variant,
            score=result.score,
            accuracy=result.accuracy,
        )

        return result

    async def compare_variants(
        self,
        agent_type: str,
        variants: list[str],
        benchmark_name: str,
        test_cases: list[dict],
    ) -> dict[str, BenchmarkResult]:
        """Compare multiple prompt variants on a benchmark.

        Args:
            agent_type: Agent class name
            variants: List of variant names to compare
            benchmark_name: Benchmark to run
            test_cases: Test cases

        Returns:
            Dict mapping variant name to BenchmarkResult
        """
        results = {}

        for variant in variants:
            result = await self.run_benchmark(
                agent_type=agent_type,
                benchmark_name=benchmark_name,
                test_cases=test_cases,
                prompt_variant=variant,
            )
            results[variant] = result

        # Log comparison
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].score,
            reverse=True,
        )

        self.log.info(
            "Variant comparison complete",
            agent_type=agent_type,
            benchmark=benchmark_name,
            rankings=[(v, r.score) for v, r in sorted_results],
        )

        return results

    def detect_regression(
        self,
        agent_type: str,
        benchmark_name: str,
        threshold: float = 0.05,
    ) -> bool | None:
        """Detect if recent performance regressed.

        Args:
            agent_type: Agent class name
            benchmark_name: Benchmark to check
            threshold: Minimum score drop to count as regression

        Returns:
            True if regression detected, False if not, None if insufficient data
        """
        # Get relevant results
        relevant = [
            r
            for r in self._results_history
            if r.agent_type == agent_type and r.benchmark_name == benchmark_name
        ]

        if len(relevant) < 2:
            return None

        # Compare latest to previous
        relevant.sort(key=lambda r: r.timestamp)
        latest = relevant[-1]
        previous = relevant[-2]

        if previous.score - latest.score > threshold:
            self.log.warning(
                "Regression detected",
                agent_type=agent_type,
                benchmark=benchmark_name,
                previous_score=previous.score,
                latest_score=latest.score,
                drop=previous.score - latest.score,
            )
            return True

        return False

    def get_history(
        self,
        agent_type: str | None = None,
        benchmark_name: str | None = None,
        limit: int = 100,
    ) -> list[BenchmarkResult]:
        """Get benchmark history with optional filtering.

        Args:
            agent_type: Filter by agent type
            benchmark_name: Filter by benchmark name
            limit: Maximum results to return

        Returns:
            List of BenchmarkResults
        """
        results = self._results_history

        if agent_type:
            results = [r for r in results if r.agent_type == agent_type]
        if benchmark_name:
            results = [r for r in results if r.benchmark_name == benchmark_name]

        return results[-limit:]
