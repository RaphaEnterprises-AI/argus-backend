"""
Subgraph Isolation for Multi-Agent Systems.

Implements isolated execution contexts for different agent domains.
Each subagent works with only its relevant context, saving 40-50% tokens.

Based on Anthropic's architecture:
- Context isolation between domains
- Parallel execution of independent subgraphs
- Result synthesis by supervisor
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AgentDomain(Enum):
    """Agent specialization domains."""

    CODE_ANALYSIS = "code_analysis"
    UI_TESTING = "ui_testing"
    API_TESTING = "api_testing"
    DATABASE_TESTING = "database_testing"
    SELF_HEALING = "self_healing"
    REPORTING = "reporting"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class DomainConfig:
    """Configuration for a domain subgraph."""

    domain: AgentDomain
    agents: list[str]  # Agent names in this domain
    max_context_tokens: int = 50_000
    max_concurrent_agents: int = 3
    model: str = "claude-sonnet-4-5"
    tools: list[str] = field(default_factory=list)
    requires_shared_context: bool = False


@dataclass
class SubgraphResult:
    """Result from a subgraph execution."""

    domain: AgentDomain
    success: bool
    output: Any
    tokens_used: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    agent_name: str | None = None
    error: str | None = None


@dataclass
class IsolatedContext:
    """Isolated context for a subgraph execution."""

    domain: AgentDomain
    task_id: str
    messages: list[dict] = field(default_factory=list)
    shared_data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, role: str, content: str):
        """Add a message to the isolated context."""
        self.messages.append({"role": role, "content": content})

    def get_context_for_agent(self, agent_name: str) -> list[dict]:
        """Get context filtered for a specific agent."""
        # Include system messages and recent exchanges
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        recent_msgs = self.messages[-6:]  # Last 3 exchanges

        return system_msgs + recent_msgs


# Default domain configurations
DEFAULT_DOMAIN_CONFIGS: dict[AgentDomain, DomainConfig] = {
    AgentDomain.CODE_ANALYSIS: DomainConfig(
        domain=AgentDomain.CODE_ANALYSIS,
        agents=["CodeAnalyzerAgent"],
        max_context_tokens=60_000,
        model="claude-sonnet-4-5",
        tools=["Read", "Glob", "Grep"],
    ),
    AgentDomain.UI_TESTING: DomainConfig(
        domain=AgentDomain.UI_TESTING,
        agents=["UITesterAgent"],
        max_context_tokens=80_000,  # Needs more for screenshots
        model="claude-sonnet-4-5",
        tools=["browser_pool", "screenshot", "click", "type"],
    ),
    AgentDomain.API_TESTING: DomainConfig(
        domain=AgentDomain.API_TESTING,
        agents=["APITesterAgent"],
        max_context_tokens=40_000,
        model="claude-sonnet-4-5",
        tools=["http_request", "schema_validate"],
    ),
    AgentDomain.SELF_HEALING: DomainConfig(
        domain=AgentDomain.SELF_HEALING,
        agents=["SelfHealerAgent"],
        max_context_tokens=60_000,
        model="claude-sonnet-4-5",
        tools=["Read", "Edit", "Grep"],
        requires_shared_context=True,  # Needs failure history
    ),
    AgentDomain.SECURITY: DomainConfig(
        domain=AgentDomain.SECURITY,
        agents=["SecurityAnalyzerAgent"],
        max_context_tokens=80_000,
        model="claude-opus-4-5",  # Use best model for security
        tools=["Read", "Grep", "vulnerability_scan"],
    ),
}


class SubgraphIsolator:
    """
    Manages isolated subgraph execution for different agent domains.

    Key benefits:
    - Context isolation saves 40-50% tokens
    - Parallel execution of independent domains
    - Clean separation of concerns
    """

    def __init__(self, configs: dict[AgentDomain, DomainConfig] | None = None):
        self.log = logger.bind(component="subgraph_isolator")
        self.configs = configs or DEFAULT_DOMAIN_CONFIGS

        # Active contexts by task_id and domain
        self._contexts: dict[str, dict[AgentDomain, IsolatedContext]] = {}

        # Metrics
        self.metrics = {
            "total_executions": 0,
            "parallel_executions": 0,
            "tokens_saved": 0,
            "by_domain": {d.value: {"count": 0, "tokens": 0} for d in AgentDomain},
        }

    def create_context(
        self,
        task_id: str,
        domain: AgentDomain,
        initial_context: str | None = None,
    ) -> IsolatedContext:
        """Create an isolated context for a domain."""
        if task_id not in self._contexts:
            self._contexts[task_id] = {}

        config = self.configs.get(domain)
        if not config:
            config = DomainConfig(domain=domain, agents=[])

        context = IsolatedContext(
            domain=domain,
            task_id=task_id,
        )

        # Add domain-specific system message
        system_prompt = self._build_domain_system_prompt(domain, config)
        context.add_message("system", system_prompt)

        # Add initial context if provided
        if initial_context:
            context.add_message("user", initial_context)

        self._contexts[task_id][domain] = context

        self.log.debug(
            "Created isolated context",
            task_id=task_id,
            domain=domain.value,
        )

        return context

    def _build_domain_system_prompt(self, domain: AgentDomain, config: DomainConfig) -> str:
        """Build a domain-specific system prompt."""
        prompts = {
            AgentDomain.CODE_ANALYSIS: """You are a code analysis specialist.
Your role is to analyze codebases and identify testable surfaces, patterns, and potential issues.
Focus on: file structure, function signatures, dependencies, and test coverage gaps.
Provide structured, actionable analysis.""",

            AgentDomain.UI_TESTING: """You are a UI testing specialist.
Your role is to execute browser-based tests and validate user interface behavior.
Focus on: element interactions, visual verification, user flows, and accessibility.
Report issues with precise selectors and reproducible steps.""",

            AgentDomain.API_TESTING: """You are an API testing specialist.
Your role is to test API endpoints and validate responses against schemas.
Focus on: request/response validation, error handling, edge cases, and performance.
Document findings with exact requests and responses.""",

            AgentDomain.SELF_HEALING: """You are a test self-healing specialist.
Your role is to analyze test failures and fix them automatically.
Focus on: selector changes, timing issues, data dependencies, and environment variations.
Provide fixes with high confidence and clear reasoning.""",

            AgentDomain.SECURITY: """You are a security analysis specialist.
Your role is to identify security vulnerabilities in code and configurations.
Focus on: OWASP Top 10, injection attacks, authentication issues, and data exposure.
Classify findings by severity and provide remediation guidance.""",
        }

        base_prompt = prompts.get(domain, f"You are a specialist in {domain.value}.")

        # Add tool context
        if config.tools:
            tool_list = ", ".join(config.tools)
            base_prompt += f"\n\nAvailable tools: {tool_list}"

        return base_prompt

    def get_context(self, task_id: str, domain: AgentDomain) -> IsolatedContext | None:
        """Get an existing isolated context."""
        return self._contexts.get(task_id, {}).get(domain)

    async def execute_in_domain(
        self,
        task_id: str,
        domain: AgentDomain,
        task: str,
        execute_fn: Callable[[IsolatedContext, str], Any],
        shared_data: dict[str, Any] | None = None,
    ) -> SubgraphResult:
        """
        Execute a task in an isolated domain context.

        Args:
            task_id: Unique task identifier
            domain: Target domain
            task: Task description
            execute_fn: Async function that executes the task
            shared_data: Optional data to share with the subgraph

        Returns:
            SubgraphResult with output and metrics
        """
        start_time = datetime.utcnow()

        # Get or create context
        context = self.get_context(task_id, domain)
        if not context:
            context = self.create_context(task_id, domain, task)
        else:
            context.add_message("user", task)

        # Add shared data if provided
        if shared_data:
            context.shared_data.update(shared_data)

        self.metrics["total_executions"] += 1
        self.metrics["by_domain"][domain.value]["count"] += 1

        try:
            result = await execute_fn(context, task)

            duration = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Extract tokens if available
            tokens_used = result.get("tokens_used", 0) if isinstance(result, dict) else 0
            self.metrics["by_domain"][domain.value]["tokens"] += tokens_used

            return SubgraphResult(
                domain=domain,
                success=True,
                output=result,
                tokens_used=tokens_used,
                duration_ms=int(duration),
            )

        except Exception as e:
            self.log.error(
                "Domain execution failed",
                task_id=task_id,
                domain=domain.value,
                error=str(e),
            )
            return SubgraphResult(
                domain=domain,
                success=False,
                output=None,
                error=str(e),
            )

    async def execute_parallel(
        self,
        task_id: str,
        domain_tasks: dict[AgentDomain, tuple[str, Callable]],
        shared_data: dict[str, Any] | None = None,
    ) -> dict[AgentDomain, SubgraphResult]:
        """
        Execute tasks in parallel across multiple domains.

        Args:
            task_id: Unique task identifier
            domain_tasks: Dict of domain -> (task, execute_fn)
            shared_data: Optional data shared across all domains

        Returns:
            Dict of domain -> SubgraphResult
        """
        self.metrics["parallel_executions"] += 1

        async def execute_one(domain: AgentDomain, task: str, fn: Callable):
            return domain, await self.execute_in_domain(
                task_id, domain, task, fn, shared_data
            )

        # Execute all domains in parallel
        tasks = [
            execute_one(domain, task, fn)
            for domain, (task, fn) in domain_tasks.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        result_dict = {}
        for result in results:
            if isinstance(result, Exception):
                self.log.error("Parallel execution error", error=str(result))
            else:
                domain, subgraph_result = result
                result_dict[domain] = subgraph_result

        self.log.info(
            "Parallel execution complete",
            task_id=task_id,
            domains=list(domain_tasks.keys()),
            success_count=sum(1 for r in result_dict.values() if r.success),
        )

        return result_dict

    def cleanup_task(self, task_id: str):
        """Clean up all contexts for a task."""
        if task_id in self._contexts:
            del self._contexts[task_id]

    def get_token_savings_estimate(self) -> dict[str, Any]:
        """
        Estimate token savings from isolation.

        Based on research showing 40-50% savings from context isolation.
        """
        total_domain_tokens = sum(
            self.metrics["by_domain"][d]["tokens"]
            for d in self.metrics["by_domain"]
        )

        # Without isolation, each domain would see ~2x the tokens
        # (from other domains' context bleeding in)
        estimated_without_isolation = total_domain_tokens * 2
        tokens_saved = estimated_without_isolation - total_domain_tokens

        return {
            "total_tokens_used": total_domain_tokens,
            "estimated_without_isolation": estimated_without_isolation,
            "tokens_saved": tokens_saved,
            "savings_percentage": (
                tokens_saved / estimated_without_isolation * 100
                if estimated_without_isolation > 0
                else 0
            ),
        }


class DomainOrchestrator:
    """
    Orchestrates work across multiple isolated domains.

    Implements the supervisor pattern from Anthropic's architecture:
    - Decomposes tasks into domain-specific subtasks
    - Routes to appropriate isolated subgraphs
    - Synthesizes results from multiple domains
    """

    def __init__(self, isolator: SubgraphIsolator | None = None):
        self.log = logger.bind(component="domain_orchestrator")
        self.isolator = isolator or SubgraphIsolator()

    def classify_task_domains(self, task: str) -> list[AgentDomain]:
        """
        Classify which domains are relevant for a task.

        Uses keyword matching and patterns to identify domains.
        """
        task_lower = task.lower()
        domains = []

        # Domain keyword mapping
        domain_keywords = {
            AgentDomain.CODE_ANALYSIS: [
                "analyze", "code", "function", "class", "module",
                "testable", "surface", "structure", "dependency",
            ],
            AgentDomain.UI_TESTING: [
                "ui", "browser", "click", "button", "form",
                "page", "element", "screenshot", "visual",
            ],
            AgentDomain.API_TESTING: [
                "api", "endpoint", "request", "response", "http",
                "rest", "graphql", "schema", "json",
            ],
            AgentDomain.SELF_HEALING: [
                "heal", "fix", "repair", "selector", "timeout",
                "flaky", "broken", "failure", "retry",
            ],
            AgentDomain.SECURITY: [
                "security", "vulnerability", "injection", "xss",
                "authentication", "authorization", "sensitive",
            ],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in task_lower for kw in keywords):
                domains.append(domain)

        # Default to code analysis if no specific domain found
        if not domains:
            domains.append(AgentDomain.CODE_ANALYSIS)

        self.log.debug(
            "Task domains classified",
            task_preview=task[:100],
            domains=[d.value for d in domains],
        )

        return domains

    async def execute_task(
        self,
        task_id: str,
        task: str,
        domain_executors: dict[AgentDomain, Callable],
        synthesize_fn: Callable[[dict[AgentDomain, SubgraphResult]], Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a task across relevant domains and synthesize results.

        Args:
            task_id: Unique task identifier
            task: Task description
            domain_executors: Mapping of domains to executor functions
            synthesize_fn: Optional function to synthesize results

        Returns:
            Combined results from all relevant domains
        """
        # Identify relevant domains
        domains = self.classify_task_domains(task)

        # Filter to domains we have executors for
        available_domains = [d for d in domains if d in domain_executors]

        if not available_domains:
            self.log.warning("No available executors for task domains")
            return {"error": "No available executors", "domains": domains}

        # Build domain tasks
        domain_tasks = {
            domain: (task, domain_executors[domain])
            for domain in available_domains
        }

        # Execute in parallel
        results = await self.isolator.execute_parallel(task_id, domain_tasks)

        # Synthesize results
        if synthesize_fn:
            synthesized = await synthesize_fn(results)
            return {
                "synthesized": synthesized,
                "domain_results": {d.value: r for d, r in results.items()},
            }

        return {
            "domain_results": {d.value: r for d, r in results.items()},
        }
