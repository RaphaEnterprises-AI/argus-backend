"""
Enterprise 5-Layer Guardrail Stack.

Based on best practices from regulated, high-stakes AI deployments.
Safety comes from layered guardrails that assume failure, monitor
continuously, and keep a human hand on the override.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable
import asyncio
import hashlib
import re

import structlog

logger = structlog.get_logger()


class RiskTier(Enum):
    """Risk classification for actions."""

    LOW = "low"  # Auto-approve (e.g., read operations)
    MEDIUM = "medium"  # Notify, proceed (e.g., modify test files)
    HIGH = "high"  # Require explicit approval (e.g., production changes)
    CRITICAL = "critical"  # Block until manual review (e.g., data deletion)


class GuardrailAction(Enum):
    """Actions a guardrail can take."""

    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    REQUIRE_APPROVAL = "require_approval"
    MODIFY = "modify"  # Allow but modify the request


@dataclass
class GuardrailResult:
    """Result of guardrail evaluation."""

    action: GuardrailAction
    layer: str
    reason: str | None = None
    modified_input: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def allowed(cls, layer: str = "default") -> "GuardrailResult":
        return cls(action=GuardrailAction.ALLOW, layer=layer)

    @classmethod
    def blocked(cls, reason: str, layer: str = "default") -> "GuardrailResult":
        return cls(action=GuardrailAction.BLOCK, layer=layer, reason=reason)

    @classmethod
    def warn(cls, reason: str, layer: str = "default") -> "GuardrailResult":
        return cls(action=GuardrailAction.WARN, layer=layer, reason=reason)

    @classmethod
    def require_approval(cls, reason: str, layer: str = "default") -> "GuardrailResult":
        return cls(action=GuardrailAction.REQUIRE_APPROVAL, layer=layer, reason=reason)

    @property
    def is_allowed(self) -> bool:
        return self.action in (GuardrailAction.ALLOW, GuardrailAction.WARN, GuardrailAction.MODIFY)

    @property
    def is_blocked(self) -> bool:
        return self.action == GuardrailAction.BLOCK

    @property
    def requires_approval(self) -> bool:
        return self.action == GuardrailAction.REQUIRE_APPROVAL


@dataclass
class AgentRequest:
    """Request from an agent that needs guardrail evaluation."""

    agent_id: str
    agent_name: str
    action_type: str  # e.g., "tool_call", "file_write", "api_request"
    action_name: str  # e.g., "Bash", "Edit", "database_query"
    input_data: Any
    context: dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    is_production: bool = False

    def get_fingerprint(self) -> str:
        """Generate a unique fingerprint for this request."""
        content = f"{self.agent_id}:{self.action_type}:{self.action_name}:{str(self.input_data)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class AgentIdentity:
    """Identity and permissions for an agent."""

    agent_id: str
    agent_name: str
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    max_budget_usd: float = 10.0
    rate_limit_per_minute: int = 60
    can_access_production: bool = False
    requires_approval_for: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    def is_valid(self) -> bool:
        """Check if identity is still valid."""
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True


class ActionClassifier:
    """Classifies actions into risk tiers."""

    # Actions by risk level
    LOW_RISK_ACTIONS = {
        "Read",
        "Glob",
        "Grep",
        "LS",
        "git status",
        "git log",
        "git diff",
        "npm list",
        "pip list",
    }

    MEDIUM_RISK_ACTIONS = {
        "Edit",
        "Write",
        "NotebookEdit",
        "git add",
        "git commit",
        "npm install",
        "pip install",
    }

    HIGH_RISK_ACTIONS = {
        "Bash",  # General bash is high risk
        "git push",
        "git reset",
        "rm",
        "delete",
        "drop",
        "truncate",
    }

    CRITICAL_RISK_ACTIONS = {
        "git push --force",
        "rm -rf",
        "DROP TABLE",
        "DELETE FROM",
        "production deploy",
        "database migration",
    }

    # Patterns that indicate higher risk
    RISK_PATTERNS = {
        RiskTier.HIGH: [
            r"--force",
            r"--hard",
            r"-rf\s",
            r"sudo\s",
            r"production",
            r"prod\.",
        ],
        RiskTier.CRITICAL: [
            r"DROP\s+(TABLE|DATABASE)",
            r"DELETE\s+FROM.*WHERE\s*1\s*=\s*1",
            r"TRUNCATE",
            r"rm\s+-rf\s+/",
        ],
    }

    def classify(self, request: AgentRequest) -> RiskTier:
        """Classify the risk tier of an action."""
        action_name = request.action_name
        input_str = str(request.input_data).lower()

        # Check critical patterns first
        for pattern in self.RISK_PATTERNS.get(RiskTier.CRITICAL, []):
            if re.search(pattern, input_str, re.IGNORECASE):
                return RiskTier.CRITICAL

        # Check high risk patterns
        for pattern in self.RISK_PATTERNS.get(RiskTier.HIGH, []):
            if re.search(pattern, input_str, re.IGNORECASE):
                return RiskTier.HIGH

        # Check action name against known categories
        if action_name in self.CRITICAL_RISK_ACTIONS:
            return RiskTier.CRITICAL

        if action_name in self.HIGH_RISK_ACTIONS:
            return RiskTier.HIGH

        if action_name in self.MEDIUM_RISK_ACTIONS:
            return RiskTier.MEDIUM

        if action_name in self.LOW_RISK_ACTIONS:
            return RiskTier.LOW

        # Production flag bumps risk
        if request.is_production:
            return RiskTier.HIGH

        # Default to medium for unknown actions
        return RiskTier.MEDIUM


class GuardrailStack:
    """
    Enterprise 5-layer guardrail implementation.

    Layers:
    L1: Identity & Access - Validate agent identity and permissions
    L2: Input Validation - Check for prompt injection, sanitize inputs
    L3: Execution Boundaries - Rate limits, quotas, sandboxing
    L4: Output Filtering - Content policy, PII redaction, hallucination
    L5: Human-in-the-Loop - Risk-tiered approval workflows
    """

    def __init__(
        self,
        enable_hitl: bool = True,
        approval_callback: Callable[[AgentRequest, RiskTier], bool] | None = None,
    ):
        self.log = logger.bind(component="guardrail_stack")
        self.enable_hitl = enable_hitl
        self.approval_callback = approval_callback

        # Identity store
        self.identities: dict[str, AgentIdentity] = {}

        # Rate limiting
        self.rate_limits: dict[str, list[datetime]] = {}

        # Budget tracking
        self.budget_spent: dict[str, float] = {}

        # Action classifier
        self.classifier = ActionClassifier()

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "allowed": 0,
            "blocked": 0,
            "approvals_required": 0,
            "approvals_granted": 0,
            "by_layer": {f"L{i}": {"blocked": 0, "warned": 0} for i in range(1, 6)},
        }

    def register_agent(self, identity: AgentIdentity):
        """Register an agent identity."""
        self.identities[identity.agent_id] = identity
        self.budget_spent[identity.agent_id] = 0.0
        self.rate_limits[identity.agent_id] = []
        self.log.info(
            "Agent registered",
            agent_id=identity.agent_id,
            agent_name=identity.agent_name,
            allowed_tools=list(identity.allowed_tools)[:5],
        )

    async def process(self, request: AgentRequest) -> GuardrailResult:
        """
        Process a request through all guardrail layers.

        Returns the first blocking result, or allowed if all pass.
        """
        self.metrics["total_requests"] += 1

        # L1: Identity & Access
        result = await self._check_identity(request)
        if not result.is_allowed:
            self.metrics["blocked"] += 1
            self.metrics["by_layer"]["L1"]["blocked"] += 1
            return result

        # L2: Input Validation
        result = await self._validate_input(request)
        if not result.is_allowed:
            self.metrics["blocked"] += 1
            self.metrics["by_layer"]["L2"]["blocked"] += 1
            return result

        # L3: Execution Boundaries
        result = await self._check_boundaries(request)
        if not result.is_allowed:
            self.metrics["blocked"] += 1
            self.metrics["by_layer"]["L3"]["blocked"] += 1
            return result

        # L4: Output Filtering (pre-check for known patterns)
        result = await self._precheck_output(request)
        if not result.is_allowed:
            self.metrics["blocked"] += 1
            self.metrics["by_layer"]["L4"]["blocked"] += 1
            return result

        # L5: Human-in-the-Loop (risk-based)
        result = await self._check_hitl(request)
        if result.requires_approval:
            self.metrics["approvals_required"] += 1
            if self.approval_callback:
                risk_tier = self.classifier.classify(request)
                approved = self.approval_callback(request, risk_tier)
                if approved:
                    self.metrics["approvals_granted"] += 1
                    result = GuardrailResult.allowed("L5_approved")
                else:
                    self.metrics["blocked"] += 1
                    self.metrics["by_layer"]["L5"]["blocked"] += 1
                    return GuardrailResult.blocked("Human approval denied", "L5")
            else:
                # No callback, block by default for high-risk
                self.metrics["blocked"] += 1
                return result

        self.metrics["allowed"] += 1
        return GuardrailResult.allowed("all_layers")

    async def _check_identity(self, request: AgentRequest) -> GuardrailResult:
        """L1: Identity & Access Control."""
        identity = self.identities.get(request.agent_id)

        if not identity:
            return GuardrailResult.blocked(
                f"Unknown agent identity: {request.agent_id}", "L1"
            )

        if not identity.is_valid():
            return GuardrailResult.blocked(
                f"Agent identity expired: {request.agent_id}", "L1"
            )

        # Check tool permissions
        tool_name = request.action_name
        if identity.denied_tools and tool_name in identity.denied_tools:
            return GuardrailResult.blocked(
                f"Tool '{tool_name}' denied for agent {request.agent_name}", "L1"
            )

        if identity.allowed_tools and tool_name not in identity.allowed_tools:
            # If allowed_tools is set, it's a whitelist
            return GuardrailResult.blocked(
                f"Tool '{tool_name}' not in allowed list for agent {request.agent_name}",
                "L1",
            )

        # Check production access
        if request.is_production and not identity.can_access_production:
            return GuardrailResult.blocked(
                f"Agent {request.agent_name} cannot access production", "L1"
            )

        return GuardrailResult.allowed("L1")

    async def _validate_input(self, request: AgentRequest) -> GuardrailResult:
        """L2: Input Validation."""
        input_str = str(request.input_data)

        # Check for prompt injection patterns
        injection_patterns = [
            r"ignore\s+(previous|all|above)\s+instructions",
            r"disregard\s+(your|all)\s+(rules|instructions)",
            r"you\s+are\s+now\s+in\s+developer\s+mode",
            r"system:\s*you\s+are",
            r"</?(system|user|assistant)>",
            r"IMPORTANT:\s*ignore",
        ]

        for pattern in injection_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                self.log.warning(
                    "Prompt injection detected",
                    agent_id=request.agent_id,
                    pattern=pattern,
                )
                return GuardrailResult.blocked(
                    "Potential prompt injection detected", "L2"
                )

        # Check for dangerous shell commands in Bash actions
        if request.action_type == "tool_call" and request.action_name == "Bash":
            dangerous_patterns = [
                r"curl.*\|\s*(ba)?sh",  # Pipe to shell
                r"wget.*-O\s*-.*\|\s*(ba)?sh",
                r"eval\s*\$",
                r"base64\s+-d.*\|",
                r">\s*/dev/sd[a-z]",  # Direct disk write
            ]
            for pattern in dangerous_patterns:
                if re.search(pattern, input_str, re.IGNORECASE):
                    return GuardrailResult.blocked(
                        f"Dangerous command pattern detected: {pattern}", "L2"
                    )

        return GuardrailResult.allowed("L2")

    async def _check_boundaries(self, request: AgentRequest) -> GuardrailResult:
        """L3: Execution Boundaries (rate limits, quotas)."""
        identity = self.identities.get(request.agent_id)
        if not identity:
            return GuardrailResult.allowed("L3")  # Already checked in L1

        # Rate limiting
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        # Clean old entries
        self.rate_limits[request.agent_id] = [
            ts
            for ts in self.rate_limits.get(request.agent_id, [])
            if ts > window_start
        ]

        # Check rate
        current_rate = len(self.rate_limits[request.agent_id])
        if current_rate >= identity.rate_limit_per_minute:
            return GuardrailResult.blocked(
                f"Rate limit exceeded ({current_rate}/{identity.rate_limit_per_minute} per minute)",
                "L3",
            )

        # Record this request
        self.rate_limits[request.agent_id].append(now)

        # Budget check
        total_spent = self.budget_spent.get(request.agent_id, 0.0)
        if total_spent + request.estimated_cost > identity.max_budget_usd:
            return GuardrailResult.blocked(
                f"Budget exceeded (${total_spent:.2f} + ${request.estimated_cost:.2f} > ${identity.max_budget_usd:.2f})",
                "L3",
            )

        return GuardrailResult.allowed("L3")

    async def _precheck_output(self, request: AgentRequest) -> GuardrailResult:
        """L4: Output pre-filtering (check for known bad patterns in input)."""
        # This is a pre-check; full output filtering happens post-execution
        input_str = str(request.input_data)

        # Check for PII exposure attempts
        pii_patterns = [
            r"password\s*=",
            r"api[_-]?key\s*=",
            r"secret\s*=",
            r"token\s*=",
            r"bearer\s+[a-zA-Z0-9\-_]+",
        ]

        for pattern in pii_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                self.log.warning(
                    "Potential credential exposure",
                    agent_id=request.agent_id,
                    pattern=pattern,
                )
                # Warn but don't block - might be legitimate
                return GuardrailResult.warn(
                    "Request may expose sensitive credentials", "L4"
                )

        return GuardrailResult.allowed("L4")

    async def _check_hitl(self, request: AgentRequest) -> GuardrailResult:
        """L5: Human-in-the-Loop for high-risk actions."""
        if not self.enable_hitl:
            return GuardrailResult.allowed("L5")

        risk_tier = self.classifier.classify(request)

        if risk_tier == RiskTier.CRITICAL:
            return GuardrailResult.require_approval(
                f"Critical action requires manual review: {request.action_name}", "L5"
            )

        if risk_tier == RiskTier.HIGH:
            identity = self.identities.get(request.agent_id)
            # Check if this action is in the agent's requires_approval_for set
            if identity and request.action_name in identity.requires_approval_for:
                return GuardrailResult.require_approval(
                    f"High-risk action requires approval: {request.action_name}", "L5"
                )

        return GuardrailResult.allowed("L5")

    def record_cost(self, agent_id: str, cost: float):
        """Record cost incurred by an agent."""
        if agent_id in self.budget_spent:
            self.budget_spent[agent_id] += cost

    def get_metrics(self) -> dict[str, Any]:
        """Get guardrail metrics."""
        return {
            **self.metrics,
            "block_rate": (
                self.metrics["blocked"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0
                else 0
            ),
            "approval_grant_rate": (
                self.metrics["approvals_granted"] / self.metrics["approvals_required"]
                if self.metrics["approvals_required"] > 0
                else 0
            ),
        }

    async def filter_output(self, output: str) -> tuple[str, list[str]]:
        """
        L4 post-execution: Filter agent output for sensitive content.

        Returns:
            Tuple of (filtered_output, list of redaction reasons)
        """
        redactions = []
        filtered = output

        # Redact potential API keys
        api_key_pattern = r"(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([a-zA-Z0-9\-_]{20,})['\"]?"
        if re.search(api_key_pattern, output, re.IGNORECASE):
            filtered = re.sub(
                api_key_pattern,
                r"\1=[REDACTED]",
                filtered,
                flags=re.IGNORECASE,
            )
            redactions.append("API key/secret redacted")

        # Redact email addresses (partial)
        email_pattern = r"([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
        if re.search(email_pattern, output):
            filtered = re.sub(
                email_pattern,
                r"[EMAIL]@\2",
                filtered,
            )
            redactions.append("Email addresses partially redacted")

        return filtered, redactions
