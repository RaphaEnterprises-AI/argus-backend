"""3-layer throttler for agent swarm execution.

Layer 1: Semaphore-based concurrency limiting for AI calls.
Layer 2: Per-org swarm count limiting.
Layer 3: Cost budget enforcement per swarm and per org per hour.

Supports tiered configurations (FREE, PRO, ENTERPRISE) for scaling
from 8 to 100 concurrent workers per swarm.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class ThrottlerTier(str, Enum):
    """Organization throttling tiers."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class ThrottlerConfig:
    """Configuration for the 3-layer swarm throttler."""

    # Layer 1: Concurrent AI call limit (semaphore)
    max_concurrent_calls: int = 8

    # Layer 2: Worker and swarm limits
    max_workers_per_swarm: int = 8
    max_concurrent_swarms_per_org: int = 3

    # Layer 3: Cost budgets (USD)
    max_cost_per_swarm: float = 2.00
    max_cost_per_org_per_hour: float = 10.00


# Pre-built configs for each tier
TIER_CONFIGS: dict[ThrottlerTier, ThrottlerConfig] = {
    ThrottlerTier.FREE: ThrottlerConfig(
        max_concurrent_calls=8,
        max_workers_per_swarm=8,
        max_concurrent_swarms_per_org=3,
        max_cost_per_swarm=2.00,
        max_cost_per_org_per_hour=10.00,
    ),
    ThrottlerTier.PRO: ThrottlerConfig(
        max_concurrent_calls=25,
        max_workers_per_swarm=50,
        max_concurrent_swarms_per_org=5,
        max_cost_per_swarm=10.00,
        max_cost_per_org_per_hour=50.00,
    ),
    ThrottlerTier.ENTERPRISE: ThrottlerConfig(
        max_concurrent_calls=50,
        max_workers_per_swarm=100,
        max_concurrent_swarms_per_org=10,
        max_cost_per_swarm=50.00,
        max_cost_per_org_per_hour=200.00,
    ),
}


class SwarmThrottler:
    """3-layer throttler controlling concurrency, swarm counts, and cost budgets.

    Layer 1 - Semaphore: Limits total concurrent AI calls across all swarms.
    Layer 2 - Swarm tracking: Limits concurrent swarms per organization.
    Layer 3 - Cost tracking: Enforces per-swarm and per-org-per-hour budgets.

    Supports per-org tier overrides via set_org_tier().
    """

    def __init__(self, config: ThrottlerConfig | None = None) -> None:
        self.config = config or ThrottlerConfig()

        # Layer 1: Global concurrency semaphore (default tier)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)

        # Per-org semaphores for tiered concurrency
        self._org_semaphores: dict[str, asyncio.Semaphore] = {}

        # Layer 2: org_id -> set of active swarm_ids
        self._active_swarms: dict[str, set[str]] = {}

        # Layer 3: swarm_id -> accumulated cost (USD)
        self._swarm_costs: dict[str, float] = {}
        # org_id -> list of (timestamp, cost) tuples for hourly tracking
        self._org_hourly_costs: dict[str, list[tuple[float, float]]] = {}

        # Tier overrides: org_id -> ThrottlerTier
        self._org_tiers: dict[str, ThrottlerTier] = {}

    # ── Tier management ───────────────────────────────────────────────

    def set_org_tier(self, org_id: str, tier: ThrottlerTier) -> None:
        """Set the throttling tier for an organization.

        Creates a per-org semaphore with the tier's concurrency limit.
        """
        self._org_tiers[org_id] = tier
        tier_config = TIER_CONFIGS[tier]
        self._org_semaphores[org_id] = asyncio.Semaphore(tier_config.max_concurrent_calls)

    def _get_config_for_org(self, org_id: str) -> ThrottlerConfig:
        """Get the effective ThrottlerConfig for an org (tier override or default)."""
        tier = self._org_tiers.get(org_id)
        if tier is not None:
            return TIER_CONFIGS[tier]
        return self.config

    # ── Layer 2: Swarm reservation ──────────────────────────────────────

    def reserve_swarm(self, org_id: str, swarm_id: str) -> None:
        """Reserve a swarm slot for the given organization.

        Raises:
            RuntimeError: If the org has reached its concurrent swarm limit.
        """
        cfg = self._get_config_for_org(org_id)

        if org_id not in self._active_swarms:
            self._active_swarms[org_id] = set()

        if len(self._active_swarms[org_id]) >= cfg.max_concurrent_swarms_per_org:
            raise RuntimeError(
                f"Organization {org_id} has reached the maximum of "
                f"{cfg.max_concurrent_swarms_per_org} concurrent swarms"
            )

        self._active_swarms[org_id].add(swarm_id)
        self._swarm_costs[swarm_id] = 0.0

    def release_swarm(self, org_id: str, swarm_id: str) -> None:
        """Release a swarm slot for the given organization."""
        if org_id in self._active_swarms:
            self._active_swarms[org_id].discard(swarm_id)
            if not self._active_swarms[org_id]:
                del self._active_swarms[org_id]

        self._swarm_costs.pop(swarm_id, None)

    # ── Layer 3: Cost budget ────────────────────────────────────────────

    def check_budget(self, org_id: str, swarm_id: str, additional_cost: float = 0.0) -> None:
        """Check if the swarm and org budgets can accommodate the additional cost.

        Raises:
            RuntimeError: If either budget would be exceeded.
        """
        cfg = self._get_config_for_org(org_id)

        # Check per-swarm budget
        current_swarm_cost = self._swarm_costs.get(swarm_id, 0.0)
        if current_swarm_cost + additional_cost > cfg.max_cost_per_swarm:
            raise RuntimeError(
                f"Swarm {swarm_id} would exceed budget: "
                f"${current_swarm_cost + additional_cost:.2f} > "
                f"${cfg.max_cost_per_swarm:.2f}"
            )

        # Check per-org hourly budget
        now = time.time()
        one_hour_ago = now - 3600.0
        hourly_entries = self._org_hourly_costs.get(org_id, [])

        # Prune entries older than 1 hour
        hourly_entries = [(ts, cost) for ts, cost in hourly_entries if ts > one_hour_ago]
        self._org_hourly_costs[org_id] = hourly_entries

        org_hourly_total = sum(cost for _, cost in hourly_entries) + additional_cost
        if org_hourly_total > cfg.max_cost_per_org_per_hour:
            raise RuntimeError(
                f"Organization {org_id} would exceed hourly budget: "
                f"${org_hourly_total:.2f} > "
                f"${cfg.max_cost_per_org_per_hour:.2f}"
            )

    def record_cost(self, org_id: str, swarm_id: str, cost: float) -> None:
        """Record an incurred cost against both swarm and org budgets."""
        self._swarm_costs[swarm_id] = self._swarm_costs.get(swarm_id, 0.0) + cost

        if org_id not in self._org_hourly_costs:
            self._org_hourly_costs[org_id] = []
        self._org_hourly_costs[org_id].append((time.time(), cost))

    # ── Layer 1 + combined: Worker acquisition ──────────────────────────

    @asynccontextmanager
    async def acquire_worker(
        self, org_id: str, swarm_id: str
    ) -> AsyncIterator[None]:
        """Acquire a worker slot, enforcing all 3 throttling layers.

        Uses per-org semaphore if a tier is set, otherwise the global semaphore.

        Usage::

            async with throttler.acquire_worker(org_id, swarm_id):
                # AI call happens here — concurrency is limited
                result = await call_ai(...)
                throttler.record_cost(org_id, swarm_id, result.cost)
        """
        # Layer 3: Pre-flight budget check (zero additional — just verify not already over)
        self.check_budget(org_id, swarm_id)

        # Layer 1: Acquire semaphore slot (per-org if tiered, else global)
        sem = self._org_semaphores.get(org_id, self._semaphore)
        async with sem:
            yield


# ── Singleton ───────────────────────────────────────────────────────────

_throttler_instance: SwarmThrottler | None = None


def get_swarm_throttler(config: ThrottlerConfig | None = None) -> SwarmThrottler:
    """Get or create the singleton SwarmThrottler instance."""
    global _throttler_instance
    if _throttler_instance is None:
        _throttler_instance = SwarmThrottler(config)
    return _throttler_instance
