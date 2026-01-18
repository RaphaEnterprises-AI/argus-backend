"""Feature Mesh Integration - Connects Discovery to Visual AI and Self-Healing.

This module implements the "feature mesh" architecture where discovery insights
automatically flow to other Argus systems:

1. Discovery → Visual AI: Auto-create baselines from discovered pages
2. Discovery → Self-Healing: Share selector alternatives for resilient tests

This creates a feedback loop where:
- Discovery finds pages and elements
- Visual AI monitors them for regressions
- Self-Healing uses alternative selectors when tests break
- Patterns flow back to improve future discoveries
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


@dataclass
class FeatureMeshConfig:
    """Configuration for feature mesh integrations."""

    # Visual AI integration
    auto_create_baselines: bool = True
    baseline_viewports: list[dict[str, int]] = field(default_factory=lambda: [
        {"name": "mobile", "width": 375, "height": 667},
        {"name": "desktop", "width": 1440, "height": 900},
    ])
    baseline_browser: str = "chromium"

    # Self-Healing integration
    share_selectors: bool = True
    min_selector_stability: float = 0.5
    max_alternatives_per_element: int = 5

    # General
    async_processing: bool = True


class FeatureMeshIntegration:
    """Connects Discovery to Visual AI and Self-Healing systems.

    ★ Insight ─────────────────────────────────────────
    The "feature mesh" is an architectural pattern where AI systems share
    learnings automatically. Discovery finds elements → Visual AI monitors
    them → Self-Healing uses alternatives when tests break → all learnings
    feed back to improve future discoveries.
    ─────────────────────────────────────────────────────
    """

    def __init__(self, config: FeatureMeshConfig | None = None):
        self.config = config or FeatureMeshConfig()
        self.supabase = get_supabase_client()
        self.log = logger.bind(component="feature_mesh")

    # =========================================================================
    # Discovery → Visual AI Integration
    # =========================================================================

    async def create_baselines_from_discovery(
        self,
        session_id: str,
        project_id: str,
        pages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Auto-create visual baselines from discovered pages.

        Each discovered page becomes a visual baseline that can be monitored
        for regressions. This creates proactive visual monitoring coverage.

        Args:
            session_id: Discovery session ID
            project_id: Project ID for organizing baselines
            pages: List of discovered page data

        Returns:
            Summary of created baselines
        """
        self.log.info(
            "Creating visual baselines from discovery",
            session_id=session_id,
            page_count=len(pages)
        )

        created = []
        skipped = []
        errors = []

        for page in pages:
            url = page.get("url")
            title = page.get("title") or page.get("page_title")
            category = page.get("category", "other")

            if not url:
                continue

            # Generate baseline name from page info
            baseline_name = self._generate_baseline_name(url, title, category)

            try:
                # Check if baseline already exists
                existing = await self.supabase.select(
                    "visual_baselines",
                    filters={
                        "project_id": f"eq.{project_id}",
                        "url": f"eq.{url}",
                    }
                )

                if existing.get("data"):
                    skipped.append({
                        "url": url,
                        "reason": "baseline_exists",
                        "baseline_id": existing["data"][0]["id"]
                    })
                    continue

                # Create baseline record (actual screenshot capture happens async)
                baseline_id = f"bl_{uuid.uuid4().hex[:12]}"

                baseline_record = {
                    "id": baseline_id,
                    "name": baseline_name,
                    "project_id": project_id,
                    "url": url,
                    "version": 1,
                    "status": "pending_capture",  # Will be captured by background job
                    "source": "discovery",
                    "discovery_session_id": session_id,
                    "category": category,
                    "metadata": {
                        "page_title": title,
                        "category": category,
                        "discovered_at": datetime.now(UTC).isoformat(),
                        "elements_count": len(page.get("actions", [])),
                    },
                    "viewport_width": self.config.baseline_viewports[0]["width"],
                    "viewport_height": self.config.baseline_viewports[0]["height"],
                    "browser": self.config.baseline_browser,
                    "created_at": datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                }

                result = await self.supabase.insert("visual_baselines", baseline_record)

                if result.get("error"):
                    errors.append({
                        "url": url,
                        "error": str(result["error"])
                    })
                else:
                    created.append({
                        "url": url,
                        "baseline_id": baseline_id,
                        "name": baseline_name,
                    })

            except Exception as e:
                errors.append({
                    "url": url,
                    "error": str(e)
                })

        summary = {
            "session_id": session_id,
            "project_id": project_id,
            "total_pages": len(pages),
            "baselines_created": len(created),
            "baselines_skipped": len(skipped),
            "errors": len(errors),
            "created": created,
            "skipped": skipped,
            "errors_detail": errors,
        }

        self.log.info(
            "Baselines created from discovery",
            created=len(created),
            skipped=len(skipped),
            errors=len(errors)
        )

        # Link baselines back to discovery session
        await self._link_baselines_to_session(session_id, created)

        return summary

    async def _link_baselines_to_session(
        self,
        session_id: str,
        baselines: list[dict]
    ):
        """Link created baselines back to discovery session for tracking."""
        if not baselines:
            return

        baseline_ids = [b["baseline_id"] for b in baselines]

        await self.supabase.update(
            "discovery_sessions",
            {"id": f"eq.{session_id}"},
            {"visual_baseline_ids": baseline_ids}
        )

    def _generate_baseline_name(
        self,
        url: str,
        title: str | None,
        category: str
    ) -> str:
        """Generate a descriptive baseline name from page info."""
        if title:
            # Clean and truncate title
            name = title.strip()[:50]
        else:
            # Use URL path
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path.strip("/")
            if path:
                name = path.replace("/", " > ")[:50]
            else:
                name = parsed.netloc

        # Add category prefix for organization
        category_prefix = {
            "auth_login": "[Login]",
            "auth_signup": "[Signup]",
            "dashboard": "[Dashboard]",
            "landing": "[Landing]",
            "settings": "[Settings]",
            "profile": "[Profile]",
            "checkout": "[Checkout]",
        }.get(category, "")

        if category_prefix:
            name = f"{category_prefix} {name}"

        return name

    # =========================================================================
    # Discovery → Self-Healing Integration
    # =========================================================================

    async def share_selectors_with_self_healer(
        self,
        session_id: str,
        project_id: str,
        elements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Share discovered selector alternatives with Self-Healing system.

        When discovery finds elements, it often identifies multiple ways to
        select them (CSS, XPath, text, aria-label, etc.). These alternatives
        are invaluable for self-healing when primary selectors break.

        ★ Insight ─────────────────────────────────────────
        Self-healing works by trying alternative selectors when the primary
        one fails. Discovery provides a rich source of alternatives because
        it analyzes the full DOM context, not just a single test scenario.
        ─────────────────────────────────────────────────────

        Args:
            session_id: Discovery session ID
            project_id: Project ID
            elements: List of discovered element data

        Returns:
            Summary of shared selectors
        """
        self.log.info(
            "Sharing selectors with self-healer",
            session_id=session_id,
            element_count=len(elements)
        )

        shared = []
        errors = []

        for element in elements:
            try:
                selector = element.get("selector")
                if not selector:
                    continue

                # Extract alternative selectors from element data
                alternatives = self._extract_alternative_selectors(element)

                if not alternatives:
                    continue

                # Generate fingerprint for deduplication
                fingerprint = self._generate_selector_fingerprint(selector, alternatives)

                # Store in selector_alternatives table
                record = {
                    "id": str(uuid.uuid4()),
                    "project_id": project_id,
                    "primary_selector": selector,
                    "alternatives": alternatives,
                    "fingerprint": fingerprint,
                    "source": "discovery",
                    "discovery_session_id": session_id,
                    "element_type": element.get("category", element.get("type", "unknown")),
                    "element_label": element.get("label") or element.get("purpose"),
                    "page_url": element.get("page_url"),
                    "stability_score": element.get("stability_score", 0.5),
                    "usage_count": 0,
                    "success_count": 0,
                    "created_at": datetime.now(UTC).isoformat(),
                    "updated_at": datetime.now(UTC).isoformat(),
                }

                # Upsert (update if fingerprint exists)
                result = await self._upsert_selector_alternative(record)

                if result:
                    shared.append({
                        "selector": selector,
                        "alternatives_count": len(alternatives),
                        "element_type": record["element_type"],
                    })

            except Exception as e:
                errors.append({
                    "selector": element.get("selector", "unknown"),
                    "error": str(e)
                })

        summary = {
            "session_id": session_id,
            "project_id": project_id,
            "total_elements": len(elements),
            "selectors_shared": len(shared),
            "errors": len(errors),
            "shared": shared,
            "errors_detail": errors,
        }

        self.log.info(
            "Selectors shared with self-healer",
            shared=len(shared),
            errors=len(errors)
        )

        return summary

    def _extract_alternative_selectors(
        self,
        element: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract alternative selectors from discovered element data.

        Builds a prioritized list of alternative selectors based on
        different strategies: CSS, XPath, text, aria-label, data attributes.
        """
        alternatives = []

        # Check for explicitly provided alternatives
        if element.get("alternative_selectors"):
            for alt in element["alternative_selectors"][:self.config.max_alternatives_per_element]:
                alternatives.append({
                    "selector": alt,
                    "strategy": "discovered",
                    "confidence": 0.8,
                })

        # Check for XPath
        xpath = element.get("xpath")
        if xpath:
            alternatives.append({
                "selector": xpath,
                "strategy": "xpath",
                "confidence": 0.7,
            })

        # Check for text content (for buttons, links)
        label = element.get("label") or element.get("text_content")
        if label and len(label) <= 50:
            # Build text-based selectors
            tag = element.get("tag_name", "*")
            text_selector = f"//{tag}[contains(text(), '{label[:30]}')]"
            alternatives.append({
                "selector": text_selector,
                "strategy": "text_content",
                "confidence": 0.6,
            })

        # Check for aria-label
        aria_label = element.get("aria_label")
        if aria_label:
            alternatives.append({
                "selector": f"[aria-label='{aria_label}']",
                "strategy": "aria_label",
                "confidence": 0.9,  # High confidence for accessibility selectors
            })

        # Check for role
        role = element.get("role")
        if role and label:
            alternatives.append({
                "selector": f"[role='{role}'][aria-label*='{label[:20]}']",
                "strategy": "role_label",
                "confidence": 0.85,
            })

        # Check for data attributes
        if element.get("html_attributes"):
            attrs = element["html_attributes"]
            for attr_name, attr_value in attrs.items():
                if attr_name.startswith("data-") and attr_value:
                    alternatives.append({
                        "selector": f"[{attr_name}='{attr_value}']",
                        "strategy": "data_attribute",
                        "confidence": 0.75,
                    })
                    break  # Only add one data attribute selector

        # Sort by confidence descending
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)

        return alternatives[:self.config.max_alternatives_per_element]

    def _generate_selector_fingerprint(
        self,
        primary_selector: str,
        alternatives: list[dict]
    ) -> str:
        """Generate a unique fingerprint for selector deduplication."""
        alt_str = ",".join(sorted(a["selector"] for a in alternatives))
        key = f"{primary_selector}:{alt_str}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    async def _upsert_selector_alternative(
        self,
        record: dict[str, Any]
    ) -> bool:
        """Insert or update selector alternative record."""
        try:
            # Check for existing by fingerprint
            existing = await self.supabase.select(
                "selector_alternatives",
                filters={"fingerprint": f"eq.{record['fingerprint']}"}
            )

            if existing.get("data"):
                # Update existing record
                existing_record = existing["data"][0]
                await self.supabase.update(
                    "selector_alternatives",
                    {"id": f"eq.{existing_record['id']}"},
                    {
                        "alternatives": record["alternatives"],
                        "updated_at": record["updated_at"],
                    }
                )
            else:
                # Insert new record
                await self.supabase.insert("selector_alternatives", record)

            return True

        except Exception as e:
            self.log.warning(
                "Failed to upsert selector alternative",
                error=str(e),
                fingerprint=record.get("fingerprint")
            )
            return False

    # =========================================================================
    # Self-Healing → Discovery Feedback
    # =========================================================================

    async def record_healing_feedback(
        self,
        primary_selector: str,
        used_alternative: str,
        success: bool,
        project_id: str,
    ) -> bool:
        """Record healing feedback to improve selector quality scores.

        When self-healing uses an alternative selector and it works/fails,
        that feedback is recorded to improve future recommendations.

        Args:
            primary_selector: The original broken selector
            used_alternative: The alternative that was tried
            success: Whether the alternative worked
            project_id: Project ID

        Returns:
            True if feedback was recorded
        """
        try:
            # Find the selector alternative record
            existing = await self.supabase.select(
                "selector_alternatives",
                filters={
                    "project_id": f"eq.{project_id}",
                    "primary_selector": f"eq.{primary_selector}",
                }
            )

            if not existing.get("data"):
                return False

            record = existing["data"][0]

            # Update usage counts
            usage_count = record.get("usage_count", 0) + 1
            success_count = record.get("success_count", 0) + (1 if success else 0)

            # Update alternatives with this feedback
            alternatives = record.get("alternatives", [])
            for alt in alternatives:
                if alt.get("selector") == used_alternative:
                    # Adjust confidence based on success
                    current_conf = alt.get("confidence", 0.5)
                    if success:
                        alt["confidence"] = min(1.0, current_conf + 0.05)
                    else:
                        alt["confidence"] = max(0.1, current_conf - 0.1)
                    alt["last_used"] = datetime.now(UTC).isoformat()
                    alt["usage_count"] = alt.get("usage_count", 0) + 1
                    break

            # Update record
            await self.supabase.update(
                "selector_alternatives",
                {"id": f"eq.{record['id']}"},
                {
                    "usage_count": usage_count,
                    "success_count": success_count,
                    "alternatives": alternatives,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
            )

            return True

        except Exception as e:
            self.log.warning(
                "Failed to record healing feedback",
                error=str(e)
            )
            return False

    # =========================================================================
    # Combined Integration Entry Point
    # =========================================================================

    async def process_discovery_completion(
        self,
        session_id: str,
        project_id: str,
        pages: list[dict[str, Any]],
        elements: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Process completed discovery session for all feature integrations.

        This is the main entry point called when a discovery session completes.
        It triggers all feature mesh integrations in parallel.

        Args:
            session_id: Completed discovery session ID
            project_id: Project ID
            pages: List of discovered pages
            elements: List of discovered elements

        Returns:
            Combined summary of all integrations
        """
        self.log.info(
            "Processing discovery for feature mesh",
            session_id=session_id,
            pages=len(pages),
            elements=len(elements)
        )

        results = {}

        # Run integrations in parallel
        tasks = []

        if self.config.auto_create_baselines and pages:
            tasks.append(("visual_ai", self.create_baselines_from_discovery(
                session_id, project_id, pages
            )))

        if self.config.share_selectors and elements:
            tasks.append(("self_healing", self.share_selectors_with_self_healer(
                session_id, project_id, elements
            )))

        # Execute in parallel
        for name, coro in tasks:
            try:
                results[name] = await coro
            except Exception as e:
                self.log.error(
                    f"Feature mesh integration failed: {name}",
                    error=str(e)
                )
                results[name] = {"error": str(e)}

        return {
            "session_id": session_id,
            "project_id": project_id,
            "integrations": results,
            "processed_at": datetime.now(UTC).isoformat(),
        }


# Singleton instance
_feature_mesh: FeatureMeshIntegration | None = None


def get_feature_mesh(config: FeatureMeshConfig | None = None) -> FeatureMeshIntegration:
    """Get or create the feature mesh integration instance."""
    global _feature_mesh
    if _feature_mesh is None:
        _feature_mesh = FeatureMeshIntegration(config)
    return _feature_mesh
