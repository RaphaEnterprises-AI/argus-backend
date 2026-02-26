"""DOM snapshot diffing for proactive test healing.

Compares DOM snapshots to detect selector changes that may break tests.
Used by TestHealerCICDAgent for proactive fragility scanning.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import structlog
from typing import Any

logger = structlog.get_logger()


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MOVED = "moved"  # same element, different position


@dataclass
class SelectorInfo:
    """Information about a CSS/XPath selector in the DOM."""
    selector: str
    selector_type: str  # "css", "xpath", "id", "data-testid", "role"
    element_tag: str
    element_text: str = ""
    attributes: dict = field(default_factory=dict)
    parent_path: str = ""  # Simplified parent chain
    structural_hash: str = ""  # Hash of structural context


@dataclass
class DOMSnapshot:
    """A snapshot of DOM state for a page."""
    page_url: str
    selectors: list[SelectorInfo]
    structural_hash: str  # Overall page structure hash
    captured_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "page_url": self.page_url,
            "selectors": [vars(s) for s in self.selectors],
            "structural_hash": self.structural_hash,
            "captured_at": self.captured_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DOMSnapshot":
        selectors = [SelectorInfo(**s) for s in data.get("selectors", [])]
        return cls(
            page_url=data["page_url"],
            selectors=selectors,
            structural_hash=data.get("structural_hash", ""),
            captured_at=data.get("captured_at", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SelectorChange:
    """A detected change in a selector between snapshots."""
    change_type: ChangeType
    selector: str
    selector_type: str
    old_info: SelectorInfo | None = None
    new_info: SelectorInfo | None = None
    impact_score: float = 0.0  # 0.0-1.0, higher = more likely to break tests
    details: str = ""


@dataclass
class DOMDiffResult:
    """Result of comparing two DOM snapshots."""
    page_url: str
    total_changes: int
    added_selectors: list[SelectorChange]
    removed_selectors: list[SelectorChange]
    modified_selectors: list[SelectorChange]
    moved_selectors: list[SelectorChange]
    structural_change_score: float  # 0.0-1.0, overall structural change magnitude
    high_impact_changes: list[SelectorChange]  # Changes likely to break tests


class DOMDiffer:
    """Compares DOM snapshots to detect selector changes."""

    def diff(self, old: DOMSnapshot, new: DOMSnapshot) -> DOMDiffResult:
        """Compare two DOM snapshots and identify selector changes."""
        # Build lookup maps
        old_map = {s.selector: s for s in old.selectors}
        new_map = {s.selector: s for s in new.selectors}

        added = []
        removed = []
        modified = []
        moved = []

        # Find removed and modified selectors
        for selector, old_info in old_map.items():
            if selector not in new_map:
                # Check if it was moved (same element, different selector)
                moved_match = self._find_moved_selector(old_info, new.selectors)
                if moved_match:
                    change = SelectorChange(
                        change_type=ChangeType.MOVED,
                        selector=selector,
                        selector_type=old_info.selector_type,
                        old_info=old_info,
                        new_info=moved_match,
                        impact_score=self._compute_move_impact(old_info, moved_match),
                        details=f"Selector moved from '{selector}' to '{moved_match.selector}'",
                    )
                    moved.append(change)
                else:
                    change = SelectorChange(
                        change_type=ChangeType.REMOVED,
                        selector=selector,
                        selector_type=old_info.selector_type,
                        old_info=old_info,
                        impact_score=self._compute_removal_impact(old_info),
                        details=f"Selector '{selector}' removed from DOM",
                    )
                    removed.append(change)
            else:
                new_info = new_map[selector]
                if self._has_meaningful_change(old_info, new_info):
                    change = SelectorChange(
                        change_type=ChangeType.MODIFIED,
                        selector=selector,
                        selector_type=old_info.selector_type,
                        old_info=old_info,
                        new_info=new_info,
                        impact_score=self._compute_modification_impact(old_info, new_info),
                        details=self._describe_modification(old_info, new_info),
                    )
                    modified.append(change)

        # Find added selectors
        moved_new_selectors = {m.new_info.selector for m in moved if m.new_info}
        for selector, new_info in new_map.items():
            if selector not in old_map and selector not in moved_new_selectors:
                change = SelectorChange(
                    change_type=ChangeType.ADDED,
                    selector=selector,
                    selector_type=new_info.selector_type,
                    new_info=new_info,
                    impact_score=0.1,  # Added selectors rarely break existing tests
                    details=f"New selector '{selector}' added to DOM",
                )
                added.append(change)

        all_changes = added + removed + modified + moved
        high_impact = [c for c in all_changes if c.impact_score >= 0.6]

        # Compute structural change score
        structural_score = self._compute_structural_score(old, new, all_changes)

        return DOMDiffResult(
            page_url=old.page_url,
            total_changes=len(all_changes),
            added_selectors=added,
            removed_selectors=removed,
            modified_selectors=modified,
            moved_selectors=moved,
            structural_change_score=structural_score,
            high_impact_changes=sorted(high_impact, key=lambda c: c.impact_score, reverse=True),
        )

    def _find_moved_selector(self, old_info: SelectorInfo, new_selectors: list[SelectorInfo]) -> SelectorInfo | None:
        """Try to find the same element with a different selector."""
        for new_info in new_selectors:
            # Same tag + similar text = likely the same element moved
            if (old_info.element_tag == new_info.element_tag
                and old_info.element_text and new_info.element_text
                and old_info.element_text == new_info.element_text
                and old_info.selector != new_info.selector):
                return new_info
            # Same data-testid attribute
            old_testid = old_info.attributes.get("data-testid", "")
            new_testid = new_info.attributes.get("data-testid", "")
            if old_testid and old_testid == new_testid and old_info.selector != new_info.selector:
                return new_info
        return None

    def _has_meaningful_change(self, old: SelectorInfo, new: SelectorInfo) -> bool:
        """Check if the change is meaningful (not just whitespace etc)."""
        if old.element_tag != new.element_tag:
            return True
        if old.parent_path != new.parent_path:
            return True
        # Check key attributes
        key_attrs = {"class", "role", "type", "name", "aria-label", "data-testid"}
        for attr in key_attrs:
            if old.attributes.get(attr) != new.attributes.get(attr):
                return True
        return False

    def _compute_removal_impact(self, info: SelectorInfo) -> float:
        """Compute impact score for a removed selector."""
        score = 0.8  # Removals are generally high impact
        # data-testid selectors are explicitly maintained, so removal is very impactful
        if info.selector_type == "data-testid":
            score = 0.95
        elif info.selector_type == "id":
            score = 0.9
        elif info.selector_type == "role":
            score = 0.7
        return score

    def _compute_move_impact(self, old: SelectorInfo, new: SelectorInfo) -> float:
        """Compute impact score for a moved selector."""
        score = 0.5
        if old.selector_type != new.selector_type:
            score = 0.7
        if old.parent_path != new.parent_path:
            score += 0.1
        return min(score, 1.0)

    def _compute_modification_impact(self, old: SelectorInfo, new: SelectorInfo) -> float:
        """Compute impact score for a modified selector."""
        score = 0.3
        if old.element_tag != new.element_tag:
            score += 0.3
        if old.parent_path != new.parent_path:
            score += 0.2
        old_classes = set(old.attributes.get("class", "").split())
        new_classes = set(new.attributes.get("class", "").split())
        if old_classes != new_classes:
            score += 0.1
        return min(score, 1.0)

    def _describe_modification(self, old: SelectorInfo, new: SelectorInfo) -> str:
        """Describe what changed between old and new."""
        changes = []
        if old.element_tag != new.element_tag:
            changes.append(f"tag changed: {old.element_tag} -> {new.element_tag}")
        if old.parent_path != new.parent_path:
            changes.append("parent structure changed")
        for attr in {"class", "role", "type", "name", "aria-label"}:
            old_val = old.attributes.get(attr)
            new_val = new.attributes.get(attr)
            if old_val != new_val:
                changes.append(f"{attr}: '{old_val}' -> '{new_val}'")
        return "; ".join(changes) if changes else "minor attribute changes"

    def _compute_structural_score(self, old: DOMSnapshot, new: DOMSnapshot, changes: list[SelectorChange]) -> float:
        """Compute overall structural change magnitude."""
        if not old.selectors:
            return 1.0 if new.selectors else 0.0
        total = len(old.selectors) + len(new.selectors)
        if total == 0:
            return 0.0
        change_ratio = len(changes) / (total / 2)
        hash_changed = 0.2 if old.structural_hash != new.structural_hash else 0.0
        return min(change_ratio * 0.8 + hash_changed, 1.0)

    @staticmethod
    def compute_structural_hash(selectors: list[SelectorInfo]) -> str:
        """Compute a hash representing the structural layout of selectors."""
        parts = sorted(f"{s.element_tag}:{s.selector_type}:{s.parent_path}" for s in selectors)
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
