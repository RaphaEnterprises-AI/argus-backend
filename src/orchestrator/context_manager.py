"""
Context Management for Multi-Agent Systems.

Handles:
- Context window management and compression
- Summarization of long conversations
- Token counting and optimization
- Cross-agent context sharing

Based on best practices:
- Periodically reset or prune context during long sessions
- Prefer retrieval and summaries over dumping raw logs
- Use CLAUDE.md to encode project conventions
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ContextStats:
    """Statistics about context usage."""

    total_tokens: int = 0
    message_count: int = 0
    compressed_at: datetime | None = None
    compressions_performed: int = 0
    tokens_saved: int = 0


@dataclass
class ContextConfig:
    """Configuration for context management."""

    # Token limits
    max_context_tokens: int = 100_000
    compression_threshold: int = 80_000  # Start compression at 80%
    target_after_compression: int = 40_000  # Target 40% after compression

    # Message retention
    min_recent_messages: int = 10  # Always keep last N messages
    preserve_system_messages: bool = True

    # Summarization
    summary_model: str = "claude-haiku-4-5"  # Use fast model for summaries
    max_summary_tokens: int = 2000


class ContextManager:
    """
    Manages context for long-running agent sessions.

    Implements automatic compression and summarization to keep
    context within limits while preserving critical information.
    """

    def __init__(self, config: ContextConfig | None = None):
        self.config = config or ContextConfig()
        self.log = logger.bind(component="context_manager")
        self.stats = ContextStats()

        # Cache for token counting
        self._token_cache: dict[str, int] = {}

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses a simple heuristic: ~4 characters per token for English.
        For production, use tiktoken or the actual tokenizer.
        """
        # Check cache first
        cache_key = hash(text[:100]) if len(text) > 100 else hash(text)
        if cache_key in self._token_cache:
            return self._token_cache[cache_key]

        # Simple estimation: ~4 chars per token
        # Code tends to be ~3.5, prose ~4.5
        estimated = len(text) // 4

        self._token_cache[cache_key] = estimated
        return estimated

    def count_message_tokens(self, messages: list[dict]) -> int:
        """Count total tokens in a message list."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                # Multi-part content (e.g., with images)
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += self.estimate_tokens(part.get("text", ""))
                    elif isinstance(part, dict) and part.get("type") == "image":
                        # Images are roughly 1000 tokens
                        total += 1000
            # Add overhead for message structure
            total += 10
        return total

    def needs_compression(self, messages: list[dict]) -> bool:
        """Check if context needs compression."""
        current_tokens = self.count_message_tokens(messages)
        self.stats.total_tokens = current_tokens
        self.stats.message_count = len(messages)
        return current_tokens > self.config.compression_threshold

    async def compress_context(
        self,
        messages: list[dict],
        summarize_fn: callable | None = None,
    ) -> list[dict]:
        """
        Compress context by summarizing older messages.

        Args:
            messages: List of conversation messages
            summarize_fn: Optional async function to generate summaries
                         Falls back to simple truncation if not provided

        Returns:
            Compressed message list
        """
        current_tokens = self.count_message_tokens(messages)

        if current_tokens <= self.config.compression_threshold:
            return messages

        self.log.info(
            "Compressing context",
            current_tokens=current_tokens,
            threshold=self.config.compression_threshold,
            message_count=len(messages),
        )

        # Separate messages into categories
        system_messages = []
        recent_messages = []
        older_messages = []

        for i, msg in enumerate(messages):
            if msg.get("role") == "system" and self.config.preserve_system_messages:
                system_messages.append(msg)
            elif i >= len(messages) - self.config.min_recent_messages:
                recent_messages.append(msg)
            else:
                older_messages.append(msg)

        # If we have a summarization function, use it
        if summarize_fn and older_messages:
            summary = await self._summarize_messages(older_messages, summarize_fn)
            summary_message = {
                "role": "system",
                "content": f"[Context Summary]\n{summary}\n[End Summary - Recent conversation follows]",
            }

            compressed = system_messages + [summary_message] + recent_messages
        else:
            # Simple truncation: keep system and recent only
            compressed = system_messages + recent_messages

        new_tokens = self.count_message_tokens(compressed)
        tokens_saved = current_tokens - new_tokens

        self.stats.compressions_performed += 1
        self.stats.tokens_saved += tokens_saved
        self.stats.compressed_at = datetime.utcnow()

        self.log.info(
            "Context compressed",
            original_tokens=current_tokens,
            new_tokens=new_tokens,
            tokens_saved=tokens_saved,
            messages_before=len(messages),
            messages_after=len(compressed),
        )

        return compressed

    async def _summarize_messages(
        self,
        messages: list[dict],
        summarize_fn: callable,
    ) -> str:
        """Generate a summary of messages using provided function."""
        # Build a text representation of the messages
        text_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                text_parts.append(f"{role}: {content[:500]}...")  # Truncate long messages

        conversation_text = "\n".join(text_parts)

        # Use the summarization function
        summary_prompt = f"""Summarize the following conversation, preserving:
1. Key decisions made
2. Important findings or results
3. Current task state and progress
4. Any errors or issues encountered

Keep the summary concise (under {self.config.max_summary_tokens} tokens).

Conversation:
{conversation_text}

Summary:"""

        try:
            summary = await summarize_fn(summary_prompt)
            return summary
        except Exception as e:
            self.log.warning("Summarization failed, using truncation", error=str(e))
            # Fallback: just list key points
            return f"[Previous conversation of {len(messages)} messages truncated]"

    def extract_key_context(self, messages: list[dict]) -> dict[str, Any]:
        """
        Extract key contextual information from messages.

        Useful for sharing context between agents without full history.
        """
        context = {
            "task_description": None,
            "current_file": None,
            "recent_errors": [],
            "tools_used": [],
            "decisions_made": [],
        }

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                # Extract task from first user message
                if msg.get("role") == "user" and not context["task_description"]:
                    context["task_description"] = content[:500]

                # Track errors
                if "error" in content.lower() or "failed" in content.lower():
                    context["recent_errors"].append(content[:200])

                # Track file references
                import re
                file_matches = re.findall(r"[\w/.-]+\.(py|ts|js|tsx|jsx|json|yaml|yml)", content)
                if file_matches:
                    context["current_file"] = file_matches[-1]

        # Limit error list
        context["recent_errors"] = context["recent_errors"][-3:]

        return context

    def create_handoff_context(
        self,
        messages: list[dict],
        target_agent: str,
        max_tokens: int = 5000,
    ) -> list[dict]:
        """
        Create a minimal context for handoff to another agent.

        Args:
            messages: Full message history
            target_agent: Name of the target agent
            max_tokens: Maximum tokens for handoff context

        Returns:
            Minimal message list for the target agent
        """
        # Extract key context
        key_context = self.extract_key_context(messages)

        # Build handoff message
        handoff_content = f"""You are taking over from a previous agent. Here's the context:

Task: {key_context['task_description'] or 'Not specified'}
Current file: {key_context['current_file'] or 'None'}
Recent errors: {'; '.join(key_context['recent_errors']) if key_context['recent_errors'] else 'None'}

Continue from where the previous agent left off."""

        # Include system messages and last few exchanges
        system_msgs = [m for m in messages if m.get("role") == "system"]
        recent_msgs = messages[-4:]  # Last 2 exchanges

        handoff_messages = system_msgs + [
            {"role": "system", "content": handoff_content}
        ] + recent_msgs

        # Check token count
        while self.count_message_tokens(handoff_messages) > max_tokens and len(recent_msgs) > 1:
            recent_msgs = recent_msgs[1:]
            handoff_messages = system_msgs + [
                {"role": "system", "content": handoff_content}
            ] + recent_msgs

        self.log.debug(
            "Created handoff context",
            target_agent=target_agent,
            messages=len(handoff_messages),
            tokens=self.count_message_tokens(handoff_messages),
        )

        return handoff_messages

    def get_stats(self) -> dict[str, Any]:
        """Get context management statistics."""
        return {
            "total_tokens": self.stats.total_tokens,
            "message_count": self.stats.message_count,
            "compressions_performed": self.stats.compressions_performed,
            "tokens_saved": self.stats.tokens_saved,
            "last_compression": (
                self.stats.compressed_at.isoformat()
                if self.stats.compressed_at
                else None
            ),
            "compression_ratio": (
                self.stats.tokens_saved / (self.stats.total_tokens + self.stats.tokens_saved)
                if self.stats.total_tokens > 0
                else 0
            ),
        }


class SharedContextStore:
    """
    Shared context store for cross-agent communication.

    Implements working memory tier for multi-agent coordination.
    """

    def __init__(self):
        self.log = logger.bind(component="shared_context")
        self._store: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, datetime] = {}

    def set(self, namespace: str, key: str, value: Any, ttl_seconds: int | None = None):
        """Set a value in the shared context."""
        if namespace not in self._store:
            self._store[namespace] = {}

        self._store[namespace][key] = {
            "value": value,
            "set_at": datetime.utcnow(),
            "ttl": ttl_seconds,
        }

        self.log.debug(
            "Shared context set",
            namespace=namespace,
            key=key,
            ttl=ttl_seconds,
        )

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Get a value from shared context."""
        if namespace not in self._store:
            return default

        entry = self._store[namespace].get(key)
        if not entry:
            return default

        # Check TTL
        if entry.get("ttl"):
            age = (datetime.utcnow() - entry["set_at"]).total_seconds()
            if age > entry["ttl"]:
                del self._store[namespace][key]
                return default

        return entry["value"]

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        """Get all values in a namespace."""
        if namespace not in self._store:
            return {}

        result = {}
        expired_keys = []

        for key, entry in self._store[namespace].items():
            if entry.get("ttl"):
                age = (datetime.utcnow() - entry["set_at"]).total_seconds()
                if age > entry["ttl"]:
                    expired_keys.append(key)
                    continue
            result[key] = entry["value"]

        # Clean up expired entries
        for key in expired_keys:
            del self._store[namespace][key]

        return result

    def delete(self, namespace: str, key: str):
        """Delete a value from shared context."""
        if namespace in self._store and key in self._store[namespace]:
            del self._store[namespace][key]

    def clear_namespace(self, namespace: str):
        """Clear all values in a namespace."""
        if namespace in self._store:
            del self._store[namespace]

    def list_namespaces(self) -> list[str]:
        """List all namespaces."""
        return list(self._store.keys())
