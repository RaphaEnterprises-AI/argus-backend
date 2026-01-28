"""Perplexity AI provider implementation.

Perplexity AI provides online search-augmented language models that can
access real-time information and include citations in responses.

RAP-213: Implement Perplexity provider
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .base import (
    AuthenticationError,
    BaseProvider,
    ChatMessage,
    ChatResponse,
    ContentFilterError,
    ContextLengthError,
    ModelInfo,
    ModelNotFoundError,
    ModelTier,
    ProviderError,
    RateLimitError,
    ToolCall,
)


@dataclass
class Citation:
    """Represents a citation from Perplexity's online search.

    Perplexity models that search the web return citations to sources
    used in generating the response.
    """
    url: str
    title: str | None = None
    snippet: str | None = None
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "index": self.index,
        }


@dataclass
class PerplexityResponse(ChatResponse):
    """Extended ChatResponse with Perplexity-specific fields.

    Includes citations from online search when using sonar models.
    """
    citations: list[Citation] = field(default_factory=list)
    search_context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        if self.citations:
            result["citations"] = [c.to_dict() for c in self.citations]
        if self.search_context:
            result["search_context"] = self.search_context
        return result

    def format_with_citations(self) -> str:
        """Format the response content with numbered citations.

        Returns:
            Content with citation references and a sources section
        """
        if not self.citations:
            return self.content

        content = self.content
        sources = "\n\n**Sources:**\n"
        for i, citation in enumerate(self.citations, 1):
            title = citation.title or citation.url
            sources += f"[{i}] {title}: {citation.url}\n"

        return content + sources


class PerplexityProvider(BaseProvider):
    """Provider for Perplexity AI search-augmented models.

    Perplexity AI offers models that can search the internet in real-time
    and provide citations for their responses. This is particularly useful
    for questions requiring up-to-date information.

    Example usage:
        ```python
        provider = PerplexityProvider(api_key="pplx-...")

        messages = [
            ChatMessage.system("You are a helpful research assistant."),
            ChatMessage.user("What are the latest developments in AI?"),
        ]

        response = await provider.chat(
            messages=messages,
            model="llama-3.1-sonar-large-128k-online",
            temperature=0.2,
        )

        # Access citations
        if hasattr(response, 'citations'):
            for citation in response.citations:
                print(f"Source: {citation.url}")

        print(response.format_with_citations())
        ```
    """

    # Provider metadata
    provider_id = "perplexity"
    display_name = "Perplexity AI"
    website = "https://perplexity.ai"
    key_url = "https://www.perplexity.ai/settings/api"
    description = "Search-augmented language models with real-time web access and citations"

    # Capability flags
    supports_streaming = True
    supports_tools = False  # Perplexity models don't support function calling
    supports_vision = False
    supports_computer_use = False
    is_aggregator = False

    # API configuration
    BASE_URL = "https://api.perplexity.ai"
    DEFAULT_TIMEOUT = 120.0
    ENV_KEY_NAME = "PERPLEXITY_API_KEY"

    # Model definitions with pricing (as of Jan 2025)
    MODELS = {
        "llama-3.1-sonar-small-128k-online": ModelInfo(
            model_id="llama-3.1-sonar-small-128k-online",
            provider="perplexity",
            display_name="Sonar Small Online",
            input_price_per_1m=0.20,
            output_price_per_1m=0.20,
            context_window=127072,
            max_output=4096,
            supports_vision=False,
            supports_tools=False,
            supports_streaming=True,
            tier=ModelTier.FLASH,
            description="Fast online search model - 8B parameters with web search",
            aliases=["sonar-small-online", "sonar-small"],
        ),
        "llama-3.1-sonar-large-128k-online": ModelInfo(
            model_id="llama-3.1-sonar-large-128k-online",
            provider="perplexity",
            display_name="Sonar Large Online",
            input_price_per_1m=1.00,
            output_price_per_1m=1.00,
            context_window=127072,
            max_output=4096,
            supports_vision=False,
            supports_tools=False,
            supports_streaming=True,
            tier=ModelTier.VALUE,
            description="High-quality online search model - 70B parameters with web search",
            aliases=["sonar-large-online", "sonar-large"],
        ),
        "llama-3.1-sonar-huge-128k-online": ModelInfo(
            model_id="llama-3.1-sonar-huge-128k-online",
            provider="perplexity",
            display_name="Sonar Huge Online",
            input_price_per_1m=5.00,
            output_price_per_1m=5.00,
            context_window=127072,
            max_output=4096,
            supports_vision=False,
            supports_tools=False,
            supports_streaming=True,
            tier=ModelTier.PREMIUM,
            description="Most capable online search model - 405B parameters with web search",
            aliases=["sonar-huge-online", "sonar-huge"],
        ),
        "llama-3.1-sonar-small-128k-chat": ModelInfo(
            model_id="llama-3.1-sonar-small-128k-chat",
            provider="perplexity",
            display_name="Sonar Small Chat",
            input_price_per_1m=0.20,
            output_price_per_1m=0.20,
            context_window=131072,
            max_output=4096,
            supports_vision=False,
            supports_tools=False,
            supports_streaming=True,
            tier=ModelTier.FLASH,
            description="Fast chat model - 8B parameters without web search",
            aliases=["sonar-small-chat"],
        ),
        "llama-3.1-sonar-large-128k-chat": ModelInfo(
            model_id="llama-3.1-sonar-large-128k-chat",
            provider="perplexity",
            display_name="Sonar Large Chat",
            input_price_per_1m=1.00,
            output_price_per_1m=1.00,
            context_window=131072,
            max_output=4096,
            supports_vision=False,
            supports_tools=False,
            supports_streaming=True,
            tier=ModelTier.VALUE,
            description="High-quality chat model - 70B parameters without web search",
            aliases=["sonar-large-chat"],
        ),
    }

    def __init__(self, api_key: str | None = None):
        """Initialize the Perplexity provider.

        Args:
            api_key: Perplexity API key. If None, reads from PERPLEXITY_API_KEY
                     environment variable.
        """
        super().__init__(api_key)
        self.api_key = api_key or os.getenv(self.ENV_KEY_NAME)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(self.DEFAULT_TIMEOUT),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _convert_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage objects to Perplexity API format.

        Args:
            messages: List of ChatMessage objects

        Returns:
            List of message dictionaries
        """
        result = []
        for msg in messages:
            message: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            }
            if msg.name:
                message["name"] = msg.name
            result.append(message)
        return result

    def _parse_citations(self, response_data: dict, content: str) -> list[Citation]:
        """Parse citations from Perplexity response.

        Perplexity returns citations in the response which can be extracted
        and mapped to the content.

        Args:
            response_data: Raw API response
            content: Generated content

        Returns:
            List of Citation objects
        """
        citations = []

        # Perplexity includes citations in a separate field
        raw_citations = response_data.get("citations", [])
        for i, url in enumerate(raw_citations):
            if isinstance(url, str):
                citations.append(Citation(
                    url=url,
                    title=None,
                    snippet=None,
                    index=i + 1,
                ))
            elif isinstance(url, dict):
                citations.append(Citation(
                    url=url.get("url", ""),
                    title=url.get("title"),
                    snippet=url.get("snippet"),
                    index=i + 1,
                ))

        return citations

    def _extract_search_context(self, response_data: dict) -> str | None:
        """Extract the search context/query used by Perplexity.

        Args:
            response_data: Raw API response

        Returns:
            Search context if available
        """
        return response_data.get("search_context")

    def _handle_error(self, status_code: int, response_data: dict) -> None:
        """Handle API error responses.

        Args:
            status_code: HTTP status code
            response_data: Parsed response JSON

        Raises:
            Appropriate ProviderError subclass
        """
        error = response_data.get("error", {})
        if isinstance(error, str):
            message = error
        else:
            message = error.get("message", str(response_data))

        if status_code == 401:
            raise AuthenticationError(f"Invalid API key: {message}")
        elif status_code == 429:
            retry_after = None
            if isinstance(error, dict) and "retry_after" in error:
                retry_after = float(error["retry_after"])
            raise RateLimitError(f"Rate limit exceeded: {message}", retry_after)
        elif status_code == 404:
            raise ModelNotFoundError(f"Model not found: {message}")
        elif status_code == 400:
            if "context" in message.lower() or "token" in message.lower():
                raise ContextLengthError(f"Context length exceeded: {message}")
            elif "safety" in message.lower() or "content" in message.lower():
                raise ContentFilterError(f"Content blocked: {message}")
            raise ProviderError(f"Bad request: {message}")
        else:
            raise ProviderError(f"API error ({status_code}): {message}")

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs
    ) -> PerplexityResponse:
        """Send a chat completion request to Perplexity.

        Args:
            messages: List of chat messages
            model: Model ID (e.g., "llama-3.1-sonar-large-128k-online")
            temperature: Sampling temperature (0.0-2.0), lower is recommended for search
            max_tokens: Maximum tokens to generate
            tools: Not supported by Perplexity - will be ignored
            tool_choice: Not supported by Perplexity - will be ignored
            stop_sequences: Sequences that stop generation
            **kwargs: Additional Perplexity-specific parameters:
                - search_domain_filter: List of domains to restrict search
                - search_recency_filter: "month", "week", "day", "hour"
                - return_images: Whether to return images (default: False)
                - return_related_questions: Return related questions (default: False)

        Returns:
            PerplexityResponse with generated content and citations

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
        """
        if not self.api_key:
            raise AuthenticationError("Perplexity API key not provided")

        if tools:
            # Log warning but don't fail - just ignore tools
            pass

        client = await self._get_client()
        start_time = time.time()

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._convert_messages(messages),
            "temperature": temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if stop_sequences:
            payload["stop"] = stop_sequences

        # Perplexity-specific parameters
        if "search_domain_filter" in kwargs:
            payload["search_domain_filter"] = kwargs["search_domain_filter"]
        if "search_recency_filter" in kwargs:
            payload["search_recency_filter"] = kwargs["search_recency_filter"]
        if "return_images" in kwargs:
            payload["return_images"] = kwargs["return_images"]
        if "return_related_questions" in kwargs:
            payload["return_related_questions"] = kwargs["return_related_questions"]

        # Add other kwargs
        for key, value in kwargs.items():
            if key not in payload and key not in [
                "search_domain_filter", "search_recency_filter",
                "return_images", "return_related_questions"
            ]:
                payload[key] = value

        try:
            response = await client.post("/chat/completions", json=payload)
            response_data = response.json()

            if response.status_code != 200:
                self._handle_error(response.status_code, response_data)

            # Parse response
            choice = response_data["choices"][0]
            message = choice["message"]
            usage = response_data.get("usage", {})
            content = message.get("content", "")

            # Extract citations
            citations = self._parse_citations(response_data, content)
            search_context = self._extract_search_context(response_data)

            latency_ms = (time.time() - start_time) * 1000

            return PerplexityResponse(
                content=content,
                model=response_data.get("model", model),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                finish_reason=choice.get("finish_reason", "stop"),
                raw_response=response_data,
                tool_calls=None,  # Perplexity doesn't support tools
                system_fingerprint=response_data.get("system_fingerprint"),
                latency_ms=latency_ms,
                citations=citations,
                search_context=search_context,
            )

        except httpx.TimeoutException as e:
            raise ProviderError(f"Request timeout: {e}")
        except httpx.RequestError as e:
            raise ProviderError(f"Request failed: {e}")

    async def search(
        self,
        query: str,
        model: str = "llama-3.1-sonar-large-128k-online",
        system_prompt: str | None = None,
        search_domain_filter: list[str] | None = None,
        search_recency_filter: str | None = None,
        **kwargs
    ) -> PerplexityResponse:
        """Convenience method for search-focused queries.

        This is a simplified interface for using Perplexity's online models
        for research and information retrieval.

        Args:
            query: The search query or question
            model: Model to use (defaults to sonar-large-online)
            system_prompt: Optional system prompt for search behavior
            search_domain_filter: Restrict search to specific domains
            search_recency_filter: Time filter ("month", "week", "day", "hour")
            **kwargs: Additional parameters passed to chat()

        Returns:
            PerplexityResponse with search results and citations

        Example:
            ```python
            response = await provider.search(
                "What are the latest AI developments in 2025?",
                search_recency_filter="week",
            )
            print(response.format_with_citations())
            ```
        """
        messages = []

        if system_prompt:
            messages.append(ChatMessage.system(system_prompt))
        else:
            messages.append(ChatMessage.system(
                "You are a helpful research assistant. Provide accurate, "
                "well-sourced information based on your web search results. "
                "Always cite your sources."
            ))

        messages.append(ChatMessage.user(query))

        search_kwargs = {**kwargs}
        if search_domain_filter:
            search_kwargs["search_domain_filter"] = search_domain_filter
        if search_recency_filter:
            search_kwargs["search_recency_filter"] = search_recency_filter

        return await self.chat(
            messages=messages,
            model=model,
            temperature=0.2,  # Lower temperature for search accuracy
            **search_kwargs
        )

    async def validate_key(self, api_key: str) -> bool:
        """Validate a Perplexity API key.

        Args:
            api_key: API key to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            async with httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(30.0),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as client:
                # Try a minimal request to validate the key
                response = await client.post(
                    "/chat/completions",
                    json={
                        "model": "llama-3.1-sonar-small-128k-online",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 1,
                    },
                )
                return response.status_code == 200

        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List all available Perplexity models.

        Returns:
            List of ModelInfo objects for available models
        """
        return list(self.MODELS.values())

    async def get_model_info(self, model_id: str) -> ModelInfo | None:
        """Get info for a specific model.

        Args:
            model_id: Model ID to look up

        Returns:
            ModelInfo if found, None otherwise
        """
        # Direct lookup
        if model_id in self.MODELS:
            return self.MODELS[model_id]

        # Check aliases
        for model in self.MODELS.values():
            if model_id in model.aliases:
                return model

        return None

    def is_online_model(self, model_id: str) -> bool:
        """Check if a model has online search capabilities.

        Args:
            model_id: Model ID to check

        Returns:
            True if the model can search the web
        """
        return "online" in model_id.lower()
