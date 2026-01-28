"""Cohere AI provider implementation.

Cohere provides enterprise-grade language models with strong RAG
(Retrieval-Augmented Generation) capabilities and tool use support.

RAP-214: Implement Cohere provider
"""

import json
import os
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
    mask_api_key,
    validate_temperature,
)


@dataclass
class Document:
    """Represents a document for RAG with Cohere.

    Documents can be provided to Cohere models to ground their
    responses in specific content.
    """
    id: str
    text: str
    title: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to Cohere document format."""
        doc = {
            "id": self.id,
            "text": self.text,
        }
        if self.title:
            doc["title"] = self.title
        if self.url:
            doc["url"] = self.url
        return doc


@dataclass
class CitationSpan:
    """Represents a citation span in Cohere's response.

    Cohere provides detailed citation information mapping
    specific parts of the response to source documents.
    """
    start: int
    end: int
    text: str
    document_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "document_ids": self.document_ids,
        }


@dataclass
class CohereResponse(ChatResponse):
    """Extended ChatResponse with Cohere-specific fields.

    Includes citation spans, search queries, and document references
    for RAG-enabled responses.
    """
    citations: list[CitationSpan] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    search_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        if self.citations:
            result["citations"] = [c.to_dict() for c in self.citations]
        if self.documents:
            result["documents"] = [d.to_dict() for d in self.documents]
        if self.search_queries:
            result["search_queries"] = self.search_queries
        if self.search_results:
            result["search_results"] = self.search_results
        return result

    def format_with_citations(self) -> str:
        """Format the response content with inline citations.

        Returns:
            Content with citation markers and a sources section
        """
        if not self.citations or not self.documents:
            return self.content

        # Build document index
        doc_index = {doc.id: doc for doc in self.documents}

        # Format sources section
        sources = "\n\n**Sources:**\n"
        seen_docs = set()
        for i, citation in enumerate(self.citations, 1):
            for doc_id in citation.document_ids:
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    doc = doc_index.get(doc_id)
                    if doc:
                        title = doc.title or doc_id
                        url = f" ({doc.url})" if doc.url else ""
                        sources += f"- {title}{url}\n"

        return self.content + sources


class CohereProvider(BaseProvider):
    """Provider for Cohere's enterprise language models.

    Cohere offers models optimized for enterprise use cases, with
    strong support for RAG, tool use, and multi-language capabilities.

    Example usage:
        ```python
        provider = CohereProvider(api_key="...")

        messages = [
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("Summarize these documents."),
        ]

        # Basic chat
        response = await provider.chat(
            messages=messages,
            model="command-r-plus",
        )

        # With RAG documents
        documents = [
            Document(id="doc1", text="Document content...", title="Report"),
            Document(id="doc2", text="More content...", title="Analysis"),
        ]

        response = await provider.chat_with_rag(
            messages=messages,
            model="command-r-plus",
            documents=documents,
        )

        print(response.format_with_citations())
        ```
    """

    # Provider metadata
    provider_id = "cohere"
    display_name = "Cohere"
    website = "https://cohere.com"
    key_url = "https://dashboard.cohere.com/api-keys"
    description = "Enterprise language models with RAG and tool use capabilities"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = False
    supports_computer_use = False
    is_aggregator = False

    # API configuration
    BASE_URL = "https://api.cohere.ai/v1"
    DEFAULT_TIMEOUT = 120.0
    ENV_KEY_NAME = "COHERE_API_KEY"

    # Model definitions with pricing (as of Jan 2025)
    MODELS = {
        "command-r-plus": ModelInfo(
            model_id="command-r-plus",
            provider="cohere",
            display_name="Command R+",
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            context_window=128000,
            max_output=4096,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.PREMIUM,
            description="Cohere's most capable model for complex RAG and tool use",
            aliases=["command-r-plus-08-2024", "command-r-plus-latest"],
        ),
        "command-r": ModelInfo(
            model_id="command-r",
            provider="cohere",
            display_name="Command R",
            input_price_per_1m=0.50,
            output_price_per_1m=1.50,
            context_window=128000,
            max_output=4096,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.VALUE,
            description="Balanced model for RAG and conversational tasks",
            aliases=["command-r-08-2024", "command-r-latest"],
        ),
        "command-r7b-12-2024": ModelInfo(
            model_id="command-r7b-12-2024",
            provider="cohere",
            display_name="Command R 7B",
            input_price_per_1m=0.0375,
            output_price_per_1m=0.15,
            context_window=128000,
            max_output=4096,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.FLASH,
            description="Fast and efficient model for simpler tasks",
            aliases=["command-r-7b", "command-light"],
        ),
        "command-a-03-2025": ModelInfo(
            model_id="command-a-03-2025",
            provider="cohere",
            display_name="Command A",
            input_price_per_1m=2.50,
            output_price_per_1m=10.00,
            context_window=256000,
            max_output=8192,
            supports_vision=False,
            supports_tools=True,
            supports_streaming=True,
            tier=ModelTier.PREMIUM,
            description="Advanced model with extended context and capabilities",
            aliases=["command-a", "command-a-latest"],
        ),
    }

    def __init__(self, api_key: str | None = None):
        """Initialize the Cohere provider.

        Args:
            api_key: Cohere API key. If None, reads from COHERE_API_KEY
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
                    "X-Client-Name": "argus-e2e-testing-agent",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[str | None, list[dict], str]:
        """Convert ChatMessage objects to Cohere API format.

        Cohere uses a different message format with a separate preamble
        for system messages and a chat_history array.

        Args:
            messages: List of ChatMessage objects

        Returns:
            Tuple of (preamble, chat_history, message)
        """
        preamble = None
        chat_history = []
        current_message = ""

        for i, msg in enumerate(messages):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if msg.role == "system":
                # Cohere uses preamble for system message
                preamble = content
            elif i == len(messages) - 1 and msg.role == "user":
                # Last user message is the current message
                current_message = content
            else:
                # Map roles to Cohere format
                role = "USER" if msg.role == "user" else "CHATBOT"
                chat_history.append({
                    "role": role,
                    "message": content,
                })

        return preamble, chat_history, current_message

    def _convert_tools(self, tools: list[dict] | None) -> list[dict] | None:
        """Convert tools to Cohere API format.

        Args:
            tools: List of tool definitions

        Returns:
            Tools in Cohere format
        """
        if not tools:
            return None

        converted = []
        for tool in tools:
            if "type" in tool and tool["type"] == "function":
                # Convert from OpenAI format
                func = tool["function"]
                cohere_tool = {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameter_definitions": {},
                }

                params = func.get("parameters", {})
                properties = params.get("properties", {})
                required = params.get("required", [])

                for prop_name, prop_def in properties.items():
                    cohere_tool["parameter_definitions"][prop_name] = {
                        "type": prop_def.get("type", "string"),
                        "description": prop_def.get("description", ""),
                        "required": prop_name in required,
                    }

                converted.append(cohere_tool)
            else:
                # Assume already in Cohere format or simple format
                converted.append({
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameter_definitions": tool.get("parameter_definitions", tool.get("parameters", {})),
                })

        return converted

    def _parse_citations(self, response_data: dict) -> list[CitationSpan]:
        """Parse citations from Cohere response.

        Args:
            response_data: Raw API response

        Returns:
            List of CitationSpan objects
        """
        citations = []
        raw_citations = response_data.get("citations", [])

        for citation in raw_citations:
            citations.append(CitationSpan(
                start=citation.get("start", 0),
                end=citation.get("end", 0),
                text=citation.get("text", ""),
                document_ids=citation.get("document_ids", []),
            ))

        return citations

    def _parse_documents(self, response_data: dict) -> list[Document]:
        """Parse documents from Cohere response.

        Args:
            response_data: Raw API response

        Returns:
            List of Document objects
        """
        documents = []
        raw_docs = response_data.get("documents", [])

        for doc in raw_docs:
            documents.append(Document(
                id=doc.get("id", ""),
                text=doc.get("text", doc.get("snippet", "")),
                title=doc.get("title"),
                url=doc.get("url"),
            ))

        return documents

    def _handle_error(self, status_code: int, response_data: dict) -> None:
        """Handle API error responses.

        Args:
            status_code: HTTP status code
            response_data: Parsed response JSON

        Raises:
            Appropriate ProviderError subclass
        """
        message = response_data.get("message", str(response_data))

        if status_code == 401:
            raise AuthenticationError(f"Invalid API key: {message}")
        elif status_code == 429:
            raise RateLimitError(f"Rate limit exceeded: {message}")
        elif status_code == 404:
            raise ModelNotFoundError(f"Model not found: {message}")
        elif status_code == 400:
            if "context" in message.lower() or "token" in message.lower() or "length" in message.lower():
                raise ContextLengthError(f"Context length exceeded: {message}")
            elif "safety" in message.lower() or "blocked" in message.lower():
                raise ContentFilterError(f"Content blocked: {message}")
            raise ProviderError(f"Bad request: {message}")
        else:
            raise ProviderError(f"API error ({status_code}): {message}")

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs
    ) -> CohereResponse:
        """Send a chat completion request to Cohere.

        Args:
            messages: List of chat messages
            model: Model ID (e.g., "command-r-plus")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool definitions
            tool_choice: Tool choice strategy (not directly supported by Cohere)
            stop_sequences: Sequences that stop generation
            **kwargs: Additional Cohere-specific parameters:
                - documents: List of Document objects for RAG
                - connectors: List of connector configurations
                - citation_quality: "accurate" or "fast"
                - prompt_truncation: "AUTO" or "OFF"

        Returns:
            CohereResponse with generated content and RAG information

        Raises:
            AuthenticationError: If API key is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If model doesn't exist
        """
        if not self.api_key:
            raise AuthenticationError("Cohere API key not provided")

        client = await self._get_client()
        start_time = time.time()

        # Convert messages to Cohere format
        preamble, chat_history, current_message = self._convert_messages(messages)

        # Build request payload
        payload: dict[str, Any] = {
            "model": model,
            "message": current_message,
            "temperature": temperature,
        }

        if preamble:
            payload["preamble"] = preamble

        if chat_history:
            payload["chat_history"] = chat_history

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = self._convert_tools(tools)

        if stop_sequences:
            payload["stop_sequences"] = stop_sequences

        # Handle documents for RAG
        documents = kwargs.get("documents", [])
        if documents:
            payload["documents"] = [
                doc.to_dict() if isinstance(doc, Document) else doc
                for doc in documents
            ]

        # Cohere-specific parameters
        if "connectors" in kwargs:
            payload["connectors"] = kwargs["connectors"]
        if "citation_quality" in kwargs:
            payload["citation_quality"] = kwargs["citation_quality"]
        if "prompt_truncation" in kwargs:
            payload["prompt_truncation"] = kwargs["prompt_truncation"]

        try:
            response = await client.post("/chat", json=payload)
            response_data = response.json()

            if response.status_code != 200:
                self._handle_error(response.status_code, response_data)

            # Extract content and metadata
            content = response_data.get("text", "")
            meta = response_data.get("meta", {})
            tokens = meta.get("tokens", {})

            # Parse tool calls if present
            tool_calls = None
            raw_tool_calls = response_data.get("tool_calls", [])
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(
                        id=f"call_{i}",
                        name=tc.get("name", ""),
                        arguments=tc.get("parameters", {}),
                    )
                    for i, tc in enumerate(raw_tool_calls)
                ]

            # Parse RAG-specific fields
            citations = self._parse_citations(response_data)
            parsed_documents = self._parse_documents(response_data)
            search_queries = response_data.get("search_queries", [])
            search_results = response_data.get("search_results", [])

            latency_ms = (time.time() - start_time) * 1000

            # Determine finish reason
            finish_reason = "stop"
            if response_data.get("finish_reason"):
                finish_reason = response_data["finish_reason"].lower()
            elif tool_calls:
                finish_reason = "tool_calls"

            return CohereResponse(
                content=content,
                model=model,
                input_tokens=tokens.get("input_tokens", 0),
                output_tokens=tokens.get("output_tokens", 0),
                finish_reason=finish_reason,
                raw_response=response_data,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
                citations=citations,
                documents=parsed_documents,
                search_queries=[q.get("text", q) if isinstance(q, dict) else q for q in search_queries],
                search_results=search_results,
            )

        except httpx.TimeoutException as e:
            raise ProviderError(f"Request timeout: {e}")
        except httpx.RequestError as e:
            raise ProviderError(f"Request failed: {e}")

    async def chat_with_rag(
        self,
        messages: list[ChatMessage],
        model: str,
        documents: list[Document],
        temperature: float = 0.3,
        citation_quality: str = "accurate",
        **kwargs
    ) -> CohereResponse:
        """Convenience method for RAG-enabled chat.

        This provides a streamlined interface for using Cohere's
        RAG capabilities with documents.

        Args:
            messages: List of chat messages
            model: Model ID to use
            documents: List of Document objects to ground the response
            temperature: Sampling temperature
            citation_quality: "accurate" or "fast"
            **kwargs: Additional parameters passed to chat()

        Returns:
            CohereResponse with citations mapping to documents

        Example:
            ```python
            docs = [
                Document(id="1", text="Company policy states...", title="Policy"),
                Document(id="2", text="The annual report shows...", title="Report"),
            ]

            response = await provider.chat_with_rag(
                messages=[ChatMessage.user("Summarize our company policies.")],
                model="command-r-plus",
                documents=docs,
            )

            for citation in response.citations:
                print(f"'{citation.text}' -> {citation.document_ids}")
            ```
        """
        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            documents=documents,
            citation_quality=citation_quality,
            **kwargs
        )

    async def chat_with_connectors(
        self,
        messages: list[ChatMessage],
        model: str,
        connectors: list[dict],
        temperature: float = 0.3,
        **kwargs
    ) -> CohereResponse:
        """Chat with Cohere connectors for external data retrieval.

        Connectors allow Cohere to retrieve data from external sources
        like web search or custom APIs.

        Args:
            messages: List of chat messages
            model: Model ID to use
            connectors: List of connector configurations
            temperature: Sampling temperature
            **kwargs: Additional parameters passed to chat()

        Returns:
            CohereResponse with search results from connectors

        Example:
            ```python
            # Use web search connector
            response = await provider.chat_with_connectors(
                messages=[ChatMessage.user("Latest news about AI")],
                model="command-r-plus",
                connectors=[{"id": "web-search"}],
            )
            ```
        """
        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            connectors=connectors,
            **kwargs
        )

    async def validate_key(self, api_key: str) -> bool:
        """Validate a Cohere API key.

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
                # Use the check-api-key endpoint
                response = await client.post("/check-api-key")
                return response.status_code == 200

        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List all available Cohere models.

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

    async def rerank(
        self,
        query: str,
        documents: list[str | Document],
        model: str = "rerank-english-v3.0",
        top_n: int | None = None,
        **kwargs
    ) -> list[dict]:
        """Rerank documents by relevance to a query.

        This uses Cohere's rerank API to order documents by
        their semantic relevance to a query.

        Args:
            query: The search query
            documents: List of documents (strings or Document objects)
            model: Rerank model to use
            top_n: Return only top N results
            **kwargs: Additional parameters

        Returns:
            List of reranked results with relevance scores

        Example:
            ```python
            docs = ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
            results = await provider.rerank(
                query="Find relevant information",
                documents=docs,
                top_n=2,
            )
            for result in results:
                print(f"Index: {result['index']}, Score: {result['relevance_score']}")
            ```
        """
        if not self.api_key:
            raise AuthenticationError("Cohere API key not provided")

        client = await self._get_client()

        # Convert documents to strings
        doc_texts = []
        for doc in documents:
            if isinstance(doc, Document):
                doc_texts.append(doc.text)
            else:
                doc_texts.append(str(doc))

        payload: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": doc_texts,
        }

        if top_n is not None:
            payload["top_n"] = top_n

        payload.update(kwargs)

        try:
            response = await client.post("/rerank", json=payload)
            response_data = response.json()

            if response.status_code != 200:
                self._handle_error(response.status_code, response_data)

            return response_data.get("results", [])

        except httpx.TimeoutException as e:
            raise ProviderError(f"Request timeout: {e}")
        except httpx.RequestError as e:
            raise ProviderError(f"Request failed: {e}")
