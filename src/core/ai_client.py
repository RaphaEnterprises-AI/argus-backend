"""
Centralized AI Client Factory with Langfuse Instrumentation.

This module provides a SINGLE SOURCE OF TRUTH for all AI client creation.
All AI calls in the codebase should use these factory functions instead of
creating clients directly. This ensures:

1. Automatic Langfuse tracing for ALL AI calls
2. Consistent configuration across the codebase
3. Centralized cost tracking
4. Easy provider switching

Usage:
    # Instead of:
    #   client = anthropic.Anthropic()
    #   response = client.messages.create(...)

    # Use:
    from src.core.ai_client import get_anthropic_client, trace_ai_call

    client = get_anthropic_client()
    response = client.messages.create(...)  # Automatically traced!

    # Or use the decorator for any AI function:
    @trace_ai_call(name="my_analysis")
    async def analyze_code(code: str) -> str:
        ...

Environment Variables:
    LANGFUSE_ENABLED: Set to "false" to disable tracing
    LANGFUSE_PUBLIC_KEY: Langfuse public key
    LANGFUSE_SECRET_KEY: Langfuse secret key
    LANGFUSE_HOST: Langfuse server URL
"""

import functools
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

import structlog

logger = structlog.get_logger(__name__)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])

# =============================================================================
# Langfuse Setup (Lazy initialization)
# =============================================================================

_langfuse_client = None
_langfuse_enabled = None


def _is_langfuse_enabled() -> bool:
    """Check if Langfuse is enabled and configured."""
    global _langfuse_enabled

    if _langfuse_enabled is not None:
        return _langfuse_enabled

    # Check environment
    if os.environ.get("LANGFUSE_ENABLED", "true").lower() == "false":
        _langfuse_enabled = False
        logger.info("Langfuse disabled via LANGFUSE_ENABLED=false")
        return False

    # Check for credentials
    if not os.environ.get("LANGFUSE_PUBLIC_KEY") or not os.environ.get("LANGFUSE_SECRET_KEY"):
        _langfuse_enabled = False
        logger.warning("Langfuse disabled - missing credentials")
        return False

    _langfuse_enabled = True
    return True


def _get_langfuse():
    """Get or create the Langfuse client singleton."""
    global _langfuse_client

    if not _is_langfuse_enabled():
        return None

    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        logger.info(
            "Langfuse client initialized",
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )

        return _langfuse_client

    except ImportError:
        logger.warning("Langfuse package not installed")
        _langfuse_enabled = False
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse: {e}")
        return None


# =============================================================================
# Instrumented Client Factories
# =============================================================================

def get_anthropic_client(
    api_key: str | None = None,
    trace_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Get an Anthropic client with automatic Langfuse instrumentation.

    All calls made through this client are automatically traced to Langfuse.

    Args:
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        trace_name: Name for the Langfuse trace
        user_id: User ID for trace attribution
        session_id: Session ID for grouping traces
        tags: Tags for filtering in Langfuse
        metadata: Additional metadata to attach

    Returns:
        Instrumented Anthropic client

    Example:
        client = get_anthropic_client(user_id="org-123", tags=["test-run"])
        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            messages=[{"role": "user", "content": "Hello"}],
        )
        # Automatically traced to Langfuse!
    """
    import anthropic

    # Get API key
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No Anthropic API key provided")

    # Create base client
    client = anthropic.Anthropic(api_key=key)

    # Wrap with Langfuse instrumentation if enabled
    langfuse = _get_langfuse()
    if langfuse:
        try:
            # Use Langfuse's observe decorator pattern for the client
            # We wrap the messages.create method
            original_create = client.messages.create

            @functools.wraps(original_create)
            def instrumented_create(*args, **kwargs):
                return _trace_anthropic_call(
                    langfuse=langfuse,
                    func=original_create,
                    args=args,
                    kwargs=kwargs,
                    trace_name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )

            client.messages.create = instrumented_create

            # Also wrap async version if it exists
            if hasattr(client.messages, "acreate"):
                original_acreate = client.messages.acreate

                @functools.wraps(original_acreate)
                async def instrumented_acreate(*args, **kwargs):
                    return await _trace_anthropic_call_async(
                        langfuse=langfuse,
                        func=original_acreate,
                        args=args,
                        kwargs=kwargs,
                        trace_name=trace_name,
                        user_id=user_id,
                        session_id=session_id,
                        tags=tags,
                        metadata=metadata,
                    )

                client.messages.acreate = instrumented_acreate

        except Exception as e:
            logger.warning(f"Failed to instrument Anthropic client: {e}")

    return client


def get_async_anthropic_client(
    api_key: str | None = None,
    trace_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Get an async Anthropic client with automatic Langfuse instrumentation.

    Same as get_anthropic_client but returns AsyncAnthropic.
    """
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No Anthropic API key provided")

    client = anthropic.AsyncAnthropic(api_key=key)

    langfuse = _get_langfuse()
    if langfuse:
        try:
            original_create = client.messages.create

            @functools.wraps(original_create)
            async def instrumented_create(*args, **kwargs):
                return await _trace_anthropic_call_async(
                    langfuse=langfuse,
                    func=original_create,
                    args=args,
                    kwargs=kwargs,
                    trace_name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )

            client.messages.create = instrumented_create

        except Exception as e:
            logger.warning(f"Failed to instrument async Anthropic client: {e}")

    return client


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    trace_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Get an OpenAI client with automatic Langfuse instrumentation.

    Also works with OpenAI-compatible APIs (OpenRouter, Together, etc.)
    by specifying base_url.

    Args:
        api_key: OpenAI API key
        base_url: Custom base URL for OpenAI-compatible APIs
        trace_name: Name for the Langfuse trace
        user_id: User ID for trace attribution
        session_id: Session ID for grouping traces
        tags: Tags for filtering
        metadata: Additional metadata

    Returns:
        Instrumented OpenAI client
    """
    import openai

    key = api_key or os.environ.get("OPENAI_API_KEY")

    # Create client with optional custom base URL
    client_kwargs = {}
    if key:
        client_kwargs["api_key"] = key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.OpenAI(**client_kwargs)

    langfuse = _get_langfuse()
    if langfuse:
        try:
            original_create = client.chat.completions.create

            @functools.wraps(original_create)
            def instrumented_create(*args, **kwargs):
                return _trace_openai_call(
                    langfuse=langfuse,
                    func=original_create,
                    args=args,
                    kwargs=kwargs,
                    trace_name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )

            client.chat.completions.create = instrumented_create

        except Exception as e:
            logger.warning(f"Failed to instrument OpenAI client: {e}")

    return client


def get_async_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    trace_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """Get an async OpenAI client with automatic Langfuse instrumentation."""
    import openai

    key = api_key or os.environ.get("OPENAI_API_KEY")

    client_kwargs = {}
    if key:
        client_kwargs["api_key"] = key
    if base_url:
        client_kwargs["base_url"] = base_url

    client = openai.AsyncOpenAI(**client_kwargs)

    langfuse = _get_langfuse()
    if langfuse:
        try:
            original_create = client.chat.completions.create

            @functools.wraps(original_create)
            async def instrumented_create(*args, **kwargs):
                return await _trace_openai_call_async(
                    langfuse=langfuse,
                    func=original_create,
                    args=args,
                    kwargs=kwargs,
                    trace_name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                    tags=tags,
                    metadata=metadata,
                )

            client.chat.completions.create = instrumented_create

        except Exception as e:
            logger.warning(f"Failed to instrument async OpenAI client: {e}")

    return client


def get_openrouter_client(
    api_key: str | None = None,
    trace_name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Get an OpenRouter client (OpenAI-compatible) with Langfuse instrumentation.

    OpenRouter provides access to 400+ models via a single API key.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")

    return get_openai_client(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        trace_name=trace_name or "openrouter",
        user_id=user_id,
        session_id=session_id,
        tags=["openrouter", *(tags or [])],
        metadata=metadata,
    )


# =============================================================================
# LangChain Integration
# =============================================================================

def get_langchain_llm(
    provider: str = "anthropic",
    model: str | None = None,
    api_key: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    **kwargs,
):
    """
    Get a LangChain LLM with Langfuse callback handler pre-configured.

    Args:
        provider: "anthropic", "openai", "google", "groq", "together"
        model: Model ID (provider-specific)
        api_key: API key (or use env var)
        user_id: User ID for Langfuse trace
        session_id: Session ID for Langfuse trace
        tags: Tags for Langfuse
        metadata: Additional metadata
        **kwargs: Additional kwargs passed to LLM constructor

    Returns:
        Tuple of (llm, callback_handler) - use callback_handler in invoke()

    Example:
        llm, handler = get_langchain_llm(
            provider="anthropic",
            model="claude-sonnet-4-5-20250514",
            user_id="org-123",
        )

        response = await llm.ainvoke(
            messages,
            config={"callbacks": [handler]}
        )
    """
    from src.orchestrator.langfuse_integration import get_langfuse_handler

    # Get callback handler
    handler = get_langfuse_handler(
        user_id=user_id,
        session_id=session_id,
        tags=tags,
        metadata=metadata,
    )

    # Create LLM based on provider
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=model or "claude-sonnet-4-5-20250514",
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            **kwargs,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model or "gpt-4o",
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            google_api_key=api_key or os.environ.get("GOOGLE_API_KEY"),
            **kwargs,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            **kwargs,
        )

    elif provider == "together":
        from langchain_together import ChatTogether

        llm = ChatTogether(
            model=model or "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key=api_key or os.environ.get("TOGETHER_API_KEY"),
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return llm, handler


# =============================================================================
# Decorator for Any AI Function
# =============================================================================

def trace_ai_call(
    name: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Decorator to trace any function as an AI generation in Langfuse.

    Use this for custom AI integrations or when wrapping external libraries.

    Args:
        name: Name for the generation (defaults to function name)
        user_id: User ID for trace
        session_id: Session ID for trace
        tags: Tags for filtering
        metadata: Additional metadata

    Example:
        @trace_ai_call(name="custom_analysis", tags=["analysis"])
        async def analyze_with_custom_model(prompt: str) -> str:
            # Your custom AI call here
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            langfuse = _get_langfuse()
            if not langfuse:
                return await func(*args, **kwargs)

            trace = langfuse.trace(
                name=name or func.__name__,
                user_id=user_id,
                session_id=session_id,
                tags=tags or [],
                metadata=metadata or {},
            )

            generation = trace.generation(
                name=name or func.__name__,
                metadata={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
            )

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                generation.end(
                    output=str(result)[:1000] if result else None,
                    metadata={"latency_ms": (time.time() - start_time) * 1000},
                )
                return result
            except Exception as e:
                generation.end(
                    level="ERROR",
                    status_message=str(e),
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            langfuse = _get_langfuse()
            if not langfuse:
                return func(*args, **kwargs)

            trace = langfuse.trace(
                name=name or func.__name__,
                user_id=user_id,
                session_id=session_id,
                tags=tags or [],
                metadata=metadata or {},
            )

            generation = trace.generation(
                name=name or func.__name__,
                metadata={"args": str(args)[:500], "kwargs": str(kwargs)[:500]},
            )

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                generation.end(
                    output=str(result)[:1000] if result else None,
                    metadata={"latency_ms": (time.time() - start_time) * 1000},
                )
                return result
            except Exception as e:
                generation.end(
                    level="ERROR",
                    status_message=str(e),
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


@contextmanager
def trace_ai_span(
    name: str,
    user_id: str | None = None,
    session_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    """
    Context manager for tracing a block of AI operations.

    Example:
        with trace_ai_span("code_analysis", user_id="org-123") as span:
            result = await some_ai_call()
            span.update(output=result)
    """
    langfuse = _get_langfuse()

    if not langfuse:
        yield None
        return

    trace = langfuse.trace(
        name=name,
        user_id=user_id,
        session_id=session_id,
        tags=tags or [],
        metadata=metadata or {},
    )

    span = trace.span(name=name)
    start_time = time.time()

    try:
        yield span
        span.end(
            metadata={"latency_ms": (time.time() - start_time) * 1000},
        )
    except Exception as e:
        span.end(
            level="ERROR",
            status_message=str(e),
        )
        raise


# =============================================================================
# Internal Tracing Helpers
# =============================================================================

def _trace_anthropic_call(
    langfuse,
    func,
    args,
    kwargs,
    trace_name,
    user_id,
    session_id,
    tags,
    metadata,
):
    """Trace a synchronous Anthropic API call."""
    model = kwargs.get("model", "unknown")

    trace = langfuse.trace(
        name=trace_name or "anthropic_call",
        user_id=user_id,
        session_id=session_id,
        tags=["anthropic", f"model:{model}", *(tags or [])],
        metadata=metadata or {},
    )

    generation = trace.generation(
        name="anthropic_messages_create",
        model=model,
        input=kwargs.get("messages"),
        model_parameters={
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
        },
    )

    start_time = time.time()
    try:
        response = func(*args, **kwargs)

        # Extract usage
        usage = {}
        if hasattr(response, "usage"):
            usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens,
            }

        # Extract content
        output = None
        if hasattr(response, "content") and response.content:
            output = response.content[0].text if hasattr(response.content[0], "text") else str(response.content)

        generation.end(
            output=output,
            usage=usage,
            metadata={"latency_ms": (time.time() - start_time) * 1000},
        )

        return response

    except Exception as e:
        generation.end(
            level="ERROR",
            status_message=str(e),
        )
        raise


async def _trace_anthropic_call_async(
    langfuse,
    func,
    args,
    kwargs,
    trace_name,
    user_id,
    session_id,
    tags,
    metadata,
):
    """Trace an async Anthropic API call."""
    model = kwargs.get("model", "unknown")

    trace = langfuse.trace(
        name=trace_name or "anthropic_call",
        user_id=user_id,
        session_id=session_id,
        tags=["anthropic", f"model:{model}", *(tags or [])],
        metadata=metadata or {},
    )

    generation = trace.generation(
        name="anthropic_messages_create",
        model=model,
        input=kwargs.get("messages"),
        model_parameters={
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
        },
    )

    start_time = time.time()
    try:
        response = await func(*args, **kwargs)

        usage = {}
        if hasattr(response, "usage"):
            usage = {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens,
            }

        output = None
        if hasattr(response, "content") and response.content:
            output = response.content[0].text if hasattr(response.content[0], "text") else str(response.content)

        generation.end(
            output=output,
            usage=usage,
            metadata={"latency_ms": (time.time() - start_time) * 1000},
        )

        return response

    except Exception as e:
        generation.end(
            level="ERROR",
            status_message=str(e),
        )
        raise


def _trace_openai_call(
    langfuse,
    func,
    args,
    kwargs,
    trace_name,
    user_id,
    session_id,
    tags,
    metadata,
):
    """Trace a synchronous OpenAI API call."""
    model = kwargs.get("model", "unknown")

    trace = langfuse.trace(
        name=trace_name or "openai_call",
        user_id=user_id,
        session_id=session_id,
        tags=["openai", f"model:{model}", *(tags or [])],
        metadata=metadata or {},
    )

    generation = trace.generation(
        name="openai_chat_completions_create",
        model=model,
        input=kwargs.get("messages"),
        model_parameters={
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
        },
    )

    start_time = time.time()
    try:
        response = func(*args, **kwargs)

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }

        output = None
        if hasattr(response, "choices") and response.choices:
            output = response.choices[0].message.content

        generation.end(
            output=output,
            usage=usage,
            metadata={"latency_ms": (time.time() - start_time) * 1000},
        )

        return response

    except Exception as e:
        generation.end(
            level="ERROR",
            status_message=str(e),
        )
        raise


async def _trace_openai_call_async(
    langfuse,
    func,
    args,
    kwargs,
    trace_name,
    user_id,
    session_id,
    tags,
    metadata,
):
    """Trace an async OpenAI API call."""
    model = kwargs.get("model", "unknown")

    trace = langfuse.trace(
        name=trace_name or "openai_call",
        user_id=user_id,
        session_id=session_id,
        tags=["openai", f"model:{model}", *(tags or [])],
        metadata=metadata or {},
    )

    generation = trace.generation(
        name="openai_chat_completions_create",
        model=model,
        input=kwargs.get("messages"),
        model_parameters={
            "max_tokens": kwargs.get("max_tokens"),
            "temperature": kwargs.get("temperature"),
        },
    )

    start_time = time.time()
    try:
        response = await func(*args, **kwargs)

        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }

        output = None
        if hasattr(response, "choices") and response.choices:
            output = response.choices[0].message.content

        generation.end(
            output=output,
            usage=usage,
            metadata={"latency_ms": (time.time() - start_time) * 1000},
        )

        return response

    except Exception as e:
        generation.end(
            level="ERROR",
            status_message=str(e),
        )
        raise


# =============================================================================
# Flush Function
# =============================================================================

def flush_langfuse():
    """Flush all pending Langfuse events. Call before app shutdown."""
    langfuse = _get_langfuse()
    if langfuse:
        try:
            langfuse.flush()
            logger.debug("Langfuse flushed")
        except Exception as e:
            logger.warning(f"Failed to flush Langfuse: {e}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Client factories
    "get_anthropic_client",
    "get_async_anthropic_client",
    "get_openai_client",
    "get_async_openai_client",
    "get_openrouter_client",
    "get_langchain_llm",
    # Decorators and context managers
    "trace_ai_call",
    "trace_ai_span",
    # Utilities
    "flush_langfuse",
]
