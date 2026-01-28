"""Azure OpenAI Provider implementation (RAP-206).

This module provides integration with Azure OpenAI Service for enterprise deployments.

Azure OpenAI offers:
- Enterprise security (VNet, Private Link, Managed Identity)
- Regional data residency (compliance: GDPR, HIPAA, etc.)
- Microsoft ecosystem integration (Azure AD, RBAC)
- Same OpenAI models with Azure SLA

Key Differences from Direct OpenAI:
- Uses "deployments" instead of model names (you create deployments in Azure Portal)
- Requires Azure endpoint URL (e.g., https://your-resource.openai.azure.com/)
- API versioning is explicit (e.g., 2024-08-01-preview)
- Pricing may differ slightly based on region

Configuration:
    Set environment variables:
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key
    - AZURE_OPENAI_ENDPOINT: Your Azure resource endpoint URL

Example:
    provider = AzureOpenAIProvider(
        api_key="abc123...",
        endpoint="https://my-openai.openai.azure.com/",
    )

    # List deployments
    deployments = await provider.list_models()

    # Chat using deployment name
    response = await provider.chat(
        messages=[ChatMessage.user("Hello!")],
        model="gpt-4o-deployment",  # Your deployment name
    )
"""

import os
import time
from typing import Any

import structlog

from src.core.providers.base import (
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

logger = structlog.get_logger()

# Default API version - use the latest stable version
DEFAULT_API_VERSION = "2024-08-01-preview"

# Mapping of Azure deployment model versions to pricing and capabilities
# Note: Actual pricing depends on your Azure contract
AZURE_MODEL_INFO = {
    "gpt-4o": {
        "display_name": "GPT-4o (Azure)",
        "input_price_per_1m": 2.50,
        "output_price_per_1m": 10.00,
        "context_window": 128000,
        "max_output": 16384,
        "supports_vision": True,
        "supports_tools": True,
        "tier": ModelTier.PREMIUM,
    },
    "gpt-4o-mini": {
        "display_name": "GPT-4o Mini (Azure)",
        "input_price_per_1m": 0.15,
        "output_price_per_1m": 0.60,
        "context_window": 128000,
        "max_output": 16384,
        "supports_vision": True,
        "supports_tools": True,
        "tier": ModelTier.VALUE,
    },
    "gpt-4-turbo": {
        "display_name": "GPT-4 Turbo (Azure)",
        "input_price_per_1m": 10.00,
        "output_price_per_1m": 30.00,
        "context_window": 128000,
        "max_output": 4096,
        "supports_vision": True,
        "supports_tools": True,
        "tier": ModelTier.PREMIUM,
    },
    "gpt-4": {
        "display_name": "GPT-4 (Azure)",
        "input_price_per_1m": 30.00,
        "output_price_per_1m": 60.00,
        "context_window": 8192,
        "max_output": 4096,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.EXPERT,
    },
    "gpt-35-turbo": {
        "display_name": "GPT-3.5 Turbo (Azure)",
        "input_price_per_1m": 0.50,
        "output_price_per_1m": 1.50,
        "context_window": 16385,
        "max_output": 4096,
        "supports_vision": False,
        "supports_tools": True,
        "tier": ModelTier.FLASH,
    },
    "o1": {
        "display_name": "o1 (Azure)",
        "input_price_per_1m": 15.00,
        "output_price_per_1m": 60.00,
        "context_window": 200000,
        "max_output": 100000,
        "supports_vision": True,
        "supports_tools": False,
        "tier": ModelTier.EXPERT,
    },
    "o1-mini": {
        "display_name": "o1 Mini (Azure)",
        "input_price_per_1m": 3.00,
        "output_price_per_1m": 12.00,
        "context_window": 128000,
        "max_output": 65536,
        "supports_vision": True,
        "supports_tools": False,
        "tier": ModelTier.PREMIUM,
    },
    "text-embedding-3-large": {
        "display_name": "Text Embedding 3 Large (Azure)",
        "input_price_per_1m": 0.13,
        "output_price_per_1m": 0.0,
        "context_window": 8191,
        "max_output": 0,
        "supports_vision": False,
        "supports_tools": False,
        "tier": ModelTier.FLASH,
    },
    "text-embedding-3-small": {
        "display_name": "Text Embedding 3 Small (Azure)",
        "input_price_per_1m": 0.02,
        "output_price_per_1m": 0.0,
        "context_window": 8191,
        "max_output": 0,
        "supports_vision": False,
        "supports_tools": False,
        "tier": ModelTier.FLASH,
    },
}


class AzureOpenAIProvider(BaseProvider):
    """Azure OpenAI Service provider.

    Provides access to OpenAI models through Azure's enterprise cloud platform.

    Features:
    - Enterprise security (VNet, Private Link, Managed Identity)
    - Regional deployment for data residency
    - Microsoft ecosystem integration
    - Azure SLA and support

    Attributes:
        provider_id: "azure_openai"
        display_name: "Azure OpenAI"
        website: "https://azure.microsoft.com/en-us/products/ai-services/openai-service"
        key_url: "https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub"
    """

    provider_id = "azure_openai"
    display_name = "Azure OpenAI"
    website = "https://azure.microsoft.com/en-us/products/ai-services/openai-service"
    key_url = "https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub"
    description = (
        "Access OpenAI models through Microsoft Azure with enterprise security, "
        "regional data residency, and Azure integration."
    )

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_computer_use = False
    is_aggregator = False

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str = DEFAULT_API_VERSION,
        default_deployment: str | None = None,
    ):
        """Initialize the Azure OpenAI provider.

        Args:
            api_key: Azure OpenAI API key. If None, reads from AZURE_OPENAI_API_KEY env var.
            endpoint: Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/).
                     If None, reads from AZURE_OPENAI_ENDPOINT env var.
            api_version: Azure OpenAI API version (default: 2024-08-01-preview)
            default_deployment: Default deployment name to use when model is not specified
        """
        super().__init__(api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"))
        self.endpoint = (
            endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT") or ""
        ).rstrip("/")
        self.api_version = api_version
        self.default_deployment = default_deployment
        self._client = None
        self._deployment_model_map: dict[str, str] = {}

    @property
    def client(self):
        """Lazy-load the OpenAI client configured for Azure."""
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required for Azure OpenAI. "
                    "Install with: pip install openai"
                )

            if not self.api_key:
                raise AuthenticationError(
                    "Azure OpenAI API key not configured. "
                    "Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter."
                )

            if not self.endpoint:
                raise AuthenticationError(
                    "Azure OpenAI endpoint not configured. "
                    "Set AZURE_OPENAI_ENDPOINT environment variable or pass endpoint parameter."
                )

            self._client = AsyncAzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )

        return self._client

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request to Azure OpenAI.

        Args:
            messages: List of chat messages
            model: Deployment name (not the underlying model name)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Tool definitions for function calling
            tool_choice: Tool choice strategy
            stop_sequences: Sequences that stop generation
            **kwargs: Additional Azure-specific parameters

        Returns:
            ChatResponse with generated content

        Raises:
            AuthenticationError: If API key or endpoint is invalid
            RateLimitError: If rate limit exceeded
            ModelNotFoundError: If deployment doesn't exist
            ContentFilterError: If content blocked by Azure's filters
        """
        start_time = time.time()

        # Validate and clamp temperature to valid range (0.0-2.0)
        validated_temp = validate_temperature(temperature)

        # Use default deployment if model not specified
        deployment_name = model or self.default_deployment
        if not deployment_name:
            raise ModelNotFoundError(
                "No deployment specified. Pass model parameter or set default_deployment."
            )

        # Convert ChatMessage objects to dicts
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted_messages.append(msg.to_dict())
            else:
                formatted_messages.append(msg)

        # Build request parameters
        params: dict[str, Any] = {
            "model": deployment_name,  # In Azure, this is the deployment name
            "messages": formatted_messages,
            "temperature": validated_temp,
        }

        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        if stop_sequences:
            params["stop"] = stop_sequences

        if tools:
            params["tools"] = self._convert_tools(tools)
            if tool_choice:
                params["tool_choice"] = tool_choice

        # Add any extra kwargs (e.g., response_format for JSON mode)
        params.update(kwargs)

        try:
            response = await self.client.chat.completions.create(**params)

            latency_ms = (time.time() - start_time) * 1000

            # Extract tool calls if present
            parsed_tool_calls = None
            if response.choices[0].message.tool_calls:
                parsed_tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=self._parse_tool_arguments(tc.function.arguments),
                    )
                    for tc in response.choices[0].message.tool_calls
                ]

            return ChatResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason or "stop",
                tool_calls=parsed_tool_calls,
                system_fingerprint=response.system_fingerprint,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except Exception as e:
            self._handle_error(e)

    async def validate_key(self, api_key: str | None = None) -> bool:
        """Validate Azure OpenAI credentials by listing deployments.

        Args:
            api_key: API key to validate (uses instance key if None)

        Returns:
            True if credentials are valid
        """
        test_key = api_key or self.api_key

        if not test_key or not self.endpoint:
            return False

        try:
            # Create a temporary client for validation
            from openai import AsyncAzureOpenAI

            client = AsyncAzureOpenAI(
                api_key=test_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint,
            )

            # Try to list deployments - this validates both key and endpoint
            # We use the Azure REST API directly since openai SDK doesn't expose deployments
            import httpx

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    f"{self.endpoint}/openai/deployments?api-version={self.api_version}",
                    headers={"api-key": test_key},
                    timeout=10.0,
                )

                if response.status_code == 200:
                    logger.info(
                        "Azure OpenAI key validated successfully",
                        endpoint=self.endpoint,
                    )
                    return True
                elif response.status_code == 401:
                    logger.warning(
                        "Azure OpenAI key validation failed: Invalid API key",
                        endpoint=self.endpoint,
                    )
                    return False
                elif response.status_code == 404:
                    logger.warning(
                        "Azure OpenAI key validation failed: Invalid endpoint",
                        endpoint=self.endpoint,
                    )
                    return False
                else:
                    logger.warning(
                        "Azure OpenAI key validation failed",
                        status_code=response.status_code,
                        endpoint=self.endpoint,
                    )
                    return False

        except Exception as e:
            logger.error(
                "Azure OpenAI key validation error",
                error=str(e),
                endpoint=self.endpoint,
            )
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List available deployments from Azure OpenAI.

        Note: Azure doesn't have a dynamic model list like OpenAI.
        Instead, you create "deployments" in the Azure Portal that use
        specific model versions. This method lists your deployments.

        Returns:
            List of ModelInfo objects for available deployments
        """
        if not self.api_key or not self.endpoint:
            logger.warning(
                "Cannot list Azure models: missing credentials",
                has_key=bool(self.api_key),
                has_endpoint=bool(self.endpoint),
            )
            return []

        try:
            import httpx

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(
                    f"{self.endpoint}/openai/deployments?api-version={self.api_version}",
                    headers={"api-key": self.api_key},
                    timeout=10.0,
                )

                if response.status_code != 200:
                    logger.warning(
                        "Failed to list Azure OpenAI deployments",
                        status_code=response.status_code,
                    )
                    return []

                data = response.json()
                deployments = data.get("data", [])

                models = []
                for deployment in deployments:
                    deployment_name = deployment.get("id", "")
                    model_name = deployment.get("model", "")
                    status = deployment.get("status", "")

                    # Only include successfully deployed models
                    if status != "succeeded":
                        continue

                    # Cache the deployment -> model mapping
                    self._deployment_model_map[deployment_name] = model_name

                    # Get model info from our static mapping, or create a generic entry
                    base_info = AZURE_MODEL_INFO.get(model_name, {})

                    models.append(
                        ModelInfo(
                            model_id=deployment_name,
                            provider=self.provider_id,
                            display_name=base_info.get(
                                "display_name",
                                f"{model_name} (Azure Deployment: {deployment_name})",
                            ),
                            input_price_per_1m=base_info.get("input_price_per_1m", 0.0),
                            output_price_per_1m=base_info.get("output_price_per_1m", 0.0),
                            context_window=base_info.get("context_window", 4096),
                            max_output=base_info.get("max_output", 4096),
                            supports_vision=base_info.get("supports_vision", False),
                            supports_tools=base_info.get("supports_tools", True),
                            supports_streaming=True,
                            supports_computer_use=False,
                            tier=base_info.get("tier", ModelTier.STANDARD),
                            description=(
                                f"Azure deployment '{deployment_name}' using model {model_name}"
                            ),
                            aliases=[model_name] if model_name != deployment_name else [],
                        )
                    )

                logger.info(
                    "Listed Azure OpenAI deployments",
                    count=len(models),
                    deployments=[m.model_id for m in models],
                )

                return models

        except Exception as e:
            logger.error(
                "Error listing Azure OpenAI deployments",
                error=str(e),
            )
            return []

    async def get_deployment_model(self, deployment_name: str) -> str | None:
        """Get the underlying model name for a deployment.

        Args:
            deployment_name: Name of the deployment

        Returns:
            Model name (e.g., "gpt-4o") or None if not found
        """
        # Check cache first
        if deployment_name in self._deployment_model_map:
            return self._deployment_model_map[deployment_name]

        # Fetch fresh deployment list
        await self.list_models()

        return self._deployment_model_map.get(deployment_name)

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert generic tool format to OpenAI format.

        Args:
            tools: List of tool definitions (Anthropic or generic format)

        Returns:
            Tools in OpenAI format
        """
        converted = []
        for tool in tools:
            # Already in OpenAI format
            if "type" in tool and tool["type"] == "function":
                converted.append(tool)
            # Anthropic format
            elif "name" in tool and "input_schema" in tool:
                converted.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool["input_schema"],
                    },
                })
            # Generic format
            elif "function" in tool:
                converted.append({
                    "type": "function",
                    "function": tool["function"],
                })
            else:
                # Pass through unknown format
                converted.append(tool)

        return converted

    def _parse_tool_arguments(self, arguments: str) -> dict[str, Any]:
        """Parse tool call arguments from JSON string.

        Args:
            arguments: JSON string of arguments

        Returns:
            Parsed dictionary
        """
        import json

        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}

    def _handle_error(self, error: Exception) -> None:
        """Convert OpenAI/Azure errors to provider errors.

        Args:
            error: Original exception

        Raises:
            Appropriate ProviderError subclass
        """
        from openai import (
            APIConnectionError,
            APIStatusError,
        )
        from openai import (
            AuthenticationError as OpenAIAuthError,
        )
        from openai import (
            RateLimitError as OpenAIRateLimit,
        )

        error_str = str(error)

        if isinstance(error, OpenAIAuthError):
            raise AuthenticationError(
                f"Azure OpenAI authentication failed: {error_str}"
            )

        if isinstance(error, OpenAIRateLimit):
            # Try to extract retry-after from error
            retry_after = None
            if hasattr(error, "response"):
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    try:
                        retry_after = float(retry_after)
                    except ValueError:
                        retry_after = None

            raise RateLimitError(
                f"Azure OpenAI rate limit exceeded: {error_str}",
                retry_after=retry_after,
            )

        if isinstance(error, APIStatusError):
            status_code = error.status_code

            if status_code == 404:
                raise ModelNotFoundError(
                    f"Azure OpenAI deployment not found: {error_str}"
                )

            if status_code == 400:
                # Check for specific error types
                if "content_filter" in error_str.lower():
                    raise ContentFilterError(
                        f"Content blocked by Azure OpenAI content filter: {error_str}"
                    )
                if "context_length" in error_str.lower() or "maximum" in error_str.lower():
                    raise ContextLengthError(
                        f"Input exceeds Azure deployment context window: {error_str}"
                    )

        if isinstance(error, APIConnectionError):
            raise ProviderError(
                f"Failed to connect to Azure OpenAI: {error_str}. "
                f"Check your endpoint URL: {self.endpoint}"
            )

        # Re-raise as generic provider error
        raise ProviderError(f"Azure OpenAI error: {error_str}")

    async def create_embedding(
        self,
        input_text: str | list[str],
        model: str,
        dimensions: int | None = None,
    ) -> dict[str, Any]:
        """Create embeddings using Azure OpenAI.

        Args:
            input_text: Text or list of texts to embed
            model: Deployment name for embedding model
            dimensions: Optional dimension override (for text-embedding-3 models)

        Returns:
            Dictionary with embeddings and usage info
        """
        params: dict[str, Any] = {
            "model": model,
            "input": input_text,
        }

        if dimensions:
            params["dimensions"] = dimensions

        try:
            response = await self.client.embeddings.create(**params)

            return {
                "embeddings": [item.embedding for item in response.data],
                "model": response.model,
                "total_tokens": response.usage.total_tokens,
            }

        except Exception as e:
            self._handle_error(e)

    def __repr__(self) -> str:
        """String representation with masked API key for security."""
        endpoint_display = (
            self.endpoint[:30] + "..." if len(self.endpoint) > 30 else self.endpoint
        )
        return (
            f"<AzureOpenAIProvider("
            f"endpoint='{endpoint_display}', "
            f"api_key={mask_api_key(self.api_key)}, "
            f"api_version='{self.api_version}'"
            f")>"
        )
