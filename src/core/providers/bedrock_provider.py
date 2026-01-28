"""AWS Bedrock Provider for Argus E2E Testing Agent.

This module provides integration with AWS Bedrock for accessing foundation models
including Claude (Anthropic), Titan (Amazon), and Llama (Meta).

AWS Bedrock Benefits:
- Unified AWS billing and security (IAM, VPC, CloudTrail)
- No need to manage API keys - use IAM roles
- Data stays in your AWS account
- Enterprise compliance (SOC2, HIPAA, FedRAMP)
- Cross-region inference for reliability

Authentication:
- IAM Role (recommended): Uses boto3 credential chain
- Access Keys: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
- Profile: AWS_PROFILE environment variable

Configuration:
- AWS_BEDROCK_REGION: AWS region (e.g., us-east-1, us-west-2)
- AWS_ACCESS_KEY_ID: Optional if using IAM roles
- AWS_SECRET_ACCESS_KEY: Optional if using IAM roles
"""

import asyncio
import json
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
    validate_temperature,
)

logger = structlog.get_logger()


class BedrockError(ProviderError):
    """Bedrock-specific errors."""
    pass


class ThrottlingError(RateLimitError):
    """Raised when Bedrock throttles requests."""
    pass


class ServiceQuotaError(ProviderError):
    """Raised when service quota is exceeded."""
    pass


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider for foundation model access.

    Supports multiple model families:
    - Claude (Anthropic): claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
    - Titan (Amazon): titan-text-express, titan-text-lite, titan-embed
    - Llama (Meta): llama3-2-1b, llama3-2-3b, llama3-2-11b, llama3-2-90b

    Example:
        ```python
        provider = BedrockProvider(region="us-east-1")

        # List available models
        models = await provider.list_models()

        # Chat with Claude via Bedrock
        response = await provider.chat(
            messages=[ChatMessage.user("Hello!")],
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
        ```
    """

    # Provider metadata
    provider_id = "aws_bedrock"
    display_name = "AWS Bedrock"
    website = "https://aws.amazon.com/bedrock/"
    key_url = "https://console.aws.amazon.com/iam/home#/security_credentials"
    description = "Access foundation models via AWS with IAM security and unified billing"

    # Capability flags
    supports_streaming = True
    supports_tools = True
    supports_vision = True
    supports_computer_use = True  # Claude on Bedrock supports computer use
    is_aggregator = True  # Provides access to multiple model providers

    # Model family prefixes for routing
    MODEL_FAMILIES = {
        "anthropic": "anthropic",
        "amazon": "amazon",
        "meta": "meta",
        "cohere": "cohere",
        "ai21": "ai21",
        "stability": "stability",
        "mistral": "mistral",
    }

    # Known models with their pricing (per 1M tokens)
    # Prices as of January 2026 for us-east-1
    KNOWN_MODELS: dict[str, dict[str, Any]] = {
        # Claude 4 models (latest)
        "anthropic.claude-sonnet-4-20250514-v1:0": {
            "display_name": "Claude Sonnet 4",
            "input_price": 3.00,
            "output_price": 15.00,
            "context_window": 200000,
            "max_output": 8192,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": True,
            "tier": ModelTier.PREMIUM,
        },
        "anthropic.claude-opus-4-20250514-v1:0": {
            "display_name": "Claude Opus 4",
            "input_price": 15.00,
            "output_price": 75.00,
            "context_window": 200000,
            "max_output": 8192,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": True,
            "tier": ModelTier.EXPERT,
        },
        # Claude 3.5 models
        "anthropic.claude-3-5-sonnet-20241022-v2:0": {
            "display_name": "Claude 3.5 Sonnet v2",
            "input_price": 3.00,
            "output_price": 15.00,
            "context_window": 200000,
            "max_output": 8192,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": True,
            "tier": ModelTier.PREMIUM,
        },
        "anthropic.claude-3-5-haiku-20241022-v1:0": {
            "display_name": "Claude 3.5 Haiku",
            "input_price": 0.80,
            "output_price": 4.00,
            "context_window": 200000,
            "max_output": 8192,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.VALUE,
        },
        # Claude 3 models (legacy)
        "anthropic.claude-3-opus-20240229-v1:0": {
            "display_name": "Claude 3 Opus",
            "input_price": 15.00,
            "output_price": 75.00,
            "context_window": 200000,
            "max_output": 4096,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.EXPERT,
        },
        "anthropic.claude-3-sonnet-20240229-v1:0": {
            "display_name": "Claude 3 Sonnet",
            "input_price": 3.00,
            "output_price": 15.00,
            "context_window": 200000,
            "max_output": 4096,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.STANDARD,
        },
        "anthropic.claude-3-haiku-20240307-v1:0": {
            "display_name": "Claude 3 Haiku",
            "input_price": 0.25,
            "output_price": 1.25,
            "context_window": 200000,
            "max_output": 4096,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.FLASH,
        },
        # Amazon Titan models
        "amazon.titan-text-premier-v1:0": {
            "display_name": "Titan Text Premier",
            "input_price": 0.50,
            "output_price": 1.50,
            "context_window": 32000,
            "max_output": 3072,
            "supports_vision": False,
            "supports_tools": False,
            "supports_computer_use": False,
            "tier": ModelTier.VALUE,
        },
        "amazon.titan-text-express-v1": {
            "display_name": "Titan Text Express",
            "input_price": 0.20,
            "output_price": 0.60,
            "context_window": 8000,
            "max_output": 2048,
            "supports_vision": False,
            "supports_tools": False,
            "supports_computer_use": False,
            "tier": ModelTier.FLASH,
        },
        "amazon.titan-text-lite-v1": {
            "display_name": "Titan Text Lite",
            "input_price": 0.15,
            "output_price": 0.20,
            "context_window": 4000,
            "max_output": 2048,
            "supports_vision": False,
            "supports_tools": False,
            "supports_computer_use": False,
            "tier": ModelTier.FLASH,
        },
        # Meta Llama models
        "meta.llama3-2-90b-instruct-v1:0": {
            "display_name": "Llama 3.2 90B Instruct",
            "input_price": 0.72,
            "output_price": 0.72,
            "context_window": 128000,
            "max_output": 4096,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.STANDARD,
        },
        "meta.llama3-2-11b-instruct-v1:0": {
            "display_name": "Llama 3.2 11B Instruct",
            "input_price": 0.16,
            "output_price": 0.16,
            "context_window": 128000,
            "max_output": 4096,
            "supports_vision": True,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.FLASH,
        },
        "meta.llama3-2-3b-instruct-v1:0": {
            "display_name": "Llama 3.2 3B Instruct",
            "input_price": 0.15,
            "output_price": 0.15,
            "context_window": 128000,
            "max_output": 4096,
            "supports_vision": False,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.FLASH,
        },
        "meta.llama3-2-1b-instruct-v1:0": {
            "display_name": "Llama 3.2 1B Instruct",
            "input_price": 0.10,
            "output_price": 0.10,
            "context_window": 128000,
            "max_output": 4096,
            "supports_vision": False,
            "supports_tools": False,
            "supports_computer_use": False,
            "tier": ModelTier.FLASH,
        },
        # Mistral models
        "mistral.mistral-large-2407-v1:0": {
            "display_name": "Mistral Large 2",
            "input_price": 2.00,
            "output_price": 6.00,
            "context_window": 128000,
            "max_output": 8192,
            "supports_vision": False,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.STANDARD,
        },
        "mistral.mistral-small-2402-v1:0": {
            "display_name": "Mistral Small",
            "input_price": 0.10,
            "output_price": 0.30,
            "context_window": 32000,
            "max_output": 8192,
            "supports_vision": False,
            "supports_tools": True,
            "supports_computer_use": False,
            "tier": ModelTier.FLASH,
        },
    }

    # Model ID aliases for convenience
    MODEL_ALIASES: dict[str, str] = {
        # Claude 4 aliases
        "claude-sonnet-4": "anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-opus-4": "anthropic.claude-opus-4-20250514-v1:0",
        # Claude 3.5 aliases
        "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        "claude-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
        # Claude 3 aliases
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        # Titan aliases
        "titan-premier": "amazon.titan-text-premier-v1:0",
        "titan-express": "amazon.titan-text-express-v1",
        "titan-lite": "amazon.titan-text-lite-v1",
        # Llama aliases
        "llama-3-2-90b": "meta.llama3-2-90b-instruct-v1:0",
        "llama-3-2-11b": "meta.llama3-2-11b-instruct-v1:0",
        "llama-3-2-3b": "meta.llama3-2-3b-instruct-v1:0",
        "llama-3-2-1b": "meta.llama3-2-1b-instruct-v1:0",
        # Mistral aliases
        "mistral-large": "mistral.mistral-large-2407-v1:0",
        "mistral-small": "mistral.mistral-small-2402-v1:0",
    }

    def __init__(
        self,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        profile_name: str | None = None,
    ):
        """Initialize Bedrock provider.

        Authentication priority:
        1. Explicit access_key_id + secret_access_key
        2. AWS_PROFILE environment variable or profile_name parameter
        3. IAM role (when running on AWS EC2/ECS/Lambda)
        4. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

        Args:
            region: AWS region for Bedrock (e.g., us-east-1, us-west-2).
                   If None, uses AWS_BEDROCK_REGION or AWS_REGION env var.
            access_key_id: AWS access key ID. Optional if using IAM roles.
            secret_access_key: AWS secret access key. Optional if using IAM roles.
            session_token: AWS session token for temporary credentials.
            profile_name: AWS profile name from ~/.aws/credentials.
        """
        # Don't pass api_key to parent - we use AWS credentials
        super().__init__(api_key=None)

        self.region = region or os.getenv("AWS_BEDROCK_REGION") or os.getenv("AWS_REGION", "us-east-1")
        self.access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.session_token = session_token or os.getenv("AWS_SESSION_TOKEN")
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")

        # Lazy-load boto3 clients
        self._bedrock_client = None
        self._bedrock_runtime_client = None

    @property
    def bedrock_client(self):
        """Get or create Bedrock control plane client (for listing models)."""
        if self._bedrock_client is None:
            import boto3

            session_kwargs = {}
            if self.profile_name:
                session_kwargs["profile_name"] = self.profile_name

            session = boto3.Session(**session_kwargs)

            client_kwargs = {"region_name": self.region}
            if self.access_key_id and self.secret_access_key:
                client_kwargs["aws_access_key_id"] = self.access_key_id
                client_kwargs["aws_secret_access_key"] = self.secret_access_key
                if self.session_token:
                    client_kwargs["aws_session_token"] = self.session_token

            self._bedrock_client = session.client("bedrock", **client_kwargs)

        return self._bedrock_client

    @property
    def bedrock_runtime_client(self):
        """Get or create Bedrock runtime client (for inference)."""
        if self._bedrock_runtime_client is None:
            import boto3

            session_kwargs = {}
            if self.profile_name:
                session_kwargs["profile_name"] = self.profile_name

            session = boto3.Session(**session_kwargs)

            client_kwargs = {"region_name": self.region}
            if self.access_key_id and self.secret_access_key:
                client_kwargs["aws_access_key_id"] = self.access_key_id
                client_kwargs["aws_secret_access_key"] = self.secret_access_key
                if self.session_token:
                    client_kwargs["aws_session_token"] = self.session_token

            self._bedrock_runtime_client = session.client("bedrock-runtime", **client_kwargs)

        return self._bedrock_runtime_client

    def _resolve_model_id(self, model: str) -> str:
        """Resolve model alias to full Bedrock model ID.

        Args:
            model: Model ID or alias

        Returns:
            Full Bedrock model ID
        """
        return self.MODEL_ALIASES.get(model, model)

    def _get_model_family(self, model_id: str) -> str:
        """Determine the model family from model ID.

        Args:
            model_id: Bedrock model ID

        Returns:
            Model family string (anthropic, amazon, meta, etc.)
        """
        for prefix in self.MODEL_FAMILIES:
            if model_id.startswith(prefix):
                return prefix
        return "unknown"

    def _format_messages_anthropic(
        self,
        messages: list[ChatMessage],
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Format messages for Anthropic Claude models.

        Args:
            messages: List of chat messages
            tools: Optional tool definitions

        Returns:
            Request body for Anthropic Bedrock API
        """
        # Separate system message from conversation
        system_content = None
        conversation = []

        for msg in messages:
            if msg.role == "system":
                # Claude on Bedrock uses top-level system field
                if isinstance(msg.content, str):
                    system_content = msg.content
                else:
                    # Handle multimodal system content (rare but possible)
                    system_content = json.dumps(msg.content)
            else:
                # Convert content to Anthropic format
                content = msg.content
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    # Convert image content if needed
                    formatted_content = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "image_url":
                                # Convert from OpenAI format to Anthropic format
                                url = item.get("image_url", {}).get("url", "")
                                if url.startswith("data:"):
                                    # Parse data URL
                                    import base64
                                    header, data = url.split(",", 1)
                                    media_type = header.split(";")[0].split(":")[1]
                                    formatted_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data,
                                        }
                                    })
                                else:
                                    # URL-based image
                                    formatted_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": url,
                                        }
                                    })
                            else:
                                formatted_content.append(item)
                        else:
                            formatted_content.append({"type": "text", "text": str(item)})
                    content = formatted_content

                conversation.append({
                    "role": msg.role,
                    "content": content,
                })

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": conversation,
        }

        if system_content:
            body["system"] = system_content

        if tools:
            # Convert tools to Anthropic format
            body["tools"] = [
                {
                    "name": tool.get("name", tool.get("function", {}).get("name")),
                    "description": tool.get("description", tool.get("function", {}).get("description", "")),
                    "input_schema": tool.get("input_schema", tool.get("function", {}).get("parameters", {})),
                }
                for tool in tools
            ]

        return body

    def _format_messages_titan(
        self,
        messages: list[ChatMessage],
    ) -> dict[str, Any]:
        """Format messages for Amazon Titan models.

        Args:
            messages: List of chat messages

        Returns:
            Request body for Titan Bedrock API
        """
        # Titan uses a simpler format with inputText
        # Combine all messages into a single prompt
        parts = []

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            if msg.role == "system":
                parts.append(f"System: {content}")
            elif msg.role == "user":
                parts.append(f"User: {content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {content}")

        # Add assistant prompt prefix
        parts.append("Assistant:")

        return {
            "inputText": "\n\n".join(parts),
            "textGenerationConfig": {},
        }

    def _format_messages_llama(
        self,
        messages: list[ChatMessage],
    ) -> dict[str, Any]:
        """Format messages for Meta Llama models.

        Args:
            messages: List of chat messages

        Returns:
            Request body for Llama Bedrock API
        """
        # Llama uses a chat format similar to Claude
        # Build the prompt in Llama's expected format
        formatted_messages = []

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            formatted_messages.append({
                "role": msg.role,
                "content": content,
            })

        return {
            "prompt": self._build_llama_prompt(formatted_messages),
        }

    def _build_llama_prompt(self, messages: list[dict]) -> str:
        """Build a Llama-format prompt from messages.

        Args:
            messages: List of message dicts with role and content

        Returns:
            Formatted prompt string for Llama
        """
        # Llama 3 chat format
        prompt = "<|begin_of_text|>"

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

        # Add assistant prompt
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt

    def _format_messages_mistral(
        self,
        messages: list[ChatMessage],
    ) -> dict[str, Any]:
        """Format messages for Mistral models.

        Args:
            messages: List of chat messages

        Returns:
            Request body for Mistral Bedrock API
        """
        # Mistral uses a chat format
        formatted_messages = []

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            formatted_messages.append({
                "role": msg.role,
                "content": content,
            })

        return {
            "messages": formatted_messages,
        }

    def _parse_response_anthropic(self, response_body: dict) -> tuple[str, int, int, str, list[ToolCall] | None]:
        """Parse response from Anthropic Claude models.

        Args:
            response_body: Parsed JSON response from Bedrock

        Returns:
            Tuple of (content, input_tokens, output_tokens, finish_reason, tool_calls)
        """
        content_blocks = response_body.get("content", [])
        tool_calls = []
        text_content = []

        for block in content_blocks:
            if block.get("type") == "text":
                text_content.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))

        usage = response_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Map stop_reason to standard finish_reason
        stop_reason = response_body.get("stop_reason", "end_turn")
        finish_reason_map = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "content_filtered": "content_filter",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason)

        return (
            "\n".join(text_content),
            input_tokens,
            output_tokens,
            finish_reason,
            tool_calls if tool_calls else None,
        )

    def _parse_response_titan(self, response_body: dict) -> tuple[str, int, int, str, None]:
        """Parse response from Amazon Titan models.

        Args:
            response_body: Parsed JSON response from Bedrock

        Returns:
            Tuple of (content, input_tokens, output_tokens, finish_reason, None)
        """
        results = response_body.get("results", [{}])
        if results:
            content = results[0].get("outputText", "")
            completion_reason = results[0].get("completionReason", "FINISH")
        else:
            content = ""
            completion_reason = "FINISH"

        # Titan doesn't always return token counts
        input_tokens = response_body.get("inputTextTokenCount", 0)
        output_tokens = results[0].get("tokenCount", 0) if results else 0

        finish_reason_map = {
            "FINISH": "stop",
            "LENGTH": "length",
            "CONTENT_FILTERED": "content_filter",
        }
        finish_reason = finish_reason_map.get(completion_reason, "stop")

        return content, input_tokens, output_tokens, finish_reason, None

    def _parse_response_llama(self, response_body: dict) -> tuple[str, int, int, str, None]:
        """Parse response from Meta Llama models.

        Args:
            response_body: Parsed JSON response from Bedrock

        Returns:
            Tuple of (content, input_tokens, output_tokens, finish_reason, None)
        """
        content = response_body.get("generation", "")

        # Llama may return these in different fields
        input_tokens = response_body.get("prompt_token_count", 0)
        output_tokens = response_body.get("generation_token_count", 0)

        stop_reason = response_body.get("stop_reason", "stop")
        finish_reason_map = {
            "stop": "stop",
            "length": "length",
            "end_of_text": "stop",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason)

        return content, input_tokens, output_tokens, finish_reason, None

    def _parse_response_mistral(self, response_body: dict) -> tuple[str, int, int, str, None]:
        """Parse response from Mistral models.

        Args:
            response_body: Parsed JSON response from Bedrock

        Returns:
            Tuple of (content, input_tokens, output_tokens, finish_reason, None)
        """
        outputs = response_body.get("outputs", [{}])
        if outputs:
            content = outputs[0].get("text", "")
            stop_reason = outputs[0].get("stop_reason", "stop")
        else:
            content = ""
            stop_reason = "stop"

        usage = response_body.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        finish_reason_map = {
            "stop": "stop",
            "length": "length",
        }
        finish_reason = finish_reason_map.get(stop_reason, stop_reason)

        return content, input_tokens, output_tokens, finish_reason, None

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs
    ) -> ChatResponse:
        """Send a chat completion request to Bedrock.

        Args:
            messages: List of chat messages
            model: Model ID or alias (e.g., "claude-sonnet" or "anthropic.claude-3-5-sonnet-20241022-v2:0")
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions for function calling
            tool_choice: Tool choice strategy (only for Claude)
            stop_sequences: Sequences that stop generation
            **kwargs: Additional provider-specific arguments

        Returns:
            ChatResponse with generated content and metadata

        Raises:
            AuthenticationError: If AWS credentials are invalid
            ModelNotFoundError: If model doesn't exist in region
            RateLimitError: If throttled by Bedrock
            ContextLengthError: If input too long
        """
        import botocore.exceptions

        model_id = self._resolve_model_id(model)
        model_family = self._get_model_family(model_id)

        # Get default max_tokens from known models
        if max_tokens is None:
            model_info = self.KNOWN_MODELS.get(model_id, {})
            max_tokens = model_info.get("max_output", 4096)

        # Format request body based on model family
        if model_family == "anthropic":
            body = self._format_messages_anthropic(messages, tools)
            body["max_tokens"] = max_tokens
            if temperature is not None:
                body["temperature"] = temperature
            if stop_sequences:
                body["stop_sequences"] = stop_sequences
        elif model_family == "amazon":
            body = self._format_messages_titan(messages)
            body["textGenerationConfig"]["maxTokenCount"] = max_tokens
            if temperature is not None:
                body["textGenerationConfig"]["temperature"] = temperature
            if stop_sequences:
                body["textGenerationConfig"]["stopSequences"] = stop_sequences
        elif model_family == "meta":
            body = self._format_messages_llama(messages)
            body["max_gen_len"] = max_tokens
            if temperature is not None:
                body["temperature"] = temperature
        elif model_family == "mistral":
            body = self._format_messages_mistral(messages)
            body["max_tokens"] = max_tokens
            if temperature is not None:
                body["temperature"] = temperature
            if stop_sequences:
                body["stop"] = stop_sequences
        else:
            raise ModelNotFoundError(f"Unsupported model family: {model_family} for model {model_id}")

        start_time = time.time()

        try:
            # Use run_in_executor for async compatibility with boto3
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.bedrock_runtime_client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
            )

            response_body = json.loads(response["body"].read())

            # Parse response based on model family
            if model_family == "anthropic":
                content, input_tokens, output_tokens, finish_reason, tool_calls = self._parse_response_anthropic(response_body)
            elif model_family == "amazon":
                content, input_tokens, output_tokens, finish_reason, tool_calls = self._parse_response_titan(response_body)
            elif model_family == "meta":
                content, input_tokens, output_tokens, finish_reason, tool_calls = self._parse_response_llama(response_body)
            elif model_family == "mistral":
                content, input_tokens, output_tokens, finish_reason, tool_calls = self._parse_response_mistral(response_body)
            else:
                content, input_tokens, output_tokens, finish_reason, tool_calls = "", 0, 0, "stop", None

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                "Bedrock chat completed",
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
            )

            return ChatResponse(
                content=content,
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                latency_ms=latency_ms,
                raw_response=response_body,
            )

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            logger.error(
                "Bedrock API error",
                model=model_id,
                error_code=error_code,
                error_message=error_message,
            )

            if error_code == "AccessDeniedException":
                raise AuthenticationError(f"AWS credentials lack Bedrock access: {error_message}")
            elif error_code == "ResourceNotFoundException":
                raise ModelNotFoundError(f"Model not found in region {self.region}: {model_id}")
            elif error_code == "ThrottlingException":
                raise ThrottlingError(f"Bedrock rate limit exceeded: {error_message}")
            elif error_code == "ServiceQuotaExceededException":
                raise ServiceQuotaError(f"Bedrock quota exceeded: {error_message}")
            elif error_code == "ValidationException":
                if "Input is too long" in error_message or "token" in error_message.lower():
                    raise ContextLengthError(f"Input exceeds context window: {error_message}")
                raise ProviderError(f"Bedrock validation error: {error_message}")
            elif error_code == "ModelStreamErrorException":
                raise ProviderError(f"Bedrock streaming error: {error_message}")
            else:
                raise ProviderError(f"Bedrock error ({error_code}): {error_message}")

        except Exception as e:
            logger.error(
                "Unexpected Bedrock error",
                model=model_id,
                error=str(e),
            )
            raise ProviderError(f"Unexpected Bedrock error: {str(e)}")

    async def validate_key(self, api_key: str = "") -> bool:
        """Validate AWS credentials by listing foundation models.

        Note: For Bedrock, we validate the configured AWS credentials,
        not an API key. The api_key parameter is ignored.

        Returns:
            True if credentials are valid and have Bedrock access
        """
        import botocore.exceptions

        try:
            loop = asyncio.get_event_loop()
            # Try to list models - this validates credentials
            response = await loop.run_in_executor(
                None,
                lambda: self.bedrock_client.list_foundation_models(byOutputModality="TEXT")
            )

            # Check we got a valid response
            models = response.get("modelSummaries", [])
            logger.info(
                "Bedrock credentials validated",
                region=self.region,
                model_count=len(models),
            )
            return True

        except botocore.exceptions.NoCredentialsError:
            logger.warning("No AWS credentials found")
            return False
        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.warning(
                "Bedrock credential validation failed",
                error_code=error_code,
            )
            return False
        except Exception as e:
            logger.warning(
                "Bedrock validation error",
                error=str(e),
            )
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List available foundation models from Bedrock.

        Returns models that:
        1. Are available in the configured region
        2. Support text/chat generation
        3. Have known pricing information

        Returns:
            List of ModelInfo objects
        """
        import botocore.exceptions

        models = []

        try:
            loop = asyncio.get_event_loop()

            # Get models that output text (chat/completion models)
            response = await loop.run_in_executor(
                None,
                lambda: self.bedrock_client.list_foundation_models(byOutputModality="TEXT")
            )

            for model_summary in response.get("modelSummaries", []):
                model_id = model_summary.get("modelId", "")

                # Check if we have pricing info for this model
                if model_id in self.KNOWN_MODELS:
                    known = self.KNOWN_MODELS[model_id]
                    models.append(ModelInfo(
                        model_id=model_id,
                        provider=self.provider_id,
                        display_name=known.get("display_name", model_summary.get("modelName", model_id)),
                        input_price_per_1m=known.get("input_price", 0.0),
                        output_price_per_1m=known.get("output_price", 0.0),
                        context_window=known.get("context_window", 4096),
                        max_output=known.get("max_output", 2048),
                        supports_vision=known.get("supports_vision", False),
                        supports_tools=known.get("supports_tools", False),
                        supports_streaming=model_summary.get("responseStreamingSupported", True),
                        supports_computer_use=known.get("supports_computer_use", False),
                        tier=known.get("tier", ModelTier.STANDARD),
                        description=model_summary.get("modelName", ""),
                        aliases=[alias for alias, target in self.MODEL_ALIASES.items() if target == model_id],
                    ))
                else:
                    # Unknown model - add with estimated pricing
                    model_family = self._get_model_family(model_id)
                    models.append(ModelInfo(
                        model_id=model_id,
                        provider=self.provider_id,
                        display_name=model_summary.get("modelName", model_id),
                        input_price_per_1m=0.0,  # Unknown pricing
                        output_price_per_1m=0.0,
                        context_window=4096,  # Conservative default
                        max_output=2048,
                        supports_vision=model_family == "anthropic",  # Claude supports vision
                        supports_tools=model_family in ("anthropic", "mistral"),
                        supports_streaming=model_summary.get("responseStreamingSupported", True),
                        supports_computer_use=False,
                        tier=ModelTier.STANDARD,
                        description=f"Bedrock model: {model_id}",
                    ))

            logger.info(
                "Listed Bedrock models",
                region=self.region,
                model_count=len(models),
            )

            # Cache the results
            self._models_cache = models
            return models

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.error(
                "Failed to list Bedrock models",
                error_code=error_code,
                region=self.region,
            )

            # Return known models as fallback
            return [
                ModelInfo(
                    model_id=model_id,
                    provider=self.provider_id,
                    display_name=info.get("display_name", model_id),
                    input_price_per_1m=info.get("input_price", 0.0),
                    output_price_per_1m=info.get("output_price", 0.0),
                    context_window=info.get("context_window", 4096),
                    max_output=info.get("max_output", 2048),
                    supports_vision=info.get("supports_vision", False),
                    supports_tools=info.get("supports_tools", False),
                    supports_streaming=True,
                    supports_computer_use=info.get("supports_computer_use", False),
                    tier=info.get("tier", ModelTier.STANDARD),
                    description="",
                    aliases=[alias for alias, target in self.MODEL_ALIASES.items() if target == model_id],
                )
                for model_id, info in self.KNOWN_MODELS.items()
            ]

        except Exception as e:
            logger.error(
                "Unexpected error listing Bedrock models",
                error=str(e),
            )
            return []

    async def health_check(self) -> dict[str, Any]:
        """Perform a health check on Bedrock.

        Returns:
            Dictionary with health status including region and credential info
        """
        is_valid = await self.validate_key()

        return {
            "provider": self.provider_id,
            "status": "healthy" if is_valid else "unhealthy",
            "authenticated": is_valid,
            "region": self.region,
            "auth_method": self._get_auth_method(),
        }

    def _get_auth_method(self) -> str:
        """Determine which authentication method is being used."""
        if self.access_key_id and self.secret_access_key:
            return "access_keys"
        elif self.profile_name:
            return "profile"
        else:
            return "iam_role"
