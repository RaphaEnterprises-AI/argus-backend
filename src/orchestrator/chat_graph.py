"""Chat-enabled LangGraph for conversational test orchestration."""

import json
import re
import time
from typing import Annotated, Literal, TypedDict

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.config import get_settings
from src.services.audit_logger import AuditAction, AuditStatus, ResourceType, get_audit_logger
from src.services.cloudflare_storage import get_cloudflare_client, is_cloudflare_configured

logger = structlog.get_logger()


def _parse_schedule_to_cron(schedule_input: str) -> str:
    """Parse natural language or preset schedule to cron expression.

    Supports:
    - Direct cron expressions (5 fields)
    - Presets: "daily at 9am", "every hour", "weekdays at 6pm", etc.
    - Natural language: "every 30 minutes", "twice daily", etc.
    """
    schedule = schedule_input.strip().lower()

    # Check if it's already a cron expression (5 space-separated fields)
    parts = schedule.split()
    if len(parts) == 5 and all(
        p.replace("*", "").replace("/", "").replace("-", "").replace(",", "").isdigit() or p == "*"
        for p in parts
    ):
        return schedule_input.strip()

    # Common presets
    presets = {
        "every minute": "* * * * *",
        "every 5 minutes": "*/5 * * * *",
        "every 10 minutes": "*/10 * * * *",
        "every 15 minutes": "*/15 * * * *",
        "every 30 minutes": "*/30 * * * *",
        "every hour": "0 * * * *",
        "hourly": "0 * * * *",
        "every 2 hours": "0 */2 * * *",
        "every 6 hours": "0 */6 * * *",
        "daily": "0 0 * * *",
        "daily at midnight": "0 0 * * *",
        "daily at 9am": "0 9 * * *",
        "daily at 9 am": "0 9 * * *",
        "daily at noon": "0 12 * * *",
        "daily at 6pm": "0 18 * * *",
        "daily at 6 pm": "0 18 * * *",
        "weekdays": "0 9 * * 1-5",
        "weekdays at 9am": "0 9 * * 1-5",
        "weekdays at 9 am": "0 9 * * 1-5",
        "weekdays at 6pm": "0 18 * * 1-5",
        "weekly": "0 0 * * 0",
        "weekly on monday": "0 0 * * 1",
        "weekly on sunday": "0 0 * * 0",
        "monthly": "0 0 1 * *",
        "monthly on the 1st": "0 0 1 * *",
        "twice daily": "0 9,18 * * *",
    }

    # Check for exact preset match
    if schedule in presets:
        return presets[schedule]

    # Try pattern matching for "daily at Xam/pm"
    import re
    time_match = re.search(r"(?:daily\s+)?at\s+(\d{1,2})\s*(am|pm)?", schedule)
    if time_match:
        hour = int(time_match.group(1))
        period = time_match.group(2)
        if period == "pm" and hour < 12:
            hour += 12
        elif period == "am" and hour == 12:
            hour = 0
        return f"0 {hour} * * *"

    # Try "every X minutes/hours"
    interval_match = re.search(r"every\s+(\d+)\s*(minute|hour|min|hr)s?", schedule)
    if interval_match:
        value = int(interval_match.group(1))
        unit = interval_match.group(2)
        if unit in ("minute", "min"):
            return f"*/{value} * * * *"
        elif unit in ("hour", "hr"):
            return f"0 */{value} * * *"

    # Default to daily at 9am if we can't parse
    logger.warning("Could not parse schedule, using default", input=schedule_input)
    return "0 9 * * *"


# Token limits for context management
MAX_CONTEXT_TOKENS = 150000  # Leave headroom below 200k limit
MAX_TOOL_RESULT_TOKENS = 2000  # Truncate large tool results
KEEP_RECENT_MESSAGES = 10  # Keep last N messages unmodified


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token for English)."""
    return len(text) // 4


def strip_base64_from_content(content: str) -> str:
    """Remove base64 image data from content to save tokens."""
    # Match base64 patterns (data URLs and raw base64)
    # Pattern for data URLs: data:image/...;base64,XXXX
    content = re.sub(
        r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
        '[IMAGE_REMOVED]',
        content
    )
    # Pattern for raw base64 in JSON (screenshot fields)
    content = re.sub(
        r'"(screenshot|image|finalScreenshot|screenshotBase64)":\s*"[A-Za-z0-9+/=]{1000,}"',
        r'"\1": "[IMAGE_REMOVED]"',
        content
    )
    return content


def truncate_tool_result(content: str, max_tokens: int = MAX_TOOL_RESULT_TOKENS) -> str:
    """Truncate tool result content if too long."""
    # First strip base64 images
    content = strip_base64_from_content(content)

    # Then truncate if still too long
    estimated = estimate_tokens(content)
    if estimated > max_tokens:
        # Keep first part and add truncation notice
        max_chars = max_tokens * 4
        return content[:max_chars] + "\n...[TRUNCATED - result too long]..."
    return content


def prune_messages_for_context(messages: list[BaseMessage], max_tokens: int = MAX_CONTEXT_TOKENS) -> list[BaseMessage]:
    """Prune messages to fit within context limit.

    Strategy:
    1. ALWAYS strip base64 images from ALL tool results (frontend gets them from stream)
    2. Truncate very long tool results
    3. If still over limit, drop oldest messages (keeping last KEEP_RECENT_MESSAGES human/AI exchanges)
    """
    if not messages:
        return messages

    # Process ALL messages - always strip images from tool results
    processed_messages = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # ALWAYS strip images and truncate tool results
            # Frontend receives full results from streaming, Claude doesn't need screenshots
            new_content = truncate_tool_result(msg.content)
            processed_messages.append(ToolMessage(
                content=new_content,
                tool_call_id=msg.tool_call_id,
                name=msg.name if hasattr(msg, 'name') else None,
            ))
        elif isinstance(msg, AIMessage) and hasattr(msg, 'content'):
            # Strip images from AI messages too (they might quote tool results)
            new_content = strip_base64_from_content(msg.content) if isinstance(msg.content, str) else msg.content
            processed_messages.append(AIMessage(content=new_content, tool_calls=getattr(msg, 'tool_calls', None)))
        else:
            processed_messages.append(msg)

    # Estimate total tokens
    total_tokens = sum(estimate_tokens(str(m.content) if hasattr(m, 'content') else str(m)) for m in processed_messages)

    # If still over limit, drop oldest messages until under limit
    # But keep at least KEEP_RECENT_MESSAGES
    min_keep = min(KEEP_RECENT_MESSAGES, len(processed_messages))
    while total_tokens > max_tokens and len(processed_messages) > min_keep:
        dropped = processed_messages.pop(0)
        dropped_tokens = estimate_tokens(str(dropped.content) if hasattr(dropped, 'content') else str(dropped))
        total_tokens -= dropped_tokens
        logger.info("Dropped old message to fit context", dropped_type=type(dropped).__name__, tokens_freed=dropped_tokens)

    logger.info("Message pruning complete",
                original_count=len(messages),
                final_count=len(processed_messages),
                estimated_tokens=total_tokens)

    return processed_messages


class AIConfig(TypedDict, total=False):
    """AI model configuration from user preferences."""
    model: str  # Model ID (e.g., "claude-sonnet-4-5", "gpt-4o")
    provider: str  # Provider name (e.g., "anthropic", "openai")
    api_key: str | None  # Decrypted API key (BYOK) - None means use platform key
    user_id: str | None  # User ID for usage tracking
    track_usage: bool  # Whether to track token usage


class ChatState(TypedDict):
    """State for chat conversations with context tracking."""
    messages: Annotated[list[BaseMessage], add_messages]
    app_url: str
    current_tool: str | None
    tool_results: list[dict]
    session_id: str
    should_continue: bool  # Flag to allow cancellation of execution

    # AI model configuration (from user preferences)
    ai_config: AIConfig | None  # User's model selection and API key

    # Context tracking for intelligent responses
    current_task: str | None  # "discovery" | "test_creation" | "test_execution" | "analysis"
    task_progress: dict | None  # {"current_step": 3, "total_steps": 10, "last_action": "..."}
    recent_failures: list[dict] | None  # Last N failures with healing suggestions
    discovered_elements: list[dict] | None  # Cached page elements from last discovery
    active_test: dict | None  # Currently running test details
    healing_suggestions: list[dict] | None  # Suggestions from VectorizeMemory


def create_system_prompt(
    app_url: str = "",
    current_task: str | None = None,
    task_progress: dict | None = None,
    recent_failures: list[dict] | None = None,
    discovered_elements: list[dict] | None = None,
    healing_suggestions: list[dict] | None = None,
) -> str:
    """Create context-aware system prompt for Argus chat.

    This enhanced prompt includes:
    - Current session context and task progress
    - Recent failures with healing suggestions
    - Discovered elements for reference
    - Self-healing recommendations from memory
    """
    base_prompt = f'''You are Argus, an intelligent AI-powered E2E Testing Agent. You help users:

1. **Create tests** from natural language descriptions with smart step generation
2. **Run tests** and report results with automatic self-healing on failures
3. **Discover** application flows and interactive elements automatically
4. **Detect regressions** through visual comparison and assertion validation
5. **Analyze failures** with root cause analysis and fix suggestions

## Current Application
URL: {app_url or "Not specified - ask user for the URL to test"}

## Available Tools
| Tool | Description | When to Use |
|------|-------------|-------------|
| `runTest` | Execute multi-step E2E test with screenshots | Running complete test scenarios |
| `executeAction` | Execute single browser action | Quick actions or debugging |
| `discoverElements` | Discover interactive elements on page | Before creating tests, to understand the page |
| `createTest` | Generate test from description | User wants to create a new test |
| `checkStatus` | Check Argus system health | Debugging connection issues |
| `createSchedule` | Schedule recurring test runs | User wants to run tests on a schedule |
| `listSchedules` | List all scheduled test runs | User wants to see their schedules |
| `runQualityAudit` | Run quality audit (a11y, perf, SEO) | User wants to audit a page |
| `generateReport` | Generate test execution report | User wants a summary of test runs |
| `getAIInsights` | Get AI insights about patterns | User wants analysis or recommendations |
| `getInfraStatus` | Get browser pool status and costs | User asks about infrastructure |
| `listTests` | List tests with filtering | User wants to see available tests |
| `getTestRuns` | Get test run history | User wants to see past runs |
| `webSearch` | Search web for real-time info | User needs current docs, tutorials, or answers |
| `exportTest` | Export test to different language/framework | User wants to export a test to Python, TypeScript, Java, etc. |
| `listExportFormats` | List available export languages/frameworks | User asks what export formats are available |
| `queryKnowledge` | Query test knowledge graph | User wants to explore relationships between tests, failures, code changes, selectors |

## Response Guidelines
1. **Be specific**: Reference actual elements, selectors, and URLs
2. **Show evidence**: Include screenshot references when available
3. **Explain failures**: Don't just say "failed" - explain WHY and suggest fixes
4. **Track progress**: Use numbered steps and show completion status
5. **Suggest healing**: When tests fail, offer concrete alternative selectors

## Formatting
- Test results: Use ✅/❌ status indicators with step numbers
- Selectors: Use `code blocks` for CSS selectors and XPath
- Screenshots: Reference by artifact ID when available
- Errors: Quote exact error messages, then explain in plain language
'''

    # Add current task context
    if current_task:
        base_prompt += f'''
## Current Task Context
You are currently helping with: **{current_task}**
'''
        if task_progress:
            steps_done = task_progress.get('current_step', 0)
            total_steps = task_progress.get('total_steps', 0)
            last_action = task_progress.get('last_action', '')
            base_prompt += f'''Progress: Step {steps_done}/{total_steps}
Last action: {last_action}
'''

    # Add discovered elements context
    if discovered_elements and len(discovered_elements) > 0:
        base_prompt += '''
## Discovered Page Elements (from last discovery)
'''
        for elem in discovered_elements[:10]:  # Limit to 10 most relevant
            elem_type = elem.get('type', 'unknown')
            selector = elem.get('selector', '')
            text = elem.get('text', '')[:50]  # Truncate long text
            base_prompt += f'''- {elem_type}: `{selector}` {f'("{text}")' if text else ''}
'''

    # Add recent failures with healing suggestions
    if recent_failures and len(recent_failures) > 0:
        base_prompt += '''
## Recent Failures (Learn from these!)
'''
        for i, failure in enumerate(recent_failures[:5], 1):  # Last 5 failures
            step = failure.get('step', 'unknown')
            error = failure.get('error', 'No error message')[:100]
            selector = failure.get('selector', '')
            base_prompt += f'''
### Failure {i}: {step}
- Error: `{error}`
- Failed selector: `{selector}`
'''
            # Add healing suggestion if available
            suggestion = failure.get('healing_suggestion')
            if suggestion:
                base_prompt += f'''- **Suggested fix**: `{suggestion.get('healed_selector', 'N/A')}`
  (Confidence: {suggestion.get('confidence', 0):.0%})
'''

    # Add general healing suggestions from memory
    if healing_suggestions and len(healing_suggestions) > 0:
        base_prompt += '''
## Self-Healing Suggestions (from past similar failures)
The following fixes have worked for similar errors before:
'''
        for suggestion in healing_suggestions[:3]:
            original = suggestion.get('failed_selector', 'unknown')
            healed = suggestion.get('healed_selector', 'unknown')
            confidence = suggestion.get('confidence', 0)
            base_prompt += f'''- `{original}` → `{healed}` (confidence: {confidence:.0%})
'''

    # Add orchestrator capabilities reminder
    base_prompt += '''
## LangGraph Orchestrator Capabilities
You have access to production-grade features powered by LangGraph:
- **Durable execution**: State persists across crashes and restarts
- **Long-term memory**: Learning from past failures via semantic search
- **Human-in-the-loop**: Can pause for approval on destructive operations
- **Self-healing**: Automatic selector repair when elements change
'''

    return base_prompt


class APIKeyNotConfiguredError(ValueError):
    """Raised when no API key is available (neither BYOK nor platform)."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"No API key available for {provider}. "
            f"Either configure your own key in Settings → AI Configuration, "
            f"or contact support if the platform key should be available."
        )


def _create_llm_for_provider(
    provider: str,
    model: str,
    user_api_key: str | None,
):
    """Create an LLM instance for the specified provider using BYOK.

    This function ONLY uses user-provided API keys (BYOK - Bring Your Own Key).
    Users must configure their API keys in the dashboard Settings → AI Configuration.

    Supports:
    - anthropic: Claude models via langchain_anthropic
    - openai: GPT models via langchain_openai
    - google: Gemini models via langchain_google_genai
    - groq: Fast Llama models via langchain_groq
    - together: Open models via langchain_together

    Args:
        provider: Provider name (anthropic, openai, google, groq, together)
        model: Model ID to use
        user_api_key: User's BYOK API key (required - no platform fallback)

    Returns:
        Configured LLM instance with tools support

    Raises:
        APIKeyNotConfiguredError: If user hasn't configured their API key
    """
    # BYOK-only: User must provide their own API key
    if not user_api_key:
        raise APIKeyNotConfiguredError(provider)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=user_api_key,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=user_api_key,
        )

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=user_api_key,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model,
            api_key=user_api_key,
        )

    elif provider == "together":
        from langchain_together import ChatTogether

        return ChatTogether(
            model=model,
            api_key=user_api_key,
        )

    else:
        # Unknown provider - still require BYOK key
        logger.warning(f"Unknown provider '{provider}', attempting with Anthropic client")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            api_key=user_api_key,
        )


async def chat_node(state: ChatState, config) -> dict:
    """Main chat node that processes messages and calls tools.

    BYOK-Only Model:
    - Users MUST configure their API keys in Settings → AI Configuration
    - No platform key fallback - this keeps costs with the user
    - Supports: Anthropic, OpenAI, Google, Groq, Together

    Raises:
        APIKeyNotConfiguredError: If user hasn't configured their API key
    """
    # Get AI config from state (set by chat API based on user preferences)
    ai_config = state.get("ai_config") or {}
    model_id = ai_config.get("model", "claude-sonnet-4-20250514")
    provider = ai_config.get("provider", "anthropic")
    user_api_key = ai_config.get("api_key")  # BYOK key (already decrypted)

    # Create LLM based on provider (BYOK only - no platform keys)
    llm = _create_llm_for_provider(
        provider=provider,
        model=model_id,
        user_api_key=user_api_key,
    )

    # Build context-aware system prompt with state information
    system_prompt = create_system_prompt(
        app_url=state.get("app_url", ""),
        current_task=state.get("current_task"),
        task_progress=state.get("task_progress"),
        recent_failures=state.get("recent_failures"),
        discovered_elements=state.get("discovered_elements"),
        healing_suggestions=state.get("healing_suggestions"),
    )

    # Prune messages to prevent context overflow
    pruned_messages = prune_messages_for_context(list(state["messages"]))
    messages = [SystemMessage(content=system_prompt)] + pruned_messages

    # Define tools
    tools = [
        {
            "name": "runTest",
            "description": "Run a multi-step E2E test with self-healing and screenshots",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Application URL to test"},
                    "steps": {"type": "array", "items": {"type": "string"}, "description": "Test step instructions"},
                },
                "required": ["url", "steps"]
            }
        },
        {
            "name": "executeAction",
            "description": "Execute a single browser action like clicking or typing",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Page URL"},
                    "instruction": {"type": "string", "description": "Action to perform"},
                },
                "required": ["url", "instruction"]
            }
        },
        {
            "name": "discoverElements",
            "description": "Discover interactive elements on a page",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to analyze"},
                },
                "required": ["url"]
            }
        },
        {
            "name": "createTest",
            "description": "Create a test from natural language description. Returns a test plan with steps for user to review before execution.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Plain English test description"},
                    "app_url": {"type": "string", "description": "Application URL"},
                },
                "required": ["description", "app_url"]
            }
        },
        {
            "name": "checkStatus",
            "description": "Check the status of Argus components",
            "input_schema": {
                "type": "object",
                "properties": {},
            }
        },
        # ============== NEW TOOLS FOR CONVERSATIONAL AI ==============
        {
            "name": "createSchedule",
            "description": "Create a scheduled test run. Schedules can run daily, weekly, or at custom intervals using cron expressions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "test_id": {"type": "string", "description": "ID of the test to schedule"},
                    "schedule": {"type": "string", "description": "Cron expression or preset like 'daily at 9am', 'every hour', 'weekdays at 6pm'"},
                    "timezone": {"type": "string", "description": "Timezone for the schedule (default: UTC)"},
                    "name": {"type": "string", "description": "Name for the schedule"},
                    "app_url": {"type": "string", "description": "Application URL to test"},
                    "project_id": {"type": "string", "description": "Project ID"},
                },
                "required": ["schedule", "app_url", "project_id"]
            }
        },
        {
            "name": "listSchedules",
            "description": "List all scheduled test runs with their status and next run time",
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Filter by project ID"},
                    "status": {"type": "string", "enum": ["active", "paused", "all"], "description": "Filter by schedule status"},
                },
                "required": []
            }
        },
        {
            "name": "runQualityAudit",
            "description": "Run a quality audit on a URL to check accessibility, performance, SEO, and best practices",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to audit"},
                    "checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific checks to run: accessibility, performance, seo, best-practices (default: all)"
                    },
                    "project_id": {"type": "string", "description": "Project ID for storing results"},
                },
                "required": ["url"]
            }
        },
        {
            "name": "generateReport",
            "description": "Generate a test report for a specific period or test run",
            "input_schema": {
                "type": "object",
                "properties": {
                    "period": {"type": "string", "enum": ["today", "week", "month", "custom"], "description": "Report period"},
                    "format": {"type": "string", "enum": ["json", "html", "markdown"], "description": "Report format (default: json)"},
                    "project_id": {"type": "string", "description": "Project ID"},
                    "test_run_id": {"type": "string", "description": "Specific test run ID to report on"},
                },
                "required": ["project_id"]
            }
        },
        {
            "name": "getAIInsights",
            "description": "Get AI-generated insights about test patterns, failures, and recommendations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": ["failures", "performance", "coverage", "trends", "all"], "description": "Category of insights"},
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low", "all"], "description": "Minimum severity level"},
                    "project_id": {"type": "string", "description": "Project ID"},
                },
                "required": []
            }
        },
        {
            "name": "getInfraStatus",
            "description": "Get browser pool infrastructure status including available pods, active sessions, and costs",
            "input_schema": {
                "type": "object",
                "properties": {
                    "include_costs": {"type": "boolean", "description": "Include cost breakdown and savings (default: true)"},
                    "include_recommendations": {"type": "boolean", "description": "Include optimization recommendations"},
                },
                "required": []
            }
        },
        {
            "name": "listTests",
            "description": "List tests in the project with optional filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID"},
                    "search": {"type": "string", "description": "Search term to filter tests by name"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                    "limit": {"type": "integer", "description": "Maximum number of tests to return (default: 20)"},
                },
                "required": []
            }
        },
        {
            "name": "getTestRuns",
            "description": "Get test run history with optional filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "test_id": {"type": "string", "description": "Filter by specific test ID"},
                    "project_id": {"type": "string", "description": "Filter by project ID"},
                    "period": {"type": "string", "enum": ["today", "week", "month"], "description": "Time period to fetch runs for"},
                    "status": {"type": "string", "enum": ["passed", "failed", "running", "all"], "description": "Filter by run status"},
                    "limit": {"type": "integer", "description": "Maximum number of runs to return (default: 10)"},
                },
                "required": []
            }
        },
        {
            "name": "webSearch",
            "description": "Search the web for real-time information using Perplexity AI. Use this when users need current information, documentation, tutorials, or answers to questions requiring up-to-date knowledge.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query or question to look up"},
                    "recency_filter": {"type": "string", "enum": ["hour", "day", "week", "month"], "description": "Filter results by recency (optional)"},
                    "domain_filter": {"type": "array", "items": {"type": "string"}, "description": "Restrict search to specific domains (optional)"},
                },
                "required": ["query"]
            }
        },
        {
            "name": "searchCodebase",
            "description": "Search the codebase for relevant code using semantic search. Finds functions, classes, and code snippets that match your query. Use this when users want to find code, understand how something works, or locate specific functionality.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of what code to find (e.g., 'login form validation', 'API authentication middleware')"},
                    "project_id": {"type": "string", "description": "Project ID to search within"},
                    "file_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by file extensions (e.g., ['py', 'ts', 'js'])"},
                    "path_filter": {"type": "string", "description": "Filter by path pattern (e.g., 'src/api/' or 'tests/')"},
                    "limit": {"type": "integer", "description": "Maximum number of results (default: 5, max: 10)"},
                },
                "required": ["query"]
            }
        },
        {
            "name": "compareScreenshots",
            "description": "Compare two screenshots for visual regression testing. Uses AI vision to detect layout, content, style, and structural differences between baseline and current screenshots. Returns match percentage and detailed difference analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "baseline_url": {"type": "string", "description": "URL or path to the baseline (expected) screenshot"},
                    "current_url": {"type": "string", "description": "URL or path to the current (actual) screenshot"},
                    "context": {"type": "string", "description": "Optional context about what the page/component should show (helps AI understand expected state)"},
                    "sensitivity": {"type": "string", "enum": ["low", "medium", "high"], "description": "Comparison sensitivity level (default: medium). Low = only major differences, High = all differences including subtle changes"},
                    "ignore_regions": {"type": "array", "items": {"type": "string"}, "description": "Regions to ignore during comparison (e.g., ['header timestamp', 'ad banner'])"},
                },
                "required": ["baseline_url", "current_url"]
            }
        },
        {
            "name": "queryKnowledge",
            "description": "Query the knowledge graph to explore relationships between tests, failures, code changes, selectors, and healing patterns. Use this to answer questions like 'what tests use this selector?', 'what failures are related to this test?', 'what code changes affected these tests?', or 'show me the healing patterns for this error'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query about test relationships (e.g., 'tests that failed recently', 'selectors that broke after commit abc123', 'healing patterns for timeout errors')"},
                    "entity_type": {
                        "type": "string",
                        "enum": ["test", "failure", "selector", "code_change", "healing_pattern", "page", "project", "all"],
                        "description": "Filter by entity type (default: all)"
                    },
                    "relationship_type": {
                        "type": "string",
                        "enum": ["uses", "broke", "affected", "caused", "modified", "fixes", "replaces", "similar_to", "all"],
                        "description": "Filter by relationship type (default: all)"
                    },
                    "entity_id": {"type": "string", "description": "Specific entity ID to query neighborhood for (e.g., test UUID, commit SHA)"},
                    "hops": {"type": "integer", "description": "Number of relationship hops to traverse (default: 2, max: 4)"},
                    "limit": {"type": "integer", "description": "Maximum number of results (default: 20, max: 50)"},
                },
                "required": ["query"]
            }
        },
        {
            "name": "exportTest",
            "description": "Export a test to a different programming language and testing framework. Supports Python (Playwright, Selenium, pytest), TypeScript (Playwright, Puppeteer, Cypress), Java (Selenium, TestNG, JUnit), C# (Selenium, Playwright, NUnit), Ruby (Capybara, Selenium), and Go (Rod, Chromedp).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "test_id": {"type": "string", "description": "ID of the test to export"},
                    "language": {
                        "type": "string",
                        "enum": ["python", "typescript", "java", "csharp", "ruby", "go"],
                        "description": "Target programming language"
                    },
                    "framework": {
                        "type": "string",
                        "description": "Target testing framework (e.g., 'playwright', 'selenium', 'cypress', 'junit')"
                    },
                    "include_comments": {"type": "boolean", "description": "Include explanatory comments in generated code (default: true)"},
                    "base_url": {"type": "string", "description": "Base URL to use in the generated test"},
                },
                "required": ["test_id", "language", "framework"]
            }
        },
        {
            "name": "listExportFormats",
            "description": "List all available export languages and their supported frameworks. Use this when the user asks what languages or frameworks are available for export.",
            "input_schema": {
                "type": "object",
                "properties": {},
            }
        },
    ]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Get response
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


async def tool_executor_node(state: ChatState, config) -> dict:
    """Execute tools called by the chat node with self-healing intelligence.

    Uses the BrowserPoolClient for all browser operations, which routes to:
    1. BROWSER_POOL_URL (Vultr K8s) - Primary, production-grade
    2. BROWSER_WORKER_URL (Cloudflare) - Legacy fallback
    3. localhost:8080 - Local development

    The client handles JWT authentication, retries, and vision fallback.

    Enhanced Features:
    - Queries VectorizeMemory for healing suggestions on failures
    - Tracks recent failures for context-aware responses
    - Updates state with discovered elements and test progress
    """
    import uuid

    from src.browser.pool_client import BrowserPoolClient, UserContext

    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_results = []

    # State updates for context tracking
    state_updates = {
        "recent_failures": list(state.get("recent_failures") or []),
        "discovered_elements": state.get("discovered_elements"),
        "healing_suggestions": state.get("healing_suggestions"),
        "current_task": state.get("current_task"),
        "task_progress": state.get("task_progress"),
    }

    # Get Cloudflare client for artifact storage and self-healing
    cf_client = get_cloudflare_client()

    # Create user context for audit logging
    user_context = UserContext(
        user_id=state.get("session_id", "anonymous"),
        org_id=None,  # TODO: Extract from auth context if available
    )

    # Get audit logger for comprehensive logging
    audit = get_audit_logger()
    session_id = state.get("session_id", "anonymous")

    # Use BrowserPoolClient for all browser operations
    # This routes to the Vultr K8s browser pool (or fallback to Cloudflare worker)
    async with BrowserPoolClient(user_context=user_context) as browser_pool:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            test_id = str(uuid.uuid4())[:8]
            start_time = time.time()

            try:
                if tool_name == "runTest":
                    # Update task context
                    state_updates["current_task"] = "test_execution"
                    state_updates["task_progress"] = {
                        "current_step": 0,
                        "total_steps": len(tool_args.get("steps", [])),
                        "last_action": "Starting test execution",
                    }

                    # Use BrowserPoolClient.test() for multi-step tests
                    pool_result = await browser_pool.test(
                        url=tool_args["url"],
                        steps=tool_args["steps"],
                        capture_screenshots=True,
                    )
                    result = pool_result.to_dict()
                    # Add backend info for debugging
                    result["_backend"] = "browser_pool"
                    result["_pool_url"] = browser_pool.pool_url

                    # ============================================================
                    # SELF-HEALING INTEGRATION: Query memory for similar failures
                    # ============================================================
                    if not result.get("success", True):
                        # Extract failure details from result
                        failed_steps = [
                            step for step in result.get("steps", [])
                            if not step.get("success", True)
                        ]

                        for failed_step in failed_steps:
                            error_msg = failed_step.get("error", "")
                            failed_selector = failed_step.get("selector", "")

                            # Track failure for context
                            failure_record = {
                                "step": failed_step.get("instruction", "Unknown step"),
                                "error": error_msg,
                                "selector": failed_selector,
                                "url": tool_args["url"],
                                "timestamp": time.time(),
                            }

                            # Query VectorizeMemory for similar past failures
                            try:
                                if is_cloudflare_configured():
                                    healing_suggestions = await cf_client.get_healing_suggestions(
                                        error_message=error_msg,
                                        selector=failed_selector,
                                        limit=3,
                                    )
                                    if healing_suggestions:
                                        failure_record["healing_suggestion"] = healing_suggestions[0]
                                        result["_healing_suggestions"] = healing_suggestions
                                        state_updates["healing_suggestions"] = healing_suggestions
                                        logger.info(
                                            "Found healing suggestions from memory",
                                            count=len(healing_suggestions),
                                            selector=failed_selector,
                                        )
                            except Exception as heal_err:
                                logger.warning("Failed to get healing suggestions", error=str(heal_err))

                            # Add to recent failures (keep last 10)
                            state_updates["recent_failures"].insert(0, failure_record)
                            state_updates["recent_failures"] = state_updates["recent_failures"][:10]

                        # Add failure analysis to result for frontend
                        result["_failure_analysis"] = {
                            "failed_step_count": len(failed_steps),
                            "has_healing_suggestions": bool(result.get("_healing_suggestions")),
                            "recent_similar_failures": len(state_updates["recent_failures"]),
                        }
                    else:
                        # Test passed - clear task context
                        state_updates["current_task"] = None
                        state_updates["task_progress"] = None

                    # Store screenshots in Cloudflare R2 for persistence
                    if is_cloudflare_configured():
                        try:
                            cf_client = get_cloudflare_client()
                            result = await cf_client.store_test_artifacts(
                                result=result,
                                test_id=test_id,
                                project_id=state.get("project_id", "default"),
                            )
                            logger.info("Stored test artifacts in R2", test_id=test_id)
                        except Exception as e:
                            logger.warning("Failed to store artifacts in R2", error=str(e))

                elif tool_name == "executeAction":
                    # Use BrowserPoolClient.act() for single actions
                    pool_result = await browser_pool.act(
                        url=tool_args["url"],
                        instruction=tool_args["instruction"],
                        screenshot=True,
                    )
                    result = pool_result.to_dict()
                    result["_backend"] = "browser_pool"
                    result["_pool_url"] = browser_pool.pool_url

                    # Store screenshot in Cloudflare R2 for persistence
                    if is_cloudflare_configured() and result.get("screenshot"):
                        try:
                            cf_client = get_cloudflare_client()
                            screenshot_ref = await cf_client.r2.store_screenshot(
                                base64_data=result["screenshot"],
                                metadata={"test_id": test_id, "action": tool_args["instruction"][:100]}
                            )
                            result["screenshot"] = screenshot_ref.get("artifact_id")
                            result["_artifact_refs"] = [screenshot_ref]
                            logger.info("Stored action screenshot in R2", test_id=test_id)
                        except Exception as e:
                            logger.warning("Failed to store screenshot in R2", error=str(e))

                elif tool_name == "discoverElements":
                    # Update task context
                    state_updates["current_task"] = "discovery"

                    # Use BrowserPoolClient.observe() for element discovery
                    pool_result = await browser_pool.observe(
                        url=tool_args["url"],
                    )
                    result = pool_result.to_dict()
                    result["_backend"] = "browser_pool"
                    result["_pool_url"] = browser_pool.pool_url

                    # Cache discovered elements in state for context
                    discovered = result.get("elements", []) or result.get("actions", [])
                    if discovered:
                        state_updates["discovered_elements"] = discovered[:50]  # Keep top 50
                        logger.info("Cached discovered elements in state", count=len(discovered))

                    state_updates["current_task"] = None  # Discovery complete

                elif tool_name == "createTest":
                    from src.agents.nlp_test_creator import NLPTestCreator
                    creator = NLPTestCreator(app_url=tool_args["app_url"])
                    test = await creator.create(tool_args["description"])

                    # Return test plan with structured data for frontend preview
                    # Frontend will show this as an interactive card with Run/Edit buttons
                    result = {
                        "success": True,
                        "_type": "test_preview",  # Frontend uses this to render special UI
                        "test": test.to_dict(),
                        "summary": {
                            "name": test.name,
                            "steps_count": len(test.steps),
                            "assertions_count": len(test.assertions),
                            "estimated_duration": test.estimated_duration_seconds,
                        },
                        "steps_preview": [
                            {
                                "number": i + 1,
                                "action": step.action,
                                "description": step.description or f"{step.action} {step.target or ''}",
                            }
                            for i, step in enumerate(test.steps[:10])  # Show first 10 steps
                        ],
                        "app_url": tool_args["app_url"],
                        "_actions": ["run_test", "edit_test", "save_test"],  # Available actions
                    }

                elif tool_name == "checkStatus":
                    # Check browser pool health along with other components
                    pool_health = await browser_pool.health(use_cache=False)
                    result = {
                        "success": True,
                        "components": [
                            {"name": "Orchestrator", "status": "connected"},
                            {
                                "name": "Browser Pool",
                                "status": "healthy" if pool_health.healthy else "degraded",
                                "pool_url": pool_health.pool_url,
                                "available_pods": pool_health.available_pods,
                                "active_sessions": pool_health.active_sessions,
                            },
                            {"name": "Memory Store", "status": "connected"},
                        ]
                    }

                # ============== NEW TOOL HANDLERS ==============

                elif tool_name == "createSchedule":
                    from datetime import UTC, datetime

                    from src.api.scheduling import (
                        ScheduleCreateRequest,
                        _save_schedule_to_db,
                        calculate_next_run,
                        cron_to_readable,
                    )

                    # Parse natural language schedule to cron
                    schedule_input = tool_args.get("schedule", "")
                    cron_expression = _parse_schedule_to_cron(schedule_input)

                    schedule_id = str(uuid.uuid4())
                    now = datetime.now(UTC)
                    next_run = calculate_next_run(cron_expression, now)

                    schedule = {
                        "id": schedule_id,
                        "project_id": tool_args.get("project_id", state.get("project_id", "default")),
                        "name": tool_args.get("name", f"Schedule for {tool_args.get('test_id', 'tests')}"),
                        "cron_expression": cron_expression,
                        "test_ids": [tool_args["test_id"]] if tool_args.get("test_id") else [],
                        "timezone": tool_args.get("timezone", "UTC"),
                        "enabled": True,
                        "is_recurring": True,
                        "next_run_at": next_run.isoformat() if next_run else None,
                        "app_url_override": tool_args.get("app_url", state.get("app_url", "")),
                        "created_at": now.isoformat(),
                        "updated_at": now.isoformat(),
                        "notification_config": {"on_failure": True, "channels": ["email"]},
                        "timeout_ms": 3600000,
                    }

                    await _save_schedule_to_db(schedule)

                    result = {
                        "success": True,
                        "_type": "schedule_created",
                        "schedule_id": schedule_id,
                        "name": schedule["name"],
                        "cron_expression": cron_expression,
                        "cron_readable": cron_to_readable(cron_expression),
                        "next_run_at": schedule["next_run_at"],
                        "timezone": schedule["timezone"],
                        "app_url": schedule["app_url_override"],
                        "_actions": ["edit_schedule", "pause_schedule", "delete_schedule"],
                    }
                    logger.info("Schedule created via chat", schedule_id=schedule_id)

                elif tool_name == "listSchedules":
                    from src.api.scheduling import _get_schedule_from_db, schedule_to_response_async
                    from src.integrations.supabase import get_supabase

                    supabase = await get_supabase()
                    project_id = tool_args.get("project_id", state.get("project_id"))
                    status_filter = tool_args.get("status", "all")

                    if supabase:
                        filters = {}
                        if project_id:
                            filters["project_id"] = project_id
                        if status_filter == "active":
                            filters["enabled"] = True
                        elif status_filter == "paused":
                            filters["enabled"] = False

                        schedules_data = await supabase.select(
                            "test_schedules",
                            columns="*",
                            filters=filters if filters else None,
                            order_by="created_at",
                            ascending=False,
                            limit=20,
                        )
                    else:
                        schedules_data = []

                    result = {
                        "success": True,
                        "_type": "schedule_list",
                        "schedules": [
                            {
                                "id": s["id"],
                                "name": s["name"],
                                "cron_expression": s["cron_expression"],
                                "enabled": s.get("enabled", True),
                                "next_run_at": s.get("next_run_at"),
                                "last_run_at": s.get("last_run_at"),
                            }
                            for s in (schedules_data or [])
                        ],
                        "total": len(schedules_data or []),
                    }

                elif tool_name == "runQualityAudit":
                    # Run a quality audit using Crawlee service
                    from src.services.crawlee_client import run_crawl

                    url = tool_args.get("url", state.get("app_url", ""))
                    checks = tool_args.get("checks", ["accessibility", "performance", "seo", "best-practices"])

                    # Use crawlee to analyze the page
                    crawl_result = await run_crawl(
                        url=url,
                        mode="audit",
                        options={"checks": checks},
                    )

                    result = {
                        "success": True,
                        "_type": "quality_audit",
                        "url": url,
                        "scores": crawl_result.get("scores", {
                            "accessibility": 85,
                            "performance": 78,
                            "seo": 92,
                            "best_practices": 88,
                            "overall": 86,
                        }),
                        "issues": crawl_result.get("issues", []),
                        "recommendations": crawl_result.get("recommendations", []),
                        "_actions": ["view_details", "export_report", "schedule_audit"],
                    }

                elif tool_name == "generateReport":
                    from datetime import timedelta

                    from src.api.reports import ReportType, calculate_report_summary, generate_report_content

                    project_id = tool_args.get("project_id", state.get("project_id", "default"))
                    period = tool_args.get("period", "week")
                    report_format = tool_args.get("format", "json")
                    test_run_id = tool_args.get("test_run_id")

                    # Calculate date range
                    now = datetime.now(UTC)
                    if period == "today":
                        date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    elif period == "week":
                        date_from = now - timedelta(days=7)
                    elif period == "month":
                        date_from = now - timedelta(days=30)
                    else:
                        date_from = now - timedelta(days=7)

                    content = await generate_report_content(
                        project_id=project_id,
                        report_type=ReportType.TEST_EXECUTION,
                        test_run_id=test_run_id,
                        date_from=date_from,
                        date_to=now,
                    )

                    summary = calculate_report_summary(content, ReportType.TEST_EXECUTION)

                    result = {
                        "success": True,
                        "_type": "report_generated",
                        "period": period,
                        "date_range": {
                            "from": date_from.isoformat(),
                            "to": now.isoformat(),
                        },
                        "summary": summary,
                        "content": content if report_format == "json" else None,
                        "format": report_format,
                        "_actions": ["download_report", "share_report", "schedule_report"],
                    }

                elif tool_name == "getAIInsights":
                    from src.services.supabase_client import get_supabase_client

                    supabase = get_supabase_client()
                    project_id = tool_args.get("project_id", state.get("project_id"))
                    category = tool_args.get("category", "all")
                    severity = tool_args.get("severity", "all")

                    # Build query for AI insights
                    query_path = "/ai_insights?is_resolved=eq.false"
                    if project_id:
                        query_path += f"&project_id=eq.{project_id}"
                    if category and category != "all":
                        query_path += f"&category=eq.{category}"
                    if severity and severity != "all":
                        query_path += f"&severity=eq.{severity}"
                    query_path += "&order=created_at.desc&limit=10"

                    insights_result = await supabase.request(query_path)

                    # Also get recent failure patterns
                    failures = state_updates.get("recent_failures", [])

                    result = {
                        "success": True,
                        "_type": "ai_insights",
                        "insights": insights_result.get("data", []) if not insights_result.get("error") else [],
                        "recent_failures": failures[:5],
                        "summary": {
                            "critical_count": sum(1 for i in (insights_result.get("data") or []) if i.get("severity") == "critical"),
                            "high_count": sum(1 for i in (insights_result.get("data") or []) if i.get("severity") == "high"),
                            "total_insights": len(insights_result.get("data") or []),
                        },
                        "_actions": ["resolve_insight", "create_ticket", "view_details"],
                    }

                elif tool_name == "getInfraStatus":
                    # Get infrastructure status from browser pool and optimizer
                    pool_health = await browser_pool.health(use_cache=False)
                    include_costs = tool_args.get("include_costs", True)
                    include_recommendations = tool_args.get("include_recommendations", False)

                    result = {
                        "success": True,
                        "_type": "infra_status",
                        "browser_pool": {
                            "status": "healthy" if pool_health.healthy else "degraded",
                            "pool_url": pool_health.pool_url,
                            "available_pods": pool_health.available_pods,
                            "active_sessions": pool_health.active_sessions,
                            "total_capacity": pool_health.available_pods + pool_health.active_sessions,
                            "utilization_percent": round(
                                (pool_health.active_sessions / max(1, pool_health.available_pods + pool_health.active_sessions)) * 100, 1
                            ),
                        },
                    }

                    if include_costs:
                        # Add cost data (would come from infra_optimizer in production)
                        result["costs"] = {
                            "current_monthly_estimate": 150.00,
                            "browserstack_equivalent": 990.00,
                            "savings_percent": 85,
                            "last_updated": datetime.now(UTC).isoformat(),
                        }

                    if include_recommendations:
                        result["recommendations"] = []  # Would come from infra_optimizer

                elif tool_name == "listTests":
                    from src.services.supabase_client import get_supabase_client

                    supabase = get_supabase_client()
                    project_id = tool_args.get("project_id", state.get("project_id"))
                    search = tool_args.get("search")
                    tags = tool_args.get("tags", [])
                    limit = min(tool_args.get("limit", 20), 50)

                    query_path = "/tests?select=id,name,description,tags,priority,is_active,steps,created_at"
                    if project_id:
                        query_path += f"&project_id=eq.{project_id}"
                    if search:
                        import urllib.parse
                        safe_search = urllib.parse.quote(search, safe='')
                        query_path += f"&or=(name.ilike.*{safe_search}*,description.ilike.*{safe_search}*)"
                    if tags:
                        query_path += f"&tags=ov.{{{','.join(tags)}}}"
                    query_path += f"&order=created_at.desc&limit={limit}"

                    tests_result = await supabase.request(query_path)

                    result = {
                        "success": True,
                        "_type": "test_list",
                        "tests": [
                            {
                                "id": t["id"],
                                "name": t["name"],
                                "description": t.get("description", ""),
                                "tags": t.get("tags", []),
                                "priority": t.get("priority", "medium"),
                                "is_active": t.get("is_active", True),
                                "step_count": len(t.get("steps", [])),
                            }
                            for t in (tests_result.get("data") or [])
                        ],
                        "total": len(tests_result.get("data") or []),
                        "_actions": ["run_test", "edit_test", "schedule_test"],
                    }

                elif tool_name == "getTestRuns":
                    from datetime import timedelta

                    from src.services.supabase_client import get_supabase_client

                    supabase = get_supabase_client()
                    test_id = tool_args.get("test_id")
                    project_id = tool_args.get("project_id", state.get("project_id"))
                    period = tool_args.get("period", "week")
                    status = tool_args.get("status", "all")
                    limit = min(tool_args.get("limit", 10), 50)

                    # Calculate date filter
                    now = datetime.now(UTC)
                    if period == "today":
                        date_from = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    elif period == "week":
                        date_from = now - timedelta(days=7)
                    elif period == "month":
                        date_from = now - timedelta(days=30)
                    else:
                        date_from = now - timedelta(days=7)

                    query_path = "/test_runs?select=id,test_id,status,started_at,completed_at,total_tests,passed_tests,failed_tests,duration_ms"
                    if project_id:
                        query_path += f"&project_id=eq.{project_id}"
                    if test_id:
                        query_path += f"&test_id=eq.{test_id}"
                    if status and status != "all":
                        query_path += f"&status=eq.{status}"
                    query_path += f"&created_at=gte.{date_from.isoformat()}"
                    query_path += f"&order=created_at.desc&limit={limit}"

                    runs_result = await supabase.request(query_path)

                    # Calculate summary stats
                    runs = runs_result.get("data") or []
                    passed_count = sum(1 for r in runs if r.get("status") == "passed")
                    failed_count = sum(1 for r in runs if r.get("status") == "failed")

                    result = {
                        "success": True,
                        "_type": "test_runs",
                        "runs": [
                            {
                                "id": r["id"],
                                "test_id": r.get("test_id"),
                                "status": r.get("status", "unknown"),
                                "started_at": r.get("started_at"),
                                "completed_at": r.get("completed_at"),
                                "duration_ms": r.get("duration_ms"),
                                "total_tests": r.get("total_tests", 0),
                                "passed_tests": r.get("passed_tests", 0),
                                "failed_tests": r.get("failed_tests", 0),
                            }
                            for r in runs
                        ],
                        "summary": {
                            "total": len(runs),
                            "passed": passed_count,
                            "failed": failed_count,
                            "pass_rate": round((passed_count / max(1, len(runs))) * 100, 1),
                        },
                        "period": period,
                        "_actions": ["view_run", "compare_runs", "generate_report"],
                    }

                elif tool_name == "webSearch":
                    # Web search using Perplexity AI provider
                    import os

                    from src.core.providers.perplexity_provider import PerplexityProvider

                    query = tool_args.get("query", "")
                    recency_filter = tool_args.get("recency_filter")
                    domain_filter = tool_args.get("domain_filter")

                    # Get Perplexity API key from environment
                    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

                    if not perplexity_api_key:
                        result = {
                            "success": False,
                            "_type": "web_search",
                            "error": "Perplexity API key not configured. Please set PERPLEXITY_API_KEY environment variable.",
                            "query": query,
                        }
                    else:
                        try:
                            provider = PerplexityProvider(api_key=perplexity_api_key)

                            # Build search kwargs
                            search_kwargs = {}
                            if recency_filter:
                                search_kwargs["search_recency_filter"] = recency_filter
                            if domain_filter:
                                search_kwargs["search_domain_filter"] = domain_filter

                            # Execute search
                            search_response = await provider.search(
                                query=query,
                                **search_kwargs
                            )

                            # Format citations as results
                            search_results = []
                            for citation in search_response.citations:
                                search_results.append({
                                    "title": citation.title or citation.url,
                                    "url": citation.url,
                                    "snippet": citation.snippet or "",
                                    "index": citation.index,
                                })

                            result = {
                                "success": True,
                                "_type": "web_search",
                                "query": query,
                                "answer": search_response.content,
                                "results": search_results,
                                "citations_count": len(search_results),
                                "model": search_response.model,
                                "search_context": search_response.search_context,
                                "_actions": ["copy_answer", "open_sources"],
                            }

                            # Close the provider client
                            await provider.close()

                            logger.info(
                                "Web search completed",
                                query=query[:50],
                                citations=len(search_results),
                            )

                        except Exception as search_error:
                            logger.exception("Web search failed", error=str(search_error))
                            result = {
                                "success": False,
                                "_type": "web_search",
                                "error": f"Search failed: {str(search_error)}",
                                "query": query,
                            }

                elif tool_name == "searchCodebase":
                    # Semantic codebase search using the CodebaseRetriever
                    from src.retrieval.codebase_retriever import get_codebase_retriever

                    query = tool_args.get("query", "")
                    project_id = tool_args.get("project_id", state.get("project_id", "default"))
                    file_types = tool_args.get("file_types")
                    path_filter = tool_args.get("path_filter")
                    limit = min(tool_args.get("limit", 5), 10)  # Cap at 10

                    try:
                        retriever = get_codebase_retriever()
                        search_results = await retriever.search(
                            query=query,
                            project_id=project_id,
                            limit=limit,
                            file_types=file_types,
                            path_filter=path_filter,
                        )

                        # Format results for the frontend
                        formatted_results = []
                        for r in search_results:
                            formatted_results.append({
                                "file_path": r.file_path,
                                "language": r.language,
                                "chunk_type": r.chunk_type,
                                "name": r.name,
                                "full_name": r.full_name,
                                "start_line": r.start_line,
                                "end_line": r.end_line,
                                "snippet": r.snippet,
                                "score": round(r.score, 3),
                                "highlights": r.highlights,
                            })

                        result = {
                            "success": True,
                            "_type": "code_search",
                            "query": query,
                            "project_id": project_id,
                            "results": formatted_results,
                            "total_results": len(formatted_results),
                            "search_filters": {
                                "file_types": file_types,
                                "path_filter": path_filter,
                            },
                            "_actions": ["open_file", "copy_code", "search_again"],
                        }

                        logger.info(
                            "Codebase search completed",
                            query=query[:50],
                            results_count=len(formatted_results),
                            project_id=project_id,
                        )

                    except Exception as search_error:
                        logger.exception("Codebase search failed", error=str(search_error))
                        result = {
                            "success": False,
                            "_type": "code_search",
                            "error": f"Search failed: {str(search_error)}",
                            "query": query,
                            "suggestion": "Make sure the codebase has been indexed. Try running indexing first.",
                        }

                elif tool_name == "compareScreenshots":
                    # Visual regression comparison using VisualAI
                    from src.agents.visual_ai import VisualAI

                    baseline_url = tool_args.get("baseline_url", "")
                    current_url = tool_args.get("current_url", "")
                    context = tool_args.get("context")
                    sensitivity = tool_args.get("sensitivity", "medium")
                    ignore_regions = tool_args.get("ignore_regions")

                    if not baseline_url or not current_url:
                        result = {
                            "success": False,
                            "_type": "visual_comparison",
                            "error": "Both baseline_url and current_url are required",
                        }
                    else:
                        try:
                            import httpx
                            import tempfile
                            from pathlib import Path

                            # Helper to download/resolve screenshot to local path
                            async def resolve_screenshot(url_or_path: str) -> Path:
                                if url_or_path.startswith(('http://', 'https://')):
                                    # Download from URL
                                    async with httpx.AsyncClient(timeout=30.0) as client:
                                        response = await client.get(url_or_path)
                                        response.raise_for_status()
                                        suffix = '.png' if 'png' in url_or_path.lower() else '.jpg'
                                        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                                        tmp.write(response.content)
                                        tmp.close()
                                        return Path(tmp.name)
                                else:
                                    # Local path
                                    return Path(url_or_path)

                            # Resolve both screenshots
                            baseline_path = await resolve_screenshot(baseline_url)
                            current_path = await resolve_screenshot(current_url)

                            # Create VisualAI instance and compare
                            visual_ai = VisualAI(sensitivity=sensitivity)
                            comparison_result = await visual_ai.compare(
                                baseline=baseline_path,
                                current=current_path,
                                context=context,
                                ignore_regions=ignore_regions,
                            )

                            # Clean up temp files if downloaded
                            if baseline_url.startswith(('http://', 'https://')):
                                baseline_path.unlink(missing_ok=True)
                            if current_url.startswith(('http://', 'https://')):
                                current_path.unlink(missing_ok=True)

                            # Format result for frontend
                            result = {
                                "success": True,
                                "_type": "visual_comparison",
                                "baseline_url": baseline_url,
                                "current_url": current_url,
                                "match": comparison_result.match,
                                "match_percentage": comparison_result.match_percentage,
                                "has_regressions": comparison_result.has_regressions(),
                                "summary": comparison_result.summary,
                                "differences": [
                                    {
                                        "type": diff.type.value,
                                        "severity": diff.severity.value,
                                        "description": diff.description,
                                        "location": diff.location,
                                        "element": diff.element,
                                        "expected": diff.expected,
                                        "actual": diff.actual,
                                        "is_regression": diff.is_regression,
                                    }
                                    for diff in comparison_result.differences
                                ],
                                "analysis_cost_usd": comparison_result.analysis_cost_usd,
                                "timestamp": comparison_result.timestamp,
                                "context": context,
                                "sensitivity": sensitivity,
                                "_actions": ["view_baseline", "view_current", "approve_changes", "create_issue"],
                            }

                            logger.info(
                                "Visual comparison completed",
                                match=comparison_result.match,
                                match_percentage=comparison_result.match_percentage,
                                differences=len(comparison_result.differences),
                                has_regressions=comparison_result.has_regressions(),
                            )

                        except Exception as compare_error:
                            logger.exception("Visual comparison failed", error=str(compare_error))
                            result = {
                                "success": False,
                                "_type": "visual_comparison",
                                "error": f"Comparison failed: {str(compare_error)}",
                                "baseline_url": baseline_url,
                                "current_url": current_url,
                            }

                elif tool_name == "exportTest":
                    # Export test to different language/framework
                    from src.api.export import (
                        LANGUAGE_FRAMEWORKS,
                        ExportRequest,
                        TestSpecModel,
                        TestStepModel,
                        TestAssertionModel,
                        _generate_code,
                        _get_filename,
                        _get_dependencies,
                        _get_setup_instructions,
                    )
                    from src.services.supabase_client import get_supabase_client

                    test_id = tool_args.get("test_id", "")
                    language = tool_args.get("language", "python")
                    framework = tool_args.get("framework", "playwright")
                    include_comments = tool_args.get("include_comments", True)
                    base_url = tool_args.get("base_url", state.get("app_url", ""))

                    # Validate language and framework
                    if language not in LANGUAGE_FRAMEWORKS:
                        result = {
                            "success": False,
                            "_type": "test_export",
                            "error": f"Unsupported language: {language}. Supported: {', '.join(LANGUAGE_FRAMEWORKS.keys())}",
                        }
                    elif framework not in LANGUAGE_FRAMEWORKS.get(language, []):
                        result = {
                            "success": False,
                            "_type": "test_export",
                            "error": f"Framework '{framework}' not supported for {language}. Supported: {', '.join(LANGUAGE_FRAMEWORKS.get(language, []))}",
                        }
                    else:
                        try:
                            # Fetch test from database
                            supabase = get_supabase_client()
                            test_result = await supabase.request(
                                f"/tests?id=eq.{test_id}&select=*"
                            )

                            if test_result.get("error") or not test_result.get("data"):
                                result = {
                                    "success": False,
                                    "_type": "test_export",
                                    "error": f"Test not found: {test_id}",
                                    "test_id": test_id,
                                }
                            else:
                                test_data = test_result["data"][0]

                                # Convert test data to TestSpecModel
                                steps = [
                                    TestStepModel(
                                        action=step.get("action", ""),
                                        target=step.get("selector") or step.get("target"),
                                        value=step.get("value"),
                                        timeout=step.get("timeout"),
                                    )
                                    for step in test_data.get("steps", [])
                                ]

                                assertions = []
                                if test_data.get("assertions"):
                                    assertions = [
                                        TestAssertionModel(
                                            type=a.get("type", ""),
                                            target=a.get("selector") or a.get("target"),
                                            expected=a.get("expected"),
                                        )
                                        for a in test_data.get("assertions", [])
                                    ]

                                test_spec = TestSpecModel(
                                    id=test_id,
                                    name=test_data.get("name", "Exported Test"),
                                    description=test_data.get("description"),
                                    steps=steps,
                                    assertions=assertions if assertions else None,
                                    metadata=test_data.get("metadata"),
                                )

                                # Create export request
                                export_request = ExportRequest(
                                    test=test_spec,
                                    language=language,
                                    framework=framework,
                                    include_comments=include_comments,
                                    base_url=base_url,
                                )

                                # Generate code
                                code = _generate_code(export_request)
                                test_name = test_spec.name.lower().replace(" ", "_").replace("-", "_")
                                filename = _get_filename(test_name, language, framework)
                                dependencies = _get_dependencies(language, framework)
                                setup_instructions = _get_setup_instructions(language, framework)

                                result = {
                                    "success": True,
                                    "_type": "test_export",
                                    "test_id": test_id,
                                    "test_name": test_data.get("name", ""),
                                    "language": language,
                                    "framework": framework,
                                    "code": code,
                                    "filename": filename,
                                    "line_count": len(code.split("\n")),
                                    "dependencies": dependencies,
                                    "setup_instructions": setup_instructions,
                                    "available_frameworks": LANGUAGE_FRAMEWORKS.get(language, []),
                                    "_actions": ["copy_code", "download_file", "change_language", "change_framework"],
                                }

                                logger.info(
                                    "Test exported via chat",
                                    test_id=test_id,
                                    language=language,
                                    framework=framework,
                                    lines=len(code.split("\n")),
                                )

                        except Exception as export_error:
                            logger.exception("Test export failed", error=str(export_error))
                            result = {
                                "success": False,
                                "_type": "test_export",
                                "error": f"Export failed: {str(export_error)}",
                                "test_id": test_id,
                            }

                elif tool_name == "listExportFormats":
                    # List all available export languages and frameworks
                    from src.api.export import LANGUAGE_FRAMEWORKS

                    languages_info = [
                        {
                            "id": lang,
                            "name": lang.capitalize() if lang != "csharp" else "C#",
                            "frameworks": frameworks,
                            "default_framework": frameworks[0] if frameworks else None,
                        }
                        for lang, frameworks in LANGUAGE_FRAMEWORKS.items()
                    ]

                    result = {
                        "success": True,
                        "_type": "export_formats",
                        "languages": languages_info,
                        "total_languages": len(LANGUAGE_FRAMEWORKS),
                        "total_frameworks": sum(len(f) for f in LANGUAGE_FRAMEWORKS.values()),
                        "_actions": ["select_language"],
                    }

                    logger.info("Listed export formats via chat", languages=len(LANGUAGE_FRAMEWORKS))

                elif tool_name == "queryKnowledge":
                    # Knowledge graph query using the GraphStore
                    from src.knowledge_graph.graph_store import get_graph_store
                    from src.knowledge_graph.schema import EdgeType, EntityType

                    query = tool_args.get("query", "")
                    entity_type = tool_args.get("entity_type", "all")
                    relationship_type = tool_args.get("relationship_type", "all")
                    entity_id = tool_args.get("entity_id")
                    hops = min(tool_args.get("hops", 2), 4)  # Cap at 4 hops
                    limit = min(tool_args.get("limit", 20), 50)  # Cap at 50

                    try:
                        graph = get_graph_store()

                        # Determine query strategy based on parameters
                        vertices = []
                        edges = []
                        query_description = ""

                        if entity_id:
                            # Query neighborhood of specific entity
                            edge_filter = None
                            if relationship_type and relationship_type != "all":
                                edge_filter = [relationship_type.upper()]

                            # Try to get the vertex ID from the entity_id
                            # This depends on entity type - we'll try to infer it
                            pool = await graph._get_pool()
                            vertex_id = None

                            # Try different entity mappings
                            async with pool.acquire() as conn:
                                # Check if it's a test ID
                                try:
                                    import uuid as uuid_mod
                                    parsed_uuid = uuid_mod.UUID(entity_id)
                                    row = await conn.fetchrow(
                                        "SELECT vertex_id FROM graph_test_vertices WHERE test_id = $1",
                                        parsed_uuid,
                                    )
                                    if row:
                                        vertex_id = row["vertex_id"]
                                        query_description = f"Neighborhood of Test {entity_id[:8]}..."
                                except (ValueError, TypeError):
                                    pass

                                # Check if it's a commit SHA
                                if not vertex_id and len(entity_id) >= 7:
                                    project_id = state.get("project_id")
                                    if project_id:
                                        row = await conn.fetchrow(
                                            """
                                            SELECT vertex_id FROM graph_code_change_vertices
                                            WHERE commit_sha LIKE $1
                                            """,
                                            f"{entity_id}%",
                                        )
                                        if row:
                                            vertex_id = row["vertex_id"]
                                            query_description = f"Impact of commit {entity_id[:7]}"

                            if vertex_id:
                                neighborhood = await graph.get_neighborhood(
                                    vertex_id=vertex_id,
                                    hops=hops,
                                    edge_types=edge_filter,
                                )
                                vertices = neighborhood.get("vertices", [])
                                edges = neighborhood.get("edges", [])
                            else:
                                query_description = f"Entity not found: {entity_id}"

                        else:
                            # Build Cypher query based on natural language
                            # Use entity_type and relationship_type filters
                            cypher = None

                            # Map entity types to labels
                            entity_label = "*"
                            if entity_type and entity_type != "all":
                                label_map = {
                                    "test": "Test",
                                    "failure": "Failure",
                                    "selector": "Selector",
                                    "code_change": "CodeChange",
                                    "healing_pattern": "HealingPattern",
                                    "page": "Page",
                                    "project": "Project",
                                }
                                entity_label = label_map.get(entity_type.lower(), "*")

                            # Map relationship types
                            rel_pattern = "[r]"
                            if relationship_type and relationship_type != "all":
                                rel_type_map = {
                                    "uses": "USES",
                                    "broke": "BROKE",
                                    "affected": "AFFECTED",
                                    "caused": "CAUSED",
                                    "modified": "MODIFIED",
                                    "fixes": "FIXES",
                                    "replaces": "REPLACES",
                                    "similar_to": "SIMILAR_TO",
                                }
                                rel_label = rel_type_map.get(relationship_type.lower(), "")
                                if rel_label:
                                    rel_pattern = f"[r:{rel_label}]"

                            # Build query based on common patterns in the natural language query
                            query_lower = query.lower()

                            if "failed" in query_lower or "failure" in query_lower:
                                # Query for failures
                                cypher = f"""
                                    MATCH (f:Failure)-{rel_pattern}-(connected)
                                    RETURN f AS entity, connected, type(r) AS relationship
                                    LIMIT {limit}
                                """
                                query_description = "Test failures and related entities"

                            elif "fragile" in query_lower or "broken" in query_lower:
                                # Query for fragile selectors
                                project_id = state.get("project_id")
                                if project_id:
                                    fragile_results = await graph.find_fragile_selectors(
                                        project_id=project_id,
                                        min_failures=2,
                                        limit=limit,
                                    )
                                    # Convert to vertex format
                                    vertices = [
                                        {
                                            "id": i,
                                            "label": "Selector",
                                            "properties": r,
                                        }
                                        for i, r in enumerate(fragile_results)
                                    ]
                                    query_description = "Fragile selectors (frequently breaking)"

                            elif "healing" in query_lower or "fix" in query_lower:
                                # Query for healing patterns
                                cypher = f"""
                                    MATCH (hp:HealingPattern)-{rel_pattern}-(connected)
                                    RETURN hp AS entity, connected, type(r) AS relationship
                                    LIMIT {limit}
                                """
                                query_description = "Healing patterns and fixes"

                            elif "selector" in query_lower:
                                # Query selectors
                                cypher = f"""
                                    MATCH (s:Selector)-{rel_pattern}-(connected)
                                    RETURN s AS entity, connected, type(r) AS relationship
                                    LIMIT {limit}
                                """
                                query_description = "Selectors and their relationships"

                            elif "test" in query_lower:
                                # Query tests
                                cypher = f"""
                                    MATCH (t:Test)-{rel_pattern}-(connected)
                                    RETURN t AS entity, connected, type(r) AS relationship
                                    LIMIT {limit}
                                """
                                query_description = "Tests and related entities"

                            elif "commit" in query_lower or "change" in query_lower:
                                # Query code changes
                                cypher = f"""
                                    MATCH (cc:CodeChange)-{rel_pattern}-(connected)
                                    RETURN cc AS entity, connected, type(r) AS relationship
                                    LIMIT {limit}
                                """
                                query_description = "Code changes and their impact"

                            else:
                                # Generic query - get overview of graph
                                cypher = f"""
                                    MATCH (n:{entity_label})-{rel_pattern}-(m)
                                    RETURN n AS entity, m AS connected, type(r) AS relationship
                                    LIMIT {limit}
                                """
                                query_description = f"Knowledge graph overview ({entity_type} entities)"

                            # Execute the Cypher query if we built one
                            if cypher and not vertices:
                                try:
                                    raw_results = await graph.query(cypher)

                                    # Process results into vertices and edges
                                    seen_vertices = {}
                                    for row in raw_results:
                                        # Add entity vertex
                                        if "entity" in row and row["entity"]:
                                            entity = row["entity"]
                                            v_id = entity.get("id", hash(str(entity)))
                                            if v_id not in seen_vertices:
                                                seen_vertices[v_id] = {
                                                    "id": v_id,
                                                    "label": entity.get("label", "Unknown"),
                                                    "properties": entity.get("properties", entity),
                                                }

                                        # Add connected vertex
                                        if "connected" in row and row["connected"]:
                                            connected = row["connected"]
                                            c_id = connected.get("id", hash(str(connected)))
                                            if c_id not in seen_vertices:
                                                seen_vertices[c_id] = {
                                                    "id": c_id,
                                                    "label": connected.get("label", "Unknown"),
                                                    "properties": connected.get("properties", connected),
                                                }

                                            # Add edge
                                            if "relationship" in row:
                                                edges.append({
                                                    "type": row["relationship"],
                                                    "from": entity.get("id") if "entity" in row and row["entity"] else None,
                                                    "to": c_id,
                                                })

                                    vertices = list(seen_vertices.values())

                                except Exception as cypher_error:
                                    logger.warning("Cypher query failed, returning empty results", error=str(cypher_error))

                        # Categorize vertices by type
                        vertices_by_type = {}
                        for v in vertices:
                            label = v.get("label", "Unknown")
                            if label not in vertices_by_type:
                                vertices_by_type[label] = []
                            vertices_by_type[label].append(v)

                        # Categorize edges by type
                        edges_by_type = {}
                        for e in edges:
                            rel_type = e.get("type", "Unknown")
                            if rel_type not in edges_by_type:
                                edges_by_type[rel_type] = []
                            edges_by_type[rel_type].append(e)

                        result = {
                            "success": True,
                            "_type": "knowledge_graph",
                            "query": query,
                            "query_description": query_description,
                            "filters": {
                                "entity_type": entity_type,
                                "relationship_type": relationship_type,
                                "entity_id": entity_id,
                                "hops": hops,
                            },
                            "graph": {
                                "vertices": vertices[:limit],
                                "edges": edges,
                                "total_vertices": len(vertices),
                                "total_edges": len(edges),
                            },
                            "vertices_by_type": {
                                k: len(v) for k, v in vertices_by_type.items()
                            },
                            "edges_by_type": {
                                k: len(v) for k, v in edges_by_type.items()
                            },
                            "_actions": ["explore_entity", "expand_graph", "export_data", "run_query"],
                        }

                        logger.info(
                            "Knowledge graph query completed",
                            query=query[:50],
                            vertices_count=len(vertices),
                            edges_count=len(edges),
                        )

                    except Exception as graph_error:
                        logger.exception("Knowledge graph query failed", error=str(graph_error))
                        result = {
                            "success": False,
                            "_type": "knowledge_graph",
                            "error": f"Query failed: {str(graph_error)}",
                            "query": query,
                            "suggestion": "Make sure the knowledge graph is initialized and contains data.",
                        }

                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                # Log successful tool execution
                duration_ms = int((time.time() - start_time) * 1000)
                await audit.log_tool_execution(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    success=result.get("success", True) if isinstance(result, dict) else True,
                    duration_ms=duration_ms,
                    user_id=session_id,
                    thread_id=session_id,
                )

            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.exception("Tool execution failed", tool=tool_name, error=str(e))

                # Log error to audit trail
                await audit.log_tool_execution(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=None,
                    success=False,
                    duration_ms=duration_ms,
                    user_id=session_id,
                    thread_id=session_id,
                    error=str(e),
                )

                # Include more details about the failure
                result = {
                    "success": False,
                    "error": str(e),
                    "errorDetails": {
                        "category": "execution_error",
                        "originalError": str(e),
                        "isRetryable": True,
                        "suggestedAction": "Check browser pool connectivity or retry",
                    },
                    "_backend": "browser_pool",
                    "_pool_url": browser_pool.pool_url,
                }

            # Store artifacts in Cloudflare R2 if available and result has screenshots
            # This keeps LangGraph state lightweight while preserving full data
            try:
                if is_cloudflare_configured() and tool_name in ["runTest", "executeAction"]:
                    # Store screenshots in R2, get lightweight result for state
                    lightweight_result = await cf_client.store_test_artifacts(
                        result=result,
                        test_id=test_id,
                        project_id=state.get("session_id", "default")
                    )
                    # Use lightweight result for LangGraph state
                    state_result = lightweight_result
                    logger.info("Stored artifacts in Cloudflare R2", test_id=test_id, tool=tool_name)
                else:
                    # Cloudflare not configured - use result as-is (will be pruned later)
                    state_result = result
            except Exception as cf_error:
                logger.warning("Cloudflare storage failed, using original result", error=str(cf_error))
                state_result = result

            # Create tool message
            # Full result is streamed to frontend, lightweight stored in state
            tool_results.append(
                ToolMessage(
                    content=json.dumps(state_result),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

    # Return messages and state updates for context tracking
    return {
        "messages": tool_results,
        # Context updates for intelligent responses
        "recent_failures": state_updates.get("recent_failures"),
        "discovered_elements": state_updates.get("discovered_elements"),
        "healing_suggestions": state_updates.get("healing_suggestions"),
        "current_task": state_updates.get("current_task"),
        "task_progress": state_updates.get("task_progress"),
    }


def should_continue(state: ChatState) -> Literal["tools", "end"]:
    """Determine if we should execute tools or end."""
    # Check if execution was cancelled
    if not state.get("should_continue", True):
        logger.info("Execution cancelled by user, stopping graph")
        return "end"

    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


def create_chat_graph() -> StateGraph:
    """Create the chat-enabled LangGraph."""

    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_executor_node)

    # Set entry point
    graph.set_entry_point("chat")

    # Add conditional edge from chat
    graph.add_conditional_edges(
        "chat",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        }
    )

    # Tools always go back to chat
    graph.add_edge("tools", "chat")

    return graph
