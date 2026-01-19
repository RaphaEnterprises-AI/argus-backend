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


class ChatState(TypedDict):
    """State for chat conversations."""
    messages: Annotated[list[BaseMessage], add_messages]
    app_url: str
    current_tool: str | None
    tool_results: list[dict]
    session_id: str
    should_continue: bool  # Flag to allow cancellation of execution


def create_system_prompt(app_url: str = "") -> str:
    """Create system prompt for Argus chat."""
    return f'''You are Argus, an AI-powered E2E Testing Agent. You help users:

1. Create tests from natural language descriptions
2. Run tests and report results with self-healing capabilities
3. Discover application flows automatically
4. Detect visual regressions
5. Analyze codebases and generate comprehensive test plans

Current Application URL: {app_url or "Not specified"}

You have access to tools for:
- Running browser tests (runTest)
- Executing single actions (executeAction)
- Discovering page elements (discoverElements)
- Creating tests from descriptions (createTest)
- Visual comparison (visualCompare)
- Checking system status (checkStatus)

Be helpful, concise, and proactive. Format test results clearly with pass/fail status.
When showing test steps, use numbered lists. When showing code or selectors, use code blocks.
Report any self-healing that occurred during tests.

IMPORTANT: You are connected to the Argus Orchestrator with full LangGraph capabilities:
- Durable execution (survives crashes)
- Long-term memory (learns from past failures)
- Human-in-the-loop (can pause for approval)
- Multi-agent coordination (specialized agents for different tasks)
'''


async def chat_node(state: ChatState, config) -> dict:
    """Main chat node that processes messages and calls tools."""
    settings = get_settings()

    # Get API key safely
    api_key = settings.anthropic_api_key
    if api_key is None:
        raise ValueError("ANTHROPIC_API_KEY is not configured")
    if hasattr(api_key, 'get_secret_value'):
        api_key = api_key.get_secret_value()

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=api_key,
    )

    # Build messages with system prompt
    system_prompt = create_system_prompt(state.get("app_url", ""))

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
    ]

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Get response
    response = await llm_with_tools.ainvoke(messages)

    return {"messages": [response]}


async def tool_executor_node(state: ChatState, config) -> dict:
    """Execute tools called by the chat node.

    Uses the BrowserPoolClient for all browser operations, which routes to:
    1. BROWSER_POOL_URL (Vultr K8s) - Primary, production-grade
    2. BROWSER_WORKER_URL (Cloudflare) - Legacy fallback
    3. localhost:8080 - Local development

    The client handles JWT authentication, retries, and vision fallback.
    """
    import uuid

    from src.browser.pool_client import BrowserPoolClient, UserContext

    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_results = []

    # Get Cloudflare client for artifact storage
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
                    # Use BrowserPoolClient.observe() for element discovery
                    pool_result = await browser_pool.observe(
                        url=tool_args["url"],
                    )
                    result = pool_result.to_dict()
                    result["_backend"] = "browser_pool"
                    result["_pool_url"] = browser_pool.pool_url

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

    return {"messages": tool_results}


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
