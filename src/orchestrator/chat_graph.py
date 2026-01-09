"""Chat-enabled LangGraph for conversational test orchestration."""

from typing import TypedDict, Annotated, Literal, Optional, List
from datetime import datetime, timezone
import json
import re

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
import structlog

from src.config import get_settings

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


def prune_messages_for_context(messages: List[BaseMessage], max_tokens: int = MAX_CONTEXT_TOKENS) -> List[BaseMessage]:
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
    messages: Annotated[List[BaseMessage], add_messages]
    app_url: str
    current_tool: Optional[str]
    tool_results: List[dict]
    session_id: str


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
            "description": "Create a test from natural language description",
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
    """Execute tools called by the chat node."""
    import httpx
    import uuid
    from src.services.cloudflare_storage import get_cloudflare_client, is_cloudflare_configured

    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_results = []

    # Worker URL for browser automation
    settings = get_settings()
    worker_url = settings.browser_worker_url

    # Get Cloudflare client for artifact storage
    cf_client = get_cloudflare_client()

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        test_id = str(uuid.uuid4())[:8]

        try:
            if tool_name == "runTest":
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(
                        f"{worker_url}/test",
                        json={
                            "url": tool_args["url"],
                            "steps": tool_args["steps"],
                            "screenshot": True,
                        }
                    )
                    result = response.json()

            elif tool_name == "executeAction":
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{worker_url}/act",
                        json={
                            "url": tool_args["url"],
                            "instruction": tool_args["instruction"],
                            "screenshot": True,
                        }
                    )
                    result = response.json()

            elif tool_name == "discoverElements":
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{worker_url}/observe",
                        json={"url": tool_args["url"]}
                    )
                    result = response.json()

            elif tool_name == "createTest":
                from src.agents.nlp_test_creator import NLPTestCreator
                creator = NLPTestCreator(app_url=tool_args["app_url"])
                test = await creator.create(tool_args["description"])
                result = {"success": True, "test": test.to_dict()}

            elif tool_name == "checkStatus":
                result = {
                    "success": True,
                    "components": [
                        {"name": "Orchestrator", "status": "connected"},
                        {"name": "Browser Worker", "status": "connected"},
                        {"name": "Memory Store", "status": "connected"},
                    ]
                }
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.exception("Tool execution failed", tool=tool_name, error=str(e))
            result = {"error": str(e)}

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
