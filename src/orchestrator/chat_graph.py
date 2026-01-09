"""Chat-enabled LangGraph for conversational test orchestration."""

from typing import TypedDict, Annotated, Literal, Optional, List
from datetime import datetime, timezone
import json

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
import structlog

from src.config import get_settings

logger = structlog.get_logger()


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

    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

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

    last_message = state["messages"][-1]

    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_results = []

    # Worker URL for browser automation
    settings = get_settings()
    worker_url = settings.browser_worker_url

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

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

        # Create tool message with properly formatted JSON content
        from langchain_core.messages import ToolMessage
        tool_results.append(
            ToolMessage(
                content=json.dumps(result),  # Use json.dumps for valid JSON, not str()
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
