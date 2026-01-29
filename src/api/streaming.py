"""Streaming API endpoints for real-time test execution updates."""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.orchestrator.checkpointer import get_checkpointer
from src.orchestrator.graph import EnhancedTestingOrchestrator, create_enhanced_testing_graph
from src.orchestrator.langfuse_integration import get_langfuse_handler, flush_langfuse
from src.orchestrator.state import create_initial_state

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/stream", tags=["Streaming"])


class StreamTestRequest(BaseModel):
    """Request to start a streaming test run."""
    codebase_path: str = Field(..., description="Path to codebase")
    app_url: str = Field(..., description="Application URL to test")
    thread_id: str | None = Field(None, description="Thread ID for resuming")
    pr_number: int | None = Field(None, description="PR number if triggered by PR")
    changed_files: list[str] | None = Field(None, description="Files that changed")


class StreamChatRequest(BaseModel):
    """Request for streaming chat with orchestrator."""
    message: str = Field(..., description="User message")
    thread_id: str = Field(..., description="Conversation thread ID")
    app_url: str | None = Field(None, description="Application URL context")


@router.post("/test")
async def stream_test_execution(request: StreamTestRequest):
    """
    Stream test execution in real-time using Server-Sent Events.

    Events emitted:
    - start: Initial event with thread_id and metadata
    - state: Full state update after each node
    - progress: Step-by-step progress updates
    - screenshot: Base64 screenshot captures
    - update: Incremental state updates
    - error: Error events
    - complete: Final completion event
    """

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            # Initialize enhanced orchestrator (LangGraph 1.0 patterns)
            orchestrator = EnhancedTestingOrchestrator(
                codebase_path=request.codebase_path,
                app_url=request.app_url,
                pr_number=request.pr_number,
                changed_files=request.changed_files,
            )

            # Create initial state
            initial_state = create_initial_state(
                codebase_path=request.codebase_path,
                app_url=request.app_url,
                pr_number=request.pr_number,
                changed_files=request.changed_files or [],
            )

            thread_id = request.thread_id or initial_state["run_id"]

            # Create Langfuse callback handler for tracing
            langfuse_handler = get_langfuse_handler(
                session_id=thread_id,
                trace_name="streaming_test_execution",
                tags=["streaming", "e2e-test"],
                metadata={
                    "codebase_path": request.codebase_path,
                    "app_url": request.app_url,
                    "pr_number": request.pr_number,
                },
            )

            config = {"configurable": {"thread_id": thread_id}}
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]

            logger.info(
                "Starting streaming test execution",
                thread_id=thread_id,
                codebase_path=request.codebase_path,
                app_url=request.app_url,
                langfuse_enabled=langfuse_handler is not None,
            )

            # Emit start event
            yield {
                "event": "start",
                "data": json.dumps({
                    "thread_id": thread_id,
                    "run_id": initial_state["run_id"],
                    "started_at": datetime.now(UTC).isoformat(),
                    "codebase_path": request.codebase_path,
                    "app_url": request.app_url,
                })
            }

            # Track last state for diffing
            last_iteration = 0
            last_test_index = 0

            # Stream execution with multiple modes
            async for event in orchestrator.app.astream(
                initial_state,
                config,
                stream_mode=["values", "updates", "custom"]
            ):
                # Handle different stream modes
                if isinstance(event, tuple):
                    event_type, event_data = event
                else:
                    # Single mode response - determine type from content
                    event_data = event
                    if "iteration" in event or "test_results" in event:
                        event_type = "values"
                    elif isinstance(event, dict) and len(event) <= 3:
                        event_type = "updates"
                    else:
                        event_type = "values"

                if event_type == "values":
                    # Full state update
                    state = event_data if isinstance(event_data, dict) else event

                    # Emit state summary (avoid sending full state to reduce bandwidth)
                    yield {
                        "event": "state",
                        "data": json.dumps({
                            "iteration": state.get("iteration", 0),
                            "current_test_index": state.get("current_test_index", 0),
                            "total_tests": len(state.get("test_plan", [])),
                            "passed_count": state.get("passed_count", 0),
                            "failed_count": state.get("failed_count", 0),
                            "skipped_count": state.get("skipped_count", 0),
                            "total_cost": state.get("total_cost", 0),
                            "next_agent": state.get("next_agent", ""),
                            "should_continue": state.get("should_continue", True),
                            "error": state.get("error"),
                        })
                    }

                    # Emit progress if iteration changed
                    current_iteration = state.get("iteration", 0)
                    if current_iteration > last_iteration:
                        yield {
                            "event": "progress",
                            "data": json.dumps({
                                "step": current_iteration,
                                "phase": state.get("next_agent", "unknown"),
                                "message": f"Iteration {current_iteration}: {state.get('next_agent', 'processing')}",
                            })
                        }
                        last_iteration = current_iteration

                    # Emit test progress if test index changed
                    current_test_index = state.get("current_test_index", 0)
                    if current_test_index > last_test_index:
                        current_test = state.get("current_test")
                        test_name = current_test.get("name", f"Test {current_test_index}") if current_test else f"Test {current_test_index}"
                        yield {
                            "event": "test_progress",
                            "data": json.dumps({
                                "test_index": current_test_index,
                                "total_tests": len(state.get("test_plan", [])),
                                "test_name": test_name,
                                "passed": state.get("passed_count", 0),
                                "failed": state.get("failed_count", 0),
                            })
                        }
                        last_test_index = current_test_index

                    # Emit screenshots if present (only new ones)
                    screenshots = state.get("screenshots", [])
                    if screenshots:
                        yield {
                            "event": "screenshot",
                            "data": json.dumps({
                                "screenshot": screenshots[-1],
                                "step_index": state.get("current_test_index", 0),
                                "total_screenshots": len(screenshots),
                            })
                        }

                    # Emit latest test result if any
                    test_results = state.get("test_results", [])
                    if test_results:
                        latest_result = test_results[-1]
                        yield {
                            "event": "test_result",
                            "data": json.dumps({
                                "test_id": latest_result.get("test_id", ""),
                                "status": latest_result.get("status", "unknown"),
                                "duration_seconds": latest_result.get("duration_seconds", 0),
                                "error_message": latest_result.get("error_message"),
                                "assertions_passed": latest_result.get("assertions_passed", 0),
                                "assertions_failed": latest_result.get("assertions_failed", 0),
                            })
                        }

                elif event_type == "updates":
                    # Incremental state updates
                    yield {
                        "event": "update",
                        "data": json.dumps(event_data if isinstance(event_data, dict) else event)
                    }

                elif event_type == "custom":
                    # Custom events from nodes (progress, tool calls, etc)
                    custom_data = event_data if isinstance(event_data, dict) else event
                    custom_event_type = custom_data.get("type", "progress")
                    yield {
                        "event": custom_event_type,
                        "data": json.dumps(custom_data)
                    }

            # Get final state
            final_state = await orchestrator.app.aget_state(config)
            final_values = final_state.values if hasattr(final_state, 'values') else final_state

            # Emit completion
            yield {
                "event": "complete",
                "data": json.dumps({
                    "thread_id": thread_id,
                    "run_id": initial_state["run_id"],
                    "success": final_values.get("failed_count", 0) == 0 and not final_values.get("error"),
                    "passed": final_values.get("passed_count", 0),
                    "failed": final_values.get("failed_count", 0),
                    "skipped": final_values.get("skipped_count", 0),
                    "total_tests": len(final_values.get("test_plan", [])),
                    "total_cost": final_values.get("total_cost", 0),
                    "iterations": final_values.get("iteration", 0),
                    "error": final_values.get("error"),
                    "completed_at": datetime.now(UTC).isoformat(),
                })
            }

            logger.info(
                "Streaming test execution completed",
                thread_id=thread_id,
                passed=final_values.get("passed_count", 0),
                failed=final_values.get("failed_count", 0),
            )

        except Exception as e:
            logger.exception("Streaming error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now(UTC).isoformat(),
                })
            }

    return EventSourceResponse(event_generator())


@router.post("/chat")
async def stream_chat(request: StreamChatRequest):
    """
    Stream chat responses from the orchestrator.

    Routes messages through LangGraph for full orchestration.
    Emits token-by-token responses for real-time UI updates.
    """

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            from langchain_core.messages import HumanMessage


            logger.info(
                "Starting streaming chat",
                thread_id=request.thread_id,
                message_length=len(request.message),
            )

            # Get checkpointer for conversation persistence
            checkpointer = get_checkpointer()

            config = {"configurable": {"thread_id": request.thread_id}}

            # Emit acknowledgment
            yield {
                "event": "ack",
                "data": json.dumps({
                    "thread_id": request.thread_id,
                    "message_received": True,
                    "timestamp": datetime.now(UTC).isoformat(),
                })
            }

            # For chat, we need a simpler graph that handles conversation
            # This is a placeholder - in production, import a dedicated chat graph
            try:
                from src.orchestrator.chat_graph import create_chat_graph
                chat_graph = create_chat_graph()
                app = chat_graph.compile(checkpointer=checkpointer)
            except ImportError:
                # Fallback: Create a simple response
                logger.warning("Chat graph not found, using fallback response")

                # Simulate streaming response
                response_text = f"Received your message about: {request.message[:100]}. The chat functionality is being initialized. Please try again in a moment."

                # Stream tokens
                words = response_text.split()
                for i, word in enumerate(words):
                    yield {
                        "event": "token",
                        "data": json.dumps({
                            "content": word + " ",
                            "index": i,
                        })
                    }
                    await asyncio.sleep(0.05)  # Simulate typing delay

                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "content": response_text,
                        "thread_id": request.thread_id,
                        "is_fallback": True,
                    })
                }
                return

            # Get current conversation state or create new
            try:
                current_state = await app.aget_state(config)
                messages = current_state.values.get("messages", []) if current_state.values else []
            except Exception:
                messages = []

            # Add user message
            messages.append(HumanMessage(content=request.message))

            input_state = {
                "messages": messages,
                "app_url": request.app_url or "",
            }

            # Stream response with message mode for token streaming
            accumulated_content = ""

            async for event in app.astream(
                input_state,
                config,
                stream_mode=["messages", "values", "custom"]
            ):
                if isinstance(event, tuple):
                    event_type, event_data = event
                else:
                    event_data = event
                    event_type = "values"

                if event_type == "messages":
                    # LLM token streaming
                    chunk, metadata = event_data if isinstance(event_data, tuple) else (event_data, {})
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        accumulated_content += content
                        yield {
                            "event": "token",
                            "data": json.dumps({
                                "content": content,
                                "node": metadata.get("langgraph_node", "") if metadata else "",
                            })
                        }

                elif event_type == "custom":
                    # Tool calls, progress updates
                    custom_data = event_data if isinstance(event_data, dict) else {}
                    yield {
                        "event": custom_data.get("type", "update"),
                        "data": json.dumps(custom_data)
                    }

            # Get final response
            final_state = await app.aget_state(config)
            final_messages = final_state.values.get("messages", []) if final_state.values else []

            if final_messages:
                last_message = final_messages[-1]
                final_content = last_message.content if hasattr(last_message, "content") else str(last_message)
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "content": final_content,
                        "thread_id": request.thread_id,
                        "message_count": len(final_messages),
                    })
                }
            else:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "content": accumulated_content or "No response generated",
                        "thread_id": request.thread_id,
                    })
                }

        except Exception as e:
            logger.exception("Chat streaming error", error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "thread_id": request.thread_id,
                })
            }

    return EventSourceResponse(event_generator())


@router.get("/status/{thread_id}")
async def get_stream_status(thread_id: str):
    """
    Get the current status of a streaming execution.

    Returns the latest state snapshot for the given thread.
    """
    from src.config import get_settings

    checkpointer = get_checkpointer()

    try:
        config = {"configurable": {"thread_id": thread_id}}

        settings = get_settings()
        graph = create_enhanced_testing_graph(settings)
        app = graph.compile(checkpointer=checkpointer)

        state = await app.aget_state(config)

        if state and state.values:
            values = state.values
            total_tests = len(values.get("test_plan", []))
            current_test = values.get("current_test_index", 0)

            # Determine status
            if values.get("error"):
                status = "error"
            elif state.next:
                status = "running"
            elif values.get("passed_count", 0) + values.get("failed_count", 0) + values.get("skipped_count", 0) > 0:
                status = "completed"
            else:
                status = "pending"

            return {
                "found": True,
                "thread_id": thread_id,
                "run_id": values.get("run_id"),
                "status": status,
                "next_node": state.next[0] if state.next else None,
                "started_at": values.get("started_at"),
                "state_summary": {
                    "iteration": values.get("iteration", 0),
                    "passed": values.get("passed_count", 0),
                    "failed": values.get("failed_count", 0),
                    "skipped": values.get("skipped_count", 0),
                    "current_test": current_test,
                    "total_tests": total_tests,
                    "progress_percent": (current_test / total_tests * 100) if total_tests > 0 else 0,
                    "total_cost": values.get("total_cost", 0),
                    "error": values.get("error"),
                },
                "current_phase": values.get("next_agent", "unknown"),
            }
        else:
            return {
                "found": False,
                "thread_id": thread_id,
                "message": "No state found for this thread",
            }

    except Exception as e:
        logger.exception("Error getting stream status", thread_id=thread_id, error=str(e))
        return {
            "found": False,
            "thread_id": thread_id,
            "error": str(e),
        }


@router.post("/resume/{thread_id}")
async def resume_stream(thread_id: str):
    """
    Resume a paused or interrupted streaming execution.

    Continues from the last checkpoint for the given thread.
    """
    from src.config import get_settings

    async def event_generator() -> AsyncGenerator[dict, None]:
        try:
            checkpointer = get_checkpointer()
            settings = get_settings()

            config = {"configurable": {"thread_id": thread_id}}

            graph = create_enhanced_testing_graph(settings)
            app = graph.compile(checkpointer=checkpointer)

            # Get current state
            current_state = await app.aget_state(config)

            if not current_state or not current_state.values:
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": "No state found for this thread",
                        "thread_id": thread_id,
                    })
                }
                return

            # Check if already completed
            if not current_state.next:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "message": "Execution already completed",
                        "thread_id": thread_id,
                        "passed": current_state.values.get("passed_count", 0),
                        "failed": current_state.values.get("failed_count", 0),
                    })
                }
                return

            yield {
                "event": "resume",
                "data": json.dumps({
                    "thread_id": thread_id,
                    "resuming_from": current_state.next[0] if current_state.next else "unknown",
                    "timestamp": datetime.now(UTC).isoformat(),
                })
            }

            # Continue execution from checkpoint
            async for event in app.astream(
                None,  # Use None to continue from checkpoint
                config,
                stream_mode=["values", "updates", "custom"]
            ):
                if isinstance(event, tuple):
                    event_type, event_data = event
                else:
                    event_data = event
                    event_type = "values"

                if event_type == "values":
                    state = event_data if isinstance(event_data, dict) else event
                    yield {
                        "event": "state",
                        "data": json.dumps({
                            "iteration": state.get("iteration", 0),
                            "current_test_index": state.get("current_test_index", 0),
                            "passed_count": state.get("passed_count", 0),
                            "failed_count": state.get("failed_count", 0),
                            "next_agent": state.get("next_agent", ""),
                        })
                    }
                elif event_type == "updates":
                    yield {
                        "event": "update",
                        "data": json.dumps(event_data if isinstance(event_data, dict) else event)
                    }

            # Get final state
            final_state = await app.aget_state(config)
            final_values = final_state.values if hasattr(final_state, 'values') else final_state

            yield {
                "event": "complete",
                "data": json.dumps({
                    "thread_id": thread_id,
                    "success": final_values.get("failed_count", 0) == 0,
                    "passed": final_values.get("passed_count", 0),
                    "failed": final_values.get("failed_count", 0),
                    "completed_at": datetime.now(UTC).isoformat(),
                })
            }

        except Exception as e:
            logger.exception("Resume streaming error", thread_id=thread_id, error=str(e))
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "thread_id": thread_id,
                })
            }

    return EventSourceResponse(event_generator())


@router.delete("/cancel/{thread_id}")
async def cancel_stream(thread_id: str):
    """
    Cancel an ongoing streaming execution.

    Note: This marks the execution as cancelled but may not immediately
    stop in-progress operations.
    """
    from src.config import get_settings

    try:
        checkpointer = get_checkpointer()
        settings = get_settings()

        config = {"configurable": {"thread_id": thread_id}}

        graph = create_enhanced_testing_graph(settings)
        app = graph.compile(checkpointer=checkpointer)

        # Get current state
        current_state = await app.aget_state(config)

        if not current_state or not current_state.values:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Update state to stop execution
        updated_values = dict(current_state.values)
        updated_values["should_continue"] = False
        updated_values["error"] = "Cancelled by user"

        # Update the state
        await app.aupdate_state(config, updated_values)

        logger.info("Stream cancelled", thread_id=thread_id)

        return {
            "success": True,
            "thread_id": thread_id,
            "message": "Execution cancelled",
            "cancelled_at": datetime.now(UTC).isoformat(),
            "final_state": {
                "passed": updated_values.get("passed_count", 0),
                "failed": updated_values.get("failed_count", 0),
                "iteration": updated_values.get("iteration", 0),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error cancelling stream", thread_id=thread_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
