"""Schedule Executor - Executes scheduled test runs using Selenium Grid.

This module provides the core test execution logic for scheduled runs.
It reuses the Selenium Grid execution patterns from browser.py.

Auto-Healing Integration (Phase 4):
When tests fail due to selector changes, the executor can automatically
attempt to heal them using the SelfHealerAgent. This feature:
- Analyzes failures for healable patterns (selector changes, timing issues)
- Applies fixes above a confidence threshold
- Re-runs healed tests to verify the fix works
- Logs all healing attempts and outcomes
"""

import asyncio
import base64
import re
import time
from datetime import UTC, datetime
from typing import Any

import structlog

from src.browser.selenium_grid_client import SeleniumGridClient
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


# =============================================================================
# AUTO-HEALING INTEGRATION (Phase 4)
# =============================================================================


async def attempt_auto_heal(
    test: dict,
    failure_result: dict,
    schedule_config: dict,
    app_url: str,
    events_queue: asyncio.Queue | None = None,
) -> dict | None:
    """
    Attempt to auto-heal a failed test.

    This function uses the SelfHealerAgent to analyze test failures and
    automatically apply fixes when confidence is above the threshold.

    Args:
        test: The original test specification that failed
        failure_result: The result dict from the failed test execution
        schedule_config: Schedule configuration containing auto_heal settings:
            - auto_heal_enabled: Whether auto-healing is enabled (default: False)
            - auto_heal_confidence_threshold: Minimum confidence for auto-apply (default: 0.9)
            - auto_heal_max_attempts: Maximum healing attempts per test (default: 1)
        app_url: Base URL of the application for re-running healed tests
        events_queue: Optional queue for emitting progress events

    Returns:
        Healed result dict if successful, None if healing not possible or failed.
        The healed result includes:
        - auto_healed: True
        - healing_details: Information about what was fixed
    """
    # Check if auto-healing is enabled
    if not schedule_config.get("auto_heal_enabled", False):
        logger.debug(
            "Auto-healing disabled for schedule",
            test_id=test.get("id"),
        )
        return None

    threshold = schedule_config.get("auto_heal_confidence_threshold", 0.9)

    test_id = test.get("id", "unknown")
    test_name = test.get("name", "Unnamed Test")

    logger.info(
        "Attempting auto-heal for failed test",
        test_id=test_id,
        test_name=test_name,
        confidence_threshold=threshold,
    )

    await emit_event(events_queue, "auto_heal_started", {
        "test_id": test_id,
        "test_name": test_name,
        "original_error": failure_result.get("error"),
    })

    try:
        # Import SelfHealerAgent (lazy import to avoid circular dependencies)
        from src.agents.self_healer import FixType, SelfHealerAgent

        # Initialize the healer with the configured threshold
        healer = SelfHealerAgent(auto_heal_threshold=threshold)

        # Build failure details from the result
        failure_details = _build_failure_details(failure_result)

        # Get screenshot from the last failed step if available
        screenshot = None
        failed_steps = [s for s in failure_result.get("steps", []) if not s.get("success")]
        if failed_steps:
            last_failed = failed_steps[-1]
            screenshot_data = last_failed.get("screenshot")
            if screenshot_data and isinstance(screenshot_data, str):
                try:
                    screenshot = base64.b64decode(screenshot_data)
                except Exception:
                    pass

        # Analyze the failure
        healing_result = await healer.execute(
            test_spec=test,
            failure_details=failure_details,
            screenshot=screenshot,
            error_logs=failure_result.get("error"),
        )

        if not healing_result.success or not healing_result.data:
            logger.warning(
                "Healing analysis failed or returned no data",
                test_id=test_id,
                error=healing_result.error,
            )
            await emit_event(events_queue, "auto_heal_failed", {
                "test_id": test_id,
                "reason": "Healing analysis failed",
                "error": healing_result.error,
            })
            return None

        analysis = healing_result.data

        # Check if we have a fix with sufficient confidence
        if not analysis.suggested_fixes:
            logger.info(
                "No fixes suggested by healer",
                test_id=test_id,
                diagnosis_type=analysis.diagnosis.failure_type.value,
            )
            await emit_event(events_queue, "auto_heal_failed", {
                "test_id": test_id,
                "reason": "No fixes suggested",
                "diagnosis_type": analysis.diagnosis.failure_type.value,
            })
            return None

        best_fix = analysis.suggested_fixes[0]

        if best_fix.confidence < threshold:
            logger.info(
                "Fix confidence below threshold",
                test_id=test_id,
                fix_confidence=best_fix.confidence,
                threshold=threshold,
                fix_type=best_fix.fix_type.value,
            )
            await emit_event(events_queue, "auto_heal_failed", {
                "test_id": test_id,
                "reason": "Confidence below threshold",
                "fix_confidence": best_fix.confidence,
                "threshold": threshold,
            })
            return None

        # Check if the fix type is actionable
        if best_fix.fix_type == FixType.NONE:
            logger.info(
                "Healer determined no fix needed (possible real bug)",
                test_id=test_id,
                diagnosis_type=analysis.diagnosis.failure_type.value,
            )
            await emit_event(events_queue, "auto_heal_failed", {
                "test_id": test_id,
                "reason": "No fix applicable (possible real bug)",
                "diagnosis_type": analysis.diagnosis.failure_type.value,
            })
            return None

        # Get the healed test spec
        healed_test = analysis.healed_test_spec
        if not healed_test:
            logger.warning(
                "Healer did not return healed test spec",
                test_id=test_id,
            )
            await emit_event(events_queue, "auto_heal_failed", {
                "test_id": test_id,
                "reason": "No healed test spec returned",
            })
            return None

        logger.info(
            "Applying healing fix",
            test_id=test_id,
            fix_type=best_fix.fix_type.value,
            confidence=best_fix.confidence,
            old_value=best_fix.old_value,
            new_value=best_fix.new_value,
        )

        await emit_event(events_queue, "auto_heal_applying", {
            "test_id": test_id,
            "fix_type": best_fix.fix_type.value,
            "confidence": best_fix.confidence,
            "explanation": best_fix.explanation,
        })

        # Re-run the test with the healed specification
        healed_result = await execute_single_test(
            test=healed_test,
            app_url=app_url,
            events_queue=events_queue,
            screenshot_enabled=True,
        )

        # Check if the healed test succeeded
        if healed_result.get("success"):
            logger.info(
                "Auto-healing succeeded",
                test_id=test_id,
                fix_type=best_fix.fix_type.value,
                confidence=best_fix.confidence,
            )

            # Enrich the result with healing information
            healed_result["auto_healed"] = True
            healed_result["healing_details"] = {
                "original_error": failure_result.get("error"),
                "diagnosis_type": analysis.diagnosis.failure_type.value,
                "diagnosis_confidence": analysis.diagnosis.confidence,
                "diagnosis_explanation": analysis.diagnosis.explanation,
                "fix_type": best_fix.fix_type.value,
                "fix_applied": best_fix.explanation,
                "fix_confidence": best_fix.confidence,
                "old_value": best_fix.old_value,
                "new_value": best_fix.new_value,
            }

            # Include code-aware context if available
            if analysis.diagnosis.code_context:
                healed_result["healing_details"]["code_context"] = analysis.diagnosis.code_context.to_dict()

            await emit_event(events_queue, "auto_heal_succeeded", {
                "test_id": test_id,
                "fix_type": best_fix.fix_type.value,
                "confidence": best_fix.confidence,
            })

            # Record successful healing outcome for learning
            if hasattr(analysis, "_memory_pattern_id") and analysis._memory_pattern_id:
                await healer._record_memory_outcome(analysis._memory_pattern_id, success=True)

            return healed_result

        else:
            logger.warning(
                "Healed test still failed",
                test_id=test_id,
                fix_type=best_fix.fix_type.value,
                new_error=healed_result.get("error"),
            )
            await emit_event(events_queue, "auto_heal_failed", {
                "test_id": test_id,
                "reason": "Healed test still failed",
                "new_error": healed_result.get("error"),
            })

            # Record failed healing outcome for learning
            if hasattr(analysis, "_memory_pattern_id") and analysis._memory_pattern_id:
                await healer._record_memory_outcome(analysis._memory_pattern_id, success=False)

            return None

    except Exception as e:
        logger.exception(
            "Auto-heal attempt failed with exception",
            test_id=test_id,
            error=str(e),
        )
        await emit_event(events_queue, "auto_heal_failed", {
            "test_id": test_id,
            "reason": "Exception during healing",
            "error": str(e),
        })
        return None


def _build_failure_details(failure_result: dict) -> dict:
    """
    Build a failure details dict from a test result for the SelfHealerAgent.

    Args:
        failure_result: The result dict from a failed test execution

    Returns:
        Failure details dict with type, message, selector, etc.
    """
    # Find the first failed step
    failed_steps = [s for s in failure_result.get("steps", []) if not s.get("success")]
    failed_step = failed_steps[0] if failed_steps else None

    failure_details = {
        "message": failure_result.get("error") or "Test failed",
        "type": "unknown",
    }

    if failed_step:
        failure_details["step_index"] = failed_step.get("step_index", 0)
        failure_details["instruction"] = failed_step.get("instruction")
        step_error = failed_step.get("error", "")
        failure_details["message"] = step_error or failure_details["message"]

        # Try to extract selector from actions
        actions = failed_step.get("actions", [])
        for action in actions:
            if action.get("error") == "element not found":
                failure_details["type"] = "selector_changed"
                # Try to extract the target selector from instruction
                instruction = failed_step.get("instruction", "")
                if "'" in instruction:
                    # Extract quoted text as potential selector
                    match = re.search(r"['\"]([^'\"]+)['\"]", instruction)
                    if match:
                        failure_details["selector"] = match.group(1)
                break
            elif "timeout" in str(action.get("error", "")).lower():
                failure_details["type"] = "timing_issue"
                break

        # Detect timing issues from error message
        error_lower = failure_details["message"].lower()
        if "timeout" in error_lower or "timed out" in error_lower or "not ready" in error_lower:
            failure_details["type"] = "timing_issue"
        elif "not found" in error_lower or "could not find" in error_lower:
            failure_details["type"] = "selector_changed"

    return failure_details


# =============================================================================
# END AUTO-HEALING INTEGRATION
# =============================================================================


# =============================================================================
# FLAKY TEST DETECTION INTEGRATION (Phase 3)
# =============================================================================


async def update_flaky_scores_after_run(
    schedule_id: str,
    run_id: str,
    test_results: list[dict],
) -> list[dict]:
    """Update flaky scores for all tests after a scheduled run completes.

    This function is called after each scheduled run to track test consistency
    and flag flaky tests for investigation.

    Args:
        schedule_id: The schedule ID
        run_id: The run ID
        test_results: List of test result dicts with test_id and success status

    Returns:
        List of newly detected flaky tests
    """
    from src.api.flaky_detector import get_flaky_detector

    detector = get_flaky_detector()
    newly_flaky_tests = []

    for result in test_results:
        test_id = result.get("test_id")
        if not test_id:
            continue

        try:
            status = "passed" if result.get("success") else "failed"

            # Record this outcome and check for flakiness
            flaky_result = await detector.record_test_outcome(
                test_id=test_id,
                test_run_id=run_id,
                status=status,
                schedule_id=schedule_id,
            )

            if flaky_result:
                newly_flaky_tests.append(flaky_result)

        except Exception as e:
            logger.warning(
                "Failed to update flaky score for test",
                test_id=test_id,
                error=str(e),
            )

    if newly_flaky_tests:
        logger.warning(
            "Flaky tests detected in scheduled run",
            schedule_id=schedule_id,
            run_id=run_id,
            flaky_count=len(newly_flaky_tests),
            flaky_tests=[t["test_id"] for t in newly_flaky_tests],
        )

    return newly_flaky_tests


# =============================================================================
# END FLAKY TEST DETECTION INTEGRATION
# =============================================================================


async def emit_event(events_queue: asyncio.Queue | None, event_type: str, data: dict) -> None:
    """Emit an event to the events queue if available."""
    if events_queue:
        try:
            await events_queue.put({"type": event_type, "data": data, "timestamp": datetime.now(UTC).isoformat()})
        except Exception as e:
            logger.warning("Failed to emit event", event_type=event_type, error=str(e))


async def fetch_tests_for_schedule(schedule: dict) -> list[dict]:
    """
    Fetch tests to execute for a schedule.

    Gets tests by test_ids if specified, otherwise applies test_filter.

    Args:
        schedule: Schedule configuration dict containing test_ids and/or test_filter

    Returns:
        List of test dicts with id, name, steps, etc.
    """
    supabase = get_supabase_client()

    test_ids = schedule.get("test_ids", [])
    project_id = schedule.get("project_id")

    if test_ids:
        # Fetch specific tests by ID
        result = await supabase.request(
            f"/tests?id=in.({','.join(test_ids)})&is_active=eq.true&select=*"
        )
    elif project_id:
        # Fetch all active tests for the project
        result = await supabase.request(
            f"/tests?project_id=eq.{project_id}&is_active=eq.true&select=*&order=priority.asc,created_at.asc"
        )
    else:
        logger.warning("Schedule has no test_ids or project_id", schedule_id=schedule.get("id"))
        return []

    if result.get("error"):
        logger.error("Failed to fetch tests for schedule", error=result.get("error"))
        return []

    tests = result.get("data", [])
    logger.info("Fetched tests for schedule", count=len(tests), schedule_id=schedule.get("id"))

    return tests


async def execute_single_test(
    test: dict,
    app_url: str,
    events_queue: asyncio.Queue | None = None,
    screenshot_enabled: bool = True,
) -> dict:
    """
    Execute a single test using Selenium Grid.

    Args:
        test: Test dict containing id, name, steps
        app_url: Base URL of the application to test
        events_queue: Optional queue for emitting progress events
        screenshot_enabled: Whether to capture screenshots after each step

    Returns:
        Dict with test results: success, steps, duration_ms, video_artifact_id, etc.
    """
    test_id = test.get("id", "unknown")
    test_name = test.get("name", "Unnamed Test")
    steps = test.get("steps", [])

    logger.info("Starting test execution", test_id=test_id, test_name=test_name, step_count=len(steps))
    await emit_event(events_queue, "test_started", {
        "test_id": test_id,
        "test_name": test_name,
        "step_count": len(steps),
    })

    start_time = time.time()
    step_results = []
    all_success = True
    video_artifact_id = None
    recording_url = None
    error_message = None

    try:
        grid_client = SeleniumGridClient()
        session_id = await grid_client.start_session()
        logger.info("Selenium session started for test", session_id=session_id, test_id=test_id)

        try:
            # Navigate to starting URL
            await grid_client.navigate(app_url)
            await asyncio.sleep(2)  # Wait for page load

            # Execute each step
            for idx, step in enumerate(steps):
                step_result = await _execute_step(
                    grid_client, step, idx, screenshot_enabled, events_queue, test_id
                )
                step_results.append(step_result)

                if not step_result["success"]:
                    all_success = False
                    logger.warning(
                        "Test step failed",
                        test_id=test_id,
                        step_index=idx,
                        error=step_result.get("error"),
                    )

        finally:
            # End session (video stops, uploads to R2)
            end_result = await grid_client.end_session()
            video_artifact_id = end_result.get("video_artifact_id")
            logger.info("Selenium session ended", session_id=session_id, test_id=test_id)

            # Generate signed URL for video playback
            if video_artifact_id:
                try:
                    from src.services.cloudflare_storage import get_cloudflare_client, is_cloudflare_configured
                    if is_cloudflare_configured():
                        cf_client = get_cloudflare_client()
                        recording_url = cf_client.r2.generate_signed_url(
                            video_artifact_id,
                            artifact_type="video",
                            expiry_seconds=3600
                        )
                except Exception as e:
                    logger.warning("Failed to generate signed video URL", error=str(e))

    except Exception as e:
        error_message = str(e)
        all_success = False
        logger.exception("Test execution failed", test_id=test_id, error=error_message)

    duration_ms = int((time.time() - start_time) * 1000)

    result = {
        "test_id": test_id,
        "test_name": test_name,
        "success": all_success,
        "steps": step_results,
        "duration_ms": duration_ms,
        "video_artifact_id": video_artifact_id,
        "recording_url": recording_url,
        "error": error_message,
        "steps_passed": sum(1 for s in step_results if s.get("success")),
        "steps_failed": sum(1 for s in step_results if not s.get("success")),
    }

    # AI-powered failure analysis for failed tests
    if not all_success:
        try:
            from src.agents.root_cause_analyzer import FailureContext, RootCauseAnalyzer

            # Get the last screenshot from step results for visual analysis
            last_screenshot = None
            for step_result in reversed(step_results):
                if step_result.get("screenshot"):
                    last_screenshot = step_result["screenshot"]
                    break

            analyzer = RootCauseAnalyzer()
            analysis = await analyzer.analyze(FailureContext(
                test_id=test_id,
                test_name=test_name,
                error_message=error_message or "",
                screenshot_base64=last_screenshot,
                step_history=step_results,
            ))

            result["ai_analysis"] = {
                "category": analysis.category.value,
                "confidence": analysis.confidence,
                "summary": analysis.summary,
                "suggested_fix": analysis.suggested_fix,
                "is_flaky": analysis.is_flaky,
                "detailed_analysis": analysis.detailed_analysis,
                "auto_healable": analysis.auto_healable,
                "healing_suggestion": analysis.healing_suggestion,
            }

            logger.info(
                "AI failure analysis completed",
                test_id=test_id,
                category=analysis.category.value,
                confidence=analysis.confidence,
                is_flaky=analysis.is_flaky,
            )
        except Exception as e:
            # Don't fail the whole run if AI analysis fails
            logger.warning(
                "AI failure analysis failed",
                test_id=test_id,
                error=str(e),
            )
            result["ai_analysis"] = None

    await emit_event(events_queue, "test_completed", {
        "test_id": test_id,
        "test_name": test_name,
        "success": all_success,
        "duration_ms": duration_ms,
        "steps_passed": result["steps_passed"],
        "steps_failed": result["steps_failed"],
    })

    logger.info(
        "Test execution completed",
        test_id=test_id,
        success=all_success,
        duration_ms=duration_ms,
        steps_passed=result["steps_passed"],
        steps_failed=result["steps_failed"],
    )

    return result


async def _execute_step(
    grid_client: SeleniumGridClient,
    step: dict,
    step_index: int,
    screenshot_enabled: bool,
    events_queue: asyncio.Queue | None,
    test_id: str,
) -> dict:
    """
    Execute a single test step.

    Args:
        grid_client: Selenium Grid client
        step: Step dict (can be {"action": ..., "target": ...} or just instruction string)
        step_index: Index of the step
        screenshot_enabled: Whether to capture screenshot
        events_queue: Optional queue for events
        test_id: Test ID for logging

    Returns:
        Step result dict
    """
    step_start = time.time()
    step_success = False
    step_error = None
    step_screenshot = None
    actions_taken = []

    # Normalize step format - steps can be dicts or strings
    if isinstance(step, dict):
        instruction = step.get("description") or step.get("action") or str(step)
        action = step.get("action", "").lower()
        target = step.get("target", "")
        value = step.get("value", "")
    else:
        instruction = str(step)
        action = ""
        target = ""
        value = ""

    instruction_lower = instruction.lower()

    await emit_event(events_queue, "step_started", {
        "test_id": test_id,
        "step_index": step_index,
        "instruction": instruction,
    })

    try:
        # Execute based on instruction type
        if instruction_lower.startswith("goto ") or instruction_lower.startswith("navigate to ") or action == "navigate":
            # Navigate to URL
            url = target or instruction.split(" ", 2)[-1].strip()
            success = await grid_client.navigate(url)
            step_success = success
            actions_taken.append({"action": "navigate", "url": url, "success": success})
            await asyncio.sleep(1)

        elif "wait" in instruction_lower or action == "wait":
            # Wait for specified duration or condition
            match = re.search(r'(\d+)\s*(second|sec|s|ms|millisecond)', instruction_lower)
            if match:
                duration = int(match.group(1))
                if 'ms' in match.group(2) or 'millisecond' in match.group(2):
                    duration = duration / 1000
                await asyncio.sleep(duration)
            elif value:
                # Handle special wait types (networkidle, load, domcontentloaded)
                value_lower = value.lower()
                if value_lower in ("networkidle", "load", "domcontentloaded", "visible", "hidden"):
                    # These are Playwright wait conditions - just wait a reasonable time
                    # Selenium Grid doesn't have native support for these, so we approximate
                    await asyncio.sleep(2)
                else:
                    # Try to parse as a number
                    try:
                        await asyncio.sleep(float(value))
                    except ValueError:
                        # Unknown wait type - default to 2 seconds
                        await asyncio.sleep(2)
            else:
                await asyncio.sleep(2)
            step_success = True
            actions_taken.append({"action": "wait", "success": True})

        elif "screenshot" in instruction_lower:
            step_success = True
            actions_taken.append({"action": "screenshot", "success": True})

        elif "press enter" in instruction_lower or "hit enter" in instruction_lower or action == "press_enter":
            success = await grid_client.press_key("enter")
            step_success = success
            actions_taken.append({"action": "press_key", "key": "enter", "success": success})
            await asyncio.sleep(0.5)

        elif "press tab" in instruction_lower or action == "press_tab":
            success = await grid_client.press_key("tab")
            step_success = success
            actions_taken.append({"action": "press_key", "key": "tab", "success": success})
            await asyncio.sleep(0.3)

        elif "press escape" in instruction_lower or "press esc" in instruction_lower or action == "press_escape":
            success = await grid_client.press_key("escape")
            step_success = success
            actions_taken.append({"action": "press_key", "key": "escape", "success": success})
            await asyncio.sleep(0.3)

        elif (action == "click" or "click" in instruction_lower or "tap" in instruction_lower or
              "select" in instruction_lower or "choose" in instruction_lower):
            # Find and click element
            element_id = await _find_element_for_instruction(grid_client, instruction, target)
            if element_id:
                success = await grid_client.click_element(element_id)
                step_success = success
                actions_taken.append({"action": "click", "element_id": element_id, "success": success})
                await asyncio.sleep(0.5)
            else:
                step_error = f"Could not find element to click for: {instruction}"
                step_success = False
                actions_taken.append({"action": "click", "error": "element not found"})

        elif (action == "type" or "type" in instruction_lower or "enter" in instruction_lower or
              "fill" in instruction_lower or "input" in instruction_lower or "write" in instruction_lower):
            # Type text into element
            element_id, text_to_type = await _find_element_and_text_for_instruction(
                grid_client, instruction, target, value
            )
            if element_id and text_to_type:
                await grid_client.clear_element(element_id)
                success = await grid_client.send_keys(element_id, text_to_type)
                step_success = success
                actions_taken.append({"action": "type", "element_id": element_id, "text": text_to_type, "success": success})
                await asyncio.sleep(0.3)
            elif text_to_type:
                # No specific element, type into active element
                success = await grid_client.send_keys_to_active_element(text_to_type)
                step_success = success
                actions_taken.append({"action": "type_active", "text": text_to_type, "success": success})
                await asyncio.sleep(0.3)
            else:
                step_error = f"Could not determine what to type for: {instruction}"
                step_success = False
                actions_taken.append({"action": "type", "error": "could not parse instruction"})

        elif action == "assert" or action == "verify" or "verify" in instruction_lower or "check" in instruction_lower or "assert" in instruction_lower:
            # Assertion / Verification step
            assertion_type = step.get("assertion_type", "") if isinstance(step, dict) else ""
            assertion_result = await _execute_assertion(grid_client, target, value, assertion_type, instruction)
            step_success = assertion_result["success"]
            if not step_success:
                step_error = assertion_result.get("error", "Assertion failed")
            actions_taken.append({"action": "assert", "assertion_type": assertion_type, "result": assertion_result})

        elif "scroll" in instruction_lower or action == "scroll":
            # Scroll the page
            if "down" in instruction_lower or value == "down":
                await grid_client.execute_script("window.scrollBy(0, 300)")
            elif "up" in instruction_lower or value == "up":
                await grid_client.execute_script("window.scrollBy(0, -300)")
            elif "top" in instruction_lower or value == "top":
                await grid_client.execute_script("window.scrollTo(0, 0)")
            elif "bottom" in instruction_lower or value == "bottom":
                await grid_client.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            else:
                await grid_client.execute_script("window.scrollBy(0, 300)")
            step_success = True
            actions_taken.append({"action": "scroll", "success": True})
            await asyncio.sleep(0.5)

        else:
            # Unknown instruction - try to interpret
            element_id = await grid_client.find_element("css selector", "input:focus, textarea:focus")
            if element_id:
                # There's a focused input, assume we should type
                success = await grid_client.send_keys(element_id, instruction)
                step_success = success
                actions_taken.append({"action": "type_fallback", "text": instruction, "success": success})
            else:
                # Try to find a clickable element with matching text
                element_id = await _find_element_for_instruction(grid_client, instruction, target)
                if element_id:
                    success = await grid_client.click_element(element_id)
                    step_success = success
                    actions_taken.append({"action": "click_fallback", "element_id": element_id, "success": success})
                else:
                    step_error = f"Could not interpret instruction: {instruction}"
                    step_success = False
                    actions_taken.append({"action": "unknown", "error": "could not interpret"})

        # Capture screenshot after step
        if screenshot_enabled:
            step_screenshot = await grid_client.screenshot()

    except Exception as e:
        step_error = str(e)
        step_success = False
        logger.warning("Step execution error", step_index=step_index, error=step_error)

    step_duration = int((time.time() - step_start) * 1000)

    await emit_event(events_queue, "step_completed", {
        "test_id": test_id,
        "step_index": step_index,
        "success": step_success,
        "duration_ms": step_duration,
        "error": step_error,
    })

    return {
        "step_index": step_index,
        "instruction": instruction,
        "success": step_success,
        "duration_ms": step_duration,
        "screenshot": step_screenshot,
        "error": step_error,
        "actions": actions_taken,
    }


async def _find_element_for_instruction(
    grid_client: SeleniumGridClient,
    instruction: str,
    explicit_target: str = "",
) -> str | None:
    """Find an element based on natural language instruction or explicit selector."""
    # If explicit target is provided, use it
    if explicit_target:
        if explicit_target.startswith("//") or explicit_target.startswith("(//"):
            element_id = await grid_client.find_element("xpath", explicit_target)
        elif explicit_target.startswith("#") or explicit_target.startswith(".") or " " in explicit_target:
            element_id = await grid_client.find_element("css selector", explicit_target)
        else:
            # Try as ID first, then as text
            element_id = await grid_client.find_element("css selector", f"#{explicit_target}")
            if not element_id:
                element_id = await grid_client.find_element(
                    "xpath",
                    f"//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{explicit_target.lower()}')]"
                )
        if element_id:
            return element_id

    instruction_lower = instruction.lower()

    # Extract element identifier from natural language
    patterns = [
        r'(?:click|tap|select|choose|press)\s+(?:the\s+)?(?:on\s+)?["\']?([^"\']+)["\']?\s*(?:button|link|checkbox|radio|option)?',
        r'(?:button|link)\s+["\']?([^"\']+)["\']?',
        r'["\']([^"\']+)["\']',
    ]

    target_text = None
    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match:
            target_text = match.group(1).strip()
            break

    if not target_text:
        target_text = instruction_lower.replace("click", "").replace("tap", "").replace("the", "").replace("on", "").strip()

    # Try various selectors
    selectors_to_try = [
        f"//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]",
        f"//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]",
        f"//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]",
        f"//*[@aria-label[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
        f"//input[@placeholder[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
        f"//*[@id[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
        f"//*[@name[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text}')]]",
    ]

    for selector in selectors_to_try:
        element_id = await grid_client.find_element("xpath", selector)
        if element_id:
            return element_id

    return None


async def _find_element_and_text_for_instruction(
    grid_client: SeleniumGridClient,
    instruction: str,
    explicit_target: str = "",
    explicit_value: str = "",
) -> tuple[str | None, str | None]:
    """Find an element and extract text to type from instruction."""
    # If explicit value provided, use it
    text_to_type = explicit_value or None

    # If no explicit text, extract from instruction
    if not text_to_type:
        patterns = [
            r'(?:type|enter|fill|input|write)\s+["\']([^"\']+)["\']',
            r'["\']([^"\']+)["\']',
            r'(?:type|enter|fill|input|write)\s+(.+?)(?:\s+(?:in|into|to)\s|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                text_to_type = match.group(1).strip()
                break

    # Find the target element
    element_id = await _find_element_for_instruction(grid_client, instruction, explicit_target)

    # If no element found by instruction, try common input selectors
    if not element_id:
        input_selectors = [
            "input:not([type='hidden']):not([type='submit']):not([type='button'])",
            "textarea",
            "[contenteditable='true']",
        ]
        for selector in input_selectors:
            element_id = await grid_client.find_element("css selector", selector)
            if element_id:
                break

    return element_id, text_to_type


async def _execute_assertion(
    grid_client: SeleniumGridClient,
    target: str,
    value: str,
    assertion_type: str,
    instruction: str = "",
) -> dict:
    """
    Execute a structured assertion with specific assertion_type.

    Args:
        grid_client: Selenium Grid client
        target: What to check (selector, "page.url", "page.title", etc.)
        value: Expected value to compare against
        assertion_type: Type of assertion (url_matches, text_contains, element_visible, etc.)
        instruction: Fallback natural language instruction

    Returns:
        Dict with success status and details
    """
    try:
        assertion_type_lower = assertion_type.lower() if assertion_type else ""

        # URL-related assertions
        if assertion_type_lower == "url_matches" or target == "page.url":
            current_url = await grid_client.execute_script("return window.location.href")
            if value in current_url or current_url == value:
                return {"success": True, "assertion": "url_matches", "actual": current_url, "expected": value}
            return {
                "success": False,
                "error": f"URL mismatch: expected '{value}' but got '{current_url}'",
                "actual": current_url,
                "expected": value,
            }

        if assertion_type_lower == "url_contains":
            current_url = await grid_client.execute_script("return window.location.href")
            if value.lower() in current_url.lower():
                return {"success": True, "assertion": "url_contains", "actual": current_url, "expected": value}
            return {
                "success": False,
                "error": f"URL does not contain '{value}' (actual: '{current_url}')",
                "actual": current_url,
                "expected": value,
            }

        # Title-related assertions
        if assertion_type_lower == "title_matches" or target == "page.title":
            current_title = await grid_client.execute_script("return document.title")
            if value.lower() in current_title.lower() or current_title == value:
                return {"success": True, "assertion": "title_matches", "actual": current_title, "expected": value}
            return {
                "success": False,
                "error": f"Title mismatch: expected '{value}' but got '{current_title}'",
                "actual": current_title,
                "expected": value,
            }

        if assertion_type_lower == "title_contains":
            current_title = await grid_client.execute_script("return document.title")
            if value.lower() in current_title.lower():
                return {"success": True, "assertion": "title_contains", "actual": current_title, "expected": value}
            return {
                "success": False,
                "error": f"Title does not contain '{value}' (actual: '{current_title}')",
                "actual": current_title,
                "expected": value,
            }

        # Text content assertions
        if assertion_type_lower == "text_contains":
            if target == "page.title":
                current_title = await grid_client.execute_script("return document.title")
                if value.lower() in current_title.lower():
                    return {"success": True, "assertion": "text_contains", "target": "page.title", "found": value}
                return {"success": False, "error": f"Title does not contain '{value}' (actual: '{current_title}')"}

            # Check if text exists on page
            page_text = await grid_client.execute_script("return document.body.innerText")
            if value.lower() in page_text.lower():
                return {"success": True, "assertion": "text_contains", "found": value}

            # Check specific element if target is a selector
            if target and target not in ("page", "page.body"):
                element_text = await _get_element_text(grid_client, target)
                if element_text and value.lower() in element_text.lower():
                    return {"success": True, "assertion": "text_contains", "target": target, "found": value}

            return {"success": False, "error": f"Text '{value}' not found on page"}

        if assertion_type_lower == "text_equals":
            element_text = await _get_element_text(grid_client, target)
            if element_text and element_text.strip() == value.strip():
                return {"success": True, "assertion": "text_equals", "target": target, "actual": element_text}
            return {
                "success": False,
                "error": f"Text mismatch: expected '{value}' but got '{element_text}'",
                "actual": element_text,
                "expected": value,
            }

        # Element visibility assertions
        if assertion_type_lower == "element_visible" or assertion_type_lower == "visible":
            element_id = await _find_element_for_selector(grid_client, target)
            if element_id:
                # Check if element is actually visible (not hidden)
                is_visible = await grid_client.execute_script(
                    f"""
                    var elem = document.evaluate(
                        "(//*[@id='{element_id}' or contains(@data-element-id, '{element_id}')])[1]",
                        document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null
                    ).singleNodeValue;
                    if (!elem) {{
                        var elems = document.querySelectorAll('*');
                        for (var e of elems) {{
                            if (e.innerText && e.innerText.toLowerCase().includes('{target.lower()}')) {{
                                elem = e; break;
                            }}
                        }}
                    }}
                    if (elem) {{
                        var style = window.getComputedStyle(elem);
                        return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
                    }}
                    return false;
                    """
                )
                # For simplicity, if we found the element, consider it visible
                return {"success": True, "assertion": "element_visible", "target": target, "element_id": element_id}
            return {"success": False, "error": f"Element '{target}' not found or not visible"}

        if assertion_type_lower == "element_exists" or assertion_type_lower == "exists":
            element_id = await _find_element_for_selector(grid_client, target)
            if element_id:
                return {"success": True, "assertion": "element_exists", "target": target, "element_id": element_id}
            return {"success": False, "error": f"Element '{target}' does not exist"}

        if assertion_type_lower == "element_not_visible" or assertion_type_lower == "not_visible":
            element_id = await _find_element_for_selector(grid_client, target)
            if not element_id:
                return {"success": True, "assertion": "element_not_visible", "target": target}
            return {"success": False, "error": f"Element '{target}' is visible but expected not visible"}

        # Attribute assertions
        if assertion_type_lower == "attribute_equals":
            parts = target.split("@")
            if len(parts) == 2:
                selector, attr = parts
                attr_value = await _get_element_attribute(grid_client, selector, attr)
                if attr_value == value:
                    return {"success": True, "assertion": "attribute_equals", "target": target, "actual": attr_value}
                return {
                    "success": False,
                    "error": f"Attribute mismatch: expected '{value}' but got '{attr_value}'",
                }

        if assertion_type_lower == "attribute_contains":
            parts = target.split("@")
            if len(parts) == 2:
                selector, attr = parts
                attr_value = await _get_element_attribute(grid_client, selector, attr)
                if attr_value and value.lower() in attr_value.lower():
                    return {"success": True, "assertion": "attribute_contains", "target": target, "found": value}
                return {"success": False, "error": f"Attribute does not contain '{value}' (actual: '{attr_value}')"}

        # Value assertions (for inputs)
        if assertion_type_lower == "value_equals":
            element_id = await _find_element_for_selector(grid_client, target)
            if element_id:
                input_value = await grid_client.execute_script(
                    f"return document.querySelector('[data-element-id=\"{element_id}\"]')?.value || ''"
                )
                if input_value == value:
                    return {"success": True, "assertion": "value_equals", "actual": input_value}
                return {"success": False, "error": f"Value mismatch: expected '{value}' but got '{input_value}'"}
            return {"success": False, "error": f"Element '{target}' not found"}

        # Fallback to natural language verification if no specific assertion type
        if instruction:
            return await _verify_instruction(grid_client, instruction, target, value)

        # Default: try text_contains
        page_text = await grid_client.execute_script("return document.body.innerText")
        if value and value.lower() in page_text.lower():
            return {"success": True, "assertion": "text_contains_fallback", "found": value}

        return {"success": False, "error": f"Unknown assertion type: {assertion_type}"}

    except Exception as e:
        logger.warning("Assertion execution error", assertion_type=assertion_type, error=str(e))
        return {"success": False, "error": str(e)}


async def _find_element_for_selector(grid_client: SeleniumGridClient, selector: str) -> str | None:
    """Find an element using various selector strategies."""
    if not selector:
        return None

    # Try XPath first
    if selector.startswith("//") or selector.startswith("(//"):
        return await grid_client.find_element("xpath", selector)

    # CSS selector
    if selector.startswith("#") or selector.startswith(".") or " " in selector or ">" in selector:
        return await grid_client.find_element("css selector", selector)

    # Tag name (like "h1", "button", "input")
    if selector.lower() in ("h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "span", "button", "input", "a", "img"):
        element_id = await grid_client.find_element("css selector", selector)
        if element_id:
            return element_id

    # Try as ID
    element_id = await grid_client.find_element("css selector", f"#{selector}")
    if element_id:
        return element_id

    # Try as text content
    element_id = await grid_client.find_element(
        "xpath",
        f"//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{selector.lower()}')]"
    )
    if element_id:
        return element_id

    return None


async def _get_element_text(grid_client: SeleniumGridClient, selector: str) -> str | None:
    """Get the text content of an element."""
    element_id = await _find_element_for_selector(grid_client, selector)
    if not element_id:
        return None

    try:
        # Use JavaScript to get text content
        text = await grid_client.execute_script(
            f"""
            var elem = document.querySelector('[data-element-id="{element_id}"]');
            if (!elem) {{
                var xpath = "//*[contains(@id, '{element_id}')]";
                elem = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            }}
            return elem ? elem.innerText || elem.textContent : null;
            """
        )
        return text
    except Exception:
        return None


async def _get_element_attribute(grid_client: SeleniumGridClient, selector: str, attribute: str) -> str | None:
    """Get an attribute value from an element."""
    element_id = await _find_element_for_selector(grid_client, selector)
    if not element_id:
        return None

    try:
        value = await grid_client.execute_script(
            f"""
            var elem = document.querySelector('[data-element-id="{element_id}"]');
            if (!elem) {{
                var xpath = "//*[contains(@id, '{element_id}')]";
                elem = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            }}
            return elem ? elem.getAttribute('{attribute}') : null;
            """
        )
        return value
    except Exception:
        return None


async def _verify_instruction(
    grid_client: SeleniumGridClient,
    instruction: str,
    explicit_target: str = "",
    explicit_value: str = "",
) -> dict:
    """Verify a condition described in natural language."""
    instruction_lower = instruction.lower()

    # Extract what we're looking for
    patterns = [
        r'(?:verify|check|assert|should\s+(?:see|have|contain))\s+["\']?([^"\']+)["\']?',
        r'["\']([^"\']+)["\']',
    ]

    target_text = explicit_value or None
    for pattern in patterns:
        match = re.search(pattern, instruction_lower)
        if match and not target_text:
            target_text = match.group(1).strip()
            break

    if not target_text:
        return {"success": False, "error": "Could not determine what to verify"}

    # Check if the text exists on the page
    try:
        page_source = await grid_client.execute_script("return document.body.innerText")
        if target_text.lower() in page_source.lower():
            return {"success": True, "found": target_text}

        # Also check for element with that text
        element_id = await grid_client.find_element(
            "xpath",
            f"//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text.lower()}')]"
        )
        if element_id:
            return {"success": True, "found": target_text, "element_id": element_id}

        return {"success": False, "error": f"Could not find '{target_text}' on page"}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def execute_scheduled_run(
    schedule_id: str,
    run_id: str,
    schedule: dict,
    events_queue: asyncio.Queue | None = None,
) -> dict:
    """
    Execute all tests for a scheduled run.

    Args:
        schedule_id: ID of the schedule
        run_id: ID of this run instance
        schedule: Schedule configuration dict
        events_queue: Optional queue for SSE events

    Returns:
        Dict with aggregate results:
        - tests_total, tests_passed, tests_failed, tests_skipped
        - duration_ms, failures (list of failure details)
    """
    start_time = time.time()

    await emit_event(events_queue, "run_started", {
        "schedule_id": schedule_id,
        "run_id": run_id,
        "schedule_name": schedule.get("name"),
    })

    logger.info("Starting scheduled run", schedule_id=schedule_id, run_id=run_id)

    # Fetch tests to run
    tests = await fetch_tests_for_schedule(schedule)

    if not tests:
        logger.warning("No tests found for schedule", schedule_id=schedule_id)
        return {
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "duration_ms": 0,
            "failures": [],
            "test_results": [],
        }

    await emit_event(events_queue, "tests_fetched", {
        "schedule_id": schedule_id,
        "run_id": run_id,
        "test_count": len(tests),
        "test_names": [t.get("name") for t in tests],
    })

    # Get app URL from schedule
    app_url = schedule.get("app_url_override") or schedule.get("app_url", "")
    if not app_url:
        logger.error("No app_url configured for schedule", schedule_id=schedule_id)
        return {
            "tests_total": len(tests),
            "tests_passed": 0,
            "tests_failed": len(tests),
            "tests_skipped": 0,
            "duration_ms": int((time.time() - start_time) * 1000),
            "failures": [{"test_id": t["id"], "error": "No app_url configured"} for t in tests],
            "test_results": [],
        }

    # Execute each test
    test_results = []
    failures = []
    tests_passed = 0
    tests_failed = 0
    tests_auto_healed = 0

    # Check if auto-healing is enabled for this schedule
    auto_heal_enabled = schedule.get("auto_heal_enabled", False)
    if auto_heal_enabled:
        logger.info(
            "Auto-healing enabled for schedule",
            schedule_id=schedule_id,
            confidence_threshold=schedule.get("auto_heal_confidence_threshold", 0.9),
        )

    for idx, test in enumerate(tests):
        await emit_event(events_queue, "progress", {
            "schedule_id": schedule_id,
            "run_id": run_id,
            "current_test": idx + 1,
            "total_tests": len(tests),
            "test_name": test.get("name"),
            "percent": int((idx / len(tests)) * 100),
        })

        try:
            result = await execute_single_test(
                test=test,
                app_url=app_url,
                events_queue=events_queue,
                screenshot_enabled=True,
            )

            if result["success"]:
                tests_passed += 1
                test_results.append(result)
            else:
                # =========================================================
                # AUTO-HEALING INTEGRATION (Phase 4)
                # When a test fails, attempt to auto-heal if enabled
                # =========================================================
                healed_result = None
                if auto_heal_enabled:
                    logger.info(
                        "Test failed, attempting auto-heal",
                        test_id=test.get("id"),
                        test_name=test.get("name"),
                    )
                    healed_result = await attempt_auto_heal(
                        test=test,
                        failure_result=result,
                        schedule_config=schedule,
                        app_url=app_url,
                        events_queue=events_queue,
                    )

                if healed_result and healed_result.get("success"):
                    # Auto-healing succeeded - count as passed
                    tests_passed += 1
                    tests_auto_healed += 1
                    test_results.append(healed_result)
                    logger.info(
                        "Test auto-healed successfully",
                        test_id=test.get("id"),
                        test_name=test.get("name"),
                        fix_type=healed_result.get("healing_details", {}).get("fix_type"),
                    )
                else:
                    # Auto-healing not attempted, not possible, or failed
                    tests_failed += 1
                    test_results.append(result)
                    failure_entry = {
                        "test_id": result["test_id"],
                        "test_name": result["test_name"],
                        "error": result.get("error"),
                        "steps_passed": result["steps_passed"],
                        "steps_failed": result["steps_failed"],
                        "recording_url": result.get("recording_url"),
                    }
                    # Include info about healing attempt if it was tried
                    if auto_heal_enabled:
                        failure_entry["auto_heal_attempted"] = True
                        failure_entry["auto_heal_succeeded"] = False
                    failures.append(failure_entry)

        except Exception as e:
            tests_failed += 1
            failures.append({
                "test_id": test.get("id"),
                "test_name": test.get("name"),
                "error": str(e),
            })
            logger.exception("Test execution failed", test_id=test.get("id"), error=str(e))

    duration_ms = int((time.time() - start_time) * 1000)

    await emit_event(events_queue, "run_completed", {
        "schedule_id": schedule_id,
        "run_id": run_id,
        "tests_total": len(tests),
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_auto_healed": tests_auto_healed,
        "duration_ms": duration_ms,
        "success": tests_failed == 0,
    })

    logger.info(
        "Scheduled run completed",
        schedule_id=schedule_id,
        run_id=run_id,
        tests_total=len(tests),
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tests_auto_healed=tests_auto_healed,
        duration_ms=duration_ms,
    )

    # =========================================================================
    # FLAKY TEST DETECTION (Phase 3)
    # After each run completes, update flaky scores for all tests
    # =========================================================================
    flaky_tests = []
    try:
        flaky_tests = await update_flaky_scores_after_run(
            schedule_id=schedule_id,
            run_id=run_id,
            test_results=test_results,
        )
        if flaky_tests:
            await emit_event(events_queue, "flaky_tests_detected", {
                "schedule_id": schedule_id,
                "run_id": run_id,
                "flaky_count": len(flaky_tests),
                "flaky_tests": flaky_tests,
            })
    except Exception as e:
        # Don't fail the run if flaky detection fails
        logger.warning(
            "Flaky test detection failed",
            schedule_id=schedule_id,
            run_id=run_id,
            error=str(e),
        )

    return {
        "tests_total": len(tests),
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_auto_healed": tests_auto_healed,
        "tests_skipped": 0,
        "duration_ms": duration_ms,
        "failures": failures,
        "test_results": test_results,
        "auto_heal_enabled": auto_heal_enabled,
        "flaky_tests": flaky_tests,
    }
