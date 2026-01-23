"""Schedule Executor - Executes scheduled test runs using Selenium Grid.

This module provides the core test execution logic for scheduled runs.
It reuses the Selenium Grid execution patterns from browser.py.
"""

import asyncio
import re
import time
from datetime import UTC, datetime
from typing import Any

import structlog

from src.browser.selenium_grid_client import SeleniumGridClient
from src.services.supabase_client import get_supabase_client

logger = structlog.get_logger()


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
            # Wait for specified duration
            match = re.search(r'(\d+)\s*(second|sec|s|ms|millisecond)', instruction_lower)
            if match:
                duration = int(match.group(1))
                if 'ms' in match.group(2) or 'millisecond' in match.group(2):
                    duration = duration / 1000
                await asyncio.sleep(duration)
            elif value:
                await asyncio.sleep(float(value))
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

        elif "verify" in instruction_lower or "check" in instruction_lower or "assert" in instruction_lower or action == "verify":
            # Verification step
            verification_result = await _verify_instruction(grid_client, instruction, target, value)
            step_success = verification_result["success"]
            if not step_success:
                step_error = verification_result.get("error", "Verification failed")
            actions_taken.append({"action": "verify", "result": verification_result})

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
            test_results.append(result)

            if result["success"]:
                tests_passed += 1
            else:
                tests_failed += 1
                failures.append({
                    "test_id": result["test_id"],
                    "test_name": result["test_name"],
                    "error": result.get("error"),
                    "steps_passed": result["steps_passed"],
                    "steps_failed": result["steps_failed"],
                    "recording_url": result.get("recording_url"),
                })

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
        duration_ms=duration_ms,
    )

    return {
        "tests_total": len(tests),
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "tests_skipped": 0,
        "duration_ms": duration_ms,
        "failures": failures,
        "test_results": test_results,
    }
