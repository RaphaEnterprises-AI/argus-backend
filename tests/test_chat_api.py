"""
Chat API Testing Script for Argus

This script tests the /chat endpoints to diagnose timeout and screenshot issues.

Usage:
    # Run with pytest
    pytest tests/test_chat_api.py -v

    # Run directly for debugging
    python tests/test_chat_api.py

Issues being diagnosed:
1. Timeout errors (30000ms client-side vs 60-180s backend)
2. Malformed screenshot URLs (data:image/png;base64,screenshot_xxx)
3. 401 authentication errors on /api/v1/projects
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Callable

import httpx
import pytest

# Default to local backend, override with env var
BASE_URL = os.getenv("ARGUS_API_URL", "http://localhost:8000")
PRODUCTION_URL = "https://argus-brain-production.up.railway.app"

# Test configuration
TIMEOUT_SECONDS = 200  # Match backend's 180s timeout for runTest + buffer


class ChatAPITester:
    """Test client for Argus Chat API."""

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.thread_id = None

    async def health_check(self) -> dict:
        """Check if the API is healthy."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.base_url}/health")
            return {"status": response.status_code, "body": response.json()}

    async def send_message(
        self,
        message: str,
        thread_id: str | None = None,
        app_url: str | None = None
    ) -> dict:
        """Send a non-streaming chat message."""
        payload = {
            "messages": [{"role": "user", "content": message}],
            "thread_id": thread_id or self.thread_id,
            "app_url": app_url,
        }

        start = time.time()
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self.base_url}/api/v1/chat/message",
                json=payload
            )
            elapsed = time.time() - start

        result = {
            "status": response.status_code,
            "elapsed_seconds": elapsed,
            "body": response.json() if response.status_code == 200 else response.text,
        }

        # Store thread_id for future messages
        if response.status_code == 200:
            self.thread_id = response.json().get("thread_id")

        return result

    async def send_message_streaming(
        self,
        message: str,
        thread_id: str | None = None,
        app_url: str | None = None,
        on_event: Callable | None = None
    ) -> dict:
        """Send a streaming chat message and collect all events."""
        payload = {
            "messages": [{"role": "user", "content": message}],
            "thread_id": thread_id or self.thread_id,
            "app_url": app_url,
        }

        events = []
        text_chunks = []
        tool_calls = []
        tool_results = []

        start = time.time()
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/v1/chat/stream",
                json=payload
            ) as response:
                # Get thread ID from header
                self.thread_id = response.headers.get("X-Thread-Id")

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # Parse Vercel AI SDK format: <type>:<json>
                    if ":" in line:
                        event_type = line[0]
                        event_data = line[2:]  # Skip "X:"

                        try:
                            parsed_data = json.loads(event_data)
                        except json.JSONDecodeError:
                            parsed_data = event_data

                        event = {"type": event_type, "data": parsed_data}
                        events.append(event)

                        if on_event:
                            on_event(event)

                        # Categorize events
                        if event_type == "0":  # Text
                            text_chunks.append(parsed_data)
                        elif event_type == "9":  # Tool call
                            tool_calls.append(parsed_data)
                        elif event_type == "a":  # Tool result
                            tool_results.append(parsed_data)

        elapsed = time.time() - start

        return {
            "status": response.status_code,
            "elapsed_seconds": elapsed,
            "thread_id": self.thread_id,
            "text": "".join(text_chunks),
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "events": events,
        }

    async def get_history(self, thread_id: str | None = None) -> dict:
        """Get chat history for a thread."""
        tid = thread_id or self.thread_id
        if not tid:
            return {"error": "No thread_id available"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.base_url}/api/v1/chat/history/{tid}"
            )
            return {"status": response.status_code, "body": response.json()}

    async def check_browser_worker(self) -> dict:
        """Directly check the browser worker status."""
        worker_url = "https://argus-api.samuelvinay-kumar.workers.dev"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Try a simple health check or status endpoint
            try:
                response = await client.get(f"{worker_url}/")
                return {
                    "status": response.status_code,
                    "body": response.text[:500] if response.text else None
                }
            except Exception as e:
                return {"error": str(e)}


def analyze_screenshot_issues(tool_result: dict) -> list[str]:
    """Analyze a tool result for screenshot issues."""
    issues = []

    result_str = json.dumps(tool_result) if isinstance(tool_result, dict) else str(tool_result)

    # Check for malformed data URLs
    if "data:image/png;base64,screenshot_" in result_str:
        issues.append(
            "MALFORMED SCREENSHOT URL: Found 'data:image/png;base64,screenshot_xxx' - "
            "This is a filename reference, not actual base64 data. "
            "The browser worker is returning a reference instead of the image data."
        )

    # Check for missing base64 data
    if "data:image/png;base64," in result_str:
        # Extract what comes after base64,
        import re
        matches = re.findall(r'data:image/png;base64,([^"\']+)', result_str)
        for match in matches:
            if len(match) < 100:  # Real base64 images are much longer
                issues.append(
                    f"SHORT BASE64 DATA: '{match[:50]}...' - "
                    "Base64 image data should be thousands of characters"
                )

    # Check for ERR_INVALID_URL patterns
    if "ERR_INVALID_URL" in result_str:
        issues.append(
            "ERR_INVALID_URL detected - The screenshot URL format is invalid for browser rendering"
        )

    return issues


# ============================================================================
# Pytest Test Cases
# ============================================================================

@pytest.fixture
def tester():
    """Create a tester instance."""
    return ChatAPITester(BASE_URL)


@pytest.fixture
def production_tester():
    """Create a tester for production."""
    return ChatAPITester(PRODUCTION_URL)


@pytest.mark.asyncio
async def test_health_check(tester: ChatAPITester):
    """Test that the API is healthy."""
    result = await tester.health_check()
    assert result["status"] == 200, f"Health check failed: {result}"


@pytest.mark.asyncio
async def test_simple_chat_message(tester: ChatAPITester):
    """Test a simple chat message without tool execution."""
    result = await tester.send_message("What can you help me with?")

    assert result["status"] == 200, f"Chat failed: {result}"
    assert "message" in result["body"], "No message in response"
    assert result["elapsed_seconds"] < 30, "Simple chat should be fast"


@pytest.mark.asyncio
async def test_chat_with_check_status(tester: ChatAPITester):
    """Test chat with checkStatus tool (fast, no browser)."""
    result = await tester.send_message("Check the system status")

    assert result["status"] == 200, f"Chat failed: {result}"
    # Tool calls may or may not be present depending on LLM decision
    print(f"\nResult: {json.dumps(result['body'], indent=2)}")


@pytest.mark.asyncio
async def test_streaming_chat(tester: ChatAPITester):
    """Test streaming chat endpoint."""
    events_received = []

    def on_event(event):
        events_received.append(event)
        print(f"Event: {event['type']} - {str(event['data'])[:100]}")

    result = await tester.send_message_streaming(
        "Say hello and tell me what you can do",
        on_event=on_event
    )

    assert result["status"] == 200, f"Stream failed: {result}"
    assert len(events_received) > 0, "No events received"
    assert result["text"], "No text content streamed"


@pytest.mark.asyncio
@pytest.mark.timeout(200)  # 200 second timeout for browser tests
async def test_discover_elements(tester: ChatAPITester):
    """Test discoverElements tool (60s timeout in backend)."""
    result = await tester.send_message_streaming(
        "Discover the elements on https://example.com",
        app_url="https://example.com"
    )

    print(f"\nElapsed: {result['elapsed_seconds']:.1f}s")
    print(f"Tool calls: {len(result['tool_calls'])}")
    print(f"Tool results: {len(result['tool_results'])}")

    # Check for screenshot issues
    for tool_result in result["tool_results"]:
        issues = analyze_screenshot_issues(tool_result)
        for issue in issues:
            print(f"\n⚠️ ISSUE: {issue}")

    assert result["status"] == 200


@pytest.mark.asyncio
@pytest.mark.timeout(200)
async def test_execute_action(tester: ChatAPITester):
    """Test executeAction tool (60s timeout in backend)."""
    result = await tester.send_message_streaming(
        "Go to https://example.com and take a screenshot",
        app_url="https://example.com"
    )

    print(f"\nElapsed: {result['elapsed_seconds']:.1f}s")

    # Analyze tool results for screenshot issues
    for tool_result in result["tool_results"]:
        issues = analyze_screenshot_issues(tool_result)
        if issues:
            print("\n❌ SCREENSHOT ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ No screenshot issues detected")

    assert result["status"] == 200


@pytest.mark.asyncio
@pytest.mark.timeout(300)  # 300 second timeout for run test (180s backend + buffer)
async def test_run_simple_test(tester: ChatAPITester):
    """Test runTest tool (180s timeout in backend) - THIS IS WHERE 30s CLIENT TIMEOUT FAILS."""
    result = await tester.send_message_streaming(
        "Run a test on https://example.com with steps: 1. Verify the page loads 2. Check for heading",
        app_url="https://example.com"
    )

    print(f"\nElapsed: {result['elapsed_seconds']:.1f}s")
    print(f"Tool calls: {result['tool_calls']}")

    # This is the key test - if this passes but frontend fails, it's a client timeout issue
    if result["elapsed_seconds"] > 30:
        print("\n⚠️ WARNING: This test took longer than 30s - frontend would timeout!")
        print("   The frontend needs timeout >= 180s for runTest operations")

    # Analyze for screenshot issues
    for tool_result in result["tool_results"]:
        issues = analyze_screenshot_issues(tool_result)
        for issue in issues:
            print(f"\n❌ {issue}")


@pytest.mark.asyncio
async def test_browser_worker_directly():
    """Test the browser worker directly to check screenshot format."""
    worker_url = "https://argus-api.samuelvinay-kumar.workers.dev"

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Test observe endpoint
        response = await client.post(
            f"{worker_url}/observe",
            json={"url": "https://example.com"}
        )

        print(f"\nWorker response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Result keys: {list(result.keys())}")

            # Check screenshot format
            if "screenshot" in result:
                screenshot = result["screenshot"]
                if isinstance(screenshot, str):
                    if screenshot.startswith("data:image"):
                        # Check if it has actual base64 data
                        if len(screenshot) < 200:
                            print(f"❌ Screenshot too short: {screenshot}")
                        else:
                            print(f"✅ Screenshot looks valid (length: {len(screenshot)})")
                    elif screenshot.startswith("screenshot_"):
                        print(f"❌ Screenshot is a filename reference: {screenshot}")
                    else:
                        print(f"⚠️ Unknown screenshot format: {screenshot[:100]}")


# ============================================================================
# Direct run mode for debugging
# ============================================================================

async def main():
    """Run tests directly for debugging."""
    print("=" * 60)
    print("Argus Chat API Diagnostic Tests")
    print("=" * 60)

    # Determine which URL to use
    url = os.getenv("ARGUS_API_URL", BASE_URL)
    print(f"\nTarget: {url}")

    tester = ChatAPITester(url)

    # Test 1: Health check
    print("\n" + "-" * 40)
    print("Test 1: Health Check")
    print("-" * 40)
    try:
        result = await tester.health_check()
        print(f"Status: {result['status']}")
        print(f"Response: {result['body']}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 2: Simple chat
    print("\n" + "-" * 40)
    print("Test 2: Simple Chat (no tools)")
    print("-" * 40)
    try:
        result = await tester.send_message("Hello! What are your capabilities?")
        print(f"Status: {result['status']}")
        print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
        if result["status"] == 200:
            print(f"Response: {result['body']['message'][:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 3: Streaming chat with tool
    print("\n" + "-" * 40)
    print("Test 3: Streaming Chat with Tool Execution")
    print("-" * 40)
    try:
        print("Sending: 'Check system status'")
        result = await tester.send_message_streaming("Check the system status")
        print(f"Status: {result['status']}")
        print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
        print(f"Text: {result['text'][:200]}..." if result['text'] else "No text")
        print(f"Tool calls: {len(result['tool_calls'])}")
        print(f"Tool results: {len(result['tool_results'])}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 4: Browser worker direct test
    print("\n" + "-" * 40)
    print("Test 4: Browser Worker Direct Test")
    print("-" * 40)
    try:
        await test_browser_worker_directly()
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test 5: Long-running test (to verify timeout issue)
    print("\n" + "-" * 40)
    print("Test 5: Long-Running Test (runTest with 180s backend timeout)")
    print("-" * 40)
    print("⚠️  This test may take up to 3 minutes...")
    try:
        result = await tester.send_message_streaming(
            "Run a simple test on https://example.com: navigate to the page and verify it loads",
            app_url="https://example.com"
        )
        print(f"Status: {result['status']}")
        print(f"Elapsed: {result['elapsed_seconds']:.1f}s")

        if result['elapsed_seconds'] > 30:
            print("\n⚠️  TEST TOOK LONGER THAN 30s")
            print("   Your frontend's 30000ms timeout would have failed!")
            print("   Increase frontend timeout to at least 180000ms (3 minutes)")

        # Check for screenshot issues
        for tr in result["tool_results"]:
            issues = analyze_screenshot_issues(tr)
            if issues:
                print("\n❌ Screenshot issues found:")
                for issue in issues:
                    print(f"   {issue}")

    except httpx.TimeoutException as e:
        print(f"❌ Timeout after {TIMEOUT_SECONDS}s: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Diagnostic Summary")
    print("=" * 60)
    print("""
Issues to check:

1. TIMEOUT ERROR (30000ms):
   - Backend tool timeouts: runTest=180s, executeAction=60s, discoverElements=60s
   - Your frontend/client has 30s timeout which is too short
   - SOLUTION: Increase client timeout to at least 180000ms

2. MALFORMED SCREENSHOT URLs:
   - If you see 'data:image/png;base64,screenshot_xxx'
   - The browser worker is returning a filename, not base64 data
   - Check browser worker implementation for screenshot handling

3. 401 ERROR on /api/v1/projects:
   - The projects endpoint requires authentication
   - Pass API key via X-API-Key header or JWT via Authorization header
   - Or set enforce_authentication=False in config for development
""")


if __name__ == "__main__":
    asyncio.run(main())
