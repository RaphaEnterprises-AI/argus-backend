# E2E Testing Agent Chrome Extension

This Chrome extension enables browser automation through a real browser instance,
similar to Claude in Chrome and Google Antigravity.

## Benefits Over Playwright/Selenium

- **Real Browser Session**: Uses your existing Chrome with cookies, auth, and extensions
- **No Bot Detection**: Websites cannot distinguish from a real user
- **Direct DOM Access**: Full access to page content and console logs
- **Framework Agnostic**: Works with any web application
- **Live Debugging**: See exactly what the agent is doing in your browser

## Installation

### 1. Load Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select this `extension/` directory

### 2. Configure the Extension

The extension connects to the Python agent via WebSocket on `ws://localhost:8765`.
This is configurable in `background.js` if needed.

### 3. Verify Installation

1. Click the extension icon in Chrome toolbar
2. You should see "E2E Testing Agent" popup
3. Status will show "Disconnected" until the Python agent starts

## Usage with Python Agent

### Basic Usage

```python
from src.tools import create_browser, AutomationFramework

async def main():
    # Create extension bridge (starts WebSocket server)
    async with await create_browser("extension") as browser:
        # Navigate to a page
        await browser.goto("https://example.com")

        # Click elements
        await browser.click("#login-button")

        # Fill forms
        await browser.fill("#email", "test@example.com")
        await browser.fill("#password", "secret123")

        # Take screenshots
        screenshot = await browser.screenshot()

        # Get console logs (extension-specific feature!)
        logs = await browser.get_console_logs()
        print(f"Console logs: {logs}")

        # Execute JavaScript
        result = await browser.evaluate("document.title")
```

### Using ExtensionBridge Directly

```python
from src.tools import ExtensionBridge

async def main():
    bridge = ExtensionBridge(port=8765, host="localhost")
    await bridge.start()  # Starts WebSocket server

    # Wait for Chrome extension to connect
    # (extension auto-connects when Chrome starts)

    try:
        await bridge.goto("https://example.com")

        # Get page info
        info = await bridge.get_page_info()
        print(f"Page: {info['title']} - {info['url']}")

        # Query elements
        elements = await bridge.query_selector_all("button")
        print(f"Found {len(elements)} buttons")

        # Create new tab
        tab_id = await bridge.create_tab("https://google.com")

        # Get all tabs
        tabs = await bridge.get_tabs()

    finally:
        await bridge.stop()
```

### Console Log Monitoring

One unique feature of the extension is real-time console log capture:

```python
async def main():
    bridge = ExtensionBridge()
    await bridge.start()

    # Register handler for console logs
    async def log_handler(log_entry):
        print(f"[{log_entry['level']}] {log_entry['args']}")

    bridge.on_console_log(log_handler)

    await bridge.goto("https://example.com")
    # Any console.log, console.error etc. from the page will trigger log_handler
```

## Architecture

```
┌─────────────────┐     WebSocket      ┌──────────────────┐
│  Python Agent   │◄──────────────────►│ Chrome Extension │
│  (ExtensionBridge)                   │  (background.js) │
└─────────────────┘   ws://localhost:8765  └────────┬─────────┘
                                                    │
                                           Chrome Messaging API
                                                    │
                                           ┌────────▼─────────┐
                                           │  Content Script  │
                                           │  (content.js)    │
                                           └────────┬─────────┘
                                                    │
                                              DOM Access
                                                    │
                                           ┌────────▼─────────┐
                                           │    Web Page      │
                                           │  (any website)   │
                                           └──────────────────┘
```

## Extension Commands

The extension supports these commands from the Python agent:

| Command | Description | Parameters |
|---------|-------------|------------|
| `navigate` | Navigate to URL | `url`, `waitUntil` |
| `click` | Click element | `selector` |
| `fill` | Fill input field | `selector`, `value` |
| `type` | Type text character by character | `selector`, `text`, `delay` |
| `hover` | Hover over element | `selector` |
| `select` | Select dropdown option | `selector`, `value` |
| `pressKey` | Press keyboard key | `key` |
| `scroll` | Scroll the page | `x`, `y` |
| `evaluate` | Execute JavaScript | `script` |
| `screenshot` | Capture screenshot | `fullPage` |
| `getPageInfo` | Get page URL/title | - |
| `querySelector` | Query single element | `selector` |
| `querySelectorAll` | Query all elements | `selector` |
| `waitForSelector` | Wait for element | `selector`, `timeout` |
| `createTab` | Create new tab | `url` |
| `closeTab` | Close tab | `tabId` |
| `getTabs` | List all tabs | - |
| `getConsoleLogs` | Get captured console logs | - |

## Element Selection

The content script supports multiple selector strategies:

1. **CSS Selectors**: Standard CSS selectors
   ```python
   await browser.click("#login-button")
   await browser.click(".submit-btn")
   await browser.click("button[type='submit']")
   ```

2. **XPath**: XPath expressions (prefix with `//`)
   ```python
   await browser.click("//button[contains(text(), 'Login')]")
   ```

3. **data-testid**: Automatically tried for matching
   ```python
   await browser.click("login-button")  # Matches [data-testid="login-button"]
   ```

4. **Text Content**: Partial text match as fallback
   ```python
   await browser.click("Login")  # Finds element containing "Login"
   ```

## Troubleshooting

### Extension Not Connecting

1. Ensure the Python agent is running with WebSocket server
2. Check Chrome DevTools (F12) → Console for errors
3. Verify the extension is enabled at `chrome://extensions/`
4. Check if port 8765 is available

### Commands Timing Out

1. Increase timeout in ExtensionBridge:
   ```python
   response = await bridge._send_command("click", {"selector": "#btn"}, timeout=60.0)
   ```
2. Use `waitForSelector` before interacting with dynamic elements

### Console Logs Not Capturing

1. Console interception only works on pages loaded after extension install
2. Refresh the page if extension was just installed
3. Check that content script is injected (look for `[E2E Agent]` in console)

## Security Considerations

- The WebSocket server only binds to `localhost` by default
- No external connections are accepted
- The extension requires explicit user installation
- Content scripts only run on pages you navigate to

## Development

To modify the extension:

1. Edit files in `extension/` directory
2. Go to `chrome://extensions/`
3. Click the refresh icon on the extension
4. Changes take effect immediately

For background script changes, you may need to reload the extension.

## Comparison with Other Tools

| Feature | Extension Bridge | Playwright | Selenium | Computer Use |
|---------|-----------------|------------|----------|--------------|
| Real browser session | Yes | No | No | Yes |
| No bot detection | Yes | Detectable | Detectable | Yes |
| Speed | Fast | Fastest | Fast | Slow |
| Console logs | Yes | Limited | No | No |
| Cookies/auth preserved | Yes | No | No | Yes |
| Works offline | Yes | Yes | Yes | No |
| Visual-based | No | No | No | Yes |

## License

MIT License - see project root for details.
