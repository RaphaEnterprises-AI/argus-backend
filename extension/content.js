/**
 * E2E Testing Agent - Content Script
 *
 * Runs in the context of web pages and provides:
 * - DOM access and manipulation
 * - Element interaction (click, fill, hover, etc.)
 * - Console log interception
 * - Element waiting and querying
 * - Script evaluation
 */

(function() {
  'use strict';

  // Avoid double injection
  if (window.__e2eTestingAgentInjected) return;
  window.__e2eTestingAgentInjected = true;

  console.log('[E2E Agent] Content script loaded');

  /**
   * Intercept console logs and forward to background script
   */
  function interceptConsoleLogs() {
    const originalConsole = {};
    ['log', 'warn', 'error', 'info', 'debug'].forEach(level => {
      originalConsole[level] = console[level];
      console[level] = function(...args) {
        // Forward to background script
        try {
          chrome.runtime.sendMessage({
            type: 'consoleLog',
            level,
            args: args.map(arg => {
              try {
                if (typeof arg === 'object') {
                  return JSON.stringify(arg, null, 2);
                }
                return String(arg);
              } catch (e) {
                return '[Unserializable]';
              }
            })
          });
        } catch (e) {
          // Extension context may be invalid
        }

        // Call original
        originalConsole[level].apply(console, args);
      };
    });
  }

  interceptConsoleLogs();

  /**
   * Find element by selector with multiple strategies
   */
  function findElement(selector) {
    // Try CSS selector first
    let element = document.querySelector(selector);
    if (element) return element;

    // Try XPath
    if (selector.startsWith('//') || selector.startsWith('(')) {
      const result = document.evaluate(
        selector,
        document,
        null,
        XPathResult.FIRST_ORDERED_NODE_TYPE,
        null
      );
      if (result.singleNodeValue) return result.singleNodeValue;
    }

    // Try data-testid
    element = document.querySelector(`[data-testid="${selector}"]`);
    if (element) return element;

    // Try by text content (partial match)
    const allElements = document.querySelectorAll('*');
    for (const el of allElements) {
      if (el.textContent && el.textContent.includes(selector) && el.children.length === 0) {
        return el;
      }
    }

    return null;
  }

  /**
   * Wait for an element to appear
   */
  async function waitForSelector(selector, timeout = 10000) {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      const element = findElement(selector);
      if (element) {
        return { success: true, data: getElementInfo(element) };
      }
      await new Promise(r => setTimeout(r, 100));
    }

    return { success: false, error: `Timeout waiting for: ${selector}` };
  }

  /**
   * Get information about an element
   */
  function getElementInfo(element) {
    const rect = element.getBoundingClientRect();
    return {
      tagName: element.tagName.toLowerCase(),
      id: element.id || null,
      className: element.className || null,
      textContent: element.textContent?.slice(0, 100) || null,
      value: element.value || null,
      href: element.href || null,
      src: element.src || null,
      isVisible: !!(rect.width && rect.height),
      rect: {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height
      },
      attributes: Object.fromEntries(
        Array.from(element.attributes || []).map(attr => [attr.name, attr.value])
      )
    };
  }

  /**
   * Click an element
   */
  async function clickElement(selector) {
    const element = findElement(selector);
    if (!element) {
      return { success: false, error: `Element not found: ${selector}` };
    }

    // Scroll into view if needed
    element.scrollIntoView({ behavior: 'instant', block: 'center' });
    await new Promise(r => setTimeout(r, 100));

    // Create and dispatch events
    const rect = element.getBoundingClientRect();
    const centerX = rect.x + rect.width / 2;
    const centerY = rect.y + rect.height / 2;

    const mousedown = new MouseEvent('mousedown', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: centerX,
      clientY: centerY
    });

    const mouseup = new MouseEvent('mouseup', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: centerX,
      clientY: centerY
    });

    const click = new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: centerX,
      clientY: centerY
    });

    element.dispatchEvent(mousedown);
    element.dispatchEvent(mouseup);
    element.dispatchEvent(click);

    // Also try native click
    if (typeof element.click === 'function') {
      element.click();
    }

    return { success: true, data: getElementInfo(element) };
  }

  /**
   * Fill an input field
   */
  async function fillElement(selector, value) {
    const element = findElement(selector);
    if (!element) {
      return { success: false, error: `Element not found: ${selector}` };
    }

    // Focus the element
    element.focus();

    // Clear existing value
    element.value = '';
    element.dispatchEvent(new Event('input', { bubbles: true }));

    // Set new value
    element.value = value;

    // Dispatch events
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));

    return { success: true, data: { value: element.value } };
  }

  /**
   * Type text character by character
   */
  async function typeText(selector, text, delay = 50) {
    const element = findElement(selector);
    if (!element) {
      return { success: false, error: `Element not found: ${selector}` };
    }

    element.focus();

    for (const char of text) {
      const keydown = new KeyboardEvent('keydown', {
        key: char,
        code: `Key${char.toUpperCase()}`,
        bubbles: true
      });

      const keypress = new KeyboardEvent('keypress', {
        key: char,
        code: `Key${char.toUpperCase()}`,
        bubbles: true
      });

      const input = new InputEvent('input', {
        data: char,
        inputType: 'insertText',
        bubbles: true
      });

      const keyup = new KeyboardEvent('keyup', {
        key: char,
        code: `Key${char.toUpperCase()}`,
        bubbles: true
      });

      element.dispatchEvent(keydown);
      element.dispatchEvent(keypress);
      element.value += char;
      element.dispatchEvent(input);
      element.dispatchEvent(keyup);

      if (delay > 0) {
        await new Promise(r => setTimeout(r, delay));
      }
    }

    return { success: true, data: { value: element.value } };
  }

  /**
   * Hover over an element
   */
  async function hoverElement(selector) {
    const element = findElement(selector);
    if (!element) {
      return { success: false, error: `Element not found: ${selector}` };
    }

    element.scrollIntoView({ behavior: 'instant', block: 'center' });

    const rect = element.getBoundingClientRect();
    const mouseenter = new MouseEvent('mouseenter', {
      bubbles: true,
      clientX: rect.x + rect.width / 2,
      clientY: rect.y + rect.height / 2
    });

    const mouseover = new MouseEvent('mouseover', {
      bubbles: true,
      clientX: rect.x + rect.width / 2,
      clientY: rect.y + rect.height / 2
    });

    element.dispatchEvent(mouseenter);
    element.dispatchEvent(mouseover);

    return { success: true, data: getElementInfo(element) };
  }

  /**
   * Select an option from a dropdown
   */
  async function selectOption(selector, value) {
    const element = findElement(selector);
    if (!element) {
      return { success: false, error: `Element not found: ${selector}` };
    }

    if (element.tagName.toLowerCase() !== 'select') {
      return { success: false, error: 'Element is not a select' };
    }

    element.value = value;
    element.dispatchEvent(new Event('change', { bubbles: true }));

    return { success: true, data: { value: element.value } };
  }

  /**
   * Press a keyboard key
   */
  async function pressKey(key) {
    const keydown = new KeyboardEvent('keydown', {
      key,
      code: key,
      bubbles: true
    });

    const keyup = new KeyboardEvent('keyup', {
      key,
      code: key,
      bubbles: true
    });

    document.activeElement.dispatchEvent(keydown);
    document.activeElement.dispatchEvent(keyup);

    return { success: true };
  }

  /**
   * Scroll the page
   */
  async function scroll(x, y) {
    window.scrollBy(x, y);
    await new Promise(r => setTimeout(r, 100));
    return { success: true, data: { scrollX: window.scrollX, scrollY: window.scrollY } };
  }

  /**
   * Evaluate JavaScript in page context
   */
  function evaluateScript(script) {
    try {
      const result = eval(script);
      return { success: true, data: result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * Get page information
   */
  function getPageInfo() {
    return {
      success: true,
      data: {
        url: window.location.href,
        title: document.title,
        readyState: document.readyState,
        doctype: document.doctype ? document.doctype.name : null,
        documentElement: {
          scrollWidth: document.documentElement.scrollWidth,
          scrollHeight: document.documentElement.scrollHeight,
          clientWidth: document.documentElement.clientWidth,
          clientHeight: document.documentElement.clientHeight
        }
      }
    };
  }

  /**
   * Query for elements
   */
  function querySelector(selector) {
    const element = findElement(selector);
    if (element) {
      return { success: true, data: getElementInfo(element) };
    }
    return { success: false, error: `Element not found: ${selector}` };
  }

  function querySelectorAll(selector) {
    const elements = document.querySelectorAll(selector);
    return {
      success: true,
      data: Array.from(elements).map(getElementInfo)
    };
  }

  // Listen for messages from background script
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    (async () => {
      let result;

      switch (message.action) {
        case 'click':
          result = await clickElement(message.selector);
          break;

        case 'fill':
          result = await fillElement(message.selector, message.value);
          break;

        case 'type':
          result = await typeText(message.selector, message.text, message.delay);
          break;

        case 'hover':
          result = await hoverElement(message.selector);
          break;

        case 'select':
          result = await selectOption(message.selector, message.value);
          break;

        case 'pressKey':
          result = await pressKey(message.key);
          break;

        case 'scroll':
          result = await scroll(message.x || 0, message.y || 0);
          break;

        case 'evaluate':
          result = evaluateScript(message.script);
          break;

        case 'getPageInfo':
          result = getPageInfo();
          break;

        case 'querySelector':
          result = querySelector(message.selector);
          break;

        case 'querySelectorAll':
          result = querySelectorAll(message.selector);
          break;

        case 'waitForSelector':
          result = await waitForSelector(message.selector, message.timeout);
          break;

        default:
          result = { success: false, error: `Unknown action: ${message.action}` };
      }

      sendResponse(result);
    })();

    return true; // Keep message channel open for async response
  });

})();
