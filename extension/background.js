/**
 * E2E Testing Agent - Background Service Worker
 *
 * This service worker handles:
 * - WebSocket connection to the Python testing agent
 * - Tab management and navigation
 * - Screenshot capture
 * - Message routing to content scripts
 * - Console log collection
 * - Network request monitoring
 */

// Connection state
let wsConnection = null;
let isConnected = false;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const WS_URL = 'ws://localhost:8765';

// Tab tracking
const managedTabs = new Map();
const consoleLogsByTab = new Map();
const networkRequestsByTab = new Map();

/**
 * Connect to the Python testing agent via WebSocket
 */
function connectToAgent() {
  if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
    return;
  }

  console.log('[E2E Agent] Connecting to testing agent...');

  wsConnection = new WebSocket(WS_URL);

  wsConnection.onopen = () => {
    console.log('[E2E Agent] Connected to testing agent');
    isConnected = true;
    reconnectAttempts = 0;

    // Send connection acknowledgment
    sendToAgent({
      type: 'connected',
      timestamp: Date.now(),
      capabilities: [
        'navigate',
        'click',
        'fill',
        'screenshot',
        'console_logs',
        'network_requests',
        'dom_query',
        'evaluate'
      ]
    });
  };

  wsConnection.onmessage = async (event) => {
    try {
      const message = JSON.parse(event.data);
      console.log('[E2E Agent] Received:', message.action);

      const result = await handleAgentMessage(message);
      sendToAgent({
        type: 'response',
        requestId: message.requestId,
        success: result.success,
        data: result.data,
        error: result.error
      });
    } catch (error) {
      console.error('[E2E Agent] Error handling message:', error);
      sendToAgent({
        type: 'error',
        error: error.message
      });
    }
  };

  wsConnection.onclose = () => {
    console.log('[E2E Agent] Disconnected from testing agent');
    isConnected = false;

    // Attempt reconnection
    if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
      reconnectAttempts++;
      const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
      console.log(`[E2E Agent] Reconnecting in ${delay}ms...`);
      setTimeout(connectToAgent, delay);
    }
  };

  wsConnection.onerror = (error) => {
    console.error('[E2E Agent] WebSocket error:', error);
  };
}

/**
 * Send message to the Python testing agent
 */
function sendToAgent(message) {
  if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
    wsConnection.send(JSON.stringify(message));
  }
}

/**
 * Handle messages from the testing agent
 */
async function handleAgentMessage(message) {
  const { action, tabId, params } = message;

  switch (action) {
    case 'navigate':
      return await navigateTab(tabId, params.url, params.waitUntil);

    case 'click':
      return await sendToContentScript(tabId, { action: 'click', selector: params.selector });

    case 'fill':
      return await sendToContentScript(tabId, { action: 'fill', selector: params.selector, value: params.value });

    case 'type':
      return await sendToContentScript(tabId, { action: 'type', selector: params.selector, text: params.text, delay: params.delay });

    case 'screenshot':
      return await captureScreenshot(tabId, params);

    case 'getConsoleLogs':
      return { success: true, data: consoleLogsByTab.get(tabId) || [] };

    case 'getNetworkRequests':
      return { success: true, data: networkRequestsByTab.get(tabId) || [] };

    case 'evaluate':
      return await sendToContentScript(tabId, { action: 'evaluate', script: params.script });

    case 'querySelector':
      return await sendToContentScript(tabId, { action: 'querySelector', selector: params.selector });

    case 'querySelectorAll':
      return await sendToContentScript(tabId, { action: 'querySelectorAll', selector: params.selector });

    case 'getPageInfo':
      return await sendToContentScript(tabId, { action: 'getPageInfo' });

    case 'waitForSelector':
      return await sendToContentScript(tabId, { action: 'waitForSelector', selector: params.selector, timeout: params.timeout });

    case 'scroll':
      return await sendToContentScript(tabId, { action: 'scroll', x: params.x, y: params.y });

    case 'hover':
      return await sendToContentScript(tabId, { action: 'hover', selector: params.selector });

    case 'select':
      return await sendToContentScript(tabId, { action: 'select', selector: params.selector, value: params.value });

    case 'pressKey':
      return await sendToContentScript(tabId, { action: 'pressKey', key: params.key });

    case 'createTab':
      return await createTab(params.url);

    case 'closeTab':
      return await closeTab(tabId);

    case 'getTabs':
      return await getOpenTabs();

    case 'getActiveTab':
      return await getActiveTab();

    default:
      return { success: false, error: `Unknown action: ${action}` };
  }
}

/**
 * Navigate a tab to a URL
 */
async function navigateTab(tabId, url, waitUntil = 'load') {
  try {
    const targetTabId = tabId || (await getActiveTab()).data.id;

    await chrome.tabs.update(targetTabId, { url });

    // Wait for page load
    await new Promise((resolve) => {
      const listener = (updatedTabId, changeInfo) => {
        if (updatedTabId === targetTabId && changeInfo.status === 'complete') {
          chrome.tabs.onUpdated.removeListener(listener);
          resolve();
        }
      };
      chrome.tabs.onUpdated.addListener(listener);

      // Timeout after 30 seconds
      setTimeout(() => {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }, 30000);
    });

    return { success: true, data: { url, tabId: targetTabId } };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Capture a screenshot of a tab
 */
async function captureScreenshot(tabId, params = {}) {
  try {
    const targetTabId = tabId || (await getActiveTab()).data.id;

    // Ensure the tab is active
    await chrome.tabs.update(targetTabId, { active: true });

    // Small delay to ensure rendering
    await new Promise(r => setTimeout(r, 100));

    const dataUrl = await chrome.tabs.captureVisibleTab(null, {
      format: params.format || 'png',
      quality: params.quality || 100
    });

    return {
      success: true,
      data: {
        dataUrl,
        timestamp: Date.now()
      }
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Send a message to the content script in a tab
 */
async function sendToContentScript(tabId, message) {
  try {
    const targetTabId = tabId || (await getActiveTab()).data.id;
    const response = await chrome.tabs.sendMessage(targetTabId, message);
    return response;
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Create a new tab
 */
async function createTab(url) {
  try {
    const tab = await chrome.tabs.create({ url: url || 'about:blank' });
    managedTabs.set(tab.id, { created: Date.now(), url });
    return { success: true, data: { tabId: tab.id, url: tab.url } };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Close a tab
 */
async function closeTab(tabId) {
  try {
    await chrome.tabs.remove(tabId);
    managedTabs.delete(tabId);
    consoleLogsByTab.delete(tabId);
    networkRequestsByTab.delete(tabId);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Get all open tabs
 */
async function getOpenTabs() {
  try {
    const tabs = await chrome.tabs.query({});
    return {
      success: true,
      data: tabs.map(t => ({
        id: t.id,
        url: t.url,
        title: t.title,
        active: t.active
      }))
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Get the active tab
 */
async function getActiveTab() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab) {
      return {
        success: true,
        data: {
          id: tab.id,
          url: tab.url,
          title: tab.title
        }
      };
    }
    return { success: false, error: 'No active tab found' };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'consoleLog') {
    // Store console logs per tab
    const tabId = sender.tab.id;
    if (!consoleLogsByTab.has(tabId)) {
      consoleLogsByTab.set(tabId, []);
    }
    consoleLogsByTab.get(tabId).push({
      level: message.level,
      args: message.args,
      timestamp: Date.now()
    });

    // Forward to agent if connected
    if (isConnected) {
      sendToAgent({
        type: 'consoleLog',
        tabId,
        level: message.level,
        args: message.args,
        timestamp: Date.now()
      });
    }
  }

  return true;
});

// Listen for extension icon click
chrome.action.onClicked.addListener(() => {
  if (!isConnected) {
    connectToAgent();
  }
});

// Start connection on install/update
chrome.runtime.onInstalled.addListener(() => {
  console.log('[E2E Agent] Extension installed');
  connectToAgent();
});

// Also try to connect on startup
chrome.runtime.onStartup.addListener(() => {
  connectToAgent();
});

// Initial connection attempt
connectToAgent();
