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
const DEFAULT_WS_URL = 'ws://localhost:8765';

/**
 * Get configurable settings from chrome.storage.sync
 */
async function getSettings() {
  const defaults = {
    serverUrl: DEFAULT_WS_URL,
    apiKey: '',
    autoConnect: true,
  };
  try {
    const stored = await chrome.storage.sync.get(defaults);
    return stored;
  } catch {
    return defaults;
  }
}

// Tab tracking
const managedTabs = new Map();
const consoleLogsByTab = new Map();
const networkRequestsByTab = new Map();

/**
 * Connect to the Python testing agent via WebSocket
 */
async function connectToAgent() {
  if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
    return;
  }

  const settings = await getSettings();
  const wsUrl = settings.serverUrl || DEFAULT_WS_URL;

  console.log(`[E2E Agent] Connecting to testing agent at ${wsUrl}...`);

  wsConnection = new WebSocket(wsUrl);

  wsConnection.onopen = () => {
    console.log('[E2E Agent] Connected to testing agent');
    isConnected = true;
    reconnectAttempts = 0;

    // Send authenticated connection acknowledgment
    const authPayload = {
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
        'evaluate',
        'recording',
        'self_healing',
        'visual_comparison',
      ],
    };

    // Include API key for authentication if configured
    if (settings.apiKey) {
      authPayload.auth = { api_key: settings.apiKey };
    }

    sendToAgent(authPayload);

    // Persist connection state
    chrome.storage.local.set({ connected: true });
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
    chrome.storage.local.set({ connected: false });

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

    case 'compareScreenshot':
      return await compareScreenshot(params.baselineId, params);

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

// =========================================================================
// Network Request Monitoring via webRequest API
// =========================================================================

/**
 * Track completed network requests per tab.
 * Uses chrome.webRequest.onCompleted to capture real request data.
 */
const MAX_REQUESTS_PER_TAB = 200;

chrome.webRequest.onCompleted.addListener(
  (details) => {
    const tabId = details.tabId;
    if (tabId < 0) return; // Ignore non-tab requests (e.g., service worker)

    if (!networkRequestsByTab.has(tabId)) {
      networkRequestsByTab.set(tabId, []);
    }

    const requests = networkRequestsByTab.get(tabId);
    requests.push({
      url: details.url,
      method: details.method,
      status: details.statusCode,
      type: details.type,
      timestamp: details.timeStamp,
      fromCache: details.fromCache,
      ip: details.ip || null,
    });

    // Cap stored requests to avoid memory bloat
    if (requests.length > MAX_REQUESTS_PER_TAB) {
      requests.splice(0, requests.length - MAX_REQUESTS_PER_TAB);
    }
  },
  { urls: ["<all_urls>"] }
);

// Also capture failed requests for error tracking + event feed
chrome.webRequest.onErrorOccurred.addListener(
  (details) => {
    const tabId = details.tabId;
    if (tabId < 0) return;

    if (!networkRequestsByTab.has(tabId)) {
      networkRequestsByTab.set(tabId, []);
    }

    networkRequestsByTab.get(tabId).push({
      url: details.url,
      method: details.method,
      status: 0,
      type: details.type,
      timestamp: details.timeStamp,
      error: details.error,
    });

    // Event feed: send network errors to backend
    sendEventToBackend('network_error', {
      url: details.url,
      method: details.method,
      error: details.error,
      type: details.type,
    });
  },
  { urls: ["<all_urls>"] }
);

// =========================================================================
// Recording Management
// =========================================================================

let activeRecordingTabId = null;

/**
 * Inject recorder.js into a tab and start recording.
 */
async function startRecording(tabId, options = {}) {
  try {
    const targetTabId = tabId || (await getActiveTab()).data.id;

    // Inject the recorder script
    await chrome.scripting.executeScript({
      target: { tabId: targetTabId },
      files: ['recorder.js'],
    });

    // Small delay for script to load
    await new Promise(r => setTimeout(r, 200));

    // Start recording
    const result = await chrome.tabs.sendMessage(targetTabId, {
      action: 'startRecording',
      options,
    });

    if (result.success) {
      activeRecordingTabId = targetTabId;
      chrome.storage.local.set({
        recording: true,
        recordingTabId: targetTabId,
        recordingStartTime: Date.now(),
      });
    }

    return result;
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Stop recording and upload to Skopaq backend.
 */
async function stopRecordingAndUpload(projectId) {
  try {
    if (!activeRecordingTabId) {
      return { success: false, error: 'No active recording' };
    }

    const result = await chrome.tabs.sendMessage(activeRecordingTabId, {
      action: 'stopRecording',
    });

    if (!result.success) return result;

    const tabId = activeRecordingTabId;
    activeRecordingTabId = null;
    chrome.storage.local.set({ recording: false, recordingTabId: null });

    // Upload to backend
    const settings = await getSettings();
    const brainUrl = settings.brainUrl || 'https://skopaq-brain-production.up.railway.app';

    const uploadPayload = {
      project_id: projectId,
      recording: {
        events: result.data.events,
        metadata: result.data.metadata,
      },
    };

    const headers = { 'Content-Type': 'application/json' };
    if (settings.apiKey) {
      headers['X-API-Key'] = settings.apiKey;
    }

    const response = await fetch(`${brainUrl}/api/v1/recording/upload`, {
      method: 'POST',
      headers,
      body: JSON.stringify(uploadPayload),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return { success: false, error: `Upload failed (${response.status}): ${errorText}` };
    }

    const uploadResult = await response.json();
    return {
      success: true,
      data: {
        ...result.data.metadata,
        recording_id: uploadResult.recording_id || uploadResult.id,
        uploaded: true,
      },
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// =========================================================================
// Self-Healing - Query backend for alternative selectors
// =========================================================================

/**
 * Ask the Skopaq healing API for alternative selectors when findElement fails.
 * Returns an array of {selector, confidence} alternatives.
 */
async function requestHealing(failedSelector, pageUrl) {
  try {
    const settings = await getSettings();
    if (!settings.apiKey) return { success: false, error: 'No API key configured' };

    const brainUrl = settings.brainUrl || 'https://skopaq-brain-production.up.railway.app';

    // Use the stored org_id from settings (set via options page)
    const orgId = settings.orgId;
    if (!orgId) return { success: false, error: 'No org_id configured' };

    const response = await fetch(
      `${brainUrl}/api/v1/healing/organizations/${orgId}/suggest`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': settings.apiKey,
        },
        body: JSON.stringify({
          error_message: `Element not found: ${failedSelector}`,
          error_type: 'selector_not_found',
          selector: failedSelector,
          context: { page_url: pageUrl },
        }),
      }
    );

    if (!response.ok) {
      return { success: false, error: `Healing API returned ${response.status}` };
    }

    const data = await response.json();

    // Transform suggestions into alternative selectors
    const alternatives = (data.suggestions || [])
      .filter(s => s.fix_type === 'update_selector' && s.new_value)
      .map(s => ({ selector: s.new_value, confidence: s.confidence }));

    return { success: alternatives.length > 0, alternatives };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// =========================================================================
// Visual Comparison - Capture screenshot and compare via backend
// =========================================================================

/**
 * Capture a screenshot of the active tab and upload it to the visual AI
 * comparison endpoint with a baseline reference.
 */
async function compareScreenshot(baselineId, options = {}) {
  try {
    const settings = await getSettings();
    if (!settings.apiKey) return { success: false, error: 'No API key configured' };

    const brainUrl = settings.brainUrl || 'https://skopaq-brain-production.up.railway.app';

    // Capture the current tab screenshot
    const screenshotResult = await captureScreenshot(null, { format: 'png', quality: 100 });
    if (!screenshotResult.success) return screenshotResult;

    // Upload current screenshot as a snapshot first
    const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const currentUrl = activeTab ? activeTab.url : 'unknown';

    // The compare endpoint handles capture internally when given a URL.
    // We pass the current page URL so the backend captures it server-side.
    const response = await fetch(`${brainUrl}/api/v1/visual/compare`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': settings.apiKey,
      },
      body: JSON.stringify({
        baseline_id: baselineId,
        current_url: currentUrl,
        sensitivity: options.sensitivity || 'medium',
        browser: 'chromium',
        project_id: options.projectId || null,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return { success: false, error: `Visual compare failed (${response.status}): ${errorText}` };
    }

    const result = await response.json();
    return { success: true, data: result };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

// =========================================================================
// Event Feed - Send console/network errors to Skopaq backend
// =========================================================================

/**
 * Send an error event to the Skopaq backend Event Gateway.
 * Feeds into the AI learning loop via Kafka.
 */
async function sendEventToBackend(eventType, eventData) {
  try {
    const settings = await getSettings();

    // Only send if event feed is enabled
    if (eventType === 'console_error' && !settings.feedConsoleErrors) return;
    if (eventType === 'network_error' && !settings.feedNetworkErrors) return;
    if (!settings.apiKey) return; // Need auth for event gateway

    const brainUrl = settings.brainUrl || 'https://skopaq-brain-production.up.railway.app';

    await fetch(`${brainUrl}/events/test.failed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': settings.apiKey,
      },
      body: JSON.stringify({
        data: {
          source: 'browser_extension',
          event_type: eventType,
          ...eventData,
          timestamp: new Date().toISOString(),
        },
      }),
    });
  } catch {
    // Fire-and-forget: don't block on event feed failures
  }
}

// Listen for messages from content scripts and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Handle connect request from popup
  if (message.action === 'connect') {
    reconnectAttempts = 0;
    connectToAgent();
    return true;
  }

  // Handle recording commands from popup
  if (message.action === 'startRecording') {
    startRecording(message.tabId, message.options).then(sendResponse);
    return true; // async
  }

  if (message.action === 'stopRecording') {
    stopRecordingAndUpload(message.projectId).then(sendResponse);
    return true; // async
  }

  if (message.action === 'getRecordingStatus') {
    if (activeRecordingTabId) {
      chrome.tabs.sendMessage(activeRecordingTabId, { action: 'getRecordingStatus' })
        .then(sendResponse)
        .catch(() => sendResponse({ success: true, data: { isRecording: false } }));
    } else {
      sendResponse({ success: true, data: { isRecording: false } });
    }
    return true; // async
  }

  // Handle self-healing request from content script
  if (message.type === 'requestHealing') {
    requestHealing(message.selector, message.pageUrl).then(sendResponse);
    return true; // async
  }

  // Handle visual comparison request from content script
  if (message.type === 'compareScreenshot') {
    compareScreenshot(message.baselineId, {
      sensitivity: message.sensitivity,
      projectId: message.projectId,
    }).then(sendResponse);
    return true; // async
  }

  // Handle recording auto-stop notification from recorder.js
  if (message.type === 'recordingStopped') {
    activeRecordingTabId = null;
    chrome.storage.local.set({ recording: false, recordingTabId: null });
  }

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

    // Event feed: send console errors to backend
    if (message.level === 'error') {
      sendEventToBackend('console_error', {
        error_message: message.args.join(' '),
        page_url: sender.tab?.url || 'unknown',
        tab_id: tabId,
      });
    }
  }

  return true;
});


// Start connection on install/update
chrome.runtime.onInstalled.addListener(async () => {
  console.log('[E2E Agent] Extension installed');
  const settings = await getSettings();
  if (settings.autoConnect) {
    connectToAgent();
  }
});

// Also try to connect on startup
chrome.runtime.onStartup.addListener(async () => {
  const settings = await getSettings();
  if (settings.autoConnect) {
    connectToAgent();
  }
});

// Initial connection attempt (respects autoConnect setting)
(async () => {
  const settings = await getSettings();
  if (settings.autoConnect) {
    connectToAgent();
  }
})();
