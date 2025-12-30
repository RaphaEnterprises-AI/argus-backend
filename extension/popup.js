/**
 * E2E Testing Agent - Popup Script
 */

document.addEventListener('DOMContentLoaded', async () => {
  const statusEl = document.getElementById('status');
  const statusText = document.getElementById('status-text');
  const actionCount = document.getElementById('action-count');
  const tabCount = document.getElementById('tab-count');
  const logsEl = document.getElementById('logs');
  const connectBtn = document.getElementById('connect-btn');
  const testBtn = document.getElementById('test-btn');

  // Get current state from background
  async function updateStatus() {
    try {
      const tabs = await chrome.tabs.query({});
      tabCount.textContent = tabs.length;

      // Check storage for stats
      const stats = await chrome.storage.local.get(['actionCount', 'connected', 'logs']);

      if (stats.connected) {
        statusEl.className = 'status connected';
        statusText.textContent = 'Connected to Agent';
        connectBtn.textContent = 'Reconnect';
      } else {
        statusEl.className = 'status disconnected';
        statusText.textContent = 'Disconnected';
        connectBtn.textContent = 'Connect to Agent';
      }

      actionCount.textContent = stats.actionCount || 0;

      // Update logs
      if (stats.logs && stats.logs.length > 0) {
        logsEl.innerHTML = stats.logs.slice(-10).map(log => `
          <div class="log-entry">
            <span class="log-time">${new Date(log.time).toLocaleTimeString()}</span>
            <span class="log-message">${log.message}</span>
          </div>
        `).join('');
      }
    } catch (error) {
      console.error('Error updating status:', error);
    }
  }

  // Connect button
  connectBtn.addEventListener('click', async () => {
    // Send message to background to connect
    chrome.runtime.sendMessage({ action: 'connect' });
    addLog('Connecting to agent...');
    setTimeout(updateStatus, 1000);
  });

  // Test button
  testBtn.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    // Send test command to content script
    const result = await chrome.tabs.sendMessage(tab.id, { action: 'getPageInfo' });

    if (result.success) {
      addLog(`Page: ${result.data.title}`);
      addLog(`URL: ${result.data.url}`);
    } else {
      addLog(`Error: ${result.error}`);
    }
  });

  function addLog(message) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
      <span class="log-time">${new Date().toLocaleTimeString()}</span>
      <span class="log-message">${message}</span>
    `;
    logsEl.insertBefore(entry, logsEl.firstChild);

    // Keep only last 10 logs
    while (logsEl.children.length > 10) {
      logsEl.removeChild(logsEl.lastChild);
    }
  }

  // Initial update
  updateStatus();

  // Listen for updates
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local') {
      updateStatus();
    }
  });
});
