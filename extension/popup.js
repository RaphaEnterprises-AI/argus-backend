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
  const recordBtn = document.getElementById('record-btn');
  const recordingInfo = document.getElementById('recording-info');
  const settingsLink = document.getElementById('settings-link');

  let isRecording = false;

  // Get current state from background
  async function updateStatus() {
    try {
      const tabs = await chrome.tabs.query({});
      tabCount.textContent = tabs.length;

      // Check storage for stats
      const stats = await chrome.storage.local.get([
        'actionCount', 'connected', 'logs', 'recording', 'recordingStartTime',
      ]);

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

      // Update recording state
      if (stats.recording) {
        isRecording = true;
        recordBtn.textContent = 'Stop Recording';
        recordBtn.classList.add('recording');
        const elapsed = Math.floor((Date.now() - stats.recordingStartTime) / 1000);
        recordingInfo.style.display = 'block';
        recordingInfo.textContent = `Recording... ${elapsed}s`;
      } else {
        isRecording = false;
        recordBtn.textContent = 'Start Recording';
        recordBtn.classList.remove('recording');
        recordingInfo.style.display = 'none';
      }

      // Update logs using safe DOM methods
      if (stats.logs && stats.logs.length > 0) {
        logsEl.textContent = ''; // Clear safely
        stats.logs.slice(-10).forEach(log => {
          const entry = document.createElement('div');
          entry.className = 'log-entry';
          const timeSpan = document.createElement('span');
          timeSpan.className = 'log-time';
          timeSpan.textContent = new Date(log.time).toLocaleTimeString();
          const msgSpan = document.createElement('span');
          msgSpan.className = 'log-message';
          msgSpan.textContent = log.message;
          entry.appendChild(timeSpan);
          entry.appendChild(msgSpan);
          logsEl.appendChild(entry);
        });
      }
    } catch (error) {
      console.error('Error updating status:', error);
    }
  }

  // Connect button
  connectBtn.addEventListener('click', async () => {
    chrome.runtime.sendMessage({ action: 'connect' });
    addLog('Connecting to agent...');
    setTimeout(updateStatus, 1000);
  });

  // Test button
  testBtn.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    try {
      const result = await chrome.tabs.sendMessage(tab.id, { action: 'getPageInfo' });
      if (result.success) {
        addLog(`Page: ${result.data.title}`);
        addLog(`URL: ${result.data.url}`);
      } else {
        addLog(`Error: ${result.error}`);
      }
    } catch (error) {
      addLog(`Error: ${error.message}`);
    }
  });

  // Recording button
  recordBtn.addEventListener('click', async () => {
    if (isRecording) {
      // Stop recording
      addLog('Stopping recording...');
      recordBtn.disabled = true;

      const settings = await chrome.storage.sync.get({ lastProjectId: '' });
      const projectId = settings.lastProjectId || 'default';

      const result = await chrome.runtime.sendMessage({
        action: 'stopRecording',
        projectId,
      });

      recordBtn.disabled = false;

      if (result.success) {
        addLog(`Recording uploaded (${result.data.eventCount} events, ${(result.data.duration / 1000).toFixed(1)}s)`);
        if (result.data.recording_id) {
          addLog(`Recording ID: ${result.data.recording_id}`);
        }
      } else {
        addLog(`Stop failed: ${result.error}`);
      }

      updateStatus();
    } else {
      // Start recording
      addLog('Starting recording...');

      const settings = await chrome.storage.sync.get({
        maskInputs: true,
        recordCanvas: false,
        maxRecordingMs: 300000,
      });

      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      const result = await chrome.runtime.sendMessage({
        action: 'startRecording',
        tabId: tab.id,
        options: {
          maskInputs: settings.maskInputs,
          recordCanvas: settings.recordCanvas,
          maxDuration: settings.maxRecordingMs,
        },
      });

      if (result.success) {
        addLog('Recording started');
      } else {
        addLog(`Record failed: ${result.error}`);
      }

      updateStatus();
    }
  });

  // Settings link
  settingsLink.addEventListener('click', () => {
    chrome.runtime.openOptionsPage();
  });

  function addLog(message) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const timeSpan = document.createElement('span');
    timeSpan.className = 'log-time';
    timeSpan.textContent = new Date().toLocaleTimeString();
    const msgSpan = document.createElement('span');
    msgSpan.className = 'log-message';
    msgSpan.textContent = message;
    entry.appendChild(timeSpan);
    entry.appendChild(msgSpan);
    logsEl.insertBefore(entry, logsEl.firstChild);

    // Keep only last 10 logs
    while (logsEl.children.length > 10) {
      logsEl.removeChild(logsEl.lastChild);
    }
  }

  // Initial update
  updateStatus();

  // Periodic update for recording timer
  setInterval(updateStatus, 2000);

  // Listen for storage changes
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local') {
      updateStatus();
    }
  });
});
