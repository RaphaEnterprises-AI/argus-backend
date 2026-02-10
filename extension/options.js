/**
 * E2E Testing Agent - Options Page Script
 *
 * Manages extension settings stored in chrome.storage.sync.
 */

const DEFAULTS = {
  serverUrl: 'ws://localhost:8765',
  apiKey: '',
  autoConnect: true,
  brainUrl: 'https://argus-brain-production.up.railway.app',
  orgId: '',
  lastProjectId: '',
  maskInputs: true,
  recordCanvas: false,
  maxRecordingMs: 300000,
  feedConsoleErrors: false,
  feedNetworkErrors: false,
};

document.addEventListener('DOMContentLoaded', async () => {
  const fields = {
    serverUrl: document.getElementById('serverUrl'),
    apiKey: document.getElementById('apiKey'),
    autoConnect: document.getElementById('autoConnect'),
    brainUrl: document.getElementById('brainUrl'),
    orgId: document.getElementById('orgId'),
    lastProjectId: document.getElementById('lastProjectId'),
    maskInputs: document.getElementById('maskInputs'),
    recordCanvas: document.getElementById('recordCanvas'),
    maxRecordingMs: document.getElementById('maxRecordingMs'),
    feedConsoleErrors: document.getElementById('feedConsoleErrors'),
    feedNetworkErrors: document.getElementById('feedNetworkErrors'),
  };

  const statusEl = document.getElementById('status');

  // Load saved settings
  const settings = await chrome.storage.sync.get(DEFAULTS);

  fields.serverUrl.value = settings.serverUrl;
  fields.apiKey.value = settings.apiKey;
  fields.autoConnect.checked = settings.autoConnect;
  fields.brainUrl.value = settings.brainUrl;
  fields.orgId.value = settings.orgId;
  fields.lastProjectId.value = settings.lastProjectId;
  fields.maskInputs.checked = settings.maskInputs;
  fields.recordCanvas.checked = settings.recordCanvas;
  fields.maxRecordingMs.value = settings.maxRecordingMs;
  fields.feedConsoleErrors.checked = settings.feedConsoleErrors;
  fields.feedNetworkErrors.checked = settings.feedNetworkErrors;

  // Save
  document.getElementById('saveBtn').addEventListener('click', async () => {
    const newSettings = {
      serverUrl: fields.serverUrl.value || DEFAULTS.serverUrl,
      apiKey: fields.apiKey.value,
      autoConnect: fields.autoConnect.checked,
      brainUrl: fields.brainUrl.value || DEFAULTS.brainUrl,
      orgId: fields.orgId.value,
      lastProjectId: fields.lastProjectId.value,
      maskInputs: fields.maskInputs.checked,
      recordCanvas: fields.recordCanvas.checked,
      maxRecordingMs: parseInt(fields.maxRecordingMs.value) || DEFAULTS.maxRecordingMs,
      feedConsoleErrors: fields.feedConsoleErrors.checked,
      feedNetworkErrors: fields.feedNetworkErrors.checked,
    };

    await chrome.storage.sync.set(newSettings);

    statusEl.textContent = 'Settings saved successfully!';
    statusEl.className = 'status success';
    setTimeout(() => { statusEl.className = 'status'; }, 3000);
  });

  // Reset
  document.getElementById('resetBtn').addEventListener('click', async () => {
    await chrome.storage.sync.set(DEFAULTS);

    fields.serverUrl.value = DEFAULTS.serverUrl;
    fields.apiKey.value = DEFAULTS.apiKey;
    fields.autoConnect.checked = DEFAULTS.autoConnect;
    fields.brainUrl.value = DEFAULTS.brainUrl;
    fields.orgId.value = DEFAULTS.orgId;
    fields.lastProjectId.value = DEFAULTS.lastProjectId;
    fields.maskInputs.checked = DEFAULTS.maskInputs;
    fields.recordCanvas.checked = DEFAULTS.recordCanvas;
    fields.maxRecordingMs.value = DEFAULTS.maxRecordingMs;
    fields.feedConsoleErrors.checked = DEFAULTS.feedConsoleErrors;
    fields.feedNetworkErrors.checked = DEFAULTS.feedNetworkErrors;

    statusEl.textContent = 'Settings reset to defaults.';
    statusEl.className = 'status success';
    setTimeout(() => { statusEl.className = 'status'; }, 3000);
  });
});
