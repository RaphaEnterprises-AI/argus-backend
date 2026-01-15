/**
 * Argus Browser Worker
 *
 * MCP-compatible browser automation service using Playwright.
 * Provides /observe, /act, /test endpoints for AI-driven browser control.
 */

const express = require('express');
const { chromium, firefox, webkit } = require('playwright');
const { v4: uuidv4 } = require('uuid');
const promClient = require('prom-client');
const winston = require('winston');

// Logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [new winston.transports.Console()]
});

// Configuration
const CONFIG = {
  port: parseInt(process.env.PORT || '3000'),
  metricsPort: parseInt(process.env.METRICS_PORT || '9090'),
  poolSize: parseInt(process.env.BROWSER_POOL_SIZE || '2'),
  sessionTimeout: parseInt(process.env.SESSION_TIMEOUT_MS || '300000'),
  defaultTimeout: parseInt(process.env.DEFAULT_TIMEOUT_MS || '30000'),
  headless: process.env.HEADLESS !== 'false',
  podName: process.env.POD_NAME || 'local',
  podIp: process.env.POD_IP || 'localhost'
};

// Express apps
const app = express();
const metricsApp = express();
app.use(express.json({ limit: '50mb' }));

// Prometheus metrics
const register = new promClient.Registry();
promClient.collectDefaultMetrics({ register });

const metrics = {
  sessions: new promClient.Counter({
    name: 'browser_sessions_total',
    help: 'Total browser sessions created',
    registers: [register]
  }),
  activeSessions: new promClient.Gauge({
    name: 'browser_sessions_active',
    help: 'Currently active browser sessions',
    registers: [register]
  }),
  actionDuration: new promClient.Histogram({
    name: 'browser_action_duration_seconds',
    help: 'Browser action duration in seconds',
    labelNames: ['action', 'status'],
    buckets: [0.1, 0.5, 1, 2, 5, 10, 30, 60],
    registers: [register]
  }),
  errors: new promClient.Counter({
    name: 'browser_errors_total',
    help: 'Total browser errors',
    labelNames: ['type'],
    registers: [register]
  })
};

// Browser pool
const browserPool = [];
const sessions = new Map();

// Browser types
const BROWSERS = {
  chromium: chromium,
  firefox: firefox,
  webkit: webkit,
  chrome: chromium
};

/**
 * Initialize browser pool
 */
async function initBrowserPool() {
  logger.info('Initializing browser pool', { poolSize: CONFIG.poolSize });

  for (let i = 0; i < CONFIG.poolSize; i++) {
    try {
      const browser = await chromium.launch({
        headless: CONFIG.headless,
        args: [
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--disable-dev-shm-usage',
          '--disable-gpu',
          '--disable-software-rasterizer',
          '--disable-extensions',
          '--disable-background-timer-throttling',
          '--disable-backgrounding-occluded-windows',
          '--disable-renderer-backgrounding'
        ]
      });

      browserPool.push({
        id: i,
        browser,
        inUse: false,
        lastUsed: Date.now()
      });

      logger.info('Browser initialized', { poolIndex: i });
    } catch (error) {
      logger.error('Failed to initialize browser', { poolIndex: i, error: error.message });
    }
  }

  logger.info('Browser pool initialized', { size: browserPool.length });
}

/**
 * Get an available browser from the pool
 */
function acquireBrowser() {
  const available = browserPool.find(b => !b.inUse && b.browser.isConnected());
  if (available) {
    available.inUse = true;
    available.lastUsed = Date.now();
    return available;
  }
  return null;
}

/**
 * Release a browser back to the pool
 */
function releaseBrowser(poolEntry) {
  if (poolEntry) {
    poolEntry.inUse = false;
    poolEntry.lastUsed = Date.now();
  }
}

/**
 * Extract interactive elements from page
 */
async function extractElements(page) {
  return await page.evaluate(() => {
    const elements = [];
    const interactiveSelectors = [
      'a[href]',
      'button',
      'input',
      'select',
      'textarea',
      '[role="button"]',
      '[role="link"]',
      '[role="checkbox"]',
      '[role="radio"]',
      '[role="switch"]',
      '[role="tab"]',
      '[role="menuitem"]',
      '[onclick]',
      '[type="submit"]'
    ].join(', ');

    document.querySelectorAll(interactiveSelectors).forEach((el, idx) => {
      const rect = el.getBoundingClientRect();

      // Skip invisible elements
      if (rect.width === 0 || rect.height === 0) return;
      if (window.getComputedStyle(el).visibility === 'hidden') return;
      if (window.getComputedStyle(el).display === 'none') return;

      // Build selector with priority
      let selector = '';
      let confidence = 0.5;

      if (el.id) {
        selector = `#${el.id}`;
        confidence = 0.95;
      } else if (el.getAttribute('data-testid')) {
        selector = `[data-testid="${el.getAttribute('data-testid')}"]`;
        confidence = 0.95;
      } else if (el.getAttribute('data-test')) {
        selector = `[data-test="${el.getAttribute('data-test')}"]`;
        confidence = 0.9;
      } else if (el.name && el.tagName !== 'A') {
        selector = `[name="${el.name}"]`;
        confidence = 0.85;
      } else if (el.getAttribute('aria-label')) {
        selector = `[aria-label="${el.getAttribute('aria-label')}"]`;
        confidence = 0.8;
      } else {
        // Fallback to text content
        const text = el.textContent?.trim().slice(0, 30);
        if (text) {
          selector = `text="${text}"`;
          confidence = 0.7;
        } else {
          selector = `${el.tagName.toLowerCase()}:nth-of-type(${idx + 1})`;
          confidence = 0.5;
        }
      }

      // Determine element type
      let type = el.tagName.toLowerCase();
      if (el.type) type = el.type;
      if (el.role) type = el.role;

      elements.push({
        selector,
        type,
        tagName: el.tagName.toLowerCase(),
        text: el.textContent?.trim().slice(0, 100) || '',
        value: el.value || '',
        placeholder: el.placeholder || '',
        description: el.getAttribute('aria-label') || el.title || el.placeholder || el.textContent?.trim().slice(0, 50) || '',
        confidence,
        bounds: {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          width: Math.round(rect.width),
          height: Math.round(rect.height)
        },
        attributes: {
          id: el.id || null,
          class: el.className || null,
          name: el.name || null,
          href: el.href || null,
          type: el.type || null,
          role: el.role || null
        }
      });
    });

    return elements;
  });
}

/**
 * Parse action instruction and execute
 */
async function executeAction(page, instruction) {
  const lower = instruction.toLowerCase();
  const actions = [];

  // Click action
  if (lower.includes('click')) {
    const match = instruction.match(/click\s+(?:on\s+)?(?:the\s+)?["']?([^"']+?)["']?\s*$/i) ||
                  instruction.match(/click\s+(?:on\s+)?(?:the\s+)?(.+)/i);

    if (match) {
      const target = match[1].trim();
      const selectors = [
        `text="${target}"`,
        `button:has-text("${target}")`,
        `a:has-text("${target}")`,
        `[aria-label="${target}"]`,
        `[title="${target}"]`,
        `#${target}`,
        `[data-testid="${target}"]`,
        target // Direct selector
      ];

      for (const sel of selectors) {
        try {
          await page.click(sel, { timeout: 5000 });
          actions.push({ action: 'click', selector: sel, success: true });
          break;
        } catch (e) {
          // Try next selector
        }
      }

      if (actions.length === 0) {
        actions.push({ action: 'click', selector: target, success: false, error: 'Element not found' });
      }
    }
  }

  // Type/Fill action
  else if (lower.includes('type') || lower.includes('fill') || lower.includes('enter')) {
    const match = instruction.match(/(?:type|fill|enter)\s+["']([^"']+)["']\s+(?:in|into)\s+(?:the\s+)?(.+)/i) ||
                  instruction.match(/(?:type|fill|enter)\s+(.+?)\s+(?:in|into)\s+(?:the\s+)?(.+)/i);

    if (match) {
      const [, value, target] = match;
      const selectors = [
        `input[placeholder*="${target}" i]`,
        `input[name*="${target}" i]`,
        `textarea[placeholder*="${target}" i]`,
        `[aria-label*="${target}" i]`,
        `#${target}`,
        target
      ];

      for (const sel of selectors) {
        try {
          await page.fill(sel, value, { timeout: 5000 });
          actions.push({ action: 'fill', selector: sel, value, success: true });
          break;
        } catch (e) {
          // Try next selector
        }
      }

      if (actions.length === 0) {
        actions.push({ action: 'fill', selector: target, value, success: false, error: 'Element not found' });
      }
    }
  }

  // Navigate action
  else if (lower.includes('navigate') || lower.includes('go to') || lower.includes('open')) {
    const match = instruction.match(/(?:navigate|go)\s+to\s+(.+)/i) ||
                  instruction.match(/open\s+(.+)/i);

    if (match) {
      let targetUrl = match[1].trim();
      if (!targetUrl.startsWith('http')) {
        const currentUrl = new URL(page.url());
        targetUrl = new URL(targetUrl, currentUrl.origin).href;
      }

      try {
        await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: CONFIG.defaultTimeout });
        actions.push({ action: 'navigate', url: targetUrl, success: true });
      } catch (e) {
        actions.push({ action: 'navigate', url: targetUrl, success: false, error: e.message });
      }
    }
  }

  // Press key action
  else if (lower.includes('press')) {
    const match = instruction.match(/press\s+(.+)/i);
    if (match) {
      const key = match[1].trim();
      try {
        await page.keyboard.press(key);
        actions.push({ action: 'press', key, success: true });
      } catch (e) {
        actions.push({ action: 'press', key, success: false, error: e.message });
      }
    }
  }

  // Wait action
  else if (lower.includes('wait')) {
    const match = instruction.match(/wait\s+(?:for\s+)?(\d+)\s*(?:ms|milliseconds|seconds?)?/i);
    if (match) {
      let ms = parseInt(match[1]);
      if (lower.includes('second')) ms *= 1000;
      await page.waitForTimeout(ms);
      actions.push({ action: 'wait', duration: ms, success: true });
    }
  }

  return actions;
}

// Health check endpoint
app.get('/health', (req, res) => {
  const healthyBrowsers = browserPool.filter(b => b.browser.isConnected()).length;
  const healthy = healthyBrowsers > 0;

  res.status(healthy ? 200 : 503).json({
    status: healthy ? 'healthy' : 'unhealthy',
    pod: CONFIG.podName,
    poolSize: CONFIG.poolSize,
    healthyBrowsers,
    activeSessions: sessions.size,
    timestamp: new Date().toISOString()
  });
});

// Readiness check endpoint
app.get('/ready', (req, res) => {
  const available = browserPool.filter(b => !b.inUse && b.browser.isConnected()).length;

  if (available > 0) {
    res.json({ ready: true, available });
  } else {
    res.status(503).json({ ready: false, available: 0 });
  }
});

// OBSERVE endpoint - Discover interactive elements (MCP compatible)
app.post('/observe', async (req, res) => {
  const timer = metrics.actionDuration.startTimer({ action: 'observe' });
  const { url, instruction } = req.body;

  if (!url) {
    return res.status(400).json({ success: false, error: 'URL is required' });
  }

  const poolEntry = acquireBrowser();
  if (!poolEntry) {
    metrics.errors.inc({ type: 'no_browser' });
    return res.status(503).json({ success: false, error: 'No browsers available' });
  }

  try {
    const context = await poolEntry.browser.newContext({
      viewport: { width: 1920, height: 1080 },
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    });

    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'networkidle', timeout: CONFIG.defaultTimeout });

    const actions = await extractElements(page);
    const pageTitle = await page.title();
    const pageUrl = page.url();

    await context.close();
    releaseBrowser(poolEntry);

    timer({ status: 'success' });
    metrics.sessions.inc();

    logger.info('Observe completed', { url, actionsFound: actions.length });

    res.json({
      success: true,
      url: pageUrl,
      title: pageTitle,
      actions,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    releaseBrowser(poolEntry);
    timer({ status: 'error' });
    metrics.errors.inc({ type: 'observe' });

    logger.error('Observe failed', { url, error: error.message });

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// ACT endpoint - Execute browser action (MCP compatible)
app.post('/act', async (req, res) => {
  const timer = metrics.actionDuration.startTimer({ action: 'act' });
  const { url, instruction, selfHeal = true, screenshot = true } = req.body;

  if (!url || !instruction) {
    return res.status(400).json({ success: false, error: 'URL and instruction are required' });
  }

  const poolEntry = acquireBrowser();
  if (!poolEntry) {
    metrics.errors.inc({ type: 'no_browser' });
    return res.status(503).json({ success: false, error: 'No browsers available' });
  }

  try {
    const context = await poolEntry.browser.newContext({
      viewport: { width: 1920, height: 1080 }
    });

    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'networkidle', timeout: CONFIG.defaultTimeout });

    const actions = await executeAction(page, instruction);

    let screenshotData = null;
    if (screenshot) {
      screenshotData = await page.screenshot({ type: 'png', encoding: 'base64' });
    }

    const success = actions.length > 0 && actions.every(a => a.success);
    const finalUrl = page.url();

    await context.close();
    releaseBrowser(poolEntry);

    timer({ status: success ? 'success' : 'error' });
    metrics.sessions.inc();

    logger.info('Act completed', { url, instruction, success });

    res.json({
      success,
      message: success ? 'Action completed successfully' : 'Action failed',
      actions,
      url: finalUrl,
      screenshot: screenshotData,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    releaseBrowser(poolEntry);
    timer({ status: 'error' });
    metrics.errors.inc({ type: 'act' });

    logger.error('Act failed', { url, instruction, error: error.message });

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// TEST endpoint - Run multi-step test
app.post('/test', async (req, res) => {
  const timer = metrics.actionDuration.startTimer({ action: 'test' });
  const { url, steps, browser: browserType = 'chrome', captureScreenshots = true } = req.body;

  if (!url || !steps || !Array.isArray(steps)) {
    return res.status(400).json({ success: false, error: 'URL and steps array are required' });
  }

  const poolEntry = acquireBrowser();
  if (!poolEntry) {
    metrics.errors.inc({ type: 'no_browser' });
    return res.status(503).json({ success: false, error: 'No browsers available' });
  }

  try {
    const context = await poolEntry.browser.newContext({
      viewport: { width: 1920, height: 1080 }
    });

    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'networkidle', timeout: CONFIG.defaultTimeout });

    const stepResults = [];
    let allPassed = true;

    for (let i = 0; i < steps.length; i++) {
      const stepInstruction = steps[i];
      const startTime = Date.now();

      try {
        const actions = await executeAction(page, stepInstruction);
        const stepSuccess = actions.length > 0 && actions.every(a => a.success);

        if (!stepSuccess) allPassed = false;

        stepResults.push({
          stepIndex: i,
          instruction: stepInstruction,
          success: stepSuccess,
          actions,
          duration: Date.now() - startTime,
          screenshot: captureScreenshots ? await page.screenshot({ type: 'png', encoding: 'base64' }) : null
        });
      } catch (error) {
        allPassed = false;
        stepResults.push({
          stepIndex: i,
          instruction: stepInstruction,
          success: false,
          error: error.message,
          duration: Date.now() - startTime
        });
      }
    }

    let finalScreenshot = null;
    if (captureScreenshots) {
      finalScreenshot = await page.screenshot({ type: 'png', encoding: 'base64' });
    }

    await context.close();
    releaseBrowser(poolEntry);

    timer({ status: allPassed ? 'success' : 'error' });
    metrics.sessions.inc();

    const passedSteps = stepResults.filter(s => s.success).length;

    logger.info('Test completed', { url, stepsTotal: steps.length, stepsPassed: passedSteps });

    res.json({
      success: allPassed,
      steps: stepResults,
      summary: {
        total: steps.length,
        passed: passedSteps,
        failed: steps.length - passedSteps
      },
      finalScreenshot,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    releaseBrowser(poolEntry);
    timer({ status: 'error' });
    metrics.errors.inc({ type: 'test' });

    logger.error('Test failed', { url, error: error.message });

    res.status(500).json({
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Metrics endpoint
metricsApp.get('/metrics', async (req, res) => {
  try {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  } catch (error) {
    res.status(500).end(error.message);
  }
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('Received SIGTERM, shutting down gracefully');

  // Close all browsers
  for (const entry of browserPool) {
    try {
      await entry.browser.close();
    } catch (e) {
      // Ignore close errors
    }
  }

  process.exit(0);
});

// Start servers
initBrowserPool().then(() => {
  app.listen(CONFIG.port, '0.0.0.0', () => {
    logger.info('Browser worker started', { port: CONFIG.port, pod: CONFIG.podName });
  });

  metricsApp.listen(CONFIG.metricsPort, '0.0.0.0', () => {
    logger.info('Metrics server started', { port: CONFIG.metricsPort });
  });
}).catch(error => {
  logger.error('Failed to start browser worker', { error: error.message });
  process.exit(1);
});
