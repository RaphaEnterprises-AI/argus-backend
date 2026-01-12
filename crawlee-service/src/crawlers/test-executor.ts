/**
 * Test Executor Module
 *
 * Executes test steps using Playwright and captures results.
 * Supports actions like click, type, navigate, and assertions.
 */

import { chromium, Browser, Page, BrowserContext } from 'playwright';
import { v4 as uuidv4 } from 'uuid';

export interface TestStep {
  action: string;
  selector?: string;
  value?: string;
  url?: string;
  assertion?: {
    type: string;
    expected: string;
  };
}

export interface TestExecutionConfig {
  testId: string;
  steps: TestStep[];
  baseUrl: string;
  viewport: { width: number; height: number };
  timeout: number;
  captureScreenshots: boolean;
  captureVideo: boolean;
}

export interface StepResult {
  stepIndex: number;
  action: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  screenshot?: string;
  error?: string;
  details?: Record<string, unknown>;
}

export interface TestExecutionResult {
  executionId: string;
  testId: string;
  passed: boolean;
  totalSteps: number;
  passedSteps: number;
  failedSteps: number;
  skippedSteps: number;
  stepResults: StepResult[];
  totalDuration: number;
  screenshots: string[];
  videoPath?: string;
  error?: string;
  timestamp: string;
}

let browserInstance: Browser | null = null;

async function getBrowser(): Promise<Browser> {
  if (!browserInstance || !browserInstance.isConnected()) {
    browserInstance = await chromium.launch({
      headless: true,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage'
      ]
    });
  }
  return browserInstance;
}

/**
 * Execute a single test step
 */
async function executeStep(
  page: Page,
  step: TestStep,
  stepIndex: number,
  config: TestExecutionConfig
): Promise<StepResult> {
  const startTime = Date.now();
  let screenshot: string | undefined;
  let error: string | undefined;
  let status: 'passed' | 'failed' | 'skipped' = 'passed';
  const details: Record<string, unknown> = {};

  try {
    switch (step.action.toLowerCase()) {
      case 'navigate':
      case 'goto':
        const targetUrl = step.url?.startsWith('http')
          ? step.url
          : `${config.baseUrl}${step.url || ''}`;
        await page.goto(targetUrl, { waitUntil: 'networkidle', timeout: config.timeout });
        details.url = targetUrl;
        break;

      case 'click':
        if (!step.selector) throw new Error('Selector required for click action');
        await page.click(step.selector, { timeout: config.timeout });
        details.selector = step.selector;
        break;

      case 'type':
      case 'fill':
        if (!step.selector) throw new Error('Selector required for type action');
        if (step.value === undefined) throw new Error('Value required for type action');
        await page.fill(step.selector, step.value, { timeout: config.timeout });
        details.selector = step.selector;
        details.value = step.value.slice(0, 20) + (step.value.length > 20 ? '...' : '');
        break;

      case 'press':
        if (!step.value) throw new Error('Key value required for press action');
        if (step.selector) {
          await page.press(step.selector, step.value, { timeout: config.timeout });
        } else {
          await page.keyboard.press(step.value);
        }
        details.key = step.value;
        break;

      case 'select':
        if (!step.selector) throw new Error('Selector required for select action');
        if (!step.value) throw new Error('Value required for select action');
        await page.selectOption(step.selector, step.value, { timeout: config.timeout });
        details.selector = step.selector;
        details.value = step.value;
        break;

      case 'hover':
        if (!step.selector) throw new Error('Selector required for hover action');
        await page.hover(step.selector, { timeout: config.timeout });
        details.selector = step.selector;
        break;

      case 'wait':
        if (step.selector) {
          await page.waitForSelector(step.selector, { timeout: config.timeout });
          details.selector = step.selector;
        } else if (step.value) {
          const waitTime = parseInt(step.value, 10);
          await page.waitForTimeout(waitTime);
          details.waitTime = waitTime;
        }
        break;

      case 'waitfornavigation':
        await page.waitForNavigation({ timeout: config.timeout });
        break;

      case 'scroll':
        if (step.selector) {
          await page.locator(step.selector).scrollIntoViewIfNeeded();
        } else {
          const scrollAmount = parseInt(step.value || '500', 10);
          await page.evaluate((amount) => window.scrollBy(0, amount), scrollAmount);
        }
        break;

      case 'assert':
      case 'assertion':
        if (!step.assertion) throw new Error('Assertion config required');
        const assertionResult = await executeAssertion(page, step.assertion, config.timeout);
        if (!assertionResult.passed) {
          throw new Error(assertionResult.error);
        }
        details.assertion = step.assertion;
        break;

      case 'screenshot':
        const buffer = await page.screenshot({ type: 'png' });
        screenshot = buffer.toString('base64');
        break;

      default:
        throw new Error(`Unknown action: ${step.action}`);
    }

    // Capture screenshot after step if configured
    if (config.captureScreenshots && !screenshot) {
      const buffer = await page.screenshot({ type: 'jpeg', quality: 70 });
      screenshot = buffer.toString('base64');
    }

  } catch (err) {
    status = 'failed';
    error = err instanceof Error ? err.message : 'Unknown error';

    // Always capture screenshot on failure
    try {
      const buffer = await page.screenshot({ type: 'png' });
      screenshot = buffer.toString('base64');
    } catch {
      // Ignore screenshot error
    }
  }

  return {
    stepIndex,
    action: step.action,
    status,
    duration: Date.now() - startTime,
    screenshot,
    error,
    details
  };
}

/**
 * Execute an assertion
 */
async function executeAssertion(
  page: Page,
  assertion: { type: string; expected: string },
  timeout: number
): Promise<{ passed: boolean; error?: string }> {
  try {
    switch (assertion.type.toLowerCase()) {
      case 'url':
      case 'url_contains':
        const url = page.url();
        if (!url.includes(assertion.expected)) {
          return { passed: false, error: `URL "${url}" does not contain "${assertion.expected}"` };
        }
        break;

      case 'url_equals':
        const currentUrl = page.url();
        if (currentUrl !== assertion.expected) {
          return { passed: false, error: `URL "${currentUrl}" does not equal "${assertion.expected}"` };
        }
        break;

      case 'title':
      case 'title_contains':
        const title = await page.title();
        if (!title.includes(assertion.expected)) {
          return { passed: false, error: `Title "${title}" does not contain "${assertion.expected}"` };
        }
        break;

      case 'title_equals':
        const pageTitle = await page.title();
        if (pageTitle !== assertion.expected) {
          return { passed: false, error: `Title "${pageTitle}" does not equal "${assertion.expected}"` };
        }
        break;

      case 'visible':
      case 'element_visible':
        const isVisible = await page.isVisible(assertion.expected, { timeout });
        if (!isVisible) {
          return { passed: false, error: `Element "${assertion.expected}" is not visible` };
        }
        break;

      case 'hidden':
      case 'element_hidden':
        const isHidden = await page.isHidden(assertion.expected);
        if (!isHidden) {
          return { passed: false, error: `Element "${assertion.expected}" is not hidden` };
        }
        break;

      case 'text':
      case 'text_contains':
        const text = await page.textContent(assertion.expected.split('::')[0]);
        const expectedText = assertion.expected.split('::')[1] || '';
        if (!text?.includes(expectedText)) {
          return { passed: false, error: `Text does not contain "${expectedText}"` };
        }
        break;

      case 'count':
        const [selector, expectedCount] = assertion.expected.split('::');
        const count = await page.locator(selector).count();
        if (count !== parseInt(expectedCount, 10)) {
          return { passed: false, error: `Expected ${expectedCount} elements, found ${count}` };
        }
        break;

      default:
        return { passed: false, error: `Unknown assertion type: ${assertion.type}` };
    }

    return { passed: true };

  } catch (err) {
    return {
      passed: false,
      error: err instanceof Error ? err.message : 'Assertion failed'
    };
  }
}

/**
 * Run a test execution
 */
export async function runTestExecution(config: TestExecutionConfig): Promise<TestExecutionResult> {
  const executionId = uuidv4();
  const startTime = Date.now();
  const stepResults: StepResult[] = [];
  const screenshots: string[] = [];
  let overallError: string | undefined;

  const browser = await getBrowser();

  // Create context with optional video recording
  const contextOptions: Parameters<Browser['newContext']>[0] = {
    viewport: config.viewport
  };

  if (config.captureVideo) {
    contextOptions.recordVideo = {
      dir: '/tmp/videos',
      size: config.viewport
    };
  }

  const context = await browser.newContext(contextOptions);
  const page = await context.newPage();

  let hasFailed = false;

  try {
    for (let i = 0; i < config.steps.length; i++) {
      const step = config.steps[i];

      // Skip remaining steps if a previous step failed
      if (hasFailed) {
        stepResults.push({
          stepIndex: i,
          action: step.action,
          status: 'skipped',
          duration: 0
        });
        continue;
      }

      const result = await executeStep(page, step, i, config);
      stepResults.push(result);

      if (result.screenshot) {
        screenshots.push(result.screenshot);
      }

      if (result.status === 'failed') {
        hasFailed = true;
      }
    }

  } catch (err) {
    overallError = err instanceof Error ? err.message : 'Test execution failed';
  } finally {
    await context.close();
  }

  const passedSteps = stepResults.filter(r => r.status === 'passed').length;
  const failedSteps = stepResults.filter(r => r.status === 'failed').length;
  const skippedSteps = stepResults.filter(r => r.status === 'skipped').length;

  return {
    executionId,
    testId: config.testId,
    passed: failedSteps === 0 && !overallError,
    totalSteps: config.steps.length,
    passedSteps,
    failedSteps,
    skippedSteps,
    stepResults,
    totalDuration: Date.now() - startTime,
    screenshots,
    error: overallError,
    timestamp: new Date().toISOString()
  };
}

export async function closeBrowser(): Promise<void> {
  if (browserInstance) {
    await browserInstance.close();
    browserInstance = null;
  }
}
