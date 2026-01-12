/**
 * Visual Capture Module
 *
 * Handles screenshot capture and DOM extraction for Visual AI testing.
 * Uses Playwright for browser automation.
 *
 * Note: page.evaluate() is Playwright's standard browser-context API,
 * not server-side eval - it safely executes JavaScript in the browser.
 */

import { chromium, Browser, Page } from 'playwright';
import { v4 as uuidv4 } from 'uuid';

export interface VisualCaptureConfig {
  url: string;
  viewport: { width: number; height: number };
  fullPage?: boolean;
  selector?: string;
  waitForSelector?: string;
  waitForTimeout?: number;
  deviceScaleFactor?: number;
  captureDom?: boolean;
  captureStyles?: boolean;
}

export interface ExtractedElement {
  selector: string;
  tagName: string;
  id: string | null;
  className: string | null;
  text: string | null;
  bounds: { x: number; y: number; width: number; height: number };
  isInteractive: boolean;
}

export interface ExtractedForm {
  selector: string;
  action: string;
  method: string;
  inputs: string[];
}

export interface ExtractedLink {
  href: string;
  text: string | null;
  isExternal: boolean;
}

export interface VisualCaptureResult {
  captureId: string;
  url: string;
  screenshot: string;
  viewport: { width: number; height: number };
  pageTitle: string;
  pageDescription: string | null;
  domSnapshot: string | null;
  elements: ExtractedElement[];
  forms: ExtractedForm[];
  links: ExtractedLink[];
  colorPalette: string[];
  loadTimeMs: number;
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
        '--disable-dev-shm-usage',
        '--disable-gpu'
      ]
    });
  }
  return browserInstance;
}

async function extractPageElements(page: Page): Promise<ExtractedElement[]> {
  // Playwright evaluate() runs JS in browser context (standard API)
  return await page.evaluate(() => {
    const elements: Array<{
      selector: string;
      tagName: string;
      id: string | null;
      className: string | null;
      text: string | null;
      bounds: { x: number; y: number; width: number; height: number };
      isInteractive: boolean;
    }> = [];

    const interactiveSelectors = 'a, button, input, select, textarea, [role="button"], [onclick]';
    const found = document.querySelectorAll(interactiveSelectors);

    found.forEach((el, index) => {
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      let selector = '';
      if (el.id) {
        selector = `#${el.id}`;
      } else if (el.getAttribute('data-testid')) {
        selector = `[data-testid="${el.getAttribute('data-testid')}"]`;
      } else {
        selector = `${el.tagName.toLowerCase()}:nth-of-type(${index + 1})`;
      }

      elements.push({
        selector,
        tagName: el.tagName.toLowerCase(),
        id: el.id || null,
        className: el.className && typeof el.className === 'string' ? el.className : null,
        text: el.textContent?.trim().slice(0, 50) || null,
        bounds: {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          width: Math.round(rect.width),
          height: Math.round(rect.height)
        },
        isInteractive: true
      });
    });

    return elements;
  });
}

async function extractPageForms(page: Page): Promise<ExtractedForm[]> {
  return await page.evaluate(() => {
    const forms: Array<{
      selector: string;
      action: string;
      method: string;
      inputs: string[];
    }> = [];

    const formElements = document.querySelectorAll('form');

    formElements.forEach((form, index) => {
      const inputs = Array.from(form.querySelectorAll('input, select, textarea'))
        .map(input => {
          const name = input.getAttribute('name') || input.id || input.getAttribute('placeholder') || 'unnamed';
          const type = (input as HTMLInputElement).type || input.tagName.toLowerCase();
          return `${name}:${type}`;
        });

      forms.push({
        selector: form.id ? `#${form.id}` : `form:nth-of-type(${index + 1})`,
        action: form.action || '',
        method: form.method || 'get',
        inputs
      });
    });

    return forms;
  });
}

async function extractPageLinks(page: Page, pageUrl: string): Promise<ExtractedLink[]> {
  const pageOrigin = new URL(pageUrl).origin;

  return await page.evaluate((origin) => {
    const links: Array<{ href: string; text: string | null; isExternal: boolean }> = [];
    const anchors = document.querySelectorAll('a[href]');

    anchors.forEach(a => {
      const href = (a as HTMLAnchorElement).href;
      if (!href || href.startsWith('javascript:') || href.startsWith('#')) return;

      links.push({
        href,
        text: a.textContent?.trim().slice(0, 100) || null,
        isExternal: !href.startsWith(origin)
      });
    });

    return links;
  }, pageOrigin);
}

async function extractColorPalette(page: Page): Promise<string[]> {
  return await page.evaluate(() => {
    const colors = new Set<string>();
    const elements = document.querySelectorAll('*');

    elements.forEach(el => {
      const style = window.getComputedStyle(el);
      const bgColor = style.backgroundColor;
      const textColor = style.color;

      if (bgColor && bgColor !== 'rgba(0, 0, 0, 0)' && bgColor !== 'transparent') {
        colors.add(bgColor);
      }
      if (textColor) {
        colors.add(textColor);
      }
    });

    return Array.from(colors).slice(0, 10);
  });
}

async function getDomSnapshot(page: Page): Promise<string> {
  return await page.evaluate(() => {
    function simplifyNode(node: Element, depth: number = 0): string {
      if (depth > 5) return '...';

      const tag = node.tagName.toLowerCase();
      const id = node.id ? `#${node.id}` : '';
      const classes = node.className && typeof node.className === 'string'
        ? '.' + node.className.split(' ').filter(Boolean).join('.')
        : '';

      const children = Array.from(node.children)
        .map(child => simplifyNode(child, depth + 1))
        .join('');

      if (['script', 'style', 'noscript', 'svg', 'path'].includes(tag)) {
        return '';
      }

      return `<${tag}${id}${classes}>${children}</${tag}>`;
    }

    return simplifyNode(document.body);
  });
}

export async function captureVisual(config: VisualCaptureConfig): Promise<VisualCaptureResult> {
  const captureId = uuidv4();
  const startTime = Date.now();

  const browser = await getBrowser();
  const context = await browser.newContext({
    viewport: config.viewport,
    deviceScaleFactor: config.deviceScaleFactor || 1
  });

  const page = await context.newPage();

  try {
    await page.goto(config.url, {
      waitUntil: 'networkidle',
      timeout: 30000
    });

    if (config.waitForSelector) {
      await page.waitForSelector(config.waitForSelector, {
        timeout: config.waitForTimeout || 5000
      });
    } else if (config.waitForTimeout) {
      await page.waitForTimeout(config.waitForTimeout);
    }

    const pageTitle = await page.title();
    const pageDescription = await page.$eval(
      'meta[name="description"]',
      el => el.getAttribute('content')
    ).catch(() => null);

    let screenshotBuffer: Buffer;
    if (config.selector) {
      const element = await page.$(config.selector);
      if (!element) {
        throw new Error(`Selector not found: ${config.selector}`);
      }
      screenshotBuffer = await element.screenshot({ type: 'png' });
    } else {
      screenshotBuffer = await page.screenshot({
        type: 'png',
        fullPage: config.fullPage || false
      });
    }

    const elements = await extractPageElements(page);
    const forms = await extractPageForms(page);
    const links = await extractPageLinks(page, config.url);
    const colorPalette = await extractColorPalette(page);

    let domSnapshot: string | null = null;
    if (config.captureDom) {
      domSnapshot = await getDomSnapshot(page);
    }

    const loadTimeMs = Date.now() - startTime;

    return {
      captureId,
      url: config.url,
      screenshot: screenshotBuffer.toString('base64'),
      viewport: config.viewport,
      pageTitle,
      pageDescription,
      domSnapshot,
      elements,
      forms,
      links,
      colorPalette,
      loadTimeMs,
      timestamp: new Date().toISOString()
    };

  } finally {
    await context.close();
  }
}

export async function closeBrowser(): Promise<void> {
  if (browserInstance) {
    await browserInstance.close();
    browserInstance = null;
  }
}

process.on('exit', () => {
  if (browserInstance) {
    browserInstance.close().catch(() => {});
  }
});
