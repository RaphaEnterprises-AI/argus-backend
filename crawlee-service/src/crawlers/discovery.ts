/**
 * Discovery Crawler
 *
 * Uses Crawlee with PlaywrightCrawler to perform graph-based crawling
 * of web applications, extracting pages, elements, forms, and links.
 *
 * Note: page.evaluate() is Playwright's standard API for DOM extraction
 * and runs in the browser context, not server-side eval.
 */

import { PlaywrightCrawler, Configuration, RequestQueue } from 'crawlee';
import { Page, Browser } from 'playwright';
import { v4 as uuidv4 } from 'uuid';

// Type for browser.newPage() options which includes video recording
type BrowserContextOptions = NonNullable<Parameters<Browser['newPage']>[0]>;

export interface DiscoveryConfig {
  startUrl: string;
  maxPages: number;
  maxDepth: number;
  includePatterns: string[];
  excludePatterns: string[];
  captureScreenshots: boolean;
  recordVideo: boolean;
  viewport: { width: number; height: number };
  authConfig?: {
    type: 'cookie' | 'form' | 'header';
    credentials: Record<string, string>;
  };
}

export interface DiscoveredElement {
  id: string;
  selector: string;
  xpath: string;
  tagName: string;
  type: string;
  category: 'navigation' | 'form' | 'action' | 'content' | 'interactive';
  label: string | null;
  placeholder: string | null;
  ariaLabel: string | null;
  bounds: { x: number; y: number; width: number; height: number };
  isVisible: boolean;
  isEnabled: boolean;
  isRequired: boolean;
}

export interface DiscoveredForm {
  id: string;
  selector: string;
  action: string;
  method: string;
  fields: DiscoveredElement[];
  submitButton: DiscoveredElement | null;
}

export interface DiscoveredPage {
  id: string;
  url: string;
  title: string;
  description: string | null;
  category: string;
  depth: number;
  screenshot: string | null;
  elements: DiscoveredElement[];
  forms: DiscoveredForm[];
  links: string[];
  loadTimeMs: number;
  timestamp: string;
}

export interface DiscoveryResult {
  sessionId: string;
  startUrl: string;
  pages: DiscoveredPage[];
  totalPages: number;
  totalElements: number;
  totalForms: number;
  totalLinks: number;
  duration: number;
  errors: Array<{ url: string; error: string }>;
  graph: Record<string, string[]>;
  videoArtifactId?: string;
  recordingUrl?: string;
}

/**
 * Extract interactive elements from a page using Playwright's browser-context evaluation
 */
async function extractElements(page: Page): Promise<DiscoveredElement[]> {
  // Playwright's evaluate() runs JavaScript in the browser context (not server-side eval)
  return await page.evaluate(() => {
    const elements: DiscoveredElement[] = [];

    // Interactive element selectors
    const selectors = [
      'a[href]',
      'button',
      'input',
      'select',
      'textarea',
      '[role="button"]',
      '[role="link"]',
      '[role="menuitem"]',
      '[role="tab"]',
      '[onclick]',
      '[data-testid]',
      '[data-cy]'
    ];

    const seen = new Set<Element>();

    for (const selectorStr of selectors) {
      const found = document.querySelectorAll(selectorStr);
      for (const el of found) {
        if (seen.has(el)) continue;
        seen.add(el);

        const rect = el.getBoundingClientRect();
        const tagName = el.tagName.toLowerCase();
        const inputType = (el as HTMLInputElement).type || tagName;

        // Determine category
        let category: 'navigation' | 'form' | 'action' | 'content' | 'interactive' = 'content';
        if (tagName === 'a' || el.getAttribute('role') === 'link') {
          category = 'navigation';
        } else if (tagName === 'button' || el.getAttribute('role') === 'button' || inputType === 'submit') {
          category = 'action';
        } else if (['input', 'select', 'textarea'].includes(tagName)) {
          category = 'form';
        } else if (el.getAttribute('onclick') || el.getAttribute('role')?.includes('menu')) {
          category = 'interactive';
        }

        // Generate stable selector
        let cssSelector = '';
        if (el.id) {
          cssSelector = `#${el.id}`;
        } else if (el.getAttribute('data-testid')) {
          cssSelector = `[data-testid="${el.getAttribute('data-testid')}"]`;
        } else if (el.getAttribute('data-cy')) {
          cssSelector = `[data-cy="${el.getAttribute('data-cy')}"]`;
        } else if (el.getAttribute('name')) {
          cssSelector = `${tagName}[name="${el.getAttribute('name')}"]`;
        } else {
          // Build CSS path
          const path: string[] = [];
          let current: Element | null = el;
          while (current && current !== document.body) {
            let seg = current.tagName.toLowerCase();
            if (current.id) {
              seg = `#${current.id}`;
              path.unshift(seg);
              break;
            }
            const siblings = current.parentElement?.children || [];
            const sameTagSiblings = Array.from(siblings).filter(s => s.tagName === current!.tagName);
            const index = sameTagSiblings.indexOf(current);
            if (index > 0) seg += `:nth-of-type(${index + 1})`;
            path.unshift(seg);
            current = current.parentElement;
          }
          cssSelector = path.join(' > ');
        }

        elements.push({
          id: `el-${Math.random().toString(36).substr(2, 9)}`,
          selector: cssSelector,
          xpath: '', // XPath generation simplified
          tagName,
          type: inputType,
          category,
          label: el.textContent?.trim().slice(0, 100) || null,
          placeholder: (el as HTMLInputElement).placeholder || null,
          ariaLabel: el.getAttribute('aria-label'),
          bounds: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height)
          },
          isVisible: rect.width > 0 && rect.height > 0,
          isEnabled: !(el as HTMLInputElement).disabled,
          isRequired: (el as HTMLInputElement).required || false
        });
      }
    }

    return elements;
  });
}

/**
 * Extract forms from a page
 */
async function extractForms(page: Page, elements: DiscoveredElement[]): Promise<DiscoveredForm[]> {
  const formData = await page.evaluate(() => {
    const forms: Array<{
      id: string;
      selector: string;
      action: string;
      method: string;
      bounds: { x: number; y: number; width: number; height: number };
    }> = [];

    const formElements = document.querySelectorAll('form');

    for (const form of formElements) {
      const rect = form.getBoundingClientRect();
      let selector = '';
      if (form.id) selector = `#${form.id}`;
      else if (form.getAttribute('name')) selector = `form[name="${form.getAttribute('name')}"]`;
      else if (form.getAttribute('action')) selector = `form[action="${form.getAttribute('action')}"]`;
      else selector = 'form';

      forms.push({
        id: `form-${Math.random().toString(36).substr(2, 9)}`,
        selector,
        action: form.action || '',
        method: form.method || 'get',
        bounds: {
          x: Math.round(rect.x),
          y: Math.round(rect.y),
          width: Math.round(rect.width),
          height: Math.round(rect.height)
        }
      });
    }

    return forms;
  });

  // Associate elements with forms
  return formData.map(form => {
    const formElements = elements.filter(el =>
      el.category === 'form' &&
      el.bounds.x >= form.bounds.x &&
      el.bounds.y >= form.bounds.y &&
      el.bounds.x + el.bounds.width <= form.bounds.x + form.bounds.width + 50 &&
      el.bounds.y + el.bounds.height <= form.bounds.y + form.bounds.height + 50
    );

    const submitButton = elements.find(el =>
      (el.type === 'submit' || el.tagName === 'button') &&
      el.bounds.x >= form.bounds.x &&
      el.bounds.y >= form.bounds.y
    ) || null;

    return {
      ...form,
      fields: formElements,
      submitButton
    };
  });
}

/**
 * Categorize a page based on URL and content
 */
function categorizePage(url: string, title: string, forms: DiscoveredForm[]): string {
  const urlLower = url.toLowerCase();

  // Check URL patterns
  if (urlLower.includes('/login') || urlLower.includes('/signin') || urlLower.includes('/auth')) {
    return 'auth_login';
  }
  if (urlLower.includes('/signup') || urlLower.includes('/register')) {
    return 'auth_signup';
  }
  if (urlLower.includes('/reset') || urlLower.includes('/forgot')) {
    return 'auth_reset';
  }
  if (urlLower.includes('/dashboard') || urlLower.includes('/home') || urlLower.includes('/app')) {
    return 'dashboard';
  }
  if (urlLower.includes('/settings') || urlLower.includes('/preferences')) {
    return 'settings';
  }
  if (urlLower.includes('/profile') || urlLower.includes('/account')) {
    return 'profile';
  }
  if (urlLower.includes('/checkout') || urlLower.includes('/cart') || urlLower.includes('/payment')) {
    return 'checkout';
  }
  if (urlLower.includes('/search') || urlLower.includes('/results')) {
    return 'search';
  }
  if (urlLower.match(/\/\d+$/) || urlLower.includes('/detail') || urlLower.includes('/view')) {
    return 'detail';
  }
  if (urlLower.includes('/list') || urlLower.includes('/browse')) {
    return 'list';
  }
  if (urlLower === new URL(url).origin + '/' || urlLower.endsWith('/home')) {
    return 'landing';
  }

  // Check for forms
  if (forms.length > 0) {
    const hasLoginFields = forms.some(f =>
      f.fields.some(field =>
        field.type === 'password' ||
        field.selector.includes('password') ||
        field.selector.includes('email')
      )
    );
    if (hasLoginFields) return 'auth_login';
    return 'form';
  }

  return 'other';
}

/**
 * Run a discovery crawl
 */
export async function runDiscoveryCrawl(config: DiscoveryConfig): Promise<DiscoveryResult> {
  const sessionId = uuidv4();
  const startTime = Date.now();
  const pages: DiscoveredPage[] = [];
  const errors: Array<{ url: string; error: string }> = [];
  const graph: Record<string, string[]> = {};

  // Video recording directory
  const videoDir = `/tmp/videos/${sessionId}`;
  let videoPath: string | undefined;

  // Configure Crawlee to use memory storage (no disk persistence)
  const crawlerConfig = new Configuration({
    persistStorage: false,
    purgeOnStart: true
  });

  // Create request queue
  const requestQueue = await RequestQueue.open(sessionId, { config: crawlerConfig });

  // Set up video recording directory if enabled
  if (config.recordVideo) {
    const fs = await import('fs');
    console.log(`Setting up video recording directory: ${videoDir}`);
    fs.mkdirSync(videoDir, { recursive: true });
    console.log(`Video directory created: ${fs.existsSync(videoDir)}`);
  }

  const crawler = new PlaywrightCrawler({
    requestQueue,
    maxRequestsPerCrawl: config.maxPages,
    maxConcurrency: 1, // Use single concurrency for video recording
    requestHandlerTimeoutSecs: 60,
    navigationTimeoutSecs: 30,

    launchContext: {
      launchOptions: {
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
      }
    },

    browserPoolOptions: {
      useFingerprints: false,
      preLaunchHooks: [
        async (_pageId, launchContext) => {
          launchContext.launchOptions = {
            ...launchContext.launchOptions,
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
          };
        }
      ],
      // Configure browser context options including video recording
      // pageOptions is the browser.newPage() options (BrowserContextOptions) for Playwright
      prePageCreateHooks: [
        async (_pageId, _browserController, pageOptions) => {
          if (config.recordVideo && pageOptions) {
            console.log(`Configuring video recording for session ${sessionId} in ${videoDir}`);
            // Cast to browser context options and add recordVideo
            const opts = pageOptions as BrowserContextOptions;
            opts.recordVideo = {
              dir: videoDir,
              size: { width: config.viewport.width, height: config.viewport.height }
            };
          }
        }
      ]
    },

    preNavigationHooks: [
      async ({ page }) => {
        if (config.recordVideo) {
          console.log(`Navigating with video recording enabled for session ${sessionId}`);
        }
      }
    ],

    async requestHandler({ page, request, enqueueLinks, log }) {
      const url = request.url;
      const depth = (request.userData?.depth as number) || 0;
      const pageStartTime = Date.now();

      log.info(`Crawling: ${url} (depth: ${depth})`);

      try {
        // Set viewport
        await page.setViewportSize(config.viewport);

        // Handle authentication if configured
        if (config.authConfig && depth === 0) {
          if (config.authConfig.type === 'cookie') {
            const cookies = Object.entries(config.authConfig.credentials).map(([name, value]) => ({
              name,
              value,
              domain: new URL(url).hostname,
              path: '/'
            }));
            await page.context().addCookies(cookies);
          } else if (config.authConfig.type === 'header') {
            await page.setExtraHTTPHeaders(config.authConfig.credentials);
          }
        }

        // Wait for page to be ready
        await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});

        // Get page info
        const title = await page.title();
        const description = await page.$eval(
          'meta[name="description"]',
          el => el.getAttribute('content')
        ).catch(() => null);

        // Extract elements and forms
        const elements = await extractElements(page);
        const forms = await extractForms(page, elements);

        // Categorize the page
        const category = categorizePage(url, title, forms);

        // Extract links
        const links = await page.evaluate(() => {
          const anchors = document.querySelectorAll('a[href]');
          return Array.from(anchors)
            .map(a => (a as HTMLAnchorElement).href)
            .filter(href => href.startsWith('http'));
        });

        // Capture screenshot if enabled
        let screenshot: string | null = null;
        if (config.captureScreenshots) {
          const buffer = await page.screenshot({
            type: 'jpeg',
            quality: 70,
            fullPage: false
          });
          screenshot = buffer.toString('base64');
        }

        const loadTimeMs = Date.now() - pageStartTime;

        // Store page data
        const pageData: DiscoveredPage = {
          id: uuidv4(),
          url,
          title,
          description,
          category,
          depth,
          screenshot,
          elements,
          forms,
          links,
          loadTimeMs,
          timestamp: new Date().toISOString()
        };

        pages.push(pageData);
        graph[url] = links;

        // Enqueue discovered links (respecting depth limit)
        if (depth < config.maxDepth) {
          // Build glob patterns for URL filtering
          const globs: string[] = config.includePatterns.length > 0
            ? config.includePatterns
            : [`${new URL(url).origin}/**`];

          await enqueueLinks({
            globs,
            exclude: config.excludePatterns,
            transformRequestFunction: (req) => {
              req.userData = { depth: depth + 1 };
              return req;
            }
          });
        }

      } catch (error) {
        log.error(`Error crawling ${url}: ${error}`);
        errors.push({
          url,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    },

    failedRequestHandler({ request, log }, error) {
      log.error(`Request failed: ${request.url}`);
      errors.push({
        url: request.url,
        error: error.message
      });
    }
  }, crawlerConfig);

  // Add initial URL
  await requestQueue.addRequest({
    url: config.startUrl,
    userData: { depth: 0 }
  });

  // Run the crawler
  await crawler.run();

  // Ensure browser pool is properly torn down (flushes video files)
  await crawler.teardown();

  const duration = Date.now() - startTime;

  // Get video file if recording was enabled
  let videoArtifactId: string | undefined;
  let recordingUrl: string | undefined;

  if (config.recordVideo) {
    try {
      const fs = await import('fs');
      const path = await import('path');

      // Small delay to ensure video file is fully written
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Check if video directory exists
      if (!fs.existsSync(videoDir)) {
        console.log(`Video directory does not exist: ${videoDir}`);
      } else {
        // Find video files in the video directory
        const files = fs.readdirSync(videoDir);
        console.log(`Files in video directory ${videoDir}: ${JSON.stringify(files)}`);
        const videoFiles = files.filter(f => f.endsWith('.webm') || f.endsWith('.mp4'));

        if (videoFiles.length > 0) {
          // Use the first video file (there should be one per context)
          videoPath = path.join(videoDir, videoFiles[0]);
          videoArtifactId = sessionId;
          // In production, upload to S3/R2 and return the URL
          // For now, return local path that can be served via static file server
          recordingUrl = `/videos/${sessionId}/${videoFiles[0]}`;
          console.log(`Video recorded successfully: ${videoPath}`);
        } else {
          console.log(`No video files found in ${videoDir}`);
        }
      }
    } catch (error) {
      console.error(`Failed to get video file: ${error}`);
    }
  }

  return {
    sessionId,
    startUrl: config.startUrl,
    pages,
    totalPages: pages.length,
    totalElements: pages.reduce((sum, p) => sum + p.elements.length, 0),
    totalForms: pages.reduce((sum, p) => sum + p.forms.length, 0),
    totalLinks: pages.reduce((sum, p) => sum + p.links.length, 0),
    duration,
    errors,
    graph,
    videoArtifactId,
    recordingUrl
  };
}
