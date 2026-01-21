/**
 * Argus Crawlee Microservice
 *
 * Provides browser-based crawling and visual capture capabilities
 * via HTTP API endpoints. Designed to run as a separate container
 * from the main Argus backend for better resource isolation.
 */

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import { v4 as uuidv4 } from 'uuid';
import { createLogger, format, transports } from 'winston';

import { runDiscoveryCrawl, DiscoveryConfig, DiscoveryResult } from './crawlers/discovery';
import { captureVisual, VisualCaptureConfig, VisualCaptureResult } from './crawlers/visual';
import { runTestExecution, TestExecutionConfig, TestExecutionResult } from './crawlers/test-executor';

// Logger setup
const logger = createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: format.combine(
    format.timestamp(),
    format.json()
  ),
  transports: [
    new transports.Console()
  ]
});

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);

// Middleware
app.use(helmet());
app.use(cors());
app.use(compression());
app.use(express.json({ limit: '50mb' }));

// Request logging
app.use((req: Request, res: Response, next: NextFunction) => {
  const requestId = uuidv4();
  req.headers['x-request-id'] = requestId;
  logger.info('Request received', {
    requestId,
    method: req.method,
    path: req.path,
    ip: req.ip
  });
  next();
});

// Health check
app.get('/health', (req: Request, res: Response) => {
  res.json({
    status: 'healthy',
    service: 'argus-crawlee',
    version: '1.0.0',
    timestamp: new Date().toISOString(),
    capabilities: ['discovery', 'visual-capture', 'test-execution']
  });
});

// ============================================
// Discovery Crawling Endpoints
// ============================================

interface DiscoveryRequest {
  startUrl: string;
  maxPages?: number;
  maxDepth?: number;
  includePatterns?: string[];
  excludePatterns?: string[];
  captureScreenshots?: boolean;
  recordVideo?: boolean;
  viewport?: { width: number; height: number };
  authConfig?: {
    type: 'cookie' | 'form' | 'header';
    credentials: Record<string, string>;
  };
}

app.post('/api/v1/crawl/discovery', async (req: Request, res: Response) => {
  const requestId = req.headers['x-request-id'] as string;
  const startTime = Date.now();

  try {
    const body: DiscoveryRequest = req.body;

    if (!body.startUrl) {
      return res.status(400).json({ error: 'startUrl is required' });
    }

    logger.info('Starting discovery crawl', { requestId, startUrl: body.startUrl });

    const config: DiscoveryConfig = {
      startUrl: body.startUrl,
      maxPages: body.maxPages || 50,
      maxDepth: body.maxDepth || 3,
      includePatterns: body.includePatterns || [],
      excludePatterns: body.excludePatterns || [],
      captureScreenshots: body.captureScreenshots ?? true,
      recordVideo: body.recordVideo ?? false,
      viewport: body.viewport || { width: 1920, height: 1080 },
      authConfig: body.authConfig
    };

    const result: DiscoveryResult = await runDiscoveryCrawl(config);

    const duration = Date.now() - startTime;
    logger.info('Discovery crawl completed', {
      requestId,
      pagesFound: result.pages.length,
      duration
    });

    res.json({
      success: true,
      requestId,
      duration,
      result
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Discovery crawl failed', {
      requestId,
      error: error instanceof Error ? error.message : 'Unknown error',
      duration
    });

    res.status(500).json({
      success: false,
      requestId,
      error: error instanceof Error ? error.message : 'Discovery crawl failed'
    });
  }
});

// ============================================
// Visual Capture Endpoints
// ============================================

interface VisualCaptureRequest {
  url: string;
  viewport?: { width: number; height: number };
  fullPage?: boolean;
  selector?: string;
  waitForSelector?: string;
  waitForTimeout?: number;
  deviceScaleFactor?: number;
  captureDom?: boolean;
  captureStyles?: boolean;
}

app.post('/api/v1/capture/screenshot', async (req: Request, res: Response) => {
  const requestId = req.headers['x-request-id'] as string;
  const startTime = Date.now();

  try {
    const body: VisualCaptureRequest = req.body;

    if (!body.url) {
      return res.status(400).json({ error: 'url is required' });
    }

    logger.info('Starting visual capture', { requestId, url: body.url });

    const config: VisualCaptureConfig = {
      url: body.url,
      viewport: body.viewport || { width: 1920, height: 1080 },
      fullPage: body.fullPage ?? false,
      selector: body.selector,
      waitForSelector: body.waitForSelector,
      waitForTimeout: body.waitForTimeout || 5000,
      deviceScaleFactor: body.deviceScaleFactor || 1,
      captureDom: body.captureDom ?? true,
      captureStyles: body.captureStyles ?? false
    };

    const result: VisualCaptureResult = await captureVisual(config);

    const duration = Date.now() - startTime;
    logger.info('Visual capture completed', { requestId, duration });

    res.json({
      success: true,
      requestId,
      duration,
      result
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Visual capture failed', {
      requestId,
      error: error instanceof Error ? error.message : 'Unknown error',
      duration
    });

    res.status(500).json({
      success: false,
      requestId,
      error: error instanceof Error ? error.message : 'Visual capture failed'
    });
  }
});

// Batch capture for responsive testing
app.post('/api/v1/capture/responsive', async (req: Request, res: Response) => {
  const requestId = req.headers['x-request-id'] as string;
  const startTime = Date.now();

  try {
    const { url, viewports } = req.body;

    if (!url) {
      return res.status(400).json({ error: 'url is required' });
    }

    const defaultViewports = [
      { name: 'mobile', width: 375, height: 667 },
      { name: 'tablet', width: 768, height: 1024 },
      { name: 'desktop', width: 1440, height: 900 },
      { name: 'wide', width: 1920, height: 1080 }
    ];

    const targetViewports = viewports || defaultViewports;
    const results: Record<string, VisualCaptureResult> = {};

    for (const vp of targetViewports) {
      const config: VisualCaptureConfig = {
        url,
        viewport: { width: vp.width, height: vp.height },
        fullPage: true,
        captureDom: true
      };
      results[vp.name] = await captureVisual(config);
    }

    const duration = Date.now() - startTime;
    logger.info('Responsive capture completed', {
      requestId,
      viewports: targetViewports.length,
      duration
    });

    res.json({
      success: true,
      requestId,
      duration,
      results
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Responsive capture failed', { requestId, error });

    res.status(500).json({
      success: false,
      requestId,
      error: error instanceof Error ? error.message : 'Responsive capture failed'
    });
  }
});

// ============================================
// Test Execution Endpoints
// ============================================

interface TestExecutionRequest {
  testId: string;
  steps: Array<{
    action: string;
    selector?: string;
    value?: string;
    url?: string;
    assertion?: {
      type: string;
      expected: string;
    };
  }>;
  baseUrl: string;
  viewport?: { width: number; height: number };
  timeout?: number;
  captureScreenshots?: boolean;
  captureVideo?: boolean;
}

app.post('/api/v1/execute/test', async (req: Request, res: Response) => {
  const requestId = req.headers['x-request-id'] as string;
  const startTime = Date.now();

  try {
    const body: TestExecutionRequest = req.body;

    if (!body.testId || !body.steps || !body.baseUrl) {
      return res.status(400).json({
        error: 'testId, steps, and baseUrl are required'
      });
    }

    logger.info('Starting test execution', {
      requestId,
      testId: body.testId,
      stepsCount: body.steps.length
    });

    const config: TestExecutionConfig = {
      testId: body.testId,
      steps: body.steps,
      baseUrl: body.baseUrl,
      viewport: body.viewport || { width: 1920, height: 1080 },
      timeout: body.timeout || 30000,
      captureScreenshots: body.captureScreenshots ?? true,
      captureVideo: body.captureVideo ?? false
    };

    const result: TestExecutionResult = await runTestExecution(config);

    const duration = Date.now() - startTime;
    logger.info('Test execution completed', {
      requestId,
      testId: body.testId,
      passed: result.passed,
      duration
    });

    res.json({
      success: true,
      requestId,
      duration,
      result
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Test execution failed', { requestId, error });

    res.status(500).json({
      success: false,
      requestId,
      error: error instanceof Error ? error.message : 'Test execution failed'
    });
  }
});

// ============================================
// Element Extraction Endpoint
// ============================================

app.post('/api/v1/extract/elements', async (req: Request, res: Response) => {
  const requestId = req.headers['x-request-id'] as string;
  const startTime = Date.now();

  try {
    const { url, selectors } = req.body;

    if (!url) {
      return res.status(400).json({ error: 'url is required' });
    }

    // Use visual capture with DOM extraction
    const config: VisualCaptureConfig = {
      url,
      viewport: { width: 1920, height: 1080 },
      captureDom: true,
      captureStyles: true
    };

    const result = await captureVisual(config);

    const duration = Date.now() - startTime;
    res.json({
      success: true,
      requestId,
      duration,
      elements: result.elements,
      forms: result.forms,
      links: result.links
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    logger.error('Element extraction failed', { requestId, error });

    res.status(500).json({
      success: false,
      requestId,
      error: error instanceof Error ? error.message : 'Element extraction failed'
    });
  }
});

// Error handler
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error('Unhandled error', { error: err.message, stack: err.stack });
  res.status(500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  logger.info(`Argus Crawlee Service started`, {
    port: PORT,
    environment: process.env.NODE_ENV || 'development'
  });
});

export default app;
