/**
 * Authentication Handler for Crawlee Discovery
 *
 * Handles various authentication mechanisms for web crawling:
 * - Form-based authentication
 * - Cookie-based authentication
 * - Header-based authentication
 * - OAuth token authentication
 */

import { Page, BrowserContext, Cookie } from 'playwright';

/**
 * Configuration for authentication
 */
export interface AuthConfig {
  /** Type of authentication to perform */
  type: 'form' | 'cookie' | 'header' | 'oauth';

  /** URL of the login page (for form auth) */
  loginUrl?: string;

  /** Credentials for form-based authentication */
  credentials?: {
    username: string;
    password: string;
  };

  /** CSS selectors for form elements */
  selectors?: {
    usernameField: string;
    passwordField: string;
    submitButton: string;
    /** Optional selector to verify successful login */
    successIndicator?: string;
  };

  /** Cookies to inject for cookie-based auth */
  cookies?: Array<{
    name: string;
    value: string;
    domain: string;
    path?: string;
    secure?: boolean;
    httpOnly?: boolean;
  }>;

  /** Headers to set for header-based auth */
  headers?: Record<string, string>;

  /** OAuth configuration */
  oauth?: {
    provider: string;
    token: string;
  };

  /** Timeout for authentication operations in milliseconds */
  timeout?: number;
}

/** Default timeout for authentication operations (30 seconds) */
const DEFAULT_TIMEOUT = 30000;

/** Common indicators that a user is not authenticated */
const UNAUTHENTICATED_INDICATORS = [
  // Login form indicators
  'input[type="password"]',
  'form[action*="login"]',
  'form[action*="signin"]',
  '[data-testid="login-form"]',
  '#login-form',
  '.login-form',

  // Common login page text patterns (checked via page content)
  // These are handled separately in verifyAuthentication
];

/** Common URL patterns indicating redirect to login */
const LOGIN_URL_PATTERNS = [
  '/login',
  '/signin',
  '/auth',
  '/authenticate',
  '/session/new',
  '/users/sign_in',
];

/** Session expired indicators */
const SESSION_EXPIRED_PATTERNS = [
  'session expired',
  'session has expired',
  'session timed out',
  'please log in again',
  'please sign in again',
  'your session has ended',
  'unauthorized',
];

/**
 * Logger interface for consistent logging
 */
interface Logger {
  info: (message: string, ...args: unknown[]) => void;
  warn: (message: string, ...args: unknown[]) => void;
  error: (message: string, ...args: unknown[]) => void;
  debug: (message: string, ...args: unknown[]) => void;
}

/**
 * Simple console logger implementation
 */
const defaultLogger: Logger = {
  info: (message: string, ...args: unknown[]) =>
    console.log(`[AUTH INFO] ${message}`, ...args),
  warn: (message: string, ...args: unknown[]) =>
    console.warn(`[AUTH WARN] ${message}`, ...args),
  error: (message: string, ...args: unknown[]) =>
    console.error(`[AUTH ERROR] ${message}`, ...args),
  debug: (message: string, ...args: unknown[]) =>
    console.debug(`[AUTH DEBUG] ${message}`, ...args),
};

let logger: Logger = defaultLogger;

/**
 * Set a custom logger for authentication operations
 */
export function setAuthLogger(customLogger: Logger): void {
  logger = customLogger;
}

/**
 * Set up authentication for a page based on the provided configuration
 *
 * @param page - Playwright page instance
 * @param config - Authentication configuration
 * @returns Promise<boolean> - true if authentication was successful
 */
export async function setupAuthentication(
  page: Page,
  config: AuthConfig
): Promise<boolean> {
  const timeout = config.timeout ?? DEFAULT_TIMEOUT;

  logger.info(`Setting up ${config.type} authentication`);

  try {
    switch (config.type) {
      case 'form':
        return await handleFormAuth(page, config, timeout);

      case 'cookie':
        return await handleCookieAuth(page, config);

      case 'header':
        return await handleHeaderAuth(page, config);

      case 'oauth':
        return await handleOAuthAuth(page, config);

      default:
        logger.error(`Unknown authentication type: ${config.type}`);
        return false;
    }
  } catch (error) {
    logger.error(`Authentication failed: ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}

/**
 * Handle form-based authentication
 */
async function handleFormAuth(
  page: Page,
  config: AuthConfig,
  timeout: number
): Promise<boolean> {
  if (!config.loginUrl) {
    logger.error('Form auth requires loginUrl');
    return false;
  }

  if (!config.credentials) {
    logger.error('Form auth requires credentials');
    return false;
  }

  if (!config.selectors) {
    logger.error('Form auth requires selectors');
    return false;
  }

  const { loginUrl, credentials, selectors } = config;

  logger.info(`Navigating to login page: ${loginUrl}`);

  // Navigate to login page
  await page.goto(loginUrl, {
    waitUntil: 'networkidle',
    timeout,
  });

  // Wait for the login form to be visible
  logger.debug('Waiting for username field to be visible');
  await page.waitForSelector(selectors.usernameField, {
    state: 'visible',
    timeout,
  });

  // Fill in the username
  logger.debug('Filling username field');
  await page.fill(selectors.usernameField, credentials.username);

  // Fill in the password
  logger.debug('Filling password field');
  await page.fill(selectors.passwordField, credentials.password);

  // Wait a moment for any client-side validation
  await page.waitForTimeout(100);

  // Click the submit button
  logger.debug('Clicking submit button');
  await page.click(selectors.submitButton);

  // Wait for navigation or success indicator
  if (selectors.successIndicator) {
    logger.debug(`Waiting for success indicator: ${selectors.successIndicator}`);
    try {
      await page.waitForSelector(selectors.successIndicator, {
        state: 'visible',
        timeout,
      });
      logger.info('Form authentication successful - success indicator found');
      return true;
    } catch {
      logger.warn('Success indicator not found after form submission');
      return false;
    }
  } else {
    // Wait for navigation to complete
    logger.debug('Waiting for navigation after form submission');
    try {
      await page.waitForNavigation({
        waitUntil: 'networkidle',
        timeout,
      });

      // Check if we were redirected back to login (authentication failed)
      const currentUrl = page.url();
      const isStillOnLogin = LOGIN_URL_PATTERNS.some((pattern) =>
        currentUrl.toLowerCase().includes(pattern)
      );

      if (isStillOnLogin) {
        logger.warn('Still on login page after form submission - authentication may have failed');
        return false;
      }

      logger.info('Form authentication successful - navigation completed');
      return true;
    } catch {
      // Navigation timeout might mean we're on a SPA
      logger.debug('Navigation wait timed out - checking for login indicators');
      return await verifyAuthentication(page, config);
    }
  }
}

/**
 * Handle cookie-based authentication
 */
async function handleCookieAuth(page: Page, config: AuthConfig): Promise<boolean> {
  if (!config.cookies || config.cookies.length === 0) {
    logger.error('Cookie auth requires cookies array');
    return false;
  }

  const context: BrowserContext = page.context();

  // Convert our cookie format to Playwright's Cookie type
  const playwrightCookies: Cookie[] = config.cookies.map((cookie) => ({
    name: cookie.name,
    value: cookie.value,
    domain: cookie.domain,
    path: cookie.path ?? '/',
    secure: cookie.secure ?? false,
    httpOnly: cookie.httpOnly ?? false,
    sameSite: 'Lax' as const,
    expires: -1, // Session cookie
  }));

  logger.debug(`Adding ${playwrightCookies.length} cookies to browser context`);

  await context.addCookies(playwrightCookies);

  logger.info('Cookie authentication setup complete');
  return true;
}

/**
 * Handle header-based authentication
 */
async function handleHeaderAuth(page: Page, config: AuthConfig): Promise<boolean> {
  if (!config.headers || Object.keys(config.headers).length === 0) {
    logger.error('Header auth requires headers object');
    return false;
  }

  logger.debug(`Setting ${Object.keys(config.headers).length} extra HTTP headers`);

  await page.setExtraHTTPHeaders(config.headers);

  logger.info('Header authentication setup complete');
  return true;
}

/**
 * Handle OAuth token-based authentication
 */
async function handleOAuthAuth(page: Page, config: AuthConfig): Promise<boolean> {
  if (!config.oauth) {
    logger.error('OAuth auth requires oauth configuration');
    return false;
  }

  if (!config.oauth.token) {
    logger.error('OAuth auth requires token');
    return false;
  }

  const { provider, token } = config.oauth;

  logger.debug(`Setting up OAuth authentication for provider: ${provider}`);

  // Set Authorization header with Bearer token
  await page.setExtraHTTPHeaders({
    Authorization: `Bearer ${token}`,
  });

  logger.info(`OAuth authentication setup complete for provider: ${provider}`);
  return true;
}

/**
 * Verify if the current page/session is still authenticated
 *
 * @param page - Playwright page instance
 * @param config - Authentication configuration (used for context)
 * @returns Promise<boolean> - true if still authenticated
 */
export async function verifyAuthentication(
  page: Page,
  config: AuthConfig
): Promise<boolean> {
  logger.debug('Verifying authentication status');

  try {
    // Check current URL for login redirects
    const currentUrl = page.url().toLowerCase();
    const isOnLoginPage = LOGIN_URL_PATTERNS.some((pattern) =>
      currentUrl.includes(pattern)
    );

    if (isOnLoginPage) {
      logger.debug('Current URL indicates login page - not authenticated');
      return false;
    }

    // Check for login form elements on the page
    for (const selector of UNAUTHENTICATED_INDICATORS) {
      try {
        const element = await page.$(selector);
        if (element) {
          // Found a potential login form - need to verify it's actually a login form
          // and not just a password field in user settings, etc.
          const isVisible = await element.isVisible();
          if (isVisible) {
            // Check if we're on a settings or profile page (false positive)
            const pageUrl = page.url().toLowerCase();
            const isSettingsPage =
              pageUrl.includes('/settings') ||
              pageUrl.includes('/profile') ||
              pageUrl.includes('/account') ||
              pageUrl.includes('/preferences');

            if (!isSettingsPage) {
              logger.debug(`Found login indicator: ${selector} - likely not authenticated`);
              return false;
            }
          }
        }
      } catch {
        // Selector not found, continue checking
      }
    }

    // Check page content for session expired messages
    const pageContent = await page.content();
    const contentLower = pageContent.toLowerCase();

    for (const pattern of SESSION_EXPIRED_PATTERNS) {
      if (contentLower.includes(pattern)) {
        logger.debug(`Found session expired indicator: "${pattern}" - not authenticated`);
        return false;
      }
    }

    // If we have a success indicator from form auth, check for it
    if (config.type === 'form' && config.selectors?.successIndicator) {
      try {
        const successElement = await page.$(config.selectors.successIndicator);
        if (successElement && (await successElement.isVisible())) {
          logger.debug('Success indicator found - authenticated');
          return true;
        }
      } catch {
        // Success indicator not found
      }
    }

    // No negative indicators found - assume authenticated
    logger.debug('No unauthenticated indicators found - assuming authenticated');
    return true;
  } catch (error) {
    logger.error(`Error verifying authentication: ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}

/**
 * Refresh authentication if the session has expired
 *
 * @param page - Playwright page instance
 * @param config - Authentication configuration
 * @returns Promise<boolean> - true if re-authentication was successful
 */
export async function refreshAuthentication(
  page: Page,
  config: AuthConfig
): Promise<boolean> {
  logger.info('Refreshing authentication');

  try {
    // First check if we actually need to refresh
    const isAuthenticated = await verifyAuthentication(page, config);
    if (isAuthenticated) {
      logger.debug('Already authenticated - no refresh needed');
      return true;
    }

    // Clear existing authentication state based on type
    switch (config.type) {
      case 'form':
      case 'cookie':
        // Clear all cookies for a fresh start
        logger.debug('Clearing cookies before re-authentication');
        const context = page.context();
        await context.clearCookies();
        break;

      case 'header':
        // Headers will be re-set by setupAuthentication
        break;

      case 'oauth':
        // Token will be re-set by setupAuthentication
        break;
    }

    // Re-run authentication
    logger.debug('Re-running authentication setup');
    return await setupAuthentication(page, config);
  } catch (error) {
    logger.error(`Error refreshing authentication: ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}

/**
 * Create an authentication handler that can be used as middleware in crawling
 *
 * @param config - Authentication configuration
 * @returns Object with setup, verify, and refresh methods bound to the config
 */
export function createAuthHandler(config: AuthConfig) {
  return {
    /**
     * Set up authentication for a page
     */
    setup: (page: Page) => setupAuthentication(page, config),

    /**
     * Verify if still authenticated
     */
    verify: (page: Page) => verifyAuthentication(page, config),

    /**
     * Refresh authentication if expired
     */
    refresh: (page: Page) => refreshAuthentication(page, config),

    /**
     * Middleware function for use in request handlers
     * Automatically refreshes auth if needed before proceeding
     */
    middleware: async (page: Page): Promise<boolean> => {
      const isAuth = await verifyAuthentication(page, config);
      if (!isAuth) {
        logger.info('Authentication expired, refreshing...');
        return await refreshAuthentication(page, config);
      }
      return true;
    },

    /**
     * Get the underlying config
     */
    getConfig: () => ({ ...config }),
  };
}

/**
 * Helper to create form auth config
 */
export function createFormAuthConfig(
  loginUrl: string,
  credentials: { username: string; password: string },
  selectors: {
    usernameField: string;
    passwordField: string;
    submitButton: string;
    successIndicator?: string;
  },
  timeout?: number
): AuthConfig {
  return {
    type: 'form',
    loginUrl,
    credentials,
    selectors,
    timeout,
  };
}

/**
 * Helper to create cookie auth config
 */
export function createCookieAuthConfig(
  cookies: AuthConfig['cookies']
): AuthConfig {
  return {
    type: 'cookie',
    cookies,
  };
}

/**
 * Helper to create header auth config (e.g., for API keys)
 */
export function createHeaderAuthConfig(
  headers: Record<string, string>
): AuthConfig {
  return {
    type: 'header',
    headers,
  };
}

/**
 * Helper to create OAuth/Bearer token auth config
 */
export function createOAuthConfig(
  provider: string,
  token: string
): AuthConfig {
  return {
    type: 'oauth',
    oauth: {
      provider,
      token,
    },
  };
}

/**
 * Helper to create basic auth header config
 */
export function createBasicAuthConfig(
  username: string,
  password: string
): AuthConfig {
  const credentials = Buffer.from(`${username}:${password}`).toString('base64');
  return {
    type: 'header',
    headers: {
      Authorization: `Basic ${credentials}`,
    },
  };
}
