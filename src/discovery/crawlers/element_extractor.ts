/**
 * DOM Element Extractor for Crawlee Discovery
 *
 * Extracts interactive elements from web pages for test generation.
 * Uses Playwright's page.evaluate() to run extraction in browser context.
 */

import type { Page } from 'playwright';

// ============================================================================
// Interfaces
// ============================================================================

/**
 * Bounding rectangle for an element
 */
export interface ElementBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Form field information
 */
export interface FormField {
  selector: string;
  name: string;
  type: string;
  required: boolean;
  placeholder?: string;
  value?: string;
  options?: string[]; // For select elements
}

/**
 * Extracted element with all relevant metadata
 */
export interface ExtractedElement {
  /** Element type classification */
  type: 'form' | 'button' | 'link' | 'input' | 'interactive';

  /** Stable CSS selector */
  selector: string;

  /** XPath to element */
  xpath: string;

  /** Visible text content */
  text: string;

  /** Element bounding rectangle */
  bounds: ElementBounds;

  /** All element attributes */
  attributes: Record<string, string>;

  /** Whether element is currently visible */
  isVisible: boolean;

  /** Whether element is enabled (not disabled) */
  isEnabled: boolean;

  /** ARIA label if present */
  ariaLabel?: string;

  /** ARIA/DOM role */
  role?: string;

  // Type-specific fields

  /** Form: action URL */
  action?: string;

  /** Form: HTTP method */
  method?: string;

  /** Form: contained fields */
  fields?: FormField[];

  /** Link: href attribute */
  href?: string;

  /** Link: whether link points to external domain */
  isExternal?: boolean;

  /** Input: input type (text, email, password, etc.) */
  inputType?: string;

  /** Input: field name attribute */
  name?: string;

  /** Input: whether field is required */
  required?: boolean;

  /** Input: placeholder text */
  placeholder?: string;

  /** Interactive: event handlers attached */
  hasClickHandler?: boolean;

  /** Interactive: data-action attribute */
  dataAction?: string;

  /** Tag name of the element */
  tagName: string;
}

/**
 * Element category for test organization
 */
export type ElementCategory =
  | 'navigation'
  | 'form'
  | 'action'
  | 'content'
  | 'interactive'
  | 'authentication'
  | 'commerce'
  | 'social';

// ============================================================================
// Browser Context Functions (run via page.evaluate)
// ============================================================================

/**
 * Script to be executed in browser context for element extraction
 */
const browserExtractScript = `
(function() {
  // Helper: Generate stable CSS selector
  function generateSelector(element) {
    try {
      // Priority 1: ID
      if (element.id && !element.id.match(/^\\d/) && !element.id.includes(':')) {
        return '#' + CSS.escape(element.id);
      }

      // Priority 2: data-testid
      const testId = element.getAttribute('data-testid') ||
                     element.getAttribute('data-test-id') ||
                     element.getAttribute('data-cy');
      if (testId) {
        return '[data-testid="' + testId + '"]';
      }

      // Priority 3: name attribute for inputs
      if (element.name && (element.tagName === 'INPUT' || element.tagName === 'SELECT' || element.tagName === 'TEXTAREA')) {
        return element.tagName.toLowerCase() + '[name="' + element.name + '"]';
      }

      // Priority 4: Unique class combination
      if (element.className && typeof element.className === 'string') {
        const classes = element.className.trim().split(/\\s+/).filter(c =>
          c && !c.match(/^\\d/) && !c.includes(':') && c.length < 50
        );
        if (classes.length > 0) {
          const selector = element.tagName.toLowerCase() + '.' + classes.slice(0, 3).map(c => CSS.escape(c)).join('.');
          const matches = document.querySelectorAll(selector);
          if (matches.length === 1) {
            return selector;
          }
        }
      }

      // Priority 5: aria-label
      const ariaLabel = element.getAttribute('aria-label');
      if (ariaLabel) {
        const selector = element.tagName.toLowerCase() + '[aria-label="' + ariaLabel.replace(/"/g, '\\\\"') + '"]';
        const matches = document.querySelectorAll(selector);
        if (matches.length === 1) {
          return selector;
        }
      }

      // Priority 6: Text content for buttons/links
      if ((element.tagName === 'BUTTON' || element.tagName === 'A') && element.textContent) {
        const text = element.textContent.trim().substring(0, 30);
        if (text && text.length > 0) {
          // Try to find by text - return xpath-like selector hint
          return element.tagName.toLowerCase() + ':has-text("' + text.replace(/"/g, '\\\\"') + '")';
        }
      }

      // Priority 7: Build path from ancestors
      const path = [];
      let current = element;
      while (current && current.nodeType === Node.ELEMENT_NODE && current !== document.body) {
        let selector = current.tagName.toLowerCase();

        if (current.id && !current.id.match(/^\\d/)) {
          selector = '#' + CSS.escape(current.id);
          path.unshift(selector);
          break;
        }

        // Add nth-child for disambiguation
        const parent = current.parentElement;
        if (parent) {
          const siblings = Array.from(parent.children).filter(c => c.tagName === current.tagName);
          if (siblings.length > 1) {
            const index = siblings.indexOf(current) + 1;
            selector += ':nth-child(' + index + ')';
          }
        }

        path.unshift(selector);
        current = current.parentElement;

        // Limit path depth
        if (path.length >= 5) break;
      }

      return path.join(' > ');
    } catch (e) {
      return element.tagName.toLowerCase();
    }
  }

  // Helper: Generate XPath
  function generateXPath(element) {
    try {
      if (element.id) {
        return '//*[@id="' + element.id + '"]';
      }

      const parts = [];
      let current = element;

      while (current && current.nodeType === Node.ELEMENT_NODE) {
        let part = current.tagName.toLowerCase();

        if (current.id) {
          part = '*[@id="' + current.id + '"]';
          parts.unshift(part);
          break;
        }

        const parent = current.parentElement;
        if (parent) {
          const siblings = Array.from(parent.children).filter(c => c.tagName === current.tagName);
          if (siblings.length > 1) {
            const index = siblings.indexOf(current) + 1;
            part += '[' + index + ']';
          }
        }

        parts.unshift(part);
        current = parent;

        if (parts.length >= 10) break;
      }

      return '//' + parts.join('/');
    } catch (e) {
      return '//' + element.tagName.toLowerCase();
    }
  }

  // Helper: Get bounding rectangle
  function getBounds(element) {
    try {
      const rect = element.getBoundingClientRect();
      return {
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        width: Math.round(rect.width),
        height: Math.round(rect.height)
      };
    } catch (e) {
      return { x: 0, y: 0, width: 0, height: 0 };
    }
  }

  // Helper: Check visibility
  function isElementVisible(element) {
    try {
      if (!element.offsetParent && element.tagName !== 'BODY' && element.tagName !== 'HTML') {
        // Check if position:fixed
        const style = window.getComputedStyle(element);
        if (style.position !== 'fixed') {
          return false;
        }
      }

      const style = window.getComputedStyle(element);
      if (style.display === 'none') return false;
      if (style.visibility === 'hidden') return false;
      if (parseFloat(style.opacity) === 0) return false;

      const rect = element.getBoundingClientRect();
      if (rect.width === 0 && rect.height === 0) return false;

      return true;
    } catch (e) {
      return false;
    }
  }

  // Helper: Check if enabled
  function isElementEnabled(element) {
    try {
      if (element.disabled) return false;
      if (element.getAttribute('aria-disabled') === 'true') return false;
      if (element.classList.contains('disabled')) return false;
      return true;
    } catch (e) {
      return true;
    }
  }

  // Helper: Get all attributes
  function getAttributes(element) {
    const attrs = {};
    try {
      for (const attr of element.attributes) {
        attrs[attr.name] = attr.value;
      }
    } catch (e) {}
    return attrs;
  }

  // Helper: Extract form fields
  function extractFormFields(form) {
    const fields = [];
    try {
      const inputs = form.querySelectorAll('input, select, textarea');
      for (const input of inputs) {
        const field = {
          selector: generateSelector(input),
          name: input.name || '',
          type: input.type || input.tagName.toLowerCase(),
          required: input.required || input.getAttribute('aria-required') === 'true',
          placeholder: input.placeholder || undefined,
          value: input.type !== 'password' ? input.value : undefined
        };

        if (input.tagName === 'SELECT') {
          field.options = Array.from(input.options).map(o => o.text);
        }

        fields.push(field);
      }
    } catch (e) {}
    return fields;
  }

  // Helper: Check if URL is external
  function isExternalUrl(href) {
    try {
      if (!href || href.startsWith('#') || href.startsWith('javascript:')) {
        return false;
      }
      const url = new URL(href, window.location.origin);
      return url.hostname !== window.location.hostname;
    } catch (e) {
      return false;
    }
  }

  // Helper: Check for click handlers
  function hasClickHandler(element) {
    try {
      if (element.onclick) return true;
      if (element.getAttribute('onclick')) return true;
      if (element.getAttribute('data-action')) return true;
      if (element.getAttribute('ng-click')) return true;
      if (element.getAttribute('@click')) return true;
      if (element.getAttribute('v-on:click')) return true;
      // Check for React synthetic events (harder to detect)
      return false;
    } catch (e) {
      return false;
    }
  }

  // Extract elements
  const elements = [];

  // 1. Extract Forms
  document.querySelectorAll('form').forEach(form => {
    try {
      elements.push({
        type: 'form',
        selector: generateSelector(form),
        xpath: generateXPath(form),
        text: '',
        bounds: getBounds(form),
        attributes: getAttributes(form),
        isVisible: isElementVisible(form),
        isEnabled: true,
        ariaLabel: form.getAttribute('aria-label') || undefined,
        role: form.getAttribute('role') || 'form',
        action: form.action || undefined,
        method: (form.method || 'GET').toUpperCase(),
        fields: extractFormFields(form),
        tagName: 'FORM'
      });
    } catch (e) {}
  });

  // 2. Extract Buttons
  const buttonSelectors = 'button, [role="button"], input[type="submit"], input[type="button"], input[type="reset"]';
  document.querySelectorAll(buttonSelectors).forEach(button => {
    try {
      elements.push({
        type: 'button',
        selector: generateSelector(button),
        xpath: generateXPath(button),
        text: button.textContent?.trim() || button.value || '',
        bounds: getBounds(button),
        attributes: getAttributes(button),
        isVisible: isElementVisible(button),
        isEnabled: isElementEnabled(button),
        ariaLabel: button.getAttribute('aria-label') || undefined,
        role: button.getAttribute('role') || 'button',
        inputType: button.type || undefined,
        tagName: button.tagName
      });
    } catch (e) {}
  });

  // 3. Extract Links
  document.querySelectorAll('a[href]').forEach(link => {
    try {
      const href = link.href || link.getAttribute('href');
      elements.push({
        type: 'link',
        selector: generateSelector(link),
        xpath: generateXPath(link),
        text: link.textContent?.trim() || '',
        bounds: getBounds(link),
        attributes: getAttributes(link),
        isVisible: isElementVisible(link),
        isEnabled: isElementEnabled(link),
        ariaLabel: link.getAttribute('aria-label') || undefined,
        role: link.getAttribute('role') || 'link',
        href: href,
        isExternal: isExternalUrl(href),
        tagName: 'A'
      });
    } catch (e) {}
  });

  // 4. Extract Inputs (not already in forms)
  const inputSelectors = 'input:not([type="submit"]):not([type="button"]):not([type="reset"]):not([type="hidden"]), select, textarea';
  document.querySelectorAll(inputSelectors).forEach(input => {
    try {
      // Skip if inside a form we already captured
      if (input.closest('form')) return;

      const extracted = {
        type: 'input',
        selector: generateSelector(input),
        xpath: generateXPath(input),
        text: '',
        bounds: getBounds(input),
        attributes: getAttributes(input),
        isVisible: isElementVisible(input),
        isEnabled: isElementEnabled(input),
        ariaLabel: input.getAttribute('aria-label') || undefined,
        role: input.getAttribute('role') || undefined,
        inputType: input.type || input.tagName.toLowerCase(),
        name: input.name || undefined,
        required: input.required || input.getAttribute('aria-required') === 'true',
        placeholder: input.placeholder || undefined,
        tagName: input.tagName
      };

      if (input.tagName === 'SELECT') {
        extracted.options = Array.from(input.options).map(o => o.text);
      }

      elements.push(extracted);
    } catch (e) {}
  });

  // 5. Extract Interactive Elements
  const interactiveSelectors = '[onclick], [data-action], [ng-click], [@click], [v-on\\\\:click]';
  document.querySelectorAll(interactiveSelectors).forEach(el => {
    try {
      // Skip if already captured as button/link
      if (el.tagName === 'BUTTON' || el.tagName === 'A' || el.tagName === 'INPUT') return;

      elements.push({
        type: 'interactive',
        selector: generateSelector(el),
        xpath: generateXPath(el),
        text: el.textContent?.trim().substring(0, 100) || '',
        bounds: getBounds(el),
        attributes: getAttributes(el),
        isVisible: isElementVisible(el),
        isEnabled: isElementEnabled(el),
        ariaLabel: el.getAttribute('aria-label') || undefined,
        role: el.getAttribute('role') || undefined,
        hasClickHandler: true,
        dataAction: el.getAttribute('data-action') || undefined,
        tagName: el.tagName
      });
    } catch (e) {}
  });

  // Also capture elements with role="button" not already captured
  document.querySelectorAll('[role="button"]:not(button):not(input):not(a)').forEach(el => {
    try {
      const existingSelector = generateSelector(el);
      if (!elements.some(e => e.selector === existingSelector)) {
        elements.push({
          type: 'interactive',
          selector: existingSelector,
          xpath: generateXPath(el),
          text: el.textContent?.trim().substring(0, 100) || '',
          bounds: getBounds(el),
          attributes: getAttributes(el),
          isVisible: isElementVisible(el),
          isEnabled: isElementEnabled(el),
          ariaLabel: el.getAttribute('aria-label') || undefined,
          role: 'button',
          hasClickHandler: hasClickHandler(el),
          tagName: el.tagName
        });
      }
    } catch (e) {}
  });

  return elements;
})()
`;

// ============================================================================
// Main Extraction Function
// ============================================================================

/**
 * Extract all interactive elements from a page
 *
 * @param page - Playwright Page instance
 * @returns Array of extracted elements
 */
export async function extractElements(page: Page): Promise<ExtractedElement[]> {
  try {
    const elements = await page.evaluate(browserExtractScript);
    return elements as ExtractedElement[];
  } catch (error) {
    console.error('Failed to extract elements:', error);
    return [];
  }
}

// ============================================================================
// Helper Functions (for use outside browser context)
// ============================================================================

/**
 * Generate a stable CSS selector for an element (server-side utility)
 * Note: This is a simplified version for external use. The browser context
 * version inside browserExtractScript is more comprehensive.
 *
 * @param element - Element data with id, attributes, tagName, className
 * @returns CSS selector string
 */
export function generateSelector(element: {
  id?: string;
  attributes?: Record<string, string>;
  tagName: string;
  className?: string;
}): string {
  try {
    // Priority 1: ID
    if (element.id && !element.id.match(/^\d/) && !element.id.includes(':')) {
      return `#${element.id}`;
    }

    // Priority 2: data-testid
    const testId = element.attributes?.['data-testid'] ||
                   element.attributes?.['data-test-id'] ||
                   element.attributes?.['data-cy'];
    if (testId) {
      return `[data-testid="${testId}"]`;
    }

    // Priority 3: name attribute
    const name = element.attributes?.['name'];
    if (name) {
      return `${element.tagName.toLowerCase()}[name="${name}"]`;
    }

    // Priority 4: Unique class
    if (element.className) {
      const classes = element.className.trim().split(/\s+/).filter(c =>
        c && !c.match(/^\d/) && !c.includes(':') && c.length < 50
      );
      if (classes.length > 0) {
        return `${element.tagName.toLowerCase()}.${classes.slice(0, 3).join('.')}`;
      }
    }

    return element.tagName.toLowerCase();
  } catch {
    return element.tagName?.toLowerCase() || '*';
  }
}

/**
 * Generate XPath for an element (server-side utility)
 *
 * @param element - Element data with id, tagName, attributes
 * @returns XPath string
 */
export function generateXPath(element: {
  id?: string;
  tagName: string;
  attributes?: Record<string, string>;
}): string {
  try {
    if (element.id) {
      return `//*[@id="${element.id}"]`;
    }

    const name = element.attributes?.['name'];
    if (name) {
      return `//${element.tagName.toLowerCase()}[@name="${name}"]`;
    }

    return `//${element.tagName.toLowerCase()}`;
  } catch {
    return '//*';
  }
}

/**
 * Get element bounds from bounding client rect
 *
 * @param rect - DOMRect or bounds object
 * @returns ElementBounds with rounded values
 */
export function getBounds(rect: DOMRect | { x: number; y: number; width: number; height: number }): ElementBounds {
  try {
    return {
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    };
  } catch {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
}

/**
 * Check if element is visible based on computed style properties
 *
 * @param element - Element data with bounds, attributes, style
 * @returns true if element is visible
 */
export function isElementVisible(element: {
  bounds?: ElementBounds;
  attributes?: Record<string, string>;
  style?: {
    display?: string;
    visibility?: string;
    opacity?: string;
  };
}): boolean {
  try {
    // Check bounds
    if (element.bounds) {
      if (element.bounds.width === 0 && element.bounds.height === 0) {
        return false;
      }
    }

    // Check style properties
    if (element.style) {
      if (element.style.display === 'none') return false;
      if (element.style.visibility === 'hidden') return false;
      if (element.style.opacity !== undefined && parseFloat(element.style.opacity) === 0) return false;
    }

    return true;
  } catch {
    return false;
  }
}

/**
 * Check if element is enabled (not disabled)
 *
 * @param element - Element data with attributes
 * @returns true if element is enabled
 */
export function isElementEnabled(element: {
  attributes?: Record<string, string>;
}): boolean {
  try {
    if (!element.attributes) return true;

    if (element.attributes['disabled'] !== undefined) return false;
    if (element.attributes['aria-disabled'] === 'true') return false;

    const className = element.attributes['class'] || '';
    if (className.includes('disabled')) return false;

    return true;
  } catch {
    return true;
  }
}

// ============================================================================
// Element Categorization
// ============================================================================

/**
 * Keywords for element categorization
 */
const CATEGORY_KEYWORDS: Record<ElementCategory, string[]> = {
  navigation: [
    'nav', 'menu', 'header', 'footer', 'breadcrumb', 'sidebar',
    'home', 'back', 'next', 'prev', 'previous', 'pagination',
    'tab', 'link', 'href'
  ],
  form: [
    'form', 'input', 'submit', 'field', 'textarea', 'select',
    'checkbox', 'radio', 'dropdown', 'search', 'filter'
  ],
  action: [
    'button', 'btn', 'click', 'action', 'trigger', 'execute',
    'save', 'cancel', 'delete', 'remove', 'add', 'create',
    'edit', 'update', 'confirm', 'apply', 'close', 'open'
  ],
  content: [
    'article', 'post', 'content', 'text', 'description', 'body',
    'main', 'section', 'card', 'item', 'list'
  ],
  interactive: [
    'toggle', 'switch', 'slider', 'drag', 'drop', 'resize',
    'expand', 'collapse', 'accordion', 'modal', 'dialog', 'popup',
    'tooltip', 'carousel', 'gallery'
  ],
  authentication: [
    'login', 'signin', 'sign-in', 'signup', 'sign-up', 'register',
    'logout', 'signout', 'sign-out', 'password', 'email', 'username',
    'auth', 'credential', 'forgot', 'reset', 'verify', 'otp', '2fa',
    'sso', 'oauth', 'google', 'facebook', 'github'
  ],
  commerce: [
    'cart', 'basket', 'checkout', 'buy', 'purchase', 'order',
    'payment', 'pay', 'price', 'product', 'shop', 'store',
    'add-to-cart', 'wishlist', 'shipping', 'billing', 'coupon',
    'discount', 'promo'
  ],
  social: [
    'share', 'like', 'comment', 'follow', 'subscribe', 'tweet',
    'post', 'feed', 'profile', 'friend', 'connect', 'invite',
    'message', 'chat', 'notification'
  ]
};

/**
 * Categorize an extracted element based on its properties
 *
 * Uses heuristics based on selector, text, and attributes to determine
 * the most appropriate category for test organization.
 *
 * @param element - The extracted element to categorize
 * @returns The element's category
 */
export function categorizeElement(element: ExtractedElement): ElementCategory {
  try {
    // Build a searchable string from element properties
    const searchableText = [
      element.selector,
      element.text,
      element.ariaLabel,
      element.role,
      element.name,
      element.action,
      element.href,
      element.dataAction,
      element.attributes?.['class'],
      element.attributes?.['id'],
      element.attributes?.['data-testid'],
      element.tagName
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();

    // Score each category
    const scores: Record<ElementCategory, number> = {
      navigation: 0,
      form: 0,
      action: 0,
      content: 0,
      interactive: 0,
      authentication: 0,
      commerce: 0,
      social: 0
    };

    for (const [category, keywords] of Object.entries(CATEGORY_KEYWORDS)) {
      for (const keyword of keywords) {
        if (searchableText.includes(keyword)) {
          scores[category as ElementCategory] += 1;
          // Bonus for exact matches
          if (
            element.text?.toLowerCase() === keyword ||
            element.ariaLabel?.toLowerCase() === keyword
          ) {
            scores[category as ElementCategory] += 2;
          }
        }
      }
    }

    // Apply type-based scoring
    switch (element.type) {
      case 'form':
        scores.form += 3;
        break;
      case 'button':
        scores.action += 2;
        break;
      case 'link':
        scores.navigation += 2;
        break;
      case 'input':
        scores.form += 2;
        break;
      case 'interactive':
        scores.interactive += 2;
        break;
    }

    // Check for authentication-specific input types
    if (element.inputType === 'password' || element.inputType === 'email') {
      scores.authentication += 3;
    }

    // Check for external links (often social)
    if (element.isExternal && element.href) {
      const socialDomains = ['twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com', 'x.com'];
      if (socialDomains.some(d => element.href?.includes(d))) {
        scores.social += 5;
      }
    }

    // Find highest scoring category
    let maxScore = 0;
    let maxCategory: ElementCategory = 'content';

    for (const [category, score] of Object.entries(scores)) {
      if (score > maxScore) {
        maxScore = score;
        maxCategory = category as ElementCategory;
      }
    }

    // Default fallbacks based on element type if no clear winner
    if (maxScore === 0) {
      switch (element.type) {
        case 'form':
          return 'form';
        case 'button':
          return 'action';
        case 'link':
          return 'navigation';
        case 'input':
          return 'form';
        case 'interactive':
          return 'interactive';
        default:
          return 'content';
      }
    }

    return maxCategory;
  } catch {
    return 'content';
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Filter elements by visibility
 *
 * @param elements - Array of extracted elements
 * @returns Only visible elements
 */
export function filterVisibleElements(elements: ExtractedElement[]): ExtractedElement[] {
  return elements.filter(el => el.isVisible);
}

/**
 * Filter elements by type
 *
 * @param elements - Array of extracted elements
 * @param types - Types to filter for
 * @returns Elements matching the specified types
 */
export function filterByType(
  elements: ExtractedElement[],
  types: ExtractedElement['type'][]
): ExtractedElement[] {
  return elements.filter(el => types.includes(el.type));
}

/**
 * Filter elements by category
 *
 * @param elements - Array of extracted elements
 * @param categories - Categories to filter for
 * @returns Elements matching the specified categories
 */
export function filterByCategory(
  elements: ExtractedElement[],
  categories: ElementCategory[]
): ExtractedElement[] {
  return elements.filter(el => categories.includes(categorizeElement(el)));
}

/**
 * Group elements by type
 *
 * @param elements - Array of extracted elements
 * @returns Elements grouped by their type
 */
export function groupByType(
  elements: ExtractedElement[]
): Record<ExtractedElement['type'], ExtractedElement[]> {
  return elements.reduce((acc, el) => {
    if (!acc[el.type]) {
      acc[el.type] = [];
    }
    acc[el.type].push(el);
    return acc;
  }, {} as Record<ExtractedElement['type'], ExtractedElement[]>);
}

/**
 * Group elements by category
 *
 * @param elements - Array of extracted elements
 * @returns Elements grouped by their category
 */
export function groupByCategory(
  elements: ExtractedElement[]
): Record<ElementCategory, ExtractedElement[]> {
  return elements.reduce((acc, el) => {
    const category = categorizeElement(el);
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(el);
    return acc;
  }, {} as Record<ElementCategory, ExtractedElement[]>);
}

/**
 * Find element by selector
 *
 * @param elements - Array of extracted elements
 * @param selector - CSS selector to search for
 * @returns Matching element or undefined
 */
export function findBySelector(
  elements: ExtractedElement[],
  selector: string
): ExtractedElement | undefined {
  return elements.find(el => el.selector === selector);
}

/**
 * Find elements containing text
 *
 * @param elements - Array of extracted elements
 * @param text - Text to search for
 * @param options - Search options (exact match, case sensitivity)
 * @returns Elements containing the specified text
 */
export function findByText(
  elements: ExtractedElement[],
  text: string,
  options: { exact?: boolean; caseSensitive?: boolean } = {}
): ExtractedElement[] {
  const { exact = false, caseSensitive = false } = options;

  return elements.filter(el => {
    if (!el.text) return false;

    const elementText = caseSensitive ? el.text : el.text.toLowerCase();
    const searchText = caseSensitive ? text : text.toLowerCase();

    if (exact) {
      return elementText === searchText;
    }
    return elementText.includes(searchText);
  });
}

/**
 * Find elements by ARIA label
 *
 * @param elements - Array of extracted elements
 * @param label - ARIA label to search for
 * @param options - Search options
 * @returns Elements with matching ARIA label
 */
export function findByAriaLabel(
  elements: ExtractedElement[],
  label: string,
  options: { exact?: boolean; caseSensitive?: boolean } = {}
): ExtractedElement[] {
  const { exact = false, caseSensitive = false } = options;

  return elements.filter(el => {
    if (!el.ariaLabel) return false;

    const elementLabel = caseSensitive ? el.ariaLabel : el.ariaLabel.toLowerCase();
    const searchLabel = caseSensitive ? label : label.toLowerCase();

    if (exact) {
      return elementLabel === searchLabel;
    }
    return elementLabel.includes(searchLabel);
  });
}

/**
 * Get authentication-related elements
 *
 * @param elements - Array of extracted elements
 * @returns Elements related to authentication
 */
export function getAuthenticationElements(elements: ExtractedElement[]): ExtractedElement[] {
  return filterByCategory(elements, ['authentication']);
}

/**
 * Get form elements with their fields
 *
 * @param elements - Array of extracted elements
 * @returns Form elements only
 */
export function getFormElements(elements: ExtractedElement[]): ExtractedElement[] {
  return filterByType(elements, ['form']);
}

/**
 * Get navigation elements
 *
 * @param elements - Array of extracted elements
 * @returns Navigation-related elements
 */
export function getNavigationElements(elements: ExtractedElement[]): ExtractedElement[] {
  return filterByCategory(elements, ['navigation']);
}

/**
 * Get actionable elements (buttons, interactive)
 *
 * @param elements - Array of extracted elements
 * @returns Actionable elements
 */
export function getActionableElements(elements: ExtractedElement[]): ExtractedElement[] {
  return filterByType(elements, ['button', 'interactive']);
}
