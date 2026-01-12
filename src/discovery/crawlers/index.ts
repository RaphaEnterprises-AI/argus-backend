/**
 * Discovery Crawlers Module
 *
 * Provides web crawling functionality for the Discovery Intelligence Platform
 * using Crawlee with Playwright, element extraction, and Claude Vision AI analysis.
 */

// Main crawler
export {
  runDiscoveryCrawl,
  buildGraphRelationships,
  type CrawlConfig,
  type CrawlResult,
  type CrawlStats,
  type DiscoveredPage,
  type PageGraph,
  type AuthConfig,
} from './crawlee_crawler';

// Element extraction
export {
  extractElements,
  extractInteractiveElements,
  extractForms,
  type DiscoveredElement,
  type ElementExtractionResult,
  type FormElement,
  type InputElement,
  type NavigationElement,
  type LinkElement,
  type ImageElement,
  type HeadingElement,
  type LandmarkElement,
} from './element_extractor';

// Page analysis
export {
  analyzeWithClaudeVision,
  classifyPageType,
  type VisionAnalysisConfig,
  type VisionAnalysisResult,
  type PageType,
  type UserFlow,
  type TestSuggestion,
  type AccessibilityIssue,
  type ContentSection,
  type InteractionPoint,
} from './page_analyzer';
