# Argus Self-Healing System Design

## Goal: 95%+ Test Pass Rate

Achieve industry-leading reliability through intelligent self-healing that automatically recovers from:
- Selector changes (DOM structure modifications)
- Timing issues (dynamic content loading)
- Visual/layout changes
- Network flakiness

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SELF-HEALING ENGINE                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────┬───────────┼───────────┬───────────────┐
        ▼               ▼           ▼           ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   SELECTOR    │ │    SMART      │ │   VISUAL      │ │   PATTERN     │
│   HEALER      │ │    WAITS      │ │   MATCHER     │ │   LEARNER     │
├───────────────┤ ├───────────────┤ ├───────────────┤ ├───────────────┤
│ • Fallbacks   │ │ • DOM Ready   │ │ • Screenshot  │ │ • Vectorize   │
│ • AI Generate │ │ • Network Idle│ │ • Element Find│ │ • Store/Query │
│ • Text Match  │ │ • Animation   │ │ • OCR Text    │ │ • Confidence  │
│ • XPath       │ │ • Custom Wait │ │ • Similarity  │ │ • Ranking     │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │               │                   │               │
        └───────────────┴───────────────────┴───────────────┘
                                    │
                        ┌───────────────────────┐
                        │   HEALING METRICS     │
                        │   & ANALYTICS         │
                        └───────────────────────┘
```

---

## Component 1: Enhanced Selector Healer

### 1.1 Multi-Strategy Selector Generation

```typescript
interface SelectorStrategy {
  name: string;
  priority: number;  // 1-10, higher = try first
  confidence: number; // 0-1, likelihood of success
  generate: (context: ElementContext) => string[];
}

const strategies: SelectorStrategy[] = [
  // Strategy 1: Data attributes (most stable)
  {
    name: "data-attributes",
    priority: 10,
    confidence: 0.95,
    generate: (ctx) => [
      `[data-testid="${ctx.testId}"]`,
      `[data-cy="${ctx.testId}"]`,
      `[data-test="${ctx.testId}"]`,
      `[data-automation="${ctx.testId}"]`,
    ]
  },

  // Strategy 2: ARIA labels (semantic, stable)
  {
    name: "aria",
    priority: 9,
    confidence: 0.90,
    generate: (ctx) => [
      `[aria-label="${ctx.ariaLabel}"]`,
      `[aria-labelledby="${ctx.ariaLabelledBy}"]`,
      `[role="${ctx.role}"]`,
    ]
  },

  // Strategy 3: Text content (human-readable)
  {
    name: "text",
    priority: 8,
    confidence: 0.85,
    generate: (ctx) => [
      `text="${ctx.text}"`,  // Playwright text selector
      `text=${ctx.text}`,
      `:has-text("${ctx.text}")`,
      `button:has-text("${ctx.text}")`,
      `a:has-text("${ctx.text}")`,
    ]
  },

  // Strategy 4: Form field attributes
  {
    name: "form-fields",
    priority: 7,
    confidence: 0.85,
    generate: (ctx) => [
      `[name="${ctx.name}"]`,
      `[placeholder="${ctx.placeholder}"]`,
      `[type="${ctx.type}"]`,
      `label:has-text("${ctx.labelText}") + input`,
      `label:has-text("${ctx.labelText}") input`,
    ]
  },

  // Strategy 5: Structural (less stable but broad)
  {
    name: "structural",
    priority: 5,
    confidence: 0.70,
    generate: (ctx) => [
      `${ctx.tagName}#${ctx.id}`,
      `${ctx.tagName}.${ctx.classes.join(".")}`,
      `${ctx.tagName}[class*="${ctx.primaryClass}"]`,
    ]
  },

  // Strategy 6: XPath (last resort, very specific)
  {
    name: "xpath",
    priority: 3,
    confidence: 0.60,
    generate: (ctx) => [
      `xpath=//${ctx.tagName}[contains(text(), "${ctx.text}")]`,
      `xpath=//${ctx.tagName}[@${ctx.attributeName}="${ctx.attributeValue}"]`,
    ]
  },

  // Strategy 7: AI-Generated (smart fallback)
  {
    name: "ai-generated",
    priority: 6,
    confidence: 0.80,
    generate: async (ctx, ai) => {
      return await ai.generateSelectors(ctx);
    }
  },
];
```

### 1.2 AI-Powered Selector Generation

When static strategies fail, use AI to analyze the page and generate new selectors:

```typescript
interface AIGeneratorConfig {
  model: string;
  maxSelectors: number;
  includeScreenshot: boolean;
}

async function aiGenerateSelectors(
  pageHTML: string,
  targetDescription: string,
  screenshot: string | null,
  config: AIGeneratorConfig
): Promise<string[]> {
  const prompt = `
You are an expert at finding HTML elements. Given the page HTML and a description of the target element, generate CSS selectors that will find it.

TARGET ELEMENT: ${targetDescription}

PAGE HTML (truncated):
${pageHTML.slice(0, 10000)}

Generate ${config.maxSelectors} CSS selectors, ordered by reliability:
1. Prefer data-testid, data-cy, aria-label attributes
2. Fall back to semantic selectors (role, type, name)
3. Use text content if other attributes unavailable
4. Avoid brittle selectors (nth-child, deep nesting)

Return ONLY a JSON array of selector strings.
`;

  const response = await callAI(prompt, {
    model: config.model,
    images: config.includeScreenshot ? [screenshot] : undefined,
  });

  return JSON.parse(response);
}
```

---

## Component 2: Smart Wait Strategies

### 2.1 Intelligent Wait System

```typescript
interface WaitStrategy {
  name: string;
  timeout: number;
  condition: (page: Page) => Promise<boolean>;
}

const waitStrategies: WaitStrategy[] = [
  // Wait for network idle
  {
    name: "network-idle",
    timeout: 10000,
    condition: async (page) => {
      return page.waitForLoadState("networkidle");
    }
  },

  // Wait for DOM stable
  {
    name: "dom-stable",
    timeout: 5000,
    condition: async (page) => {
      let previousHTML = "";
      for (let i = 0; i < 5; i++) {
        const currentHTML = await page.content();
        if (currentHTML === previousHTML) return true;
        previousHTML = currentHTML;
        await page.waitForTimeout(200);
      }
      return false;
    }
  },

  // Wait for animations complete
  {
    name: "animations-complete",
    timeout: 3000,
    condition: async (page) => {
      return page.evaluate(() => {
        return document.getAnimations().every(a =>
          a.playState === "finished" || a.playState === "idle"
        );
      });
    }
  },

  // Wait for no spinners/loaders
  {
    name: "no-loaders",
    timeout: 10000,
    condition: async (page) => {
      const loaderSelectors = [
        ".loading", ".spinner", ".loader",
        "[class*='loading']", "[class*='spinner']",
        "[aria-busy='true']", ".skeleton"
      ];
      for (const sel of loaderSelectors) {
        const visible = await page.isVisible(sel).catch(() => false);
        if (visible) return false;
      }
      return true;
    }
  },

  // Wait for specific element
  {
    name: "element-visible",
    timeout: 10000,
    condition: async (page, selector) => {
      return page.waitForSelector(selector, { state: "visible" });
    }
  },
];

// Composite wait that combines strategies
async function smartWait(
  page: Page,
  strategies: string[] = ["network-idle", "dom-stable", "no-loaders"],
  maxTimeout: number = 15000
): Promise<void> {
  const startTime = Date.now();

  for (const strategyName of strategies) {
    const strategy = waitStrategies.find(s => s.name === strategyName);
    if (!strategy) continue;

    const remainingTime = maxTimeout - (Date.now() - startTime);
    if (remainingTime <= 0) break;

    try {
      await Promise.race([
        strategy.condition(page),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error("timeout")),
          Math.min(strategy.timeout, remainingTime))
        )
      ]);
    } catch {
      // Strategy timed out, continue to next
    }
  }
}
```

---

## Component 3: Visual Element Matcher

### 3.1 Screenshot-Based Element Finding

When CSS selectors fail, use visual AI to locate elements:

```typescript
interface VisualMatchResult {
  found: boolean;
  confidence: number;
  boundingBox: { x: number; y: number; width: number; height: number } | null;
  suggestedSelector: string | null;
}

async function visualElementMatch(
  page: Page,
  elementDescription: string,
  previousScreenshot?: string
): Promise<VisualMatchResult> {
  // Take current screenshot
  const currentScreenshot = await page.screenshot({
    type: "png",
    fullPage: false
  });

  const prompt = `
Analyze this webpage screenshot and find the element described below.

ELEMENT TO FIND: ${elementDescription}

Return a JSON object with:
{
  "found": boolean,
  "confidence": number (0-1),
  "x": number (center x coordinate),
  "y": number (center y coordinate),
  "width": number,
  "height": number,
  "suggestedSelector": "CSS selector if identifiable"
}

If the element is not visible, return {"found": false, "confidence": 0}
`;

  const response = await callVisionAI(prompt, currentScreenshot);
  const result = JSON.parse(response);

  return {
    found: result.found,
    confidence: result.confidence,
    boundingBox: result.found ? {
      x: result.x,
      y: result.y,
      width: result.width,
      height: result.height
    } : null,
    suggestedSelector: result.suggestedSelector
  };
}

// Click element by visual location
async function visualClick(
  page: Page,
  elementDescription: string
): Promise<boolean> {
  const match = await visualElementMatch(page, elementDescription);

  if (!match.found || !match.boundingBox || match.confidence < 0.7) {
    return false;
  }

  // Click at the center of the found element
  await page.mouse.click(
    match.boundingBox.x + match.boundingBox.width / 2,
    match.boundingBox.y + match.boundingBox.height / 2
  );

  return true;
}
```

### 3.2 OCR-Based Text Finding

For text inputs and labels:

```typescript
async function findElementByVisibleText(
  page: Page,
  text: string,
  elementType: "button" | "link" | "input" | "any" = "any"
): Promise<string | null> {
  const screenshot = await page.screenshot({ type: "png" });

  const prompt = `
Find the ${elementType} element containing or labeled with the text: "${text}"

Return the best CSS selector to target this element, or null if not found.
Consider:
1. The element's visible text
2. Adjacent labels
3. Placeholder text
4. ARIA labels

Return ONLY a JSON object: {"selector": "..." or null}
`;

  const response = await callVisionAI(prompt, screenshot);
  const result = JSON.parse(response);
  return result.selector;
}
```

---

## Component 4: Pattern Learning with Vectorize

### 4.1 Healing Pattern Storage

```typescript
interface HealingPattern {
  id: string;
  originalSelector: string;
  healedSelector: string;
  elementDescription: string;
  pageUrl: string;
  pageContext: string;  // Surrounding HTML snippet
  confidence: number;
  successCount: number;
  failureCount: number;
  lastUsed: Date;
  embedding?: number[];
}

// Store a successful healing pattern
async function storeHealingPattern(
  env: Env,
  pattern: Omit<HealingPattern, "id" | "embedding">
): Promise<void> {
  // Generate embedding for semantic search
  const embedding = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
    text: `${pattern.elementDescription} ${pattern.pageContext}`
  });

  const id = crypto.randomUUID();

  // Store in Vectorize
  await env.VECTOR_INDEX.upsert([{
    id,
    values: embedding.data[0],
    metadata: {
      originalSelector: pattern.originalSelector,
      healedSelector: pattern.healedSelector,
      elementDescription: pattern.elementDescription,
      pageUrl: pattern.pageUrl,
      confidence: pattern.confidence,
      successCount: pattern.successCount,
    }
  }]);
}

// Query for similar healing patterns
async function findSimilarPatterns(
  env: Env,
  elementDescription: string,
  pageContext: string,
  topK: number = 5
): Promise<HealingPattern[]> {
  // Generate embedding for query
  const embedding = await env.AI.run("@cf/baai/bge-base-en-v1.5", {
    text: `${elementDescription} ${pageContext}`
  });

  // Query Vectorize
  const results = await env.VECTOR_INDEX.query(embedding.data[0], {
    topK,
    returnMetadata: true,
  });

  return results.matches.map(match => ({
    id: match.id,
    originalSelector: match.metadata.originalSelector,
    healedSelector: match.metadata.healedSelector,
    elementDescription: match.metadata.elementDescription,
    pageUrl: match.metadata.pageUrl,
    confidence: match.metadata.confidence * match.score,
    successCount: match.metadata.successCount,
    failureCount: 0,
    lastUsed: new Date(),
  }));
}
```

### 4.2 Learning from Failures

```typescript
async function updatePatternFeedback(
  env: Env,
  patternId: string,
  success: boolean
): Promise<void> {
  // Get existing pattern
  const existing = await env.VECTOR_INDEX.getByIds([patternId]);
  if (!existing.length) return;

  const pattern = existing[0];
  const newSuccessCount = pattern.metadata.successCount + (success ? 1 : 0);
  const newFailureCount = (pattern.metadata.failureCount || 0) + (success ? 0 : 1);

  // Calculate new confidence
  const totalAttempts = newSuccessCount + newFailureCount;
  const newConfidence = newSuccessCount / totalAttempts;

  // Update or remove pattern based on confidence
  if (newConfidence < 0.3 && totalAttempts > 10) {
    // Pattern is unreliable, remove it
    await env.VECTOR_INDEX.deleteByIds([patternId]);
  } else {
    // Update pattern with new stats
    await env.VECTOR_INDEX.upsert([{
      id: patternId,
      values: pattern.values,
      metadata: {
        ...pattern.metadata,
        successCount: newSuccessCount,
        failureCount: newFailureCount,
        confidence: newConfidence,
      }
    }]);
  }
}
```

---

## Component 5: Unified Healing Pipeline

### 5.1 Main Healing Function

```typescript
interface HealingResult {
  success: boolean;
  usedSelector: string | null;
  healingMethod: string;
  confidence: number;
  attempts: number;
  timeTaken: number;
}

async function healAndExecute(
  session: BrowserSession,
  action: Action,
  env: Env,
  options: HealingOptions = {}
): Promise<HealingResult> {
  const startTime = Date.now();
  let attempts = 0;

  // Phase 1: Try original selector with smart waits
  attempts++;
  try {
    await smartWait(session.page);
    await executeAction(session, action.selector, action.type, action.value);
    return {
      success: true,
      usedSelector: action.selector,
      healingMethod: "original",
      confidence: 1.0,
      attempts,
      timeTaken: Date.now() - startTime,
    };
  } catch {}

  // Phase 2: Try learned patterns from Vectorize
  const pageContext = await getPageContext(session.page, action.selector);
  const similarPatterns = await findSimilarPatterns(
    env,
    action.description || action.selector,
    pageContext
  );

  for (const pattern of similarPatterns) {
    attempts++;
    try {
      await executeAction(session, pattern.healedSelector, action.type, action.value);
      await updatePatternFeedback(env, pattern.id, true);
      return {
        success: true,
        usedSelector: pattern.healedSelector,
        healingMethod: "learned-pattern",
        confidence: pattern.confidence,
        attempts,
        timeTaken: Date.now() - startTime,
      };
    } catch {
      await updatePatternFeedback(env, pattern.id, false);
    }
  }

  // Phase 3: Try static fallback selectors
  const fallbacks = generateSelectorFallbacks(action.selector, action.description);
  for (const selector of fallbacks) {
    attempts++;
    try {
      await executeAction(session, selector, action.type, action.value);

      // Store successful healing pattern
      await storeHealingPattern(env, {
        originalSelector: action.selector,
        healedSelector: selector,
        elementDescription: action.description || "",
        pageUrl: session.page.url(),
        pageContext,
        confidence: 0.8,
        successCount: 1,
        failureCount: 0,
        lastUsed: new Date(),
      });

      return {
        success: true,
        usedSelector: selector,
        healingMethod: "static-fallback",
        confidence: 0.8,
        attempts,
        timeTaken: Date.now() - startTime,
      };
    } catch {}
  }

  // Phase 4: AI-generated selectors
  if (options.useAI !== false) {
    attempts++;
    try {
      const pageHTML = await session.page.content();
      const aiSelectors = await aiGenerateSelectors(
        pageHTML,
        action.description || action.selector,
        options.includeScreenshot ? await session.screenshot() : null,
        { model: "@cf/qwen/qwen2.5-coder-32b-instruct", maxSelectors: 5, includeScreenshot: true }
      );

      for (const selector of aiSelectors) {
        attempts++;
        try {
          await executeAction(session, selector, action.type, action.value);

          await storeHealingPattern(env, {
            originalSelector: action.selector,
            healedSelector: selector,
            elementDescription: action.description || "",
            pageUrl: session.page.url(),
            pageContext,
            confidence: 0.75,
            successCount: 1,
            failureCount: 0,
            lastUsed: new Date(),
          });

          return {
            success: true,
            usedSelector: selector,
            healingMethod: "ai-generated",
            confidence: 0.75,
            attempts,
            timeTaken: Date.now() - startTime,
          };
        } catch {}
      }
    } catch {}
  }

  // Phase 5: Visual element matching (last resort)
  if (options.useVisual !== false && action.description) {
    attempts++;
    try {
      const clicked = await visualClick(session.page, action.description);
      if (clicked) {
        return {
          success: true,
          usedSelector: null,
          healingMethod: "visual-match",
          confidence: 0.7,
          attempts,
          timeTaken: Date.now() - startTime,
        };
      }
    } catch {}
  }

  // All healing attempts failed
  return {
    success: false,
    usedSelector: null,
    healingMethod: "none",
    confidence: 0,
    attempts,
    timeTaken: Date.now() - startTime,
  };
}
```

---

## Component 6: Healing Metrics & Analytics

### 6.1 Metrics Collection

```typescript
interface HealingMetrics {
  totalAttempts: number;
  successfulHeals: number;
  failedHeals: number;

  // Breakdown by method
  byMethod: {
    original: { success: number; total: number };
    learnedPattern: { success: number; total: number };
    staticFallback: { success: number; total: number };
    aiGenerated: { success: number; total: number };
    visualMatch: { success: number; total: number };
  };

  // Performance
  averageHealingTime: number;
  averageAttempts: number;

  // Overall pass rate
  passRate: number;
}

async function recordHealingMetrics(
  env: Env,
  projectId: string,
  result: HealingResult
): Promise<void> {
  // Store in Supabase for analytics
  if (env.SUPABASE_URL && env.SUPABASE_SERVICE_KEY) {
    await fetch(`${env.SUPABASE_URL}/rest/v1/healing_metrics`, {
      method: "POST",
      headers: {
        "apikey": env.SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        project_id: projectId,
        success: result.success,
        method: result.healingMethod,
        confidence: result.confidence,
        attempts: result.attempts,
        time_taken: result.timeTaken,
        timestamp: new Date().toISOString(),
      }),
    });
  }
}

async function getHealingAnalytics(
  env: Env,
  projectId: string,
  timeRange: string = "7d"
): Promise<HealingMetrics> {
  // Query Supabase for aggregated metrics
  const response = await fetch(
    `${env.SUPABASE_URL}/rest/v1/rpc/get_healing_metrics`,
    {
      method: "POST",
      headers: {
        "apikey": env.SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        p_project_id: projectId,
        p_time_range: timeRange,
      }),
    }
  );

  return response.json();
}
```

---

## Implementation Phases

### Phase 1: Enhanced Selector Strategies (Week 1)
- [ ] Implement multi-strategy selector generation
- [ ] Add ARIA and text-based selectors
- [ ] Improve static fallback patterns

### Phase 2: Smart Waits (Week 1)
- [ ] Implement intelligent wait system
- [ ] Add network idle detection
- [ ] Add animation/loader detection

### Phase 3: Pattern Learning (Week 2)
- [ ] Set up Vectorize index for patterns
- [ ] Implement pattern storage/retrieval
- [ ] Add confidence scoring and feedback loop

### Phase 4: AI-Powered Healing (Week 2)
- [ ] Integrate AI selector generation
- [ ] Add screenshot analysis for context
- [ ] Implement visual element matching

### Phase 5: Metrics & Analytics (Week 3)
- [ ] Create healing metrics schema
- [ ] Implement metrics collection
- [ ] Build analytics dashboard

---

## Expected Results

With full implementation:

| Healing Method | Expected Success Rate |
|----------------|----------------------|
| Original selector | 60-70% |
| Learned patterns | +15-20% |
| Static fallbacks | +5-10% |
| AI-generated | +5-10% |
| Visual matching | +3-5% |
| **Total Pass Rate** | **93-97%** |

---

## Configuration Options

```typescript
interface HealingConfig {
  // Enable/disable healing phases
  enableStaticFallbacks: boolean;
  enablePatternLearning: boolean;
  enableAIGeneration: boolean;
  enableVisualMatching: boolean;

  // Timeouts
  maxHealingTime: number;  // Maximum time to spend healing (ms)
  waitTimeout: number;     // Smart wait timeout (ms)

  // AI settings
  aiModel: string;         // Model for AI selector generation
  visionModel: string;     // Model for visual matching

  // Confidence thresholds
  minPatternConfidence: number;  // Minimum confidence to use learned pattern
  minVisualConfidence: number;   // Minimum confidence for visual match

  // Pattern learning
  storeSuccessfulHeals: boolean;
  patternRetentionDays: number;
}

const defaultConfig: HealingConfig = {
  enableStaticFallbacks: true,
  enablePatternLearning: true,
  enableAIGeneration: true,
  enableVisualMatching: true,

  maxHealingTime: 30000,
  waitTimeout: 15000,

  aiModel: "@cf/qwen/qwen2.5-coder-32b-instruct",
  visionModel: "@cf/meta/llama-4-scout-17b-16e-instruct",

  minPatternConfidence: 0.5,
  minVisualConfidence: 0.7,

  storeSuccessfulHeals: true,
  patternRetentionDays: 90,
};
```
