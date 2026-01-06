/**
 * Argus Cache Utilities
 * KV-based caching for deduplication, rate limiting, and response caching
 */

// TTL constants in seconds
export const TTL = {
  DEDUP: 86400,           // 24 hours - Event deduplication
  SESSION: 86400,         // 24 hours - User sessions
  QUALITY_SCORE: 300,     // 5 minutes - Quality scores
  LLM_RESPONSE: 604800,   // 7 days - LLM response caching
  RATE_LIMIT: 120,        // 2 minutes - Rate limiting windows
  WEBHOOK_SIG: 3600,      // 1 hour - Webhook signature cache
  API_KEY: 3600,          // 1 hour - API key validation cache
} as const;

// Key prefixes for organization
export const KEY_PREFIX = {
  DEDUP: 'dedup',
  SESSION: 'session',
  QUALITY: 'api:quality',
  LLM: 'llm',
  RATE: 'rate',
  WEBHOOK: 'webhook:sig',
  API_KEY: 'apikey',
} as const;

/**
 * Cache helper class for Cloudflare KV operations
 */
export class ArgusCache {
  constructor(private kv: KVNamespace | null) {}

  /**
   * Check if cache is available
   */
  isAvailable(): boolean {
    return this.kv !== null;
  }

  /**
   * Get a value from cache
   */
  async get<T = string>(key: string): Promise<T | null> {
    if (!this.kv) return null;
    try {
      const value = await this.kv.get(key);
      if (!value) return null;
      try {
        return JSON.parse(value) as T;
      } catch {
        return value as unknown as T;
      }
    } catch (error) {
      console.error(`Cache get error for ${key}:`, error);
      return null;
    }
  }

  /**
   * Set a value in cache with TTL
   */
  async set(key: string, value: unknown, ttlSeconds: number = TTL.DEDUP): Promise<boolean> {
    if (!this.kv) return false;
    try {
      const serialized = typeof value === 'string' ? value : JSON.stringify(value);
      await this.kv.put(key, serialized, { expirationTtl: ttlSeconds });
      return true;
    } catch (error) {
      console.error(`Cache set error for ${key}:`, error);
      return false;
    }
  }

  /**
   * Delete a value from cache
   */
  async delete(key: string): Promise<boolean> {
    if (!this.kv) return false;
    try {
      await this.kv.delete(key);
      return true;
    } catch (error) {
      console.error(`Cache delete error for ${key}:`, error);
      return false;
    }
  }

  // =========================================================================
  // DEDUPLICATION
  // =========================================================================

  /**
   * Generate a deduplication fingerprint for an event
   */
  generateFingerprint(event: {
    source: string;
    message?: string;
    stackTrace?: string;
    file?: string;
    line?: number;
  }): string {
    const components = [
      event.source,
      event.message?.substring(0, 200) || '',
      event.file || '',
      event.line?.toString() || '',
      // Extract top 3 stack frames if available
      event.stackTrace?.split('\n').slice(0, 3).join('|') || '',
    ];
    return this.hashString(components.join(':'));
  }

  /**
   * Check if an event is a duplicate
   */
  async isDuplicate(fingerprint: string): Promise<boolean> {
    const key = `${KEY_PREFIX.DEDUP}:${fingerprint}`;
    const existing = await this.get(key);
    return existing !== null;
  }

  /**
   * Mark an event as seen (for deduplication)
   */
  async markAsSeen(fingerprint: string, metadata?: Record<string, unknown>): Promise<boolean> {
    const key = `${KEY_PREFIX.DEDUP}:${fingerprint}`;
    return await this.set(key, {
      firstSeen: new Date().toISOString(),
      count: 1,
      ...metadata,
    }, TTL.DEDUP);
  }

  /**
   * Check and mark atomically - returns true if new, false if duplicate
   */
  async checkAndMark(fingerprint: string, metadata?: Record<string, unknown>): Promise<boolean> {
    const isDupe = await this.isDuplicate(fingerprint);
    if (isDupe) return false;
    await this.markAsSeen(fingerprint, metadata);
    return true;
  }

  // =========================================================================
  // RATE LIMITING
  // =========================================================================

  /**
   * Check rate limit for a project/org
   * Returns { allowed: boolean, remaining: number, resetAt: number }
   */
  async checkRateLimit(
    identifier: string,
    limit: number,
    windowSeconds: number = 60
  ): Promise<{ allowed: boolean; remaining: number; resetAt: number }> {
    if (!this.kv) {
      return { allowed: true, remaining: limit, resetAt: Date.now() + windowSeconds * 1000 };
    }

    const windowKey = Math.floor(Date.now() / (windowSeconds * 1000));
    const key = `${KEY_PREFIX.RATE}:${identifier}:${windowKey}`;

    const current = await this.get<number>(key) || 0;
    const resetAt = (windowKey + 1) * windowSeconds * 1000;

    if (current >= limit) {
      return { allowed: false, remaining: 0, resetAt };
    }

    // Increment count
    await this.set(key, current + 1, windowSeconds);

    return {
      allowed: true,
      remaining: limit - current - 1,
      resetAt,
    };
  }

  /**
   * Get current rate limit status without incrementing
   */
  async getRateLimitStatus(
    identifier: string,
    limit: number,
    windowSeconds: number = 60
  ): Promise<{ count: number; remaining: number; resetAt: number }> {
    const windowKey = Math.floor(Date.now() / (windowSeconds * 1000));
    const key = `${KEY_PREFIX.RATE}:${identifier}:${windowKey}`;

    const current = await this.get<number>(key) || 0;
    const resetAt = (windowKey + 1) * windowSeconds * 1000;

    return {
      count: current,
      remaining: Math.max(0, limit - current),
      resetAt,
    };
  }

  // =========================================================================
  // LLM RESPONSE CACHING
  // =========================================================================

  /**
   * Generate cache key for LLM request
   */
  generateLLMCacheKey(params: {
    model: string;
    prompt: string;
    systemPrompt?: string;
    maxTokens?: number;
  }): string {
    const normalized = [
      params.model,
      params.systemPrompt || '',
      params.prompt,
      params.maxTokens?.toString() || '',
    ].join(':');
    return `${KEY_PREFIX.LLM}:${this.hashString(normalized)}`;
  }

  /**
   * Get cached LLM response
   */
  async getCachedLLMResponse(cacheKey: string): Promise<{
    response: string;
    cachedAt: string;
    inputTokens: number;
    outputTokens: number;
  } | null> {
    return await this.get(cacheKey);
  }

  /**
   * Cache an LLM response
   */
  async cacheLLMResponse(
    cacheKey: string,
    response: string,
    inputTokens: number,
    outputTokens: number
  ): Promise<boolean> {
    return await this.set(cacheKey, {
      response,
      cachedAt: new Date().toISOString(),
      inputTokens,
      outputTokens,
    }, TTL.LLM_RESPONSE);
  }

  // =========================================================================
  // QUALITY SCORE CACHING
  // =========================================================================

  /**
   * Get cached quality score
   */
  async getCachedQualityScore(projectId: string): Promise<{
    score: number;
    factors: Record<string, number>;
    calculatedAt: string;
  } | null> {
    const key = `${KEY_PREFIX.QUALITY}:score:${projectId}`;
    return await this.get(key);
  }

  /**
   * Cache a quality score
   */
  async cacheQualityScore(
    projectId: string,
    score: number,
    factors: Record<string, number>
  ): Promise<boolean> {
    const key = `${KEY_PREFIX.QUALITY}:score:${projectId}`;
    return await this.set(key, {
      score,
      factors,
      calculatedAt: new Date().toISOString(),
    }, TTL.QUALITY_SCORE);
  }

  // =========================================================================
  // SESSION MANAGEMENT
  // =========================================================================

  /**
   * Get session data
   */
  async getSession(sessionId: string): Promise<Record<string, unknown> | null> {
    const key = `${KEY_PREFIX.SESSION}:${sessionId}`;
    return await this.get(key);
  }

  /**
   * Set session data
   */
  async setSession(sessionId: string, data: Record<string, unknown>): Promise<boolean> {
    const key = `${KEY_PREFIX.SESSION}:${sessionId}`;
    return await this.set(key, {
      ...data,
      updatedAt: new Date().toISOString(),
    }, TTL.SESSION);
  }

  /**
   * Delete session
   */
  async deleteSession(sessionId: string): Promise<boolean> {
    const key = `${KEY_PREFIX.SESSION}:${sessionId}`;
    return await this.delete(key);
  }

  // =========================================================================
  // API KEY CACHING
  // =========================================================================

  /**
   * Get cached API key validation result
   */
  async getCachedApiKey(keyHash: string): Promise<{
    valid: boolean;
    organizationId: string;
    scopes: string[];
    expiresAt?: string;
  } | null> {
    const key = `${KEY_PREFIX.API_KEY}:${keyHash}`;
    return await this.get(key);
  }

  /**
   * Cache API key validation result
   */
  async cacheApiKey(
    keyHash: string,
    data: {
      valid: boolean;
      organizationId: string;
      scopes: string[];
      expiresAt?: string;
    }
  ): Promise<boolean> {
    const key = `${KEY_PREFIX.API_KEY}:${keyHash}`;
    return await this.set(key, data, TTL.API_KEY);
  }

  /**
   * Invalidate cached API key
   */
  async invalidateApiKey(keyHash: string): Promise<boolean> {
    const key = `${KEY_PREFIX.API_KEY}:${keyHash}`;
    return await this.delete(key);
  }

  // =========================================================================
  // WEBHOOK SIGNATURE CACHING
  // =========================================================================

  /**
   * Check if webhook signature was recently validated
   */
  async isWebhookSignatureValid(signatureHash: string): Promise<boolean> {
    const key = `${KEY_PREFIX.WEBHOOK}:${signatureHash}`;
    return (await this.get(key)) !== null;
  }

  /**
   * Cache a validated webhook signature
   */
  async cacheWebhookSignature(signatureHash: string): Promise<boolean> {
    const key = `${KEY_PREFIX.WEBHOOK}:${signatureHash}`;
    return await this.set(key, { validatedAt: new Date().toISOString() }, TTL.WEBHOOK_SIG);
  }

  // =========================================================================
  // UTILITY METHODS
  // =========================================================================

  /**
   * Simple hash function for generating cache keys
   */
  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    // Convert to positive hex string
    return Math.abs(hash).toString(16).padStart(8, '0');
  }

  /**
   * Create a namespaced cache for a specific organization
   */
  withOrganization(organizationId: string): OrganizationCache {
    return new OrganizationCache(this, organizationId);
  }
}

/**
 * Organization-scoped cache wrapper
 */
export class OrganizationCache {
  constructor(
    private cache: ArgusCache,
    private organizationId: string
  ) {}

  private prefixKey(key: string): string {
    return `org:${this.organizationId}:${key}`;
  }

  async get<T>(key: string): Promise<T | null> {
    return this.cache.get<T>(this.prefixKey(key));
  }

  async set(key: string, value: unknown, ttlSeconds?: number): Promise<boolean> {
    return this.cache.set(this.prefixKey(key), value, ttlSeconds);
  }

  async delete(key: string): Promise<boolean> {
    return this.cache.delete(this.prefixKey(key));
  }

  async checkRateLimit(limit: number, windowSeconds?: number) {
    return this.cache.checkRateLimit(this.organizationId, limit, windowSeconds);
  }
}

/**
 * Create cache instance from KV binding
 */
export function createCache(kv: KVNamespace | null | undefined): ArgusCache {
  return new ArgusCache(kv || null);
}
