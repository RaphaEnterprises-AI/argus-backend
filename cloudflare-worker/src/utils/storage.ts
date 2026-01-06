/**
 * Argus Storage Utilities
 * R2-based object storage for screenshots, artifacts, and reports
 */

// Storage key patterns
export const STORAGE_PATHS = {
  SCREENSHOTS: (projectId: string, timestamp: string) =>
    `screenshots/${projectId}/${timestamp}.png`,
  COVERAGE: (projectId: string, commitSha: string, filename: string) =>
    `coverage/${projectId}/${commitSha}/${filename}`,
  TEST_RESULTS: (projectId: string, runId: string) =>
    `test-results/${projectId}/${runId}/results.json`,
  GENERATED_TESTS: (projectId: string, timestamp: string, filename: string) =>
    `generated-tests/${projectId}/${timestamp}/${filename}`,
  ARTIFACTS: (projectId: string, category: string, filename: string) =>
    `artifacts/${projectId}/${category}/${filename}`,
} as const;

/**
 * Storage helper class for Cloudflare R2 operations
 */
export class ArgusStorage {
  constructor(private bucket: R2Bucket | null) {}

  /**
   * Check if storage is available
   */
  isAvailable(): boolean {
    return this.bucket !== null;
  }

  // =========================================================================
  // BASIC OPERATIONS
  // =========================================================================

  /**
   * Upload a file to R2
   */
  async upload(
    key: string,
    data: ArrayBuffer | string | ReadableStream,
    options?: {
      contentType?: string;
      metadata?: Record<string, string>;
      cacheControl?: string;
    }
  ): Promise<{ success: boolean; key: string; size?: number; error?: string }> {
    if (!this.bucket) {
      return { success: false, key, error: 'Storage not available' };
    }

    try {
      const httpMetadata: R2HTTPMetadata = {};
      if (options?.contentType) httpMetadata.contentType = options.contentType;
      if (options?.cacheControl) httpMetadata.cacheControl = options.cacheControl;

      const result = await this.bucket.put(key, data, {
        httpMetadata,
        customMetadata: options?.metadata,
      });

      return {
        success: true,
        key,
        size: result?.size,
      };
    } catch (error) {
      console.error(`Storage upload error for ${key}:`, error);
      return {
        success: false,
        key,
        error: error instanceof Error ? error.message : 'Upload failed',
      };
    }
  }

  /**
   * Download a file from R2
   */
  async download(key: string): Promise<{
    success: boolean;
    data?: ArrayBuffer;
    metadata?: Record<string, string>;
    contentType?: string;
    error?: string;
  }> {
    if (!this.bucket) {
      return { success: false, error: 'Storage not available' };
    }

    try {
      const object = await this.bucket.get(key);
      if (!object) {
        return { success: false, error: 'Object not found' };
      }

      const data = await object.arrayBuffer();
      return {
        success: true,
        data,
        metadata: object.customMetadata,
        contentType: object.httpMetadata?.contentType,
      };
    } catch (error) {
      console.error(`Storage download error for ${key}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Download failed',
      };
    }
  }

  /**
   * Check if a file exists
   */
  async exists(key: string): Promise<boolean> {
    if (!this.bucket) return false;
    try {
      const object = await this.bucket.head(key);
      return object !== null;
    } catch {
      return false;
    }
  }

  /**
   * Delete a file
   */
  async delete(key: string): Promise<boolean> {
    if (!this.bucket) return false;
    try {
      await this.bucket.delete(key);
      return true;
    } catch (error) {
      console.error(`Storage delete error for ${key}:`, error);
      return false;
    }
  }

  /**
   * List files with prefix
   */
  async list(
    prefix: string,
    options?: { limit?: number; cursor?: string }
  ): Promise<{
    objects: Array<{ key: string; size: number; uploaded: Date }>;
    cursor?: string;
    truncated: boolean;
  }> {
    if (!this.bucket) {
      return { objects: [], truncated: false };
    }

    try {
      const listed = await this.bucket.list({
        prefix,
        limit: options?.limit || 100,
        cursor: options?.cursor,
      });

      return {
        objects: listed.objects.map((obj) => ({
          key: obj.key,
          size: obj.size,
          uploaded: obj.uploaded,
        })),
        cursor: listed.truncated ? listed.cursor : undefined,
        truncated: listed.truncated,
      };
    } catch (error) {
      console.error(`Storage list error for ${prefix}:`, error);
      return { objects: [], truncated: false };
    }
  }

  // =========================================================================
  // SCREENSHOT OPERATIONS
  // =========================================================================

  /**
   * Store a screenshot (base64 encoded PNG)
   */
  async storeScreenshot(
    projectId: string,
    screenshotBase64: string,
    metadata?: {
      testName?: string;
      stepNumber?: number;
      url?: string;
      browser?: string;
      device?: string;
    }
  ): Promise<{ success: boolean; url?: string; key?: string; error?: string }> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const key = STORAGE_PATHS.SCREENSHOTS(projectId, timestamp);

    // Convert base64 to ArrayBuffer
    const binaryString = atob(screenshotBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const result = await this.upload(key, bytes.buffer, {
      contentType: 'image/png',
      metadata: {
        projectId,
        timestamp,
        ...Object.fromEntries(
          Object.entries(metadata || {})
            .filter(([, v]) => v !== undefined)
            .map(([k, v]) => [k, String(v)])
        ),
      },
      cacheControl: 'public, max-age=31536000', // 1 year cache
    });

    if (result.success) {
      return {
        success: true,
        key,
        // Note: Actual public URL would need R2 public access or signed URLs
        url: `/storage/${key}`,
      };
    }

    return { success: false, error: result.error };
  }

  /**
   * Get a screenshot
   */
  async getScreenshot(key: string): Promise<{
    success: boolean;
    base64?: string;
    metadata?: Record<string, string>;
    error?: string;
  }> {
    const result = await this.download(key);
    if (!result.success || !result.data) {
      return { success: false, error: result.error };
    }

    // Convert ArrayBuffer to base64
    const bytes = new Uint8Array(result.data);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64 = btoa(binary);

    return {
      success: true,
      base64,
      metadata: result.metadata,
    };
  }

  /**
   * List screenshots for a project
   */
  async listScreenshots(
    projectId: string,
    options?: { limit?: number; cursor?: string }
  ) {
    const prefix = `screenshots/${projectId}/`;
    return this.list(prefix, options);
  }

  // =========================================================================
  // COVERAGE REPORT OPERATIONS
  // =========================================================================

  /**
   * Store a coverage report
   */
  async storeCoverageReport(
    projectId: string,
    commitSha: string,
    report: string | ArrayBuffer,
    format: 'lcov' | 'istanbul' | 'cobertura' = 'lcov'
  ): Promise<{ success: boolean; key?: string; error?: string }> {
    const filename = format === 'lcov' ? 'lcov.info' : `coverage.${format}.json`;
    const key = STORAGE_PATHS.COVERAGE(projectId, commitSha, filename);

    const contentType = format === 'lcov' ? 'text/plain' : 'application/json';

    const result = await this.upload(key, report, {
      contentType,
      metadata: {
        projectId,
        commitSha,
        format,
        uploadedAt: new Date().toISOString(),
      },
    });

    return result.success
      ? { success: true, key }
      : { success: false, error: result.error };
  }

  /**
   * Get a coverage report
   */
  async getCoverageReport(
    projectId: string,
    commitSha: string,
    format: 'lcov' | 'istanbul' | 'cobertura' = 'lcov'
  ): Promise<{ success: boolean; report?: string; error?: string }> {
    const filename = format === 'lcov' ? 'lcov.info' : `coverage.${format}.json`;
    const key = STORAGE_PATHS.COVERAGE(projectId, commitSha, filename);

    const result = await this.download(key);
    if (!result.success || !result.data) {
      return { success: false, error: result.error };
    }

    const decoder = new TextDecoder();
    return {
      success: true,
      report: decoder.decode(result.data),
    };
  }

  // =========================================================================
  // TEST RESULTS OPERATIONS
  // =========================================================================

  /**
   * Store test run results
   */
  async storeTestResults(
    projectId: string,
    runId: string,
    results: Record<string, unknown>
  ): Promise<{ success: boolean; key?: string; error?: string }> {
    const key = STORAGE_PATHS.TEST_RESULTS(projectId, runId);

    const result = await this.upload(key, JSON.stringify(results, null, 2), {
      contentType: 'application/json',
      metadata: {
        projectId,
        runId,
        storedAt: new Date().toISOString(),
      },
    });

    return result.success
      ? { success: true, key }
      : { success: false, error: result.error };
  }

  /**
   * Get test run results
   */
  async getTestResults(
    projectId: string,
    runId: string
  ): Promise<{ success: boolean; results?: Record<string, unknown>; error?: string }> {
    const key = STORAGE_PATHS.TEST_RESULTS(projectId, runId);

    const result = await this.download(key);
    if (!result.success || !result.data) {
      return { success: false, error: result.error };
    }

    try {
      const decoder = new TextDecoder();
      const results = JSON.parse(decoder.decode(result.data));
      return { success: true, results };
    } catch (error) {
      return { success: false, error: 'Failed to parse results JSON' };
    }
  }

  // =========================================================================
  // GENERATED TESTS OPERATIONS
  // =========================================================================

  /**
   * Store a generated test file
   */
  async storeGeneratedTest(
    projectId: string,
    filename: string,
    testCode: string,
    metadata?: {
      framework?: string;
      triggerEventId?: string;
      confidence?: number;
    }
  ): Promise<{ success: boolean; key?: string; error?: string }> {
    const timestamp = new Date().toISOString().split('T')[0]; // YYYY-MM-DD
    const key = STORAGE_PATHS.GENERATED_TESTS(projectId, timestamp, filename);

    const result = await this.upload(key, testCode, {
      contentType: 'text/typescript',
      metadata: {
        projectId,
        filename,
        ...Object.fromEntries(
          Object.entries(metadata || {})
            .filter(([, v]) => v !== undefined)
            .map(([k, v]) => [k, String(v)])
        ),
      },
    });

    return result.success
      ? { success: true, key }
      : { success: false, error: result.error };
  }

  /**
   * Get a generated test file
   */
  async getGeneratedTest(key: string): Promise<{
    success: boolean;
    code?: string;
    metadata?: Record<string, string>;
    error?: string;
  }> {
    const result = await this.download(key);
    if (!result.success || !result.data) {
      return { success: false, error: result.error };
    }

    const decoder = new TextDecoder();
    return {
      success: true,
      code: decoder.decode(result.data),
      metadata: result.metadata,
    };
  }

  /**
   * List generated tests for a project
   */
  async listGeneratedTests(
    projectId: string,
    options?: { limit?: number; cursor?: string }
  ) {
    const prefix = `generated-tests/${projectId}/`;
    return this.list(prefix, options);
  }

  // =========================================================================
  // UTILITY METHODS
  // =========================================================================

  /**
   * Create organization-scoped storage
   */
  withOrganization(organizationId: string): OrganizationStorage {
    return new OrganizationStorage(this, organizationId);
  }
}

/**
 * Organization-scoped storage wrapper
 */
export class OrganizationStorage {
  constructor(
    private storage: ArgusStorage,
    private organizationId: string
  ) {}

  async storeScreenshot(
    projectId: string,
    screenshot: string,
    metadata?: Record<string, unknown>
  ) {
    return this.storage.storeScreenshot(projectId, screenshot, {
      ...metadata,
      organizationId: this.organizationId,
    } as any);
  }

  async storeCoverageReport(
    projectId: string,
    commitSha: string,
    report: string,
    format?: 'lcov' | 'istanbul' | 'cobertura'
  ) {
    return this.storage.storeCoverageReport(projectId, commitSha, report, format);
  }

  async storeTestResults(
    projectId: string,
    runId: string,
    results: Record<string, unknown>
  ) {
    return this.storage.storeTestResults(projectId, runId, results);
  }
}

/**
 * Create storage instance from R2 binding
 */
export function createStorage(bucket: R2Bucket | null | undefined): ArgusStorage {
  return new ArgusStorage(bucket || null);
}
