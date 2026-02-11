import * as vscode from "vscode";
import * as https from "https";
import * as http from "http";

export interface SkopaqTest {
  id: string;
  name: string;
  type: string;
  status: string;
  last_run?: string;
  project_id?: string;
}

export interface SkopaqTestRun {
  id: string;
  test_id: string;
  status: string;
  started_at: string;
  finished_at?: string;
  error?: string;
}

export interface QualityScore {
  score: number;
  grade: string;
  coverage: number;
  reliability: number;
  flaky_count: number;
}

export class SkopaqClient {
  private baseUrl: string;
  private apiKey: string | undefined;

  constructor() {
    const config = vscode.workspace.getConfiguration("argus");
    this.baseUrl = config.get<string>("backendUrl") || "https://skopaq-brain-production.up.railway.app";
  }

  setApiKey(key: string): void {
    this.apiKey = key;
  }

  getApiKey(): string | undefined {
    return this.apiKey;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const url = new URL(`${this.baseUrl}${path}`);
    const isHttps = url.protocol === "https:";
    const lib = isHttps ? https : http;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      Accept: "application/json",
    };

    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }

    const bodyStr = body ? JSON.stringify(body) : undefined;
    if (bodyStr) {
      headers["Content-Length"] = Buffer.byteLength(bodyStr).toString();
    }

    return new Promise<T>((resolve, reject) => {
      const req = lib.request(
        url,
        {
          method,
          headers,
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
              try {
                resolve(JSON.parse(data) as T);
              } catch {
                reject(new Error(`Invalid JSON response: ${data.slice(0, 200)}`));
              }
            } else if (res.statusCode === 401) {
              reject(new Error("Authentication required. Please run 'Skopaq: Login'."));
            } else if (res.statusCode === 429) {
              reject(new Error("Rate limited. Please wait a moment and try again."));
            } else {
              reject(new Error(`API error ${res.statusCode}: ${data.slice(0, 200)}`));
            }
          });
        }
      );

      req.on("error", reject);
      req.setTimeout(30000, () => {
        req.destroy();
        reject(new Error("Request timed out"));
      });

      if (bodyStr) {
        req.write(bodyStr);
      }
      req.end();
    });
  }

  async getTests(projectId: string): Promise<SkopaqTest[]> {
    const result = await this.request<{ tests: SkopaqTest[] }>(
      "GET",
      `/api/v1/tests?project_id=${projectId}`
    );
    return result.tests || [];
  }

  async getTestRuns(
    projectId: string,
    limit = 20
  ): Promise<SkopaqTestRun[]> {
    const result = await this.request<{ test_runs: SkopaqTestRun[] }>(
      "GET",
      `/api/v1/test-runs?project_id=${projectId}&limit=${limit}`
    );
    return result.test_runs || [];
  }

  async runTest(testId: string): Promise<SkopaqTestRun> {
    return this.request<SkopaqTestRun>("POST", `/api/v1/test-runs`, {
      test_id: testId,
    });
  }

  async getQualityScore(projectId: string): Promise<QualityScore> {
    return this.request<QualityScore>(
      "GET",
      `/api/v1/quality/score?project_id=${projectId}`
    );
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this.request<{ status: string }>("GET", "/health");
      return true;
    } catch {
      return false;
    }
  }
}
