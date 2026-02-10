import type { Config } from "./config.js";

export class ArgusClient {
  private apiKey: string | null;
  private baseUrl: string;

  constructor(config: Config) {
    this.apiKey = config.argusApiKey;
    this.baseUrl = config.argusApiUrl.replace(/\/$/, "");
  }

  get available(): boolean {
    return this.apiKey !== null;
  }

  async request(
    method: string,
    path: string,
    body?: unknown,
    headers?: Record<string, string>
  ): Promise<any> {
    if (!this.apiKey) {
      return {
        error:
          "Configure ARGUS_API_KEY to enable AI features. Set it in your MCP server env config.",
      };
    }

    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": this.apiKey,
        ...headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      return {
        error: `Argus API returned ${res.status}: ${text.slice(0, 500)}`,
      };
    }

    return res.json();
  }

  async get(path: string, headers?: Record<string, string>): Promise<any> {
    return this.request("GET", path, undefined, headers);
  }

  async post(
    path: string,
    body: unknown,
    headers?: Record<string, string>
  ): Promise<any> {
    return this.request("POST", path, body, headers);
  }
}
