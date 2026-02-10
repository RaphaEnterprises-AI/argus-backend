import type { Page, CDPSession as PlaywrightCDPSession } from "playwright";

export interface NetworkRequest {
  requestId: string;
  url: string;
  method: string;
  timestamp: number;
  status?: number;
  mimeType?: string;
  responseSize?: number;
  failed?: boolean;
  failureReason?: string;
}

export interface ConsoleMessage {
  type: string;
  text: string;
  timestamp: number;
  url?: string;
  lineNumber?: number;
}

export interface JSError {
  message: string;
  stack?: string;
  timestamp: number;
  url?: string;
  lineNumber?: number;
}

export class CDPSession {
  private session: PlaywrightCDPSession | null = null;
  private requests: Map<string, NetworkRequest> = new Map();
  private consoleMessages: ConsoleMessage[] = [];
  private jsErrors: JSError[] = [];

  async attach(page: Page): Promise<void> {
    this.session = await page.context().newCDPSession(page);

    this.requests.clear();
    this.consoleMessages = [];
    this.jsErrors = [];

    await this.enableNetwork();
    await this.enableRuntime();
    await this.enablePerformance();
  }

  private async enableNetwork(): Promise<void> {
    if (!this.session) return;
    await this.session.send("Network.enable");

    this.session.on("Network.requestWillBeSent", (params: any) => {
      this.requests.set(params.requestId, {
        requestId: params.requestId,
        url: params.request.url,
        method: params.request.method,
        timestamp: params.timestamp,
      });
    });

    this.session.on("Network.responseReceived", (params: any) => {
      const req = this.requests.get(params.requestId);
      if (req) {
        req.status = params.response.status;
        req.mimeType = params.response.mimeType;
        req.responseSize = params.response.encodedDataLength;
      }
    });

    this.session.on("Network.loadingFailed", (params: any) => {
      const req = this.requests.get(params.requestId);
      if (req) {
        req.failed = true;
        req.failureReason = params.errorText;
      }
    });
  }

  private async enableRuntime(): Promise<void> {
    if (!this.session) return;
    await this.session.send("Runtime.enable");

    this.session.on("Runtime.consoleAPICalled", (params: any) => {
      const text = params.args
        .map((arg: any) => arg.value ?? arg.description ?? "")
        .join(" ");

      this.consoleMessages.push({
        type: params.type,
        text,
        timestamp: params.timestamp,
      });
    });

    this.session.on("Runtime.exceptionThrown", (params: any) => {
      const detail = params.exceptionDetails;
      const exception = detail.exception;

      this.jsErrors.push({
        message: exception?.description || detail.text || "Unknown error",
        stack: exception?.description,
        timestamp: detail.timestamp,
        url: detail.url,
        lineNumber: detail.lineNumber,
      });
    });
  }

  private async enablePerformance(): Promise<void> {
    if (!this.session) return;
    await this.session.send("Performance.enable");
  }

  // --- Query methods ---

  getRequests(): NetworkRequest[] {
    return Array.from(this.requests.values());
  }

  getFailedRequests(): NetworkRequest[] {
    return this.getRequests().filter(
      (r) => r.failed || (r.status && r.status >= 400)
    );
  }

  async getResponseBody(requestId: string): Promise<string> {
    if (!this.session) throw new Error("CDP session not attached");
    try {
      const { body } = await this.session.send("Network.getResponseBody", {
        requestId,
      });
      return body;
    } catch {
      return "(response body not available)";
    }
  }

  getConsoleMessages(): ConsoleMessage[] {
    return [...this.consoleMessages];
  }

  getJSErrors(): JSError[] {
    return [...this.jsErrors];
  }

  async getPerformanceMetrics(): Promise<Record<string, number>> {
    if (!this.session) throw new Error("CDP session not attached");
    const { metrics } = await this.session.send("Performance.getMetrics");
    const result: Record<string, number> = {};
    for (const m of metrics) {
      result[m.name] = m.value;
    }
    return result;
  }

  async startTrace(): Promise<void> {
    if (!this.session) throw new Error("CDP session not attached");
    await this.session.send("Tracing.start", {
      categories:
        "devtools.timeline,v8.execute,blink.user_timing,loading,devtools.timeline.async",
      options: "sampling-frequency=10000",
    });
  }

  async stopTrace(): Promise<{ events: any[] }> {
    if (!this.session) throw new Error("CDP session not attached");

    const chunks: any[] = [];
    this.session.on("Tracing.dataCollected", (params: any) => {
      chunks.push(...params.value);
    });

    await this.session.send("Tracing.end");

    // Wait briefly for chunks to arrive
    await new Promise((r) => setTimeout(r, 500));
    return { events: chunks };
  }

  clearCaptures(): void {
    this.requests.clear();
    this.consoleMessages = [];
    this.jsErrors = [];
  }

  getRawSession(): PlaywrightCDPSession | null {
    return this.session;
  }
}
