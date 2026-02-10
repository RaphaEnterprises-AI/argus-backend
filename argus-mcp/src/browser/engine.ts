import {
  chromium,
  type Browser,
  type BrowserContext,
  type Page,
} from "playwright";
import { CDPSession } from "./cdp.js";
import type { Config } from "../config.js";

export class BrowserEngine {
  private browser: Browser | null = null;
  private context: BrowserContext | null = null;
  private page: Page | null = null;
  private cdp: CDPSession | null = null;
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  async ensurePage(): Promise<Page> {
    if (!this.page || this.page.isClosed()) {
      await this.launch();
    }
    return this.page!;
  }

  async ensureCDP(): Promise<CDPSession> {
    await this.ensurePage();
    if (!this.cdp) {
      this.cdp = new CDPSession();
      await this.cdp.attach(this.page!);
    }
    return this.cdp;
  }

  private async launch(): Promise<void> {
    await this.close();

    this.browser = await chromium.launch({
      headless: this.config.headless,
    });

    this.context = await this.browser.newContext({
      viewport: this.config.viewport,
    });

    this.page = await this.context.newPage();
    this.page.setDefaultTimeout(this.config.defaultTimeout);

    // Attach CDP for network/console/perf capture
    this.cdp = new CDPSession();
    await this.cdp.attach(this.page);
  }

  async close(): Promise<void> {
    this.cdp = null;
    this.page = null;
    this.context = null;
    if (this.browser) {
      await this.browser.close().catch(() => {});
      this.browser = null;
    }
  }

  async newPage(): Promise<Page> {
    if (!this.browser) {
      await this.launch();
      return this.page!;
    }
    this.page = await this.context!.newPage();
    this.page.setDefaultTimeout(this.config.defaultTimeout);

    this.cdp = new CDPSession();
    await this.cdp.attach(this.page);
    return this.page;
  }

  getPage(): Page | null {
    return this.page;
  }

  getCDP(): CDPSession | null {
    return this.cdp;
  }
}
