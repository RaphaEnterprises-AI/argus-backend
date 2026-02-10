import * as fs from "node:fs";
import * as path from "node:path";
import type { Page } from "playwright";
import type { Config } from "../config.js";

export class ScreenshotManager {
  private dir: string;
  private counter = 0;

  constructor(config: Config) {
    this.dir = config.screenshotDir;
  }

  private ensureDir(): void {
    if (!fs.existsSync(this.dir)) {
      fs.mkdirSync(this.dir, { recursive: true });
    }
  }

  async capture(
    page: Page,
    opts?: { fullPage?: boolean; name?: string }
  ): Promise<{ path: string; base64: string }> {
    this.ensureDir();

    const name =
      opts?.name || `screenshot_${++this.counter}_${Date.now()}.png`;
    const filePath = path.join(this.dir, name);

    const buffer = await page.screenshot({
      path: filePath,
      fullPage: opts?.fullPage ?? false,
    });

    return {
      path: filePath,
      base64: buffer.toString("base64"),
    };
  }

  async captureElement(
    page: Page,
    selector: string,
    name?: string
  ): Promise<{ path: string; base64: string }> {
    this.ensureDir();

    const el = await page.locator(selector);
    const fileName = name || `element_${++this.counter}_${Date.now()}.png`;
    const filePath = path.join(this.dir, fileName);

    const buffer = await el.screenshot({ path: filePath });
    return {
      path: filePath,
      base64: buffer.toString("base64"),
    };
  }
}
