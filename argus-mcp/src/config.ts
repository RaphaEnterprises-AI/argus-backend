export interface Config {
  argusApiKey: string | null;
  argusApiUrl: string;
  headless: boolean;
  screenshotDir: string;
  defaultTimeout: number;
  viewport: { width: number; height: number };
}

export function loadConfig(): Config {
  return {
    argusApiKey: process.env.ARGUS_API_KEY || null,
    argusApiUrl:
      process.env.ARGUS_API_URL ||
      "https://skopaq-brain-production.up.railway.app",
    headless: process.env.ARGUS_HEADLESS !== "false",
    screenshotDir: process.env.ARGUS_SCREENSHOT_DIR || "./argus-screenshots",
    defaultTimeout: parseInt(process.env.ARGUS_TIMEOUT || "30000", 10),
    viewport: {
      width: parseInt(process.env.ARGUS_VIEWPORT_WIDTH || "1280", 10),
      height: parseInt(process.env.ARGUS_VIEWPORT_HEIGHT || "720", 10),
    },
  };
}
