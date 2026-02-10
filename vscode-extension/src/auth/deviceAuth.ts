import * as vscode from "vscode";

const SECRET_KEY = "argus.apiKey";

/**
 * Manages API key authentication for the Skopaq backend.
 *
 * For the MVP, this uses manual API key entry stored in VS Code's
 * SecretStorage. A full OAuth2 device flow can be added later.
 */
export class AuthManager {
  constructor(private readonly secrets: vscode.SecretStorage) {}

  async getApiKey(): Promise<string | undefined> {
    return this.secrets.get(SECRET_KEY);
  }

  async login(): Promise<string | undefined> {
    const key = await vscode.window.showInputBox({
      prompt: "Enter your Skopaq API key",
      placeHolder: "argus_sk_...",
      password: true,
      ignoreFocusOut: true,
      validateInput: (value) => {
        if (!value) {
          return "API key is required";
        }
        if (!value.startsWith("argus_sk_")) {
          return 'API key must start with "argus_sk_"';
        }
        return null;
      },
    });

    if (key) {
      await this.secrets.store(SECRET_KEY, key);
      vscode.window.showInformationMessage("Skopaq: Logged in successfully");
    }

    return key;
  }

  async logout(): Promise<void> {
    await this.secrets.delete(SECRET_KEY);
    vscode.window.showInformationMessage("Skopaq: Logged out");
  }

  async isLoggedIn(): Promise<boolean> {
    const key = await this.getApiKey();
    return !!key;
  }
}
