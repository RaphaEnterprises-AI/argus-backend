import * as vscode from "vscode";
import { SkopaqClient } from "./api/client";
import { AuthManager } from "./auth/deviceAuth";
import { TestExplorerProvider } from "./providers/testExplorer";
import { SkopaqCodeLensProvider } from "./providers/codeLens";
import { SkopaqDiagnostics } from "./providers/diagnostics";
import { SkopaqStatusBar } from "./providers/statusBar";

let statusBar: SkopaqStatusBar | undefined;
let diagnostics: SkopaqDiagnostics | undefined;

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  const client = new SkopaqClient();
  const auth = new AuthManager(context.secrets);
  const testExplorer = new TestExplorerProvider(client);
  const codeLens = new SkopaqCodeLensProvider();
  diagnostics = new SkopaqDiagnostics(client);
  statusBar = new SkopaqStatusBar(client);

  // Restore saved API key
  const savedKey = await auth.getApiKey();
  if (savedKey) {
    client.setApiKey(savedKey);
  }

  // Register tree view
  const treeView = vscode.window.createTreeView("skopaq.testExplorer", {
    treeDataProvider: testExplorer,
    showCollapseAll: true,
  });

  // Register CodeLens provider for common test file patterns
  const codeLensDisposable = vscode.languages.registerCodeLensProvider(
    [
      { scheme: "file", language: "python" },
      { scheme: "file", language: "typescript" },
      { scheme: "file", language: "javascript" },
      { scheme: "file", language: "java" },
      { scheme: "file", language: "go" },
      { scheme: "file", language: "rust" },
    ],
    codeLens
  );

  // Register commands
  const loginCmd = vscode.commands.registerCommand("skopaq.login", async () => {
    const key = await auth.login();
    if (key) {
      client.setApiKey(key);
      testExplorer.refresh();
      statusBar?.refresh();
    }
  });

  const logoutCmd = vscode.commands.registerCommand(
    "skopaq.logout",
    async () => {
      await auth.logout();
      client.setApiKey("");
      testExplorer.refresh();
      statusBar?.refresh();
    }
  );

  const refreshCmd = vscode.commands.registerCommand(
    "skopaq.refreshTests",
    () => {
      testExplorer.refresh();
      codeLens.refresh();
      statusBar?.refresh();

      const projectId = vscode.workspace
        .getConfiguration("skopaq")
        .get<string>("projectId");
      if (projectId) {
        diagnostics?.refreshFromTestRuns(projectId);
      }
    }
  );

  const runTestCmd = vscode.commands.registerCommand(
    "skopaq.runTest",
    async (testId: string) => {
      if (!client.getApiKey()) {
        vscode.window.showWarningMessage(
          'Please login first: run "Skopaq: Login"'
        );
        return;
      }

      try {
        await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Notification,
            title: `Running test: ${testId}`,
            cancellable: false,
          },
          async () => {
            const result = await client.runTest(testId);
            if (result.status === "passed") {
              vscode.window.showInformationMessage(
                `Test ${testId} passed`
              );
            } else {
              vscode.window.showErrorMessage(
                `Test ${testId} failed: ${result.error || "Unknown error"}`
              );
            }
            testExplorer.refresh();
          }
        );
      } catch (error) {
        const msg = error instanceof Error ? error.message : "Unknown error";
        vscode.window.showErrorMessage(`Failed to run test: ${msg}`);
      }
    }
  );

  const qualityCmd = vscode.commands.registerCommand(
    "skopaq.viewQuality",
    () => statusBar?.showDetails()
  );

  // Auto-refresh on file save
  const saveWatcher = vscode.workspace.onDidSaveTextDocument(() => {
    const autoRefresh = vscode.workspace
      .getConfiguration("skopaq")
      .get<boolean>("autoRefresh");
    if (autoRefresh) {
      testExplorer.refresh();
    }
  });

  // Start status bar auto-refresh
  statusBar.startAutoRefresh(60000);

  // Add all disposables
  context.subscriptions.push(
    treeView,
    codeLensDisposable,
    loginCmd,
    logoutCmd,
    refreshCmd,
    runTestCmd,
    qualityCmd,
    saveWatcher,
    diagnostics,
    statusBar
  );
}

export function deactivate(): void {
  diagnostics?.dispose();
  statusBar?.dispose();
}
