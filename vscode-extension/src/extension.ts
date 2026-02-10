import * as vscode from "vscode";
import { ArgusClient } from "./api/client";
import { AuthManager } from "./auth/deviceAuth";
import { TestExplorerProvider } from "./providers/testExplorer";
import { ArgusCodeLensProvider } from "./providers/codeLens";
import { ArgusDiagnostics } from "./providers/diagnostics";
import { ArgusStatusBar } from "./providers/statusBar";

let statusBar: ArgusStatusBar | undefined;
let diagnostics: ArgusDiagnostics | undefined;

export async function activate(
  context: vscode.ExtensionContext
): Promise<void> {
  const client = new ArgusClient();
  const auth = new AuthManager(context.secrets);
  const testExplorer = new TestExplorerProvider(client);
  const codeLens = new ArgusCodeLensProvider();
  diagnostics = new ArgusDiagnostics(client);
  statusBar = new ArgusStatusBar(client);

  // Restore saved API key
  const savedKey = await auth.getApiKey();
  if (savedKey) {
    client.setApiKey(savedKey);
  }

  // Register tree view
  const treeView = vscode.window.createTreeView("argus.testExplorer", {
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
  const loginCmd = vscode.commands.registerCommand("argus.login", async () => {
    const key = await auth.login();
    if (key) {
      client.setApiKey(key);
      testExplorer.refresh();
      statusBar?.refresh();
    }
  });

  const logoutCmd = vscode.commands.registerCommand(
    "argus.logout",
    async () => {
      await auth.logout();
      client.setApiKey("");
      testExplorer.refresh();
      statusBar?.refresh();
    }
  );

  const refreshCmd = vscode.commands.registerCommand(
    "argus.refreshTests",
    () => {
      testExplorer.refresh();
      codeLens.refresh();
      statusBar?.refresh();

      const projectId = vscode.workspace
        .getConfiguration("argus")
        .get<string>("projectId");
      if (projectId) {
        diagnostics?.refreshFromTestRuns(projectId);
      }
    }
  );

  const runTestCmd = vscode.commands.registerCommand(
    "argus.runTest",
    async (testId: string) => {
      if (!client.getApiKey()) {
        vscode.window.showWarningMessage(
          'Please login first: run "Argus: Login"'
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
    "argus.viewQuality",
    () => statusBar?.showDetails()
  );

  // Auto-refresh on file save
  const saveWatcher = vscode.workspace.onDidSaveTextDocument(() => {
    const autoRefresh = vscode.workspace
      .getConfiguration("argus")
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
