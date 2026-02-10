import * as vscode from "vscode";
import { SkopaqClient, SkopaqTestRun } from "../api/client";

/**
 * Shows test failures in VS Code's Problems panel.
 *
 * After each test run refresh, failed tests are added as diagnostics
 * so developers see them alongside linting errors, type errors, etc.
 */
export class SkopaqDiagnostics {
  private diagnosticCollection: vscode.DiagnosticCollection;

  constructor(private readonly client: SkopaqClient) {
    this.diagnosticCollection =
      vscode.languages.createDiagnosticCollection("argus");
  }

  dispose(): void {
    this.diagnosticCollection.dispose();
  }

  clear(): void {
    this.diagnosticCollection.clear();
  }

  /**
   * Fetch recent test run results and add failures to diagnostics.
   */
  async refreshFromTestRuns(projectId: string): Promise<void> {
    this.diagnosticCollection.clear();

    try {
      const runs = await this.client.getTestRuns(projectId, 50);
      const failures = runs.filter((r) => r.status === "failed");

      if (failures.length === 0) {
        return;
      }

      // Group failures by file (if we can infer file from test name)
      const diagnosticsByUri = new Map<string, vscode.Diagnostic[]>();

      for (const failure of failures) {
        // Try to find matching test file in the workspace
        const uri = await this.findTestFile(failure);
        if (!uri) {
          continue;
        }

        const uriStr = uri.toString();
        if (!diagnosticsByUri.has(uriStr)) {
          diagnosticsByUri.set(uriStr, []);
        }

        const diagnostic = new vscode.Diagnostic(
          new vscode.Range(0, 0, 0, 0),
          `Test failed: ${failure.error || "Unknown error"}`,
          vscode.DiagnosticSeverity.Error
        );
        diagnostic.source = "Skopaq";
        diagnostic.code = failure.test_id;

        diagnosticsByUri.get(uriStr)!.push(diagnostic);
      }

      for (const [uriStr, diagnostics] of diagnosticsByUri) {
        this.diagnosticCollection.set(vscode.Uri.parse(uriStr), diagnostics);
      }
    } catch {
      // Silently fail — diagnostics are optional UX
    }
  }

  private async findTestFile(
    run: SkopaqTestRun
  ): Promise<vscode.Uri | undefined> {
    // Search workspace for test files matching the test ID
    const files = await vscode.workspace.findFiles(
      `**/*test*`,
      "**/node_modules/**",
      10
    );

    // For MVP, return first test file found (full implementation would
    // match test IDs to specific files via the backend)
    return files[0];
  }
}
