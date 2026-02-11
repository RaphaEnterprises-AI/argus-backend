import * as vscode from "vscode";
import { SkopaqClient, SkopaqTest } from "../api/client";

export class TestExplorerProvider
  implements vscode.TreeDataProvider<TestItem>
{
  private _onDidChangeTreeData = new vscode.EventEmitter<
    TestItem | undefined | null | void
  >();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private tests: SkopaqTest[] = [];

  constructor(private readonly client: SkopaqClient) {}

  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  getTreeItem(element: TestItem): vscode.TreeItem {
    return element;
  }

  async getChildren(element?: TestItem): Promise<TestItem[]> {
    if (element) {
      // No children for individual tests in MVP
      return [];
    }

    const projectId = vscode.workspace
      .getConfiguration("skopaq")
      .get<string>("projectId");

    if (!projectId) {
      return [
        new TestItem(
          "Configure project ID in settings",
          "none",
          "info",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    }

    if (!this.client.getApiKey()) {
      return [
        new TestItem(
          'Run "Skopaq: Login" to see tests',
          "none",
          "info",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    }

    try {
      this.tests = await this.client.getTests(projectId);

      if (this.tests.length === 0) {
        return [
          new TestItem(
            "No tests found",
            "none",
            "info",
            vscode.TreeItemCollapsibleState.None
          ),
        ];
      }

      return this.tests.map(
        (test) =>
          new TestItem(
            test.name,
            test.id,
            test.status,
            vscode.TreeItemCollapsibleState.None
          )
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown error";
      return [
        new TestItem(
          `Error: ${message}`,
          "none",
          "error",
          vscode.TreeItemCollapsibleState.None
        ),
      ];
    }
  }
}

export class TestItem extends vscode.TreeItem {
  constructor(
    public readonly label: string,
    public readonly testId: string,
    public readonly status: string,
    public readonly collapsibleState: vscode.TreeItemCollapsibleState
  ) {
    super(label, collapsibleState);

    this.tooltip = `${label} (${status})`;
    this.description = status;

    // Set icon based on status
    switch (status) {
      case "passed":
        this.iconPath = new vscode.ThemeIcon(
          "pass",
          new vscode.ThemeColor("testing.iconPassed")
        );
        break;
      case "failed":
        this.iconPath = new vscode.ThemeIcon(
          "error",
          new vscode.ThemeColor("testing.iconFailed")
        );
        break;
      case "running":
        this.iconPath = new vscode.ThemeIcon("loading~spin");
        break;
      case "info":
        this.iconPath = new vscode.ThemeIcon("info");
        break;
      case "error":
        this.iconPath = new vscode.ThemeIcon("warning");
        break;
      default:
        this.iconPath = new vscode.ThemeIcon(
          "circle-outline",
          new vscode.ThemeColor("testing.iconUnset")
        );
    }

    // Make test items clickable to run
    if (testId !== "none") {
      this.command = {
        command: "skopaq.runTest",
        title: "Run Test",
        arguments: [testId],
      };
    }
  }
}
