import * as vscode from "vscode";
import { SkopaqClient, QualityScore } from "../api/client";

/**
 * Shows the Skopaq quality score in the VS Code status bar.
 *
 * Displays format: "Skopaq: A (92%)" with color coding by grade.
 * Clicking opens a detail view with the full quality breakdown.
 */
export class SkopaqStatusBar {
  private statusBarItem: vscode.StatusBarItem;
  private refreshInterval: ReturnType<typeof setInterval> | undefined;

  constructor(private readonly client: SkopaqClient) {
    this.statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      50
    );
    this.statusBarItem.command = "skopaq.viewQuality";
    this.statusBarItem.tooltip = "Click to view Skopaq quality details";
    this.setDisconnected();
    this.statusBarItem.show();
  }

  dispose(): void {
    this.statusBarItem.dispose();
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
  }

  /**
   * Start periodic refresh of quality score.
   */
  startAutoRefresh(intervalMs = 60000): void {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
    this.refreshInterval = setInterval(() => this.refresh(), intervalMs);
    this.refresh();
  }

  async refresh(): Promise<void> {
    const projectId = vscode.workspace
      .getConfiguration("skopaq")
      .get<string>("projectId");

    if (!projectId || !this.client.getApiKey()) {
      this.setDisconnected();
      return;
    }

    try {
      const score = await this.client.getQualityScore(projectId);
      this.setScore(score);
    } catch {
      this.setDisconnected();
    }
  }

  private setScore(score: QualityScore): void {
    const icon = this.getGradeIcon(score.grade);
    this.statusBarItem.text = `${icon} Skopaq: ${score.grade} (${Math.round(score.score)}%)`;
    this.statusBarItem.backgroundColor = undefined;
  }

  private setDisconnected(): void {
    this.statusBarItem.text = "$(circle-outline) Skopaq";
    this.statusBarItem.backgroundColor = undefined;
  }

  private getGradeIcon(grade: string): string {
    switch (grade.toUpperCase()) {
      case "A":
        return "$(check)";
      case "B":
        return "$(info)";
      case "C":
        return "$(warning)";
      case "D":
      case "F":
        return "$(error)";
      default:
        return "$(circle-outline)";
    }
  }

  /**
   * Show detailed quality information in a message.
   */
  async showDetails(): Promise<void> {
    const projectId = vscode.workspace
      .getConfiguration("skopaq")
      .get<string>("projectId");

    if (!projectId || !this.client.getApiKey()) {
      vscode.window.showWarningMessage(
        "Configure project ID and login first."
      );
      return;
    }

    try {
      const score = await this.client.getQualityScore(projectId);
      const message = [
        `Quality Grade: ${score.grade} (${Math.round(score.score)}%)`,
        `Coverage: ${Math.round(score.coverage)}%`,
        `Reliability: ${Math.round(score.reliability)}%`,
        `Flaky Tests: ${score.flaky_count}`,
      ].join("\n");

      vscode.window.showInformationMessage(message, { modal: true });
    } catch (error) {
      const msg = error instanceof Error ? error.message : "Unknown error";
      vscode.window.showErrorMessage(`Failed to fetch quality score: ${msg}`);
    }
  }
}
