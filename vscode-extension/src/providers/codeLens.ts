import * as vscode from "vscode";

/**
 * Provides "Run Test" and "View Coverage" CodeLens above test functions.
 *
 * Detects test functions by common naming patterns (test_, it(, describe(,
 * @Test, etc.) across multiple languages.
 */
export class SkopaqCodeLensProvider implements vscode.CodeLensProvider {
  private _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
  readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;

  // Patterns that identify test functions across languages
  private static readonly TEST_PATTERNS: RegExp[] = [
    // Python: def test_something / class TestSomething
    /^\s*(async\s+)?def\s+(test_\w+)/,
    /^\s*class\s+(Test\w+)/,
    // JavaScript/TypeScript: it(, test(, describe(
    /^\s*(it|test|describe)\s*\(/,
    // Java/Kotlin: @Test annotation
    /^\s*@Test/,
    // Go: func Test
    /^\s*func\s+(Test\w+)/,
    // Rust: #[test]
    /^\s*#\[test\]/,
  ];

  refresh(): void {
    this._onDidChangeCodeLenses.fire();
  }

  provideCodeLenses(
    document: vscode.TextDocument
  ): vscode.CodeLens[] {
    const lenses: vscode.CodeLens[] = [];

    for (let i = 0; i < document.lineCount; i++) {
      const line = document.lineAt(i);

      for (const pattern of SkopaqCodeLensProvider.TEST_PATTERNS) {
        const match = line.text.match(pattern);
        if (match) {
          const range = new vscode.Range(i, 0, i, line.text.length);
          const testName = match[2] || match[1] || "test";

          // "Run Test" lens
          lenses.push(
            new vscode.CodeLens(range, {
              title: "$(play) Run Test",
              command: "argus.runTest",
              arguments: [testName],
              tooltip: `Run ${testName} via Skopaq`,
            })
          );

          break; // Only one lens per line
        }
      }
    }

    return lenses;
  }
}
