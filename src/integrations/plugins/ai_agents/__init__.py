"""AI Agents integration plugins.

These plugins enable AI coding assistants (Cursor, Claude Code) to access
Argus test data and insights through the Model Context Protocol (MCP).

Supported integrations:
- Cursor: AI-powered code editor with MCP support
- Claude Code: Anthropic's CLI for Claude with MCP integration
"""

from .cursor_plugin import CursorPlugin
from .claude_code_plugin import ClaudeCodePlugin

__all__ = [
    "CursorPlugin",
    "ClaudeCodePlugin",
]
