"""Computer Use API integration for E2E testing."""

from .client import ComputerUseClient, UsageStats, TaskResult, ActionResult
from .screenshot import ScreenshotCapture, Screenshot, CaptureMethod
from .actions import (
    Action,
    ActionType,
    ActionResult as ActionExecutionResult,
    PlaywrightActionExecutor,
    ComputerUseActionExecutor,
    HybridActionExecutor,
    parse_test_step,
)
from .sandbox import (
    SandboxManager,
    SandboxConfig,
    SandboxInfo,
    SandboxState,
    LocalSandbox,
    build_sandbox_image,
)

__all__ = [
    # Client
    "ComputerUseClient",
    "UsageStats",
    "TaskResult",
    "ActionResult",
    # Screenshot
    "ScreenshotCapture",
    "Screenshot",
    "CaptureMethod",
    # Actions
    "Action",
    "ActionType",
    "ActionExecutionResult",
    "PlaywrightActionExecutor",
    "ComputerUseActionExecutor",
    "HybridActionExecutor",
    "parse_test_step",
    # Sandbox
    "SandboxManager",
    "SandboxConfig",
    "SandboxInfo",
    "SandboxState",
    "LocalSandbox",
    "build_sandbox_image",
]
