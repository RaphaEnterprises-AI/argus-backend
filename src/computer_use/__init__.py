"""Computer Use API integration for E2E testing."""

from .actions import (
    Action,
    ActionType,
    ComputerUseActionExecutor,
    HybridActionExecutor,
    PlaywrightActionExecutor,
    parse_test_step,
)
from .actions import (
    ActionResult as ActionExecutionResult,
)
from .client import ActionResult, ComputerUseClient, TaskResult, UsageStats
from .sandbox import (
    LocalSandbox,
    SandboxConfig,
    SandboxInfo,
    SandboxManager,
    SandboxState,
    build_sandbox_image,
)
from .screenshot import CaptureMethod, Screenshot, ScreenshotCapture

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
