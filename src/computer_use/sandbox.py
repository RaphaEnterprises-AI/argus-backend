"""Docker sandbox management for safe Computer Use execution."""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import structlog

logger = structlog.get_logger()


class SandboxState(str, Enum):
    """Sandbox container states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class SandboxConfig:
    """Configuration for sandbox container."""
    image: str = "e2e-agent:latest"
    display: str = ":99"
    resolution: str = "1920x1080x24"
    shm_size: str = "2g"
    vnc_port: int = 5900
    timeout_seconds: int = 300
    cpu_limit: float = 2.0
    memory_limit: str = "4g"


@dataclass
class SandboxInfo:
    """Information about a running sandbox."""
    container_id: str
    state: SandboxState
    display: str
    vnc_port: int
    ip_address: str | None = None
    error: str | None = None


class SandboxManager:
    """
    Manages Docker containers for safe Computer Use execution.

    CRITICAL: Always run Computer Use in an isolated Docker container
    to prevent unintended system interactions.
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self.container = None
        self._state = SandboxState.STOPPED
        self.log = logger.bind(component="sandbox")

    @property
    def state(self) -> SandboxState:
        return self._state

    async def start(
        self,
        codebase_path: str | None = None,
        app_url: str | None = None,
    ) -> SandboxInfo:
        """
        Start a sandbox container.

        Args:
            codebase_path: Path to mount as read-only codebase
            app_url: URL of the application to test

        Returns:
            SandboxInfo with container details
        """
        try:
            import docker
        except ImportError:
            raise RuntimeError(
                "Docker SDK not installed. Run: pip install docker"
            )

        self._state = SandboxState.STARTING
        self.log.info("Starting sandbox container")

        try:
            client = docker.from_env()

            # Build volumes
            volumes = {}
            if codebase_path:
                volumes[str(Path(codebase_path).absolute())] = {
                    "bind": "/app/codebase",
                    "mode": "ro",
                }

            # Build environment
            environment = {
                "DISPLAY": self.config.display,
                "RESOLUTION": self.config.resolution,
            }

            # Pass API key securely
            if os.environ.get("ANTHROPIC_API_KEY"):
                environment["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

            if app_url:
                environment["APP_URL"] = app_url

            # Start container
            self.container = client.containers.run(
                self.config.image,
                detach=True,
                environment=environment,
                volumes=volumes,
                shm_size=self.config.shm_size,
                ports={f"{self.config.vnc_port}/tcp": self.config.vnc_port},
                security_opt=["seccomp:unconfined"],
                cpu_period=100000,
                cpu_quota=int(self.config.cpu_limit * 100000),
                mem_limit=self.config.memory_limit,
            )

            # Wait for container to be ready
            await self._wait_for_ready()

            self._state = SandboxState.RUNNING

            # Get container info
            self.container.reload()
            ip_address = self.container.attrs.get("NetworkSettings", {}).get(
                "IPAddress"
            )

            info = SandboxInfo(
                container_id=self.container.id,
                state=self._state,
                display=self.config.display,
                vnc_port=self.config.vnc_port,
                ip_address=ip_address,
            )

            self.log.info(
                "Sandbox started",
                container_id=self.container.short_id,
                vnc_port=self.config.vnc_port,
            )

            return info

        except Exception as e:
            self._state = SandboxState.ERROR
            self.log.error("Failed to start sandbox", error=str(e))
            raise

    async def _wait_for_ready(self, timeout: int = 30) -> None:
        """Wait for container display to be ready."""
        for i in range(timeout):
            # Check if Xvfb is running
            exit_code, output = self.container.exec_run(
                "pgrep -x Xvfb",
                demux=True,
            )
            if exit_code == 0:
                self.log.debug("Display ready", waited_seconds=i)
                return

            await asyncio.sleep(1)

        raise TimeoutError("Sandbox display not ready within timeout")

    async def execute_command(self, command: str) -> tuple[int, str]:
        """
        Execute a command in the sandbox.

        Args:
            command: Shell command to execute

        Returns:
            Tuple of (exit_code, output)
        """
        if not self.container or self._state != SandboxState.RUNNING:
            raise RuntimeError("Sandbox not running")

        exit_code, output = self.container.exec_run(
            f"/bin/bash -c '{command}'",
            demux=True,
        )

        stdout = output[0].decode() if output[0] else ""
        stderr = output[1].decode() if output[1] else ""

        return exit_code, stdout + stderr

    async def screenshot(self) -> bytes:
        """
        Capture screenshot from sandbox display.

        Returns:
            PNG screenshot bytes
        """
        if not self.container or self._state != SandboxState.RUNNING:
            raise RuntimeError("Sandbox not running")

        # Use ImageMagick import command
        exit_code, output = self.container.exec_run(
            f"import -window root -display {self.config.display} png:-",
            demux=True,
        )

        if exit_code != 0:
            stderr = output[1].decode() if output[1] else "Unknown error"
            raise RuntimeError(f"Screenshot failed: {stderr}")

        return output[0]

    async def mouse_click(
        self,
        x: int,
        y: int,
        button: str = "left",
    ) -> None:
        """
        Execute mouse click in sandbox.

        Args:
            x: X coordinate
            y: Y coordinate
            button: "left", "right", or "middle"
        """
        button_map = {"left": 1, "middle": 2, "right": 3}
        button_num = button_map.get(button, 1)

        await self.execute_command(
            f"DISPLAY={self.config.display} xdotool mousemove {x} {y} click {button_num}"
        )

    async def type_text(self, text: str) -> None:
        """
        Type text in sandbox.

        Args:
            text: Text to type
        """
        # Escape special characters
        escaped = text.replace("'", "'\\''")
        await self.execute_command(
            f"DISPLAY={self.config.display} xdotool type '{escaped}'"
        )

    async def press_key(self, key: str) -> None:
        """
        Press a key in sandbox.

        Args:
            key: Key to press (e.g., "Return", "ctrl+c")
        """
        await self.execute_command(
            f"DISPLAY={self.config.display} xdotool key {key}"
        )

    async def stop(self) -> None:
        """Stop and remove the sandbox container."""
        if self.container:
            self.log.info("Stopping sandbox", container_id=self.container.short_id)
            try:
                self.container.stop(timeout=10)
                self.container.remove(force=True)
            except Exception as e:
                self.log.warning("Error stopping container", error=str(e))
            finally:
                self.container = None
                self._state = SandboxState.STOPPED

    async def __aenter__(self) -> "SandboxManager":
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        await self.stop()


class LocalSandbox:
    """
    Local sandbox for development without Docker.

    Uses the local system with Xvfb for display.
    WARNING: Less isolated than Docker sandbox.
    """

    def __init__(
        self,
        display: str = ":99",
        resolution: str = "1920x1080x24",
    ):
        self.display = display
        self.resolution = resolution
        self._xvfb_process = None
        self._state = SandboxState.STOPPED
        self.log = logger.bind(component="local_sandbox")

    async def start(self) -> SandboxInfo:
        """Start local Xvfb display."""
        self._state = SandboxState.STARTING
        self.log.info("Starting local Xvfb display")

        try:
            # Start Xvfb
            self._xvfb_process = await asyncio.create_subprocess_exec(
                "Xvfb",
                self.display,
                "-screen", "0", self.resolution,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Wait for display to be ready
            await asyncio.sleep(1)

            if self._xvfb_process.returncode is not None:
                raise RuntimeError("Xvfb failed to start")

            self._state = SandboxState.RUNNING

            return SandboxInfo(
                container_id="local",
                state=self._state,
                display=self.display,
                vnc_port=0,
            )

        except FileNotFoundError:
            self._state = SandboxState.ERROR
            raise RuntimeError(
                "Xvfb not installed. On Ubuntu: apt-get install xvfb"
            )
        except Exception:
            self._state = SandboxState.ERROR
            raise

    async def screenshot(self) -> bytes:
        """Capture screenshot from local display."""
        result = await asyncio.create_subprocess_exec(
            "import",
            "-window", "root",
            "-display", self.display,
            "png:-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise RuntimeError(f"Screenshot failed: {stderr.decode()}")

        return stdout

    async def stop(self) -> None:
        """Stop the local Xvfb display."""
        if self._xvfb_process:
            self._xvfb_process.terminate()
            await self._xvfb_process.wait()
            self._xvfb_process = None
            self._state = SandboxState.STOPPED
            self.log.info("Local Xvfb stopped")


async def build_sandbox_image(
    dockerfile_path: Path | None = None,
    tag: str = "e2e-agent:latest",
) -> str:
    """
    Build the sandbox Docker image.

    Args:
        dockerfile_path: Path to Dockerfile (uses default if None)
        tag: Image tag

    Returns:
        Image ID
    """
    try:
        import docker
    except ImportError:
        raise RuntimeError("Docker SDK not installed. Run: pip install docker")

    client = docker.from_env()
    log = logger.bind(component="sandbox_build")

    if dockerfile_path:
        # Use provided Dockerfile
        context_path = dockerfile_path.parent
    else:
        # Create temporary Dockerfile
        context_path = Path(tempfile.mkdtemp())
        dockerfile_path = context_path / "Dockerfile"

        dockerfile_content = """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    xvfb \\
    x11vnc \\
    fluxbox \\
    xdotool \\
    imagemagick \\
    wget \\
    gnupg \\
    && rm -rf /var/lib/apt/lists/*

# Install Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \\
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \\
    && apt-get update \\
    && apt-get install -y google-chrome-stable \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install anthropic playwright httpx pydantic structlog pillow

# Install Playwright browsers
RUN playwright install chromium

# Set up virtual display
ENV DISPLAY=:99
ENV RESOLUTION=1920x1080x24

# Create entrypoint
RUN echo '#!/bin/bash\\n\\
Xvfb :99 -screen 0 $RESOLUTION &\\n\\
sleep 1\\n\\
fluxbox &\\n\\
sleep 1\\n\\
x11vnc -display :99 -forever -nopw -quiet &\\n\\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

WORKDIR /app

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
"""
        dockerfile_path.write_text(dockerfile_content)

    log.info("Building sandbox image", tag=tag)

    try:
        image, build_logs = client.images.build(
            path=str(context_path),
            tag=tag,
            rm=True,
        )

        for log_entry in build_logs:
            if "stream" in log_entry:
                log.debug(log_entry["stream"].strip())

        log.info("Sandbox image built", image_id=image.id[:12])
        return image.id

    finally:
        # Clean up temp directory if we created it
        if not dockerfile_path:
            import shutil
            shutil.rmtree(context_path, ignore_errors=True)
