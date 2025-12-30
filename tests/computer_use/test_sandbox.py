"""Tests for the Docker sandbox module."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestSandboxState:
    """Tests for SandboxState enum."""

    def test_sandbox_states_exist(self, mock_env_vars):
        """Test all sandbox states are defined."""
        from src.computer_use.sandbox import SandboxState

        assert SandboxState.STOPPED == "stopped"
        assert SandboxState.STARTING == "starting"
        assert SandboxState.RUNNING == "running"
        assert SandboxState.ERROR == "error"


class TestSandboxConfig:
    """Tests for SandboxConfig dataclass."""

    def test_config_defaults(self, mock_env_vars):
        """Test SandboxConfig default values."""
        from src.computer_use.sandbox import SandboxConfig

        config = SandboxConfig()

        assert config.image == "e2e-agent:latest"
        assert config.display == ":99"
        assert config.resolution == "1920x1080x24"
        assert config.shm_size == "2g"
        assert config.vnc_port == 5900
        assert config.timeout_seconds == 300
        assert config.cpu_limit == 2.0
        assert config.memory_limit == "4g"

    def test_config_custom(self, mock_env_vars):
        """Test SandboxConfig with custom values."""
        from src.computer_use.sandbox import SandboxConfig

        config = SandboxConfig(
            image="custom:latest",
            display=":1",
            resolution="1280x720x24",
            cpu_limit=4.0,
        )

        assert config.image == "custom:latest"
        assert config.display == ":1"
        assert config.resolution == "1280x720x24"
        assert config.cpu_limit == 4.0


class TestSandboxInfo:
    """Tests for SandboxInfo dataclass."""

    def test_info_creation(self, mock_env_vars):
        """Test SandboxInfo creation."""
        from src.computer_use.sandbox import SandboxInfo, SandboxState

        info = SandboxInfo(
            container_id="abc123",
            state=SandboxState.RUNNING,
            display=":99",
            vnc_port=5900,
        )

        assert info.container_id == "abc123"
        assert info.state == SandboxState.RUNNING
        assert info.ip_address is None
        assert info.error is None

    def test_info_with_ip(self, mock_env_vars):
        """Test SandboxInfo with IP address."""
        from src.computer_use.sandbox import SandboxInfo, SandboxState

        info = SandboxInfo(
            container_id="abc123",
            state=SandboxState.RUNNING,
            display=":99",
            vnc_port=5900,
            ip_address="172.17.0.2",
        )

        assert info.ip_address == "172.17.0.2"


class TestSandboxManager:
    """Tests for SandboxManager class."""

    def test_manager_creation(self, mock_env_vars):
        """Test SandboxManager creation."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()

        assert manager.container is None
        assert manager.state == SandboxState.STOPPED

    def test_manager_with_config(self, mock_env_vars):
        """Test SandboxManager with custom config."""
        from src.computer_use.sandbox import SandboxManager, SandboxConfig

        config = SandboxConfig(image="custom:latest")
        manager = SandboxManager(config=config)

        assert manager.config.image == "custom:latest"

    @pytest.mark.asyncio
    async def test_start_no_docker(self, mock_env_vars):
        """Test start fails without docker module."""
        from src.computer_use.sandbox import SandboxManager

        manager = SandboxManager()

        with patch.dict("sys.modules", {"docker": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(RuntimeError, match="Docker SDK not installed"):
                    await manager.start()

    @pytest.mark.asyncio
    async def test_start_success(self, mock_env_vars):
        """Test successful sandbox start."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.short_id = "containe"
        mock_container.attrs = {"NetworkSettings": {"IPAddress": "172.17.0.2"}}
        mock_container.exec_run.return_value = (0, (b"1234", None))

        mock_docker_client = MagicMock()
        mock_docker_client.containers.run.return_value = mock_container

        with patch("docker.from_env", return_value=mock_docker_client):
            info = await manager.start()

            assert info.container_id == "container123"
            assert info.state == SandboxState.RUNNING
            assert manager.state == SandboxState.RUNNING

    @pytest.mark.asyncio
    async def test_start_with_codebase_path(self, mock_env_vars):
        """Test start with codebase path mount."""
        from src.computer_use.sandbox import SandboxManager

        manager = SandboxManager()

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.short_id = "containe"
        mock_container.attrs = {"NetworkSettings": {"IPAddress": ""}}
        mock_container.exec_run.return_value = (0, (b"1234", None))

        mock_docker_client = MagicMock()
        mock_docker_client.containers.run.return_value = mock_container

        with patch("docker.from_env", return_value=mock_docker_client):
            await manager.start(codebase_path="/tmp/app")

            # Verify volumes were passed
            call_kwargs = mock_docker_client.containers.run.call_args[1]
            assert "volumes" in call_kwargs
            assert len(call_kwargs["volumes"]) > 0

    @pytest.mark.asyncio
    async def test_start_failure(self, mock_env_vars):
        """Test sandbox start failure."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()

        mock_docker_client = MagicMock()
        mock_docker_client.containers.run.side_effect = Exception("Container error")

        with patch("docker.from_env", return_value=mock_docker_client):
            with pytest.raises(Exception, match="Container error"):
                await manager.start()

            assert manager.state == SandboxState.ERROR

    @pytest.mark.asyncio
    async def test_wait_for_ready(self, mock_env_vars):
        """Test waiting for display to be ready."""
        from src.computer_use.sandbox import SandboxManager

        manager = SandboxManager()
        manager.container = MagicMock()
        manager.container.exec_run.return_value = (0, (b"1234", None))

        await manager._wait_for_ready(timeout=5)

        manager.container.exec_run.assert_called()

    @pytest.mark.asyncio
    async def test_wait_for_ready_timeout(self, mock_env_vars):
        """Test timeout waiting for display."""
        from src.computer_use.sandbox import SandboxManager

        manager = SandboxManager()
        manager.container = MagicMock()
        manager.container.exec_run.return_value = (1, (None, b"Not running"))

        with pytest.raises(TimeoutError, match="not ready"):
            await manager._wait_for_ready(timeout=1)

    @pytest.mark.asyncio
    async def test_execute_command(self, mock_env_vars):
        """Test executing command in sandbox."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (0, (b"output", b""))

        exit_code, output = await manager.execute_command("echo hello")

        assert exit_code == 0
        assert "output" in output

    @pytest.mark.asyncio
    async def test_execute_command_not_running(self, mock_env_vars):
        """Test executing command when sandbox not running."""
        from src.computer_use.sandbox import SandboxManager

        manager = SandboxManager()

        with pytest.raises(RuntimeError, match="not running"):
            await manager.execute_command("echo hello")

    @pytest.mark.asyncio
    async def test_screenshot(self, mock_env_vars):
        """Test taking screenshot in sandbox."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (0, (b"png_data", None))

        screenshot = await manager.screenshot()

        assert screenshot == b"png_data"

    @pytest.mark.asyncio
    async def test_screenshot_failure(self, mock_env_vars):
        """Test screenshot failure."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (1, (None, b"Error"))

        with pytest.raises(RuntimeError, match="Screenshot failed"):
            await manager.screenshot()

    @pytest.mark.asyncio
    async def test_mouse_click(self, mock_env_vars):
        """Test mouse click in sandbox."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (0, (b"", b""))

        await manager.mouse_click(100, 200)

        # Verify xdotool was called
        call_args = manager.container.exec_run.call_args
        assert "xdotool" in call_args[0][0]
        assert "100" in call_args[0][0]
        assert "200" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_mouse_click_right(self, mock_env_vars):
        """Test right mouse click."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (0, (b"", b""))

        await manager.mouse_click(100, 200, button="right")

        call_args = manager.container.exec_run.call_args
        assert "click 3" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_type_text(self, mock_env_vars):
        """Test typing text in sandbox."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (0, (b"", b""))

        await manager.type_text("hello world")

        call_args = manager.container.exec_run.call_args
        assert "xdotool" in call_args[0][0]
        assert "type" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_type_text_with_quotes(self, mock_env_vars):
        """Test typing text with quotes."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (0, (b"", b""))

        await manager.type_text("it's a test")

        # Should escape quotes
        call_args = manager.container.exec_run.call_args
        assert "\\'" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_press_key(self, mock_env_vars):
        """Test pressing key in sandbox."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager._state = SandboxState.RUNNING
        manager.container.exec_run.return_value = (0, (b"", b""))

        await manager.press_key("Return")

        call_args = manager.container.exec_run.call_args
        assert "key Return" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_stop(self, mock_env_vars):
        """Test stopping sandbox."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        mock_container = MagicMock()
        mock_container.short_id = "containe"
        manager.container = mock_container
        manager._state = SandboxState.RUNNING

        await manager.stop()

        mock_container.stop.assert_called_once()
        mock_container.remove.assert_called_once()
        assert manager.container is None
        assert manager.state == SandboxState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_with_error(self, mock_env_vars):
        """Test stopping sandbox with error."""
        from src.computer_use.sandbox import SandboxManager, SandboxState

        manager = SandboxManager()
        manager.container = MagicMock()
        manager.container.short_id = "containe"
        manager.container.stop.side_effect = Exception("Stop error")
        manager._state = SandboxState.RUNNING

        # Should not raise
        await manager.stop()

        assert manager.state == SandboxState.STOPPED

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_env_vars):
        """Test sandbox as context manager."""
        from src.computer_use.sandbox import SandboxManager

        manager = SandboxManager()

        mock_container = MagicMock()
        mock_container.id = "container123"
        mock_container.short_id = "containe"
        mock_container.attrs = {"NetworkSettings": {"IPAddress": ""}}
        mock_container.exec_run.return_value = (0, (b"1234", None))

        mock_docker_client = MagicMock()
        mock_docker_client.containers.run.return_value = mock_container

        with patch("docker.from_env", return_value=mock_docker_client):
            async with manager as m:
                assert m is manager

            # Should have stopped
            mock_container.stop.assert_called()


class TestLocalSandbox:
    """Tests for LocalSandbox class."""

    def test_local_sandbox_creation(self, mock_env_vars):
        """Test LocalSandbox creation."""
        from src.computer_use.sandbox import LocalSandbox, SandboxState

        sandbox = LocalSandbox()

        assert sandbox.display == ":99"
        assert sandbox.resolution == "1920x1080x24"
        assert sandbox._state == SandboxState.STOPPED

    def test_local_sandbox_custom(self, mock_env_vars):
        """Test LocalSandbox with custom settings."""
        from src.computer_use.sandbox import LocalSandbox

        sandbox = LocalSandbox(display=":1", resolution="1280x720x24")

        assert sandbox.display == ":1"
        assert sandbox.resolution == "1280x720x24"

    @pytest.mark.asyncio
    async def test_local_sandbox_start(self, mock_env_vars):
        """Test starting local sandbox."""
        from src.computer_use.sandbox import LocalSandbox, SandboxState

        sandbox = LocalSandbox()

        mock_process = AsyncMock()
        mock_process.returncode = None

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ):
            info = await sandbox.start()

            assert info.container_id == "local"
            assert info.state == SandboxState.RUNNING
            assert sandbox._state == SandboxState.RUNNING

    @pytest.mark.asyncio
    async def test_local_sandbox_start_xvfb_failed(self, mock_env_vars):
        """Test local sandbox start when Xvfb fails."""
        from src.computer_use.sandbox import LocalSandbox, SandboxState

        sandbox = LocalSandbox()

        mock_process = AsyncMock()
        mock_process.returncode = 1  # Process exited

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ):
            with pytest.raises(RuntimeError, match="Xvfb failed to start"):
                await sandbox.start()

            assert sandbox._state == SandboxState.ERROR

    @pytest.mark.asyncio
    async def test_local_sandbox_start_no_xvfb(self, mock_env_vars):
        """Test local sandbox start when Xvfb not installed."""
        from src.computer_use.sandbox import LocalSandbox, SandboxState

        sandbox = LocalSandbox()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("Xvfb not found"),
        ):
            with pytest.raises(RuntimeError, match="Xvfb not installed"):
                await sandbox.start()

            assert sandbox._state == SandboxState.ERROR

    @pytest.mark.asyncio
    async def test_local_sandbox_screenshot(self, mock_env_vars):
        """Test taking screenshot from local sandbox."""
        from src.computer_use.sandbox import LocalSandbox

        sandbox = LocalSandbox()

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"png_data", b"")

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ):
            screenshot = await sandbox.screenshot()

            assert screenshot == b"png_data"

    @pytest.mark.asyncio
    async def test_local_sandbox_screenshot_failure(self, mock_env_vars):
        """Test screenshot failure in local sandbox."""
        from src.computer_use.sandbox import LocalSandbox

        sandbox = LocalSandbox()

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error")

        with patch(
            "asyncio.create_subprocess_exec", return_value=mock_process
        ):
            with pytest.raises(RuntimeError, match="Screenshot failed"):
                await sandbox.screenshot()

    @pytest.mark.asyncio
    async def test_local_sandbox_stop(self, mock_env_vars):
        """Test stopping local sandbox."""
        from src.computer_use.sandbox import LocalSandbox, SandboxState

        sandbox = LocalSandbox()
        mock_process = MagicMock()
        mock_process.wait = AsyncMock()
        sandbox._xvfb_process = mock_process
        sandbox._state = SandboxState.RUNNING

        await sandbox.stop()

        mock_process.terminate.assert_called_once()
        assert sandbox._xvfb_process is None
        assert sandbox._state == SandboxState.STOPPED


class TestBuildSandboxImage:
    """Tests for build_sandbox_image function."""

    @pytest.mark.asyncio
    async def test_build_no_docker(self, mock_env_vars):
        """Test build fails without docker module."""
        from src.computer_use.sandbox import build_sandbox_image

        with patch.dict("sys.modules", {"docker": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(RuntimeError, match="Docker SDK not installed"):
                    await build_sandbox_image()

    @pytest.mark.asyncio
    async def test_build_with_dockerfile_path(self, mock_env_vars):
        """Test build with provided Dockerfile."""
        from src.computer_use.sandbox import build_sandbox_image
        from pathlib import Path
        import tempfile

        mock_docker_client = MagicMock()
        mock_image = MagicMock()
        mock_image.id = "sha256:abc123"
        mock_docker_client.images.build.return_value = (mock_image, [])

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11")

            with patch("docker.from_env", return_value=mock_docker_client):
                image_id = await build_sandbox_image(
                    dockerfile_path=dockerfile, tag="test:latest"
                )

                assert image_id == "sha256:abc123"
                mock_docker_client.images.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_with_logs(self, mock_env_vars):
        """Test build with log output."""
        from src.computer_use.sandbox import build_sandbox_image
        from pathlib import Path
        import tempfile

        mock_docker_client = MagicMock()
        mock_image = MagicMock()
        mock_image.id = "sha256:def456"
        mock_docker_client.images.build.return_value = (
            mock_image,
            [{"stream": "Step 1/5: FROM python:3.11"}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = Path(tmpdir) / "Dockerfile"
            dockerfile.write_text("FROM python:3.11")

            with patch("docker.from_env", return_value=mock_docker_client):
                image_id = await build_sandbox_image(
                    dockerfile_path=dockerfile, tag="test:latest"
                )

                assert image_id == "sha256:def456"
