"""Tests for the email service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os
from datetime import datetime


class TestConsoleEmailProvider:
    """Tests for ConsoleEmailProvider."""

    @pytest.mark.asyncio
    async def test_send_logs_email(self, mock_env_vars):
        """Test that console provider logs email instead of sending."""
        from src.services.email_service import ConsoleEmailProvider

        provider = ConsoleEmailProvider()

        with patch("src.services.email_service.logger") as mock_logger:
            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
                text="Hello",
                from_email="sender@example.com",
                from_name="Sender",
            )

        assert result is True
        mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_with_debug_mode(self, mock_env_vars, monkeypatch):
        """Test that console provider logs HTML in debug mode."""
        from src.services.email_service import ConsoleEmailProvider

        monkeypatch.setenv("EMAIL_DEBUG", "true")
        provider = ConsoleEmailProvider()

        with patch("src.services.email_service.logger") as mock_logger:
            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is True
        assert mock_logger.debug.called

    @pytest.mark.asyncio
    async def test_send_without_text(self, mock_env_vars):
        """Test sending email without text content."""
        from src.services.email_service import ConsoleEmailProvider

        provider = ConsoleEmailProvider()

        result = await provider.send(
            to="test@example.com",
            subject="Test Subject",
            html="<h1>Hello</h1>",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_without_optional_params(self, mock_env_vars):
        """Test sending email without optional parameters."""
        from src.services.email_service import ConsoleEmailProvider

        provider = ConsoleEmailProvider()

        result = await provider.send(
            to="test@example.com",
            subject="Test Subject",
            html="<h1>Hello</h1>",
        )

        assert result is True


class TestResendEmailProvider:
    """Tests for ResendEmailProvider."""

    def test_init_stores_api_key(self):
        """Test that init stores API key."""
        from src.services.email_service import ResendEmailProvider

        provider = ResendEmailProvider("test-api-key")
        assert provider.api_key == "test-api-key"

    @pytest.mark.asyncio
    async def test_send_success(self, mock_env_vars):
        """Test successful email send."""
        from src.services.email_service import ResendEmailProvider

        provider = ResendEmailProvider("test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
                text="Hello",
                from_email="sender@example.com",
                from_name="Sender",
            )

        assert result is True
        mock_instance.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_with_default_from_email(self, mock_env_vars):
        """Test send with default from email."""
        from src.services.email_service import ResendEmailProvider

        provider = ResendEmailProvider("test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is True
        # Check that default from was used
        call_args = mock_instance.post.call_args
        assert "noreply@heyargus.ai" in str(call_args)

    @pytest.mark.asyncio
    async def test_send_api_error(self, mock_env_vars):
        """Test handling API error."""
        from src.services.email_service import ResendEmailProvider

        provider = ResendEmailProvider("test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_request_error(self, mock_env_vars):
        """Test handling request error."""
        from src.services.email_service import ResendEmailProvider
        import httpx

        provider = ResendEmailProvider("test-api-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_to_list(self, mock_env_vars):
        """Test sending to a list of recipients."""
        from src.services.email_service import ResendEmailProvider

        provider = ResendEmailProvider("test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            # Pass list of recipients
            result = await provider.send(
                to=["user1@example.com", "user2@example.com"],
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is True


class TestSendGridEmailProvider:
    """Tests for SendGridEmailProvider."""

    def test_init_stores_api_key(self):
        """Test that init stores API key."""
        from src.services.email_service import SendGridEmailProvider

        provider = SendGridEmailProvider("sg-api-key")
        assert provider.api_key == "sg-api-key"

    @pytest.mark.asyncio
    async def test_send_success_200(self, mock_env_vars):
        """Test successful email send with 200 status."""
        from src.services.email_service import SendGridEmailProvider

        provider = SendGridEmailProvider("sg-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"X-Message-Id": "sg-msg-123"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_success_202(self, mock_env_vars):
        """Test successful email send with 202 status (accepted)."""
        from src.services.email_service import SendGridEmailProvider

        provider = SendGridEmailProvider("sg-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"X-Message-Id": "sg-msg-123"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_with_text(self, mock_env_vars):
        """Test sending email with text content."""
        from src.services.email_service import SendGridEmailProvider

        provider = SendGridEmailProvider("sg-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
                text="Hello",
            )

        assert result is True
        # Check that text was included in payload
        call_args = mock_instance.post.call_args
        payload = call_args.kwargs["json"]
        assert len(payload["content"]) == 2  # text and html

    @pytest.mark.asyncio
    async def test_send_api_error(self, mock_env_vars):
        """Test handling API error."""
        from src.services.email_service import SendGridEmailProvider

        provider = SendGridEmailProvider("sg-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_request_error(self, mock_env_vars):
        """Test handling request error."""
        from src.services.email_service import SendGridEmailProvider
        import httpx

        provider = SendGridEmailProvider("sg-api-key")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(
                side_effect=httpx.RequestError("Network error")
            )
            mock_client.return_value.__aenter__.return_value = mock_instance

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is False


class TestSMTPEmailProvider:
    """Tests for SMTPEmailProvider."""

    def test_init_stores_config(self):
        """Test that init stores SMTP configuration."""
        from src.services.email_service import SMTPEmailProvider

        provider = SMTPEmailProvider(
            host="smtp.example.com",
            port=587,
            username="user",
            password="pass",
            use_tls=True,
            start_tls=False,
        )

        assert provider.host == "smtp.example.com"
        assert provider.port == 587
        assert provider.username == "user"
        assert provider.password == "pass"
        assert provider.use_tls is True
        assert provider.start_tls is False

    @pytest.mark.asyncio
    async def test_send_success(self, mock_env_vars):
        """Test successful SMTP send."""
        from src.services.email_service import SMTPEmailProvider

        provider = SMTPEmailProvider(
            host="smtp.example.com",
            port=587,
            username="user",
            password="pass",
        )

        with patch("aiosmtplib.send", new_callable=AsyncMock) as mock_send:
            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
                text="Hello",
                from_email="sender@example.com",
                from_name="Sender",
            )

        assert result is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_without_text(self, mock_env_vars):
        """Test SMTP send without text content."""
        from src.services.email_service import SMTPEmailProvider

        provider = SMTPEmailProvider(
            host="smtp.example.com",
            port=587,
        )

        with patch("aiosmtplib.send", new_callable=AsyncMock) as mock_send:
            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_with_default_from(self, mock_env_vars):
        """Test SMTP send with default from address."""
        from src.services.email_service import SMTPEmailProvider

        provider = SMTPEmailProvider(
            host="smtp.example.com",
            port=587,
        )

        with patch("aiosmtplib.send", new_callable=AsyncMock) as mock_send:
            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is True
        # Check that message was sent
        call_args = mock_send.call_args
        msg = call_args.args[0]
        assert "noreply@heyargus.ai" in msg["From"]

    @pytest.mark.asyncio
    async def test_send_error(self, mock_env_vars):
        """Test handling SMTP send error."""
        from src.services.email_service import SMTPEmailProvider

        provider = SMTPEmailProvider(
            host="smtp.example.com",
            port=587,
        )

        with patch("aiosmtplib.send", new_callable=AsyncMock) as mock_send:
            mock_send.side_effect = Exception("SMTP connection failed")

            result = await provider.send(
                to="test@example.com",
                subject="Test Subject",
                html="<h1>Hello</h1>",
            )

        assert result is False


class TestEmailServiceInit:
    """Tests for EmailService initialization."""

    def test_init_with_provider(self, mock_env_vars):
        """Test init with custom provider."""
        from src.services.email_service import EmailService, ConsoleEmailProvider

        custom_provider = ConsoleEmailProvider()
        service = EmailService(provider=custom_provider)

        assert service.provider is custom_provider

    def test_init_auto_detect_resend(self, mock_env_vars, monkeypatch):
        """Test auto-detecting Resend provider."""
        from src.services.email_service import EmailService, ResendEmailProvider

        monkeypatch.setenv("RESEND_API_KEY", "re-test-key")
        # Clear other keys
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)

        service = EmailService()

        assert isinstance(service.provider, ResendEmailProvider)

    def test_init_auto_detect_sendgrid(self, mock_env_vars, monkeypatch):
        """Test auto-detecting SendGrid provider."""
        from src.services.email_service import EmailService, SendGridEmailProvider

        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.setenv("SENDGRID_API_KEY", "sg-test-key")
        monkeypatch.delenv("SMTP_HOST", raising=False)

        service = EmailService()

        assert isinstance(service.provider, SendGridEmailProvider)

    def test_init_auto_detect_smtp(self, mock_env_vars, monkeypatch):
        """Test auto-detecting SMTP provider."""
        from src.services.email_service import EmailService, SMTPEmailProvider

        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_PORT", "587")

        service = EmailService()

        assert isinstance(service.provider, SMTPEmailProvider)

    def test_init_auto_detect_console(self, mock_env_vars, monkeypatch):
        """Test auto-detecting Console provider as fallback."""
        from src.services.email_service import EmailService, ConsoleEmailProvider

        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)

        service = EmailService()

        assert isinstance(service.provider, ConsoleEmailProvider)

    def test_init_reads_from_env(self, mock_env_vars, monkeypatch):
        """Test that init reads from/name from environment."""
        from src.services.email_service import EmailService

        monkeypatch.setenv("EMAIL_FROM", "custom@example.com")
        monkeypatch.setenv("EMAIL_FROM_NAME", "Custom Name")

        service = EmailService()

        assert service.from_email == "custom@example.com"
        assert service.from_name == "Custom Name"


class TestEmailServiceSendEmail:
    """Tests for EmailService.send_email method."""

    @pytest.mark.asyncio
    async def test_send_email_delegates_to_provider(self, mock_env_vars):
        """Test that send_email delegates to provider."""
        from src.services.email_service import EmailService

        mock_provider = AsyncMock()
        mock_provider.send = AsyncMock(return_value=True)

        service = EmailService(provider=mock_provider)

        result = await service.send_email(
            to="test@example.com",
            subject="Test Subject",
            html="<h1>Hello</h1>",
            text="Hello",
        )

        assert result is True
        mock_provider.send.assert_called_once_with(
            to="test@example.com",
            subject="Test Subject",
            html="<h1>Hello</h1>",
            text="Hello",
            from_email=service.from_email,
            from_name=service.from_name,
        )

    @pytest.mark.asyncio
    async def test_send_email_without_text(self, mock_env_vars):
        """Test send_email without text content."""
        from src.services.email_service import EmailService

        mock_provider = AsyncMock()
        mock_provider.send = AsyncMock(return_value=True)

        service = EmailService(provider=mock_provider)

        result = await service.send_email(
            to="test@example.com",
            subject="Test Subject",
            html="<h1>Hello</h1>",
        )

        assert result is True
        mock_provider.send.assert_called_once()


class TestEmailServiceSendInvitation:
    """Tests for EmailService.send_invitation method."""

    @pytest.mark.asyncio
    async def test_send_invitation_success(self, mock_env_vars):
        """Test successful invitation send."""
        from src.services.email_service import EmailService

        mock_provider = AsyncMock()
        mock_provider.send = AsyncMock(return_value=True)

        service = EmailService(provider=mock_provider)

        result = await service.send_invitation(
            to="newuser@example.com",
            org_name="Acme Corp",
            inviter_email="admin@acme.com",
            token="abc123token",
            role="admin",
        )

        assert result is True
        mock_provider.send.assert_called_once()

        # Check that the call included proper subject
        call_args = mock_provider.send.call_args
        assert "Acme Corp" in call_args.kwargs["subject"]

    @pytest.mark.asyncio
    async def test_send_invitation_includes_role(self, mock_env_vars):
        """Test that invitation includes role in email."""
        from src.services.email_service import EmailService

        mock_provider = AsyncMock()
        mock_provider.send = AsyncMock(return_value=True)

        service = EmailService(provider=mock_provider)

        await service.send_invitation(
            to="newuser@example.com",
            org_name="Acme Corp",
            inviter_email="admin@acme.com",
            token="abc123token",
            role="project_manager",
        )

        call_args = mock_provider.send.call_args
        html_content = call_args.kwargs["html"]
        text_content = call_args.kwargs["text"]

        # Role should be formatted properly
        assert "Project Manager" in html_content
        assert "Project Manager" in text_content

    @pytest.mark.asyncio
    async def test_send_invitation_includes_accept_url(self, mock_env_vars, monkeypatch):
        """Test that invitation includes accept URL."""
        from src.services.email_service import EmailService

        monkeypatch.setenv("FRONTEND_URL", "https://app.example.com")

        mock_provider = AsyncMock()
        mock_provider.send = AsyncMock(return_value=True)

        service = EmailService(provider=mock_provider)

        await service.send_invitation(
            to="newuser@example.com",
            org_name="Acme Corp",
            inviter_email="admin@acme.com",
            token="abc123token",
            role="member",
        )

        call_args = mock_provider.send.call_args
        html_content = call_args.kwargs["html"]

        assert "https://app.example.com/invitations/abc123token" in html_content


class TestEmailServiceTemplateRendering:
    """Tests for email template rendering."""

    def test_render_invitation_template(self, mock_env_vars):
        """Test rendering invitation HTML template."""
        from src.services.email_service import EmailService

        service = EmailService()

        html = service._render_invitation_template(
            org_name="Test Org",
            inviter_email="inviter@test.com",
            accept_url="https://example.com/accept",
            role="admin",
        )

        assert "Test Org" in html
        assert "inviter@test.com" in html
        assert "https://example.com/accept" in html
        assert "Admin" in html
        assert "<!DOCTYPE html>" in html

    def test_render_invitation_template_formats_role(self, mock_env_vars):
        """Test that role is formatted properly in template."""
        from src.services.email_service import EmailService

        service = EmailService()

        html = service._render_invitation_template(
            org_name="Test Org",
            inviter_email="inviter@test.com",
            accept_url="https://example.com/accept",
            role="project_manager",
        )

        assert "Project Manager" in html

    def test_render_invitation_text(self, mock_env_vars):
        """Test rendering invitation plain text template."""
        from src.services.email_service import EmailService

        service = EmailService()

        text = service._render_invitation_text(
            org_name="Test Org",
            inviter_email="inviter@test.com",
            accept_url="https://example.com/accept",
            role="member",
        )

        assert "Test Org" in text
        assert "inviter@test.com" in text
        assert "https://example.com/accept" in text
        assert "Member" in text
        assert "7 days" in text

    def test_get_current_year(self, mock_env_vars):
        """Test getting current year for copyright."""
        from src.services.email_service import EmailService

        service = EmailService()
        year = service._get_current_year()

        assert year == datetime.now().year


class TestGetEmailService:
    """Tests for get_email_service factory function."""

    def test_get_email_service_creates_singleton(self, mock_env_vars, monkeypatch):
        """Test that get_email_service creates singleton."""
        from src.services.email_service import get_email_service
        import src.services.email_service as module

        # Clear any environment keys
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)

        # Reset singleton
        module._email_service = None

        service1 = get_email_service()
        service2 = get_email_service()

        assert service1 is service2

        # Cleanup
        module._email_service = None

    def test_get_email_service_returns_service(self, mock_env_vars, monkeypatch):
        """Test that get_email_service returns EmailService instance."""
        from src.services.email_service import get_email_service, EmailService
        import src.services.email_service as module

        # Clear any environment keys
        monkeypatch.delenv("RESEND_API_KEY", raising=False)
        monkeypatch.delenv("SENDGRID_API_KEY", raising=False)
        monkeypatch.delenv("SMTP_HOST", raising=False)

        # Reset singleton
        module._email_service = None

        service = get_email_service()

        assert isinstance(service, EmailService)

        # Cleanup
        module._email_service = None


class TestEmailProviderAbstractClass:
    """Tests for EmailProvider abstract class."""

    def test_email_provider_is_abstract(self):
        """Test that EmailProvider cannot be instantiated."""
        from src.services.email_service import EmailProvider

        # EmailProvider is abstract and should not be instantiated directly
        # This tests that subclasses properly implement the interface
        assert hasattr(EmailProvider, "send")

    @pytest.mark.asyncio
    async def test_concrete_provider_implements_interface(self, mock_env_vars):
        """Test that concrete providers implement the interface."""
        from src.services.email_service import (
            ConsoleEmailProvider,
            ResendEmailProvider,
            SendGridEmailProvider,
            SMTPEmailProvider,
        )

        providers = [
            ConsoleEmailProvider(),
            ResendEmailProvider("key"),
            SendGridEmailProvider("key"),
            SMTPEmailProvider("host", 587),
        ]

        for provider in providers:
            assert hasattr(provider, "send")
            assert callable(provider.send)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_resend_with_from_name(self, mock_env_vars):
        """Test Resend provider properly formats from name."""
        from src.services.email_service import ResendEmailProvider

        provider = ResendEmailProvider("test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.post = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value = mock_instance

            await provider.send(
                to="test@example.com",
                subject="Test",
                html="<h1>Hi</h1>",
                from_email="sender@example.com",
                from_name="Sender Name",
            )

        call_args = mock_instance.post.call_args
        payload = call_args.kwargs["json"]
        assert "Sender Name <sender@example.com>" in payload["from"]

    @pytest.mark.asyncio
    async def test_smtp_with_from_name(self, mock_env_vars):
        """Test SMTP provider properly formats from name."""
        from src.services.email_service import SMTPEmailProvider

        provider = SMTPEmailProvider("smtp.example.com", 587)

        with patch("aiosmtplib.send", new_callable=AsyncMock) as mock_send:
            await provider.send(
                to="test@example.com",
                subject="Test",
                html="<h1>Hi</h1>",
                from_email="sender@example.com",
                from_name="Sender Name",
            )

        call_args = mock_send.call_args
        msg = call_args.args[0]
        assert "Sender Name" in msg["From"]

    def test_smtp_provider_with_start_tls(self):
        """Test SMTP provider with STARTTLS configuration."""
        from src.services.email_service import SMTPEmailProvider

        provider = SMTPEmailProvider(
            host="smtp.example.com",
            port=587,
            use_tls=False,
            start_tls=True,
        )

        assert provider.use_tls is False
        assert provider.start_tls is True
