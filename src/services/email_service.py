"""Email service with multiple provider support.

Supports:
- Resend (primary - modern, simple API)
- SendGrid (fallback)
- SMTP (fallback)
- Console/Log (development - just logs emails)
"""

import os
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib
import httpx
import structlog

logger = structlog.get_logger()


class EmailProvider(ABC):
    """Abstract base class for email providers."""

    @abstractmethod
    async def send(
        self,
        to: str,
        subject: str,
        html: str,
        text: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
    ) -> bool:
        """Send an email.

        Args:
            to: Recipient email address
            subject: Email subject
            html: HTML body content
            text: Plain text body content (optional)
            from_email: Sender email address
            from_name: Sender display name

        Returns:
            True if email was sent successfully, False otherwise
        """
        pass


class ConsoleEmailProvider(EmailProvider):
    """Development provider - logs emails to console instead of sending."""

    async def send(
        self,
        to: str,
        subject: str,
        html: str,
        text: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
    ) -> bool:
        logger.info(
            "Email sent (console/development mode)",
            to=to,
            subject=subject,
            from_email=from_email,
            from_name=from_name,
            html_length=len(html),
            text_length=len(text) if text else 0,
        )
        # In development, also print the HTML for debugging
        if os.getenv("EMAIL_DEBUG", "false").lower() == "true":
            logger.debug("Email HTML content", html=html)
        return True


class ResendEmailProvider(EmailProvider):
    """Resend.com email provider.

    See: https://resend.com/docs/api-reference/emails/send-email
    """

    API_URL = "https://api.resend.com/emails"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def send(
        self,
        to: str,
        subject: str,
        html: str,
        text: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
    ) -> bool:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Format the from field
        from_field = from_email or "noreply@heyargus.ai"
        if from_name:
            from_field = f"{from_name} <{from_field}>"

        payload = {
            "from": from_field,
            "to": [to] if isinstance(to, str) else to,
            "subject": subject,
            "html": html,
        }

        if text:
            payload["text"] = text

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        "Email sent via Resend",
                        to=to,
                        subject=subject,
                        message_id=data.get("id"),
                    )
                    return True
                else:
                    logger.error(
                        "Resend API error",
                        status_code=response.status_code,
                        response=response.text,
                        to=to,
                        subject=subject,
                    )
                    return False

        except httpx.RequestError as e:
            logger.error(
                "Resend request failed",
                error=str(e),
                to=to,
                subject=subject,
            )
            return False


class SendGridEmailProvider(EmailProvider):
    """SendGrid email provider.

    See: https://docs.sendgrid.com/api-reference/mail-send/mail-send
    """

    API_URL = "https://api.sendgrid.com/v3/mail/send"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def send(
        self,
        to: str,
        subject: str,
        html: str,
        text: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
    ) -> bool:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "personalizations": [{"to": [{"email": to}]}],
            "from": {
                "email": from_email or "noreply@heyargus.ai",
                "name": from_name or "Argus",
            },
            "subject": subject,
            "content": [{"type": "text/html", "value": html}],
        }

        if text:
            payload["content"].insert(0, {"type": "text/plain", "value": text})

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )

                # SendGrid returns 202 for accepted
                if response.status_code in (200, 202):
                    message_id = response.headers.get("X-Message-Id", "unknown")
                    logger.info(
                        "Email sent via SendGrid",
                        to=to,
                        subject=subject,
                        message_id=message_id,
                    )
                    return True
                else:
                    logger.error(
                        "SendGrid API error",
                        status_code=response.status_code,
                        response=response.text,
                        to=to,
                        subject=subject,
                    )
                    return False

        except httpx.RequestError as e:
            logger.error(
                "SendGrid request failed",
                error=str(e),
                to=to,
                subject=subject,
            )
            return False


class SMTPEmailProvider(EmailProvider):
    """SMTP email provider for self-hosted or traditional email servers."""

    def __init__(
        self,
        host: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = True,
        start_tls: bool = False,
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.start_tls = start_tls

    async def send(
        self,
        to: str,
        subject: str,
        html: str,
        text: str | None = None,
        from_email: str | None = None,
        from_name: str | None = None,
    ) -> bool:
        sender = from_email or "noreply@heyargus.ai"
        if from_name:
            sender_display = f"{from_name} <{sender}>"
        else:
            sender_display = sender

        # Create multipart message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = sender_display
        msg["To"] = to

        # Add text part if provided
        if text:
            msg.attach(MIMEText(text, "plain"))

        # Add HTML part
        msg.attach(MIMEText(html, "html"))

        try:
            # Send via SMTP
            await aiosmtplib.send(
                msg,
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                use_tls=self.use_tls,
                start_tls=self.start_tls,
                timeout=30,
            )

            logger.info(
                "Email sent via SMTP",
                to=to,
                subject=subject,
                host=self.host,
            )
            return True

        except Exception as e:
            logger.error(
                "SMTP send failed",
                error=str(e),
                to=to,
                subject=subject,
                host=self.host,
            )
            return False


class EmailService:
    """Main email service with provider abstraction and template rendering."""

    def __init__(self, provider: EmailProvider | None = None):
        """Initialize email service.

        Args:
            provider: Optional email provider. If not provided, will auto-detect
                     from environment variables.
        """
        self.provider = provider or self._get_provider()
        self.from_email = os.getenv("EMAIL_FROM", "noreply@heyargus.ai")
        self.from_name = os.getenv("EMAIL_FROM_NAME", "Argus")

    def _get_provider(self) -> EmailProvider:
        """Auto-detect and return the appropriate email provider."""
        # Try Resend first (preferred)
        resend_key = os.getenv("RESEND_API_KEY")
        if resend_key:
            logger.info("Using Resend email provider")
            return ResendEmailProvider(resend_key)

        # Try SendGrid
        sendgrid_key = os.getenv("SENDGRID_API_KEY")
        if sendgrid_key:
            logger.info("Using SendGrid email provider")
            return SendGridEmailProvider(sendgrid_key)

        # Try SMTP
        smtp_host = os.getenv("SMTP_HOST")
        if smtp_host:
            logger.info("Using SMTP email provider", host=smtp_host)
            return SMTPEmailProvider(
                host=smtp_host,
                port=int(os.getenv("SMTP_PORT", "587")),
                username=os.getenv("SMTP_USERNAME"),
                password=os.getenv("SMTP_PASSWORD"),
                use_tls=os.getenv("SMTP_USE_TLS", "true").lower() == "true",
                start_tls=os.getenv("SMTP_START_TLS", "false").lower() == "true",
            )

        # Default to console provider for development
        logger.info("Using Console email provider (development mode)")
        return ConsoleEmailProvider()

    async def send_email(
        self,
        to: str,
        subject: str,
        html: str,
        text: str | None = None,
    ) -> bool:
        """Send an email using the configured provider.

        Args:
            to: Recipient email address
            subject: Email subject
            html: HTML body content
            text: Plain text body content (optional)

        Returns:
            True if email was sent successfully, False otherwise
        """
        return await self.provider.send(
            to=to,
            subject=subject,
            html=html,
            text=text,
            from_email=self.from_email,
            from_name=self.from_name,
        )

    async def send_invitation(
        self,
        to: str,
        org_name: str,
        inviter_email: str,
        token: str,
        role: str,
    ) -> bool:
        """Send an organization invitation email.

        Args:
            to: Recipient email address
            org_name: Name of the organization
            inviter_email: Email of the person who sent the invitation
            token: Unique invitation token
            role: Role being offered (e.g., 'admin', 'member')

        Returns:
            True if email was sent successfully, False otherwise
        """
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
        accept_url = f"{frontend_url}/invitations/{token}"

        subject = f"You've been invited to join {org_name} on Argus"
        html = self._render_invitation_template(
            org_name=org_name,
            inviter_email=inviter_email,
            accept_url=accept_url,
            role=role,
        )
        text = self._render_invitation_text(
            org_name=org_name,
            inviter_email=inviter_email,
            accept_url=accept_url,
            role=role,
        )

        return await self.send_email(to=to, subject=subject, html=html, text=text)

    def _render_invitation_template(
        self,
        org_name: str,
        inviter_email: str,
        accept_url: str,
        role: str,
    ) -> str:
        """Render the HTML invitation email template."""
        # Format role for display
        role_display = role.replace("_", " ").title()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>You're Invited to {org_name}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f4f4f5; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f4f4f5;">
        <tr>
            <td align="center" style="padding: 40px 20px;">
                <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                    <!-- Header with Logo -->
                    <tr>
                        <td align="center" style="padding: 40px 40px 20px 40px;">
                            <table role="presentation" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 12px; padding: 12px 20px;">
                                        <span style="font-size: 24px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;">Argus</span>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Main Content -->
                    <tr>
                        <td style="padding: 20px 40px;">
                            <h1 style="margin: 0 0 16px 0; font-size: 28px; font-weight: 700; color: #18181b; text-align: center; line-height: 1.3;">
                                You've been invited to join<br>
                                <span style="color: #6366f1;">{org_name}</span>
                            </h1>
                        </td>
                    </tr>

                    <!-- Invitation Details Card -->
                    <tr>
                        <td style="padding: 0 40px 30px 40px;">
                            <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;">
                                <tr>
                                    <td style="padding: 24px;">
                                        <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                                            <tr>
                                                <td style="padding-bottom: 16px;">
                                                    <p style="margin: 0; font-size: 14px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Invited by</p>
                                                    <p style="margin: 4px 0 0 0; font-size: 16px; color: #18181b; font-weight: 500;">{inviter_email}</p>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding-bottom: 16px;">
                                                    <p style="margin: 0; font-size: 14px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Your Role</p>
                                                    <p style="margin: 4px 0 0 0; font-size: 16px; color: #18181b; font-weight: 500;">
                                                        <span style="display: inline-block; background-color: #dbeafe; color: #1d4ed8; padding: 4px 12px; border-radius: 9999px; font-size: 14px; font-weight: 600;">{role_display}</span>
                                                    </p>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>
                                                    <p style="margin: 0; font-size: 14px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600;">Organization</p>
                                                    <p style="margin: 4px 0 0 0; font-size: 16px; color: #18181b; font-weight: 500;">{org_name}</p>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- CTA Button -->
                    <tr>
                        <td align="center" style="padding: 0 40px 30px 40px;">
                            <table role="presentation" cellpadding="0" cellspacing="0">
                                <tr>
                                    <td style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 8px;">
                                        <a href="{accept_url}" target="_blank" style="display: inline-block; padding: 16px 48px; font-size: 16px; font-weight: 600; color: #ffffff; text-decoration: none;">
                                            Accept Invitation
                                        </a>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Expiration Notice -->
                    <tr>
                        <td style="padding: 0 40px 30px 40px;">
                            <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background-color: #fef3c7; border-radius: 8px; border: 1px solid #fcd34d;">
                                <tr>
                                    <td style="padding: 16px 20px;">
                                        <table role="presentation" cellpadding="0" cellspacing="0" width="100%">
                                            <tr>
                                                <td width="24" valign="top">
                                                    <span style="font-size: 18px;">&#9888;</span>
                                                </td>
                                                <td style="padding-left: 12px;">
                                                    <p style="margin: 0; font-size: 14px; color: #92400e; font-weight: 500;">
                                                        This invitation expires in <strong>7 days</strong>. Please accept it before it expires.
                                                    </p>
                                                </td>
                                            </tr>
                                        </table>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>

                    <!-- Alternative Link -->
                    <tr>
                        <td style="padding: 0 40px 30px 40px;">
                            <p style="margin: 0; font-size: 14px; color: #6b7280; text-align: center; line-height: 1.6;">
                                If the button above doesn't work, copy and paste this link into your browser:
                            </p>
                            <p style="margin: 8px 0 0 0; font-size: 12px; color: #6366f1; text-align: center; word-break: break-all;">
                                <a href="{accept_url}" style="color: #6366f1; text-decoration: underline;">{accept_url}</a>
                            </p>
                        </td>
                    </tr>

                    <!-- Divider -->
                    <tr>
                        <td style="padding: 0 40px;">
                            <hr style="border: none; border-top: 1px solid #e5e7eb; margin: 0;">
                        </td>
                    </tr>

                    <!-- Footer -->
                    <tr>
                        <td style="padding: 30px 40px;">
                            <p style="margin: 0 0 8px 0; font-size: 14px; color: #6b7280; text-align: center;">
                                Didn't expect this invitation? You can safely ignore this email.
                            </p>
                            <p style="margin: 0; font-size: 12px; color: #9ca3af; text-align: center;">
                                &copy; {self._get_current_year()} Argus. All rights reserved.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>"""

    def _render_invitation_text(
        self,
        org_name: str,
        inviter_email: str,
        accept_url: str,
        role: str,
    ) -> str:
        """Render the plain text invitation email."""
        role_display = role.replace("_", " ").title()

        return f"""You've been invited to join {org_name} on Argus

{inviter_email} has invited you to join {org_name} as a {role_display}.

To accept this invitation, click the link below:
{accept_url}

IMPORTANT: This invitation expires in 7 days.

If you didn't expect this invitation, you can safely ignore this email.

---
Argus - Autonomous E2E Testing
"""

    def _get_current_year(self) -> int:
        """Get the current year for copyright notice."""
        from datetime import datetime

        return datetime.now().year


# Singleton instance
_email_service: EmailService | None = None


def get_email_service() -> EmailService:
    """Get the singleton EmailService instance.

    Returns:
        The EmailService instance, creating it if necessary.
    """
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service
