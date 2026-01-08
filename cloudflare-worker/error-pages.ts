/**
 * Cloudflare Worker for Custom Error Pages
 * This handles errors at the edge when the origin is down
 */

const ERROR_HTML = (statusCode: number, title: string, message: string) => `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} - Argus</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: linear-gradient(to bottom, #09090b, #18181b);
      color: #fafafa;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .container { max-width: 480px; text-align: center; }
    .logo {
      width: 64px;
      height: 64px;
      background: linear-gradient(135deg, #7c3aed, #a855f7);
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 24px;
    }
    .logo svg { width: 36px; height: 36px; }
    .error-code {
      font-size: 72px;
      font-weight: 700;
      color: rgba(124, 58, 237, 0.3);
      line-height: 1;
      margin-bottom: 16px;
    }
    h1 { font-size: 24px; font-weight: 600; margin-bottom: 12px; }
    p { color: #a1a1aa; line-height: 1.6; margin-bottom: 24px; }
    .actions { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 32px; }
    .btn {
      padding: 12px 24px;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      text-decoration: none;
      transition: all 0.2s;
    }
    .btn-primary { background: #7c3aed; color: white; }
    .btn-primary:hover { background: #6d28d9; }
    .btn-secondary { background: transparent; color: #fafafa; border: 1px solid #3f3f46; }
    .btn-secondary:hover { background: #27272a; }
    .help { border-top: 1px solid #27272a; padding-top: 24px; }
    .help p { font-size: 14px; margin-bottom: 12px; }
    .help-links { display: flex; gap: 24px; justify-content: center; flex-wrap: wrap; }
    .help-links a { color: #a1a1aa; font-size: 14px; text-decoration: none; }
    .help-links a:hover { color: #fafafa; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: #27272a;
      padding: 8px 16px;
      border-radius: 20px;
      font-size: 12px;
      color: #a1a1aa;
      margin-bottom: 24px;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      background: #22c55e;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="logo">
      <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
        <circle cx="12" cy="12" r="3"/>
      </svg>
    </div>

    <div class="error-code">${statusCode}</div>
    <h1>${title}</h1>
    <p>${message}</p>

    <div class="status">
      <span class="status-dot"></span>
      Checking service status...
    </div>

    <div class="actions">
      <a href="/" class="btn btn-primary">Go to Homepage</a>
      <a href="https://status.heyargus.ai" target="_blank" class="btn btn-secondary">View Status</a>
    </div>

    <div class="help">
      <p>Need assistance?</p>
      <div class="help-links">
        <a href="https://status.heyargus.ai" target="_blank">System Status</a>
        <a href="https://docs.heyargus.ai" target="_blank">Documentation</a>
        <a href="mailto:support@heyargus.com">support@heyargus.com</a>
      </div>
    </div>
  </div>

  <script>
    // Auto-refresh status
    fetch('https://status.heyargus.ai/api/status')
      .then(r => r.json())
      .then(data => {
        const statusEl = document.querySelector('.status');
        if (data.status === 'operational') {
          statusEl.innerHTML = '<span class="status-dot" style="background:#22c55e"></span> All systems operational';
        } else {
          statusEl.innerHTML = '<span class="status-dot" style="background:#f59e0b"></span> Service disruption detected';
        }
      })
      .catch(() => {
        document.querySelector('.status').innerHTML = '<span class="status-dot" style="background:#71717a"></span> Checking status...';
      });
  </script>
</body>
</html>
`;

const ERROR_CONFIGS: Record<number, { title: string; message: string }> = {
  500: {
    title: 'Internal Server Error',
    message: 'Our servers encountered an unexpected error. Our team has been automatically notified and is working to resolve the issue.',
  },
  502: {
    title: 'Bad Gateway',
    message: 'We\'re having trouble connecting to our servers. This is usually temporary - please try again in a few moments.',
  },
  503: {
    title: 'Service Unavailable',
    message: 'Argus is temporarily unavailable for maintenance or experiencing high load. We\'ll be back shortly.',
  },
  504: {
    title: 'Gateway Timeout',
    message: 'The request took too long to complete. Please try again or check our status page for any ongoing issues.',
  },
  520: {
    title: 'Unknown Error',
    message: 'An unexpected error occurred at our edge. Our team has been notified.',
  },
  521: {
    title: 'Web Server is Down',
    message: 'Our origin server is currently offline. We\'re working to restore service as quickly as possible.',
  },
  522: {
    title: 'Connection Timed Out',
    message: 'We couldn\'t establish a connection to our servers. This is usually temporary.',
  },
  523: {
    title: 'Origin is Unreachable',
    message: 'We can\'t reach our servers right now. Our team is investigating the issue.',
  },
  524: {
    title: 'A Timeout Occurred',
    message: 'The connection was established but the response took too long. Please try again.',
  },
};

export default {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    // Check if this is an error page request (for testing)
    if (url.pathname.startsWith('/__error/')) {
      const statusCode = parseInt(url.pathname.split('/')[2]) || 500;
      const config = ERROR_CONFIGS[statusCode] || ERROR_CONFIGS[500];

      return new Response(ERROR_HTML(statusCode, config.title, config.message), {
        status: statusCode,
        headers: {
          'Content-Type': 'text/html;charset=UTF-8',
          'Cache-Control': 'no-store',
        },
      });
    }

    // Pass through to origin
    try {
      const response = await fetch(request);

      // Intercept 5xx errors from origin
      if (response.status >= 500 && response.status < 600) {
        const config = ERROR_CONFIGS[response.status] || ERROR_CONFIGS[500];
        return new Response(ERROR_HTML(response.status, config.title, config.message), {
          status: response.status,
          headers: {
            'Content-Type': 'text/html;charset=UTF-8',
            'Cache-Control': 'no-store',
          },
        });
      }

      return response;
    } catch (error) {
      // Origin is completely unreachable
      const config = ERROR_CONFIGS[523];
      return new Response(ERROR_HTML(523, config.title, config.message), {
        status: 523,
        headers: {
          'Content-Type': 'text/html;charset=UTF-8',
          'Cache-Control': 'no-store',
        },
      });
    }
  },
};
