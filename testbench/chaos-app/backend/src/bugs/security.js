const { isBugEnabled } = require('./registry');
const http = require('http');
const https = require('https');
const url = require('url');

/**
 * Apply security bugs as middleware modifications.
 * NOTE: This is an intentional chaos/testing app. All security
 * vulnerabilities are deliberately introduced to test Argus's
 * security scanning capabilities. No real user data is involved.
 */
function apply(req, res) {
  // sec-xss-reflected: Include raw query in response without escaping
  // (Applied in transformJson and injectScripts below)

  // sec-sqli: SQL injection simulation in login
  // (Applied in auth route override — see auth.js route)

  // sec-broken-auth: Skip JWT expiration
  // (Applied in auth middleware — see middleware/auth.js)

  // sec-idor: Skip ownership check
  // (Applied in users route — see routes/users.js)
}

/**
 * Transform JSON responses for security bugs.
 */
function transformJson(req, res, data) {
  // sec-xss-reflected: If search query present, include raw query in response
  if (isBugEnabled('sec-xss-reflected') && req.query.q) {
    if (typeof data === 'object' && data !== null) {
      data._searchQuery = req.query.q; // Raw, unescaped
      data._searchMessage = `Results for: ${req.query.q}`; // Reflected
    }
  }
  return data;
}

/**
 * Inject scripts into HTML responses for security bugs.
 */
function injectScripts(html) {
  // sec-secrets-in-code: Inject fake API key into frontend
  if (isBugEnabled('sec-secrets-in-code')) {
    const secretScript =
      '<script>window.API_KEY="sk-test-FAKE-KEY-12345";window.DB_PASSWORD="admin123";</script>';
    html = html.replace('</head>', secretScript + '</head>');
  }

  return html;
}

/**
 * SSRF endpoint handler — follows any URL including internal ones.
 * Only active when sec-ssrf bug is enabled.
 */
function handleSsrf(req, res) {
  if (!isBugEnabled('sec-ssrf')) {
    return res.status(404).json({ error: 'Endpoint not available' });
  }

  const targetUrl = req.body.url;
  if (!targetUrl) {
    return res.status(400).json({ error: 'URL required' });
  }

  // Deliberately vulnerable: follows any URL including internal ones
  const parsed = url.parse(targetUrl);
  const client = parsed.protocol === 'https:' ? https : http;

  const fetchReq = client.get(targetUrl, { timeout: 5000 }, (fetchRes) => {
    let body = '';
    fetchRes.on('data', (chunk) => (body += chunk));
    fetchRes.on('end', () => {
      res.json({
        status: fetchRes.statusCode,
        headers: fetchRes.headers,
        body: body.substring(0, 10000), // Limit response size
      });
    });
  });

  fetchReq.on('error', (err) => {
    res.status(500).json({ error: `Fetch failed: ${err.message}` });
  });

  fetchReq.on('timeout', () => {
    fetchReq.destroy();
    res.status(504).json({ error: 'Request timed out' });
  });
}

/**
 * Check if SQL injection bypass should be applied to login.
 */
function shouldBypassLogin(email) {
  if (!isBugEnabled('sec-sqli')) return false;
  // Simulate SQL injection: if input contains typical SQLi patterns, bypass auth
  return (
    email &&
    (email.includes("' OR '1'='1") ||
      email.includes('" OR "1"="1') ||
      email.includes('--') ||
      email.includes('; DROP'))
  );
}

/**
 * Get SQL-like log message for injection testing.
 */
function getSqlLog(email) {
  if (!isBugEnabled('sec-sqli')) return null;
  // Deliberately vulnerable: concatenates raw input into SQL-like string
  return `SELECT * FROM users WHERE email = '${email}'`;
}

module.exports = {
  apply,
  transformJson,
  injectScripts,
  handleSsrf,
  shouldBypassLogin,
  getSqlLog,
};
