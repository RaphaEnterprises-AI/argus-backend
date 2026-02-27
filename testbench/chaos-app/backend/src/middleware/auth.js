const jwt = require('jsonwebtoken');
const { isBugEnabled } = require('../bugs/registry');

const JWT_SECRET = process.env.JWT_SECRET || 'chaos-app-secret-key';

/**
 * JWT authentication middleware.
 * When sec-broken-auth bug is enabled, skips expiration check.
 */
function authenticate(req, res, next) {
  const authHeader = req.headers.authorization;
  const tokenFromCookie = req.cookies && req.cookies.token;
  const token =
    (authHeader && authHeader.startsWith('Bearer ')
      ? authHeader.slice(7)
      : null) || tokenFromCookie;

  if (!token) {
    return res.status(401).json({ error: 'Authentication required' });
  }

  try {
    const verifyOptions = {};

    // sec-broken-auth: Skip expiration validation
    if (isBugEnabled('sec-broken-auth')) {
      verifyOptions.ignoreExpiration = true;
    }

    const decoded = jwt.verify(token, JWT_SECRET, verifyOptions);
    req.user = decoded;
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired' });
    }
    return res.status(401).json({ error: 'Invalid token' });
  }
}

/**
 * Admin-only middleware. Must be used after authenticate.
 */
function requireAdmin(req, res, next) {
  if (!req.user || req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Admin access required' });
  }
  next();
}

/**
 * Generate a JWT token for a user.
 */
function generateToken(user) {
  return jwt.sign(
    {
      userId: user.id,
      email: user.email,
      name: user.name,
      role: user.role,
    },
    JWT_SECRET,
    { expiresIn: '1h' }
  );
}

module.exports = {
  authenticate,
  requireAdmin,
  generateToken,
  JWT_SECRET,
};
