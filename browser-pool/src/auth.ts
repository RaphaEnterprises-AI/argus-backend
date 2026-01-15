/**
 * Production-Grade Authentication for Browser Pool
 *
 * Implements JWT-based authentication with:
 * - Short-lived tokens (5 minutes)
 * - User context passthrough
 * - Audit logging
 * - Token verification
 */

import * as crypto from 'crypto';

// JWT Header for HS256
const JWT_HEADER = {
  alg: 'HS256',
  typ: 'JWT'
};

export interface TokenPayload {
  // Standard JWT claims
  iss: string;           // Issuer (argus-backend, argus-mcp)
  sub: string;           // Subject (user_id)
  aud: string;           // Audience (browser-pool)
  exp: number;           // Expiration timestamp
  iat: number;           // Issued at timestamp
  jti: string;           // Unique token ID

  // Custom claims
  org_id?: string;       // Organization ID
  email?: string;        // User email (for audit)
  action?: string;       // Requested action (observe, act, test)
  ip?: string;           // Client IP
}

export interface VerifyResult {
  valid: boolean;
  payload?: TokenPayload;
  error?: string;
}

export interface AuditEntry {
  timestamp: string;
  token_id: string;
  user_id: string;
  org_id?: string;
  email?: string;
  action: string;
  endpoint: string;
  ip: string;
  success: boolean;
  error?: string;
  duration_ms?: number;
}

/**
 * Base64URL encode (JWT-safe)
 */
function base64UrlEncode(data: string | Buffer): string {
  const base64 = Buffer.from(data).toString('base64');
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/**
 * Base64URL decode
 */
function base64UrlDecode(data: string): string {
  const padded = data + '='.repeat((4 - data.length % 4) % 4);
  const base64 = padded.replace(/-/g, '+').replace(/_/g, '/');
  return Buffer.from(base64, 'base64').toString('utf8');
}

/**
 * Sign a JWT token
 */
export function signToken(payload: Omit<TokenPayload, 'iat' | 'exp' | 'jti'>, secret: string, expiresInSeconds: number = 300): string {
  const now = Math.floor(Date.now() / 1000);

  const fullPayload: TokenPayload = {
    ...payload,
    iat: now,
    exp: now + expiresInSeconds,
    jti: crypto.randomUUID(),
  };

  const header = base64UrlEncode(JSON.stringify(JWT_HEADER));
  const body = base64UrlEncode(JSON.stringify(fullPayload));
  const signature = crypto
    .createHmac('sha256', secret)
    .update(`${header}.${body}`)
    .digest();

  return `${header}.${body}.${base64UrlEncode(signature)}`;
}

/**
 * Verify a JWT token
 */
export function verifyToken(token: string, secret: string): VerifyResult {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) {
      return { valid: false, error: 'Invalid token format' };
    }

    const [header, body, signature] = parts;

    // Verify signature
    const expectedSignature = crypto
      .createHmac('sha256', secret)
      .update(`${header}.${body}`)
      .digest();

    const actualSignature = Buffer.from(
      signature + '='.repeat((4 - signature.length % 4) % 4),
      'base64'
    );

    // Use timing-safe comparison
    if (!crypto.timingSafeEqual(expectedSignature, actualSignature)) {
      return { valid: false, error: 'Invalid signature' };
    }

    // Decode and parse payload
    const payload: TokenPayload = JSON.parse(base64UrlDecode(body));

    // Check expiration
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp < now) {
      return { valid: false, error: 'Token expired' };
    }

    // Check issued time (not in future, not too old)
    if (payload.iat > now + 60) {
      return { valid: false, error: 'Token issued in future' };
    }

    // Check audience
    if (payload.aud !== 'browser-pool') {
      return { valid: false, error: 'Invalid audience' };
    }

    return { valid: true, payload };

  } catch (error) {
    return { valid: false, error: `Token verification failed: ${error}` };
  }
}

/**
 * Audit Logger
 * In production, this would write to a database or log aggregator
 */
class AuditLogger {
  private entries: AuditEntry[] = [];
  private maxEntries = 10000;  // Keep last 10k entries in memory

  log(entry: AuditEntry): void {
    // Add to in-memory buffer
    this.entries.push(entry);

    // Trim if too large
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(-this.maxEntries / 2);
    }

    // Log to stdout for aggregation (structured JSON)
    console.log(JSON.stringify({
      type: 'AUDIT',
      ...entry
    }));
  }

  getRecent(limit: number = 100): AuditEntry[] {
    return this.entries.slice(-limit);
  }

  getByUser(userId: string, limit: number = 100): AuditEntry[] {
    return this.entries
      .filter(e => e.user_id === userId)
      .slice(-limit);
  }

  getByOrg(orgId: string, limit: number = 100): AuditEntry[] {
    return this.entries
      .filter(e => e.org_id === orgId)
      .slice(-limit);
  }
}

// Singleton audit logger
export const auditLogger = new AuditLogger();

/**
 * Authentication Middleware Factory
 * Returns an Express middleware that verifies JWT tokens
 */
export function createAuthMiddleware(secret: string, options: {
  skipPaths?: string[];
  requireOrg?: boolean;
} = {}) {
  const { skipPaths = ['/health', '/ready', '/metrics'], requireOrg = false } = options;

  return (req: any, res: any, next: any) => {
    // Skip auth for certain paths
    if (skipPaths.some(path => req.path === path || req.path.startsWith(path))) {
      return next();
    }

    // Extract token from Authorization header
    const authHeader = req.headers['authorization'];
    if (!authHeader) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: 'Missing Authorization header'
      });
    }

    // Support both "Bearer <token>" and raw token
    const token = authHeader.startsWith('Bearer ')
      ? authHeader.slice(7)
      : authHeader;

    // Legacy API key support (for backward compatibility during migration)
    if (token === process.env.BROWSER_POOL_API_KEY) {
      console.warn('DEPRECATION: Using legacy API key. Migrate to JWT tokens.');
      req.user = {
        user_id: 'legacy-api-key',
        org_id: 'default',
        legacy: true
      };
      return next();
    }

    // Verify JWT
    const result = verifyToken(token, secret);

    if (!result.valid) {
      return res.status(401).json({
        error: 'Unauthorized',
        message: result.error
      });
    }

    // Check org requirement
    if (requireOrg && !result.payload?.org_id) {
      return res.status(403).json({
        error: 'Forbidden',
        message: 'Organization ID required'
      });
    }

    // Attach user context to request
    req.user = {
      user_id: result.payload!.sub,
      org_id: result.payload!.org_id,
      email: result.payload!.email,
      token_id: result.payload!.jti,
      issuer: result.payload!.iss,
      ip: result.payload!.ip || req.ip
    };

    next();
  };
}

/**
 * Audit Middleware Factory
 * Logs all requests with user context
 */
export function createAuditMiddleware() {
  return (req: any, res: any, next: any) => {
    const startTime = Date.now();

    // Capture original end function
    const originalEnd = res.end;

    res.end = function(...args: any[]) {
      const duration = Date.now() - startTime;

      // Log audit entry
      if (req.user && req.user.user_id !== 'legacy-api-key') {
        auditLogger.log({
          timestamp: new Date().toISOString(),
          token_id: req.user.token_id || 'unknown',
          user_id: req.user.user_id,
          org_id: req.user.org_id,
          email: req.user.email,
          action: req.body?.instruction || req.method,
          endpoint: req.path,
          ip: req.user.ip || req.ip,
          success: res.statusCode < 400,
          error: res.statusCode >= 400 ? `HTTP ${res.statusCode}` : undefined,
          duration_ms: duration
        });
      }

      // Call original end
      return originalEnd.apply(this, args);
    };

    next();
  };
}

/**
 * Rate Limiter Factory (per user)
 * Limits requests per user per time window
 */
export function createRateLimiter(options: {
  windowMs?: number;
  maxRequests?: number;
} = {}) {
  const { windowMs = 60000, maxRequests = 100 } = options;

  // Map of user_id -> { count, resetTime }
  const userLimits = new Map<string, { count: number; resetTime: number }>();

  // Cleanup old entries periodically
  setInterval(() => {
    const now = Date.now();
    for (const [key, value] of userLimits.entries()) {
      if (value.resetTime < now) {
        userLimits.delete(key);
      }
    }
  }, windowMs);

  return (req: any, res: any, next: any) => {
    // Skip rate limiting if no user context
    if (!req.user || !req.user.user_id) {
      return next();
    }

    const userId = req.user.user_id;
    const now = Date.now();

    let limit = userLimits.get(userId);

    if (!limit || limit.resetTime < now) {
      // Start new window
      limit = { count: 0, resetTime: now + windowMs };
      userLimits.set(userId, limit);
    }

    limit.count++;

    // Set rate limit headers
    res.setHeader('X-RateLimit-Limit', maxRequests);
    res.setHeader('X-RateLimit-Remaining', Math.max(0, maxRequests - limit.count));
    res.setHeader('X-RateLimit-Reset', Math.ceil(limit.resetTime / 1000));

    if (limit.count > maxRequests) {
      return res.status(429).json({
        error: 'Too Many Requests',
        message: `Rate limit exceeded. Try again in ${Math.ceil((limit.resetTime - now) / 1000)} seconds.`,
        retryAfter: Math.ceil((limit.resetTime - now) / 1000)
      });
    }

    next();
  };
}
