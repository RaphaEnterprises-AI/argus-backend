const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
const path = require('path');

const chaosMiddleware = require('./middleware/chaos');
const { getActiveBugCount } = require('./bugs/registry');
const { handleSsrf } = require('./bugs/security');

const productsRouter = require('./routes/products');
const authRouter = require('./routes/auth');
const usersRouter = require('./routes/users');
const cartRouter = require('./routes/cart');
const ordersRouter = require('./routes/orders');
const chaosRouter = require('./routes/chaos');

const app = express();
const PORT = process.env.PORT || 3001;

// ------------------------------------
// Core middleware
// ------------------------------------
app.use(cors({ origin: true, credentials: true }));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());

// ------------------------------------
// Health check (always public, before auth)
// ------------------------------------
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'chaos-app',
    activeBugs: getActiveBugCount(),
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

// ------------------------------------
// Bearer token auth — everything below requires TESTBENCH_SECRET
// ------------------------------------
const TESTBENCH_SECRET = process.env.TESTBENCH_SECRET;
app.use((req, res, next) => {
  if (!TESTBENCH_SECRET) return next();
  const auth = req.headers.authorization || '';
  if (!auth.startsWith('Bearer ') || auth.slice(7) !== TESTBENCH_SECRET) {
    return res.status(401).json({ error: 'Invalid or missing bearer token' });
  }
  next();
});

// Serve static files (after auth)
const publicDir = path.join(__dirname, '..', 'public');
app.use(express.static(publicDir));

// ------------------------------------
// Chaos middleware — MUST be before routes
// ------------------------------------
app.use(chaosMiddleware);

// ------------------------------------
// API routes
// ------------------------------------
app.use('/api/products', productsRouter);
app.use('/api/auth', authRouter);
app.use('/api/users', usersRouter);
app.use('/api/cart', cartRouter);
app.use('/api/orders', ordersRouter);

// SSRF endpoint (only active when sec-ssrf bug is enabled)
app.post('/api/images/fetch', handleSsrf);

// Chaos toggle API
app.use('/chaos', chaosRouter);

// ------------------------------------
// SPA fallback — non-API routes serve index.html
// ------------------------------------
app.get('*', (req, res) => {
  // Only serve index.html for non-API routes
  if (req.path.startsWith('/api/') || req.path.startsWith('/chaos/')) {
    return res.status(404).json({ error: 'Not found' });
  }

  const indexPath = path.join(publicDir, 'index.html');
  res.sendFile(indexPath, (err) => {
    if (err) {
      // If no frontend build exists yet, return a placeholder
      res.status(200).send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chaos App - Argus Test Bench</title>
</head>
<body>
  <div id="root">
    <h1>Chaos App</h1>
    <p>Frontend not built yet. API is available at <a href="/health">/health</a></p>
    <p>Chaos controls at <a href="/chaos/bugs">/chaos/bugs</a></p>
  </div>
</body>
</html>`);
    }
  });
});

// ------------------------------------
// Error handling
// ------------------------------------
app.use((err, req, res, _next) => {
  console.error('[ERROR]', err.message);
  res.status(500).json({
    error: 'Internal Server Error',
    message:
      process.env.NODE_ENV === 'production'
        ? 'An unexpected error occurred'
        : err.message,
  });
});

// ------------------------------------
// Start server
// ------------------------------------
app.listen(PORT, () => {
  console.log(`Chaos App server running on port ${PORT}`);
  console.log(`  Health: http://localhost:${PORT}/health`);
  console.log(`  API:    http://localhost:${PORT}/api/products`);
  console.log(`  Chaos:  http://localhost:${PORT}/chaos/bugs`);
  console.log(`  Bugs:   30 toggleable bugs available`);
});

module.exports = app;
