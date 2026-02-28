const express = require('express');
const cors = require('cors');
const path = require('path');
const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');

const usersRouter = require('./routes/users');
const postsRouter = require('./routes/posts');
const commentsRouter = require('./routes/comments');
const todosRouter = require('./routes/todos');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Load OpenAPI spec
const openapiSpec = YAML.load(path.join(__dirname, 'openapi.yaml'));

// Health check (always public)
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'json-api',
    uptime: process.uptime(),
  });
});

// Bearer token auth — all routes below require TESTBENCH_SECRET
const TESTBENCH_SECRET = process.env.TESTBENCH_SECRET;
app.use((req, res, next) => {
  if (!TESTBENCH_SECRET) return next();
  const auth = req.headers.authorization || '';
  if (!auth.startsWith('Bearer ') || auth.slice(7) !== TESTBENCH_SECRET) {
    return res.status(401).json({ error: 'Invalid or missing bearer token' });
  }
  next();
});

// OpenAPI JSON endpoint
app.get('/api/v1/openapi.json', (req, res) => {
  res.json(openapiSpec);
});

// Swagger UI
app.use('/api/v1/docs', swaggerUi.serve, swaggerUi.setup(openapiSpec));

// API routes
app.use('/api/v1/users', usersRouter);
app.use('/api/v1/posts', postsRouter);
app.use('/api/v1/comments', commentsRouter);
app.use('/api/v1/todos', todosRouter);

// 404 handler for unknown routes
app.use((req, res) => {
  res.status(404).json({ error: `Route ${req.method} ${req.path} not found` });
});

// Error handler
app.use((err, req, res, _next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`Argus Testbench JSON API running on port ${PORT}`);
  console.log(`  Health:  http://localhost:${PORT}/health`);
  console.log(`  Docs:    http://localhost:${PORT}/api/v1/docs`);
  console.log(`  OpenAPI: http://localhost:${PORT}/api/v1/openapi.json`);
});

module.exports = app;
