const express = require('express');
const bcrypt = require('bcryptjs');
const db = require('../db');
const { authenticate, generateToken } = require('../middleware/auth');
const { shouldBypassLogin, getSqlLog } = require('../bugs/security');

const router = express.Router();

/**
 * POST /api/auth/register — create new user and return JWT
 */
router.post('/register', async (req, res) => {
  const { name, email, password } = req.body;

  if (!name || !email || !password) {
    return res
      .status(400)
      .json({ error: 'name, email, and password are required' });
  }

  // Check if email already exists
  const existing = db.users.find((u) => u.email === email);
  if (existing) {
    return res.status(409).json({ error: 'Email already registered' });
  }

  // Validate password length
  if (password.length < 6) {
    return res
      .status(400)
      .json({ error: 'Password must be at least 6 characters' });
  }

  const hashedPassword = await bcrypt.hash(password, 10);

  const user = {
    id: db.getNextUserId(),
    name,
    email,
    password: hashedPassword,
    role: 'user',
  };

  db.users.push(user);

  const token = generateToken(user);

  res.status(201).json({
    user: { id: user.id, name: user.name, email: user.email, role: user.role },
    token,
  });
});

/**
 * POST /api/auth/login — validate credentials and return JWT
 */
router.post('/login', async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: 'email and password are required' });
  }

  // sec-sqli: Log SQL-like message (for detection testing)
  const sqlLog = getSqlLog(email);
  if (sqlLog) {
    console.log('[CHAOS SQL]', sqlLog);
  }

  // sec-sqli: Bypass authentication for SQL injection patterns
  if (shouldBypassLogin(email)) {
    // Return the first user (simulating SQL injection success)
    const user = db.users[0];
    const token = generateToken(user);
    return res.json({
      user: {
        id: user.id,
        name: user.name,
        email: user.email,
        role: user.role,
      },
      token,
      _debug: sqlLog, // Deliberately expose SQL log
    });
  }

  // Normal authentication flow
  const user = db.users.find((u) => u.email === email);
  if (!user) {
    return res.status(401).json({ error: 'Invalid email or password' });
  }

  const validPassword = await bcrypt.compare(password, user.password);
  if (!validPassword) {
    return res.status(401).json({ error: 'Invalid email or password' });
  }

  const token = generateToken(user);

  res.json({
    user: { id: user.id, name: user.name, email: user.email, role: user.role },
    token,
  });
});

/**
 * POST /api/auth/refresh — refresh JWT token
 */
router.post('/refresh', authenticate, (req, res) => {
  const user = db.users.find((u) => u.id === req.user.userId);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  const token = generateToken(user);

  res.json({
    user: { id: user.id, name: user.name, email: user.email, role: user.role },
    token,
  });
});

/**
 * GET /api/auth/me — get current user from JWT
 */
router.get('/me', authenticate, (req, res) => {
  const user = db.users.find((u) => u.id === req.user.userId);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  res.json({
    id: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
  });
});

module.exports = router;
