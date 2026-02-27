const express = require('express');
const bcrypt = require('bcryptjs');
const db = require('../db');
const { authenticate } = require('../middleware/auth');
const { isBugEnabled } = require('../bugs/registry');

const router = express.Router();

/**
 * GET /api/users/:id — get user profile
 * When sec-idor bug is disabled, enforces ownership check.
 * When sec-idor bug is enabled, returns any user's data.
 */
router.get('/:id', authenticate, (req, res) => {
  // sec-idor: Skip ownership check
  if (!isBugEnabled('sec-idor')) {
    // Normal behavior: only allow users to view their own profile (or admin)
    if (req.user.userId !== req.params.id && req.user.role !== 'admin') {
      return res.status(403).json({ error: 'Access denied' });
    }
  }

  const user = db.users.find((u) => u.id === req.params.id);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  // Never return password hash
  res.json({
    id: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
  });
});

/**
 * PUT /api/users/:id — update user profile
 */
router.put('/:id', authenticate, async (req, res) => {
  // Ownership check (always enforced for updates)
  if (req.user.userId !== req.params.id && req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Access denied' });
  }

  const user = db.users.find((u) => u.id === req.params.id);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  const { name, email, password } = req.body;

  if (name) user.name = name;
  if (email) {
    // Check for duplicate email
    const existing = db.users.find(
      (u) => u.email === email && u.id !== user.id
    );
    if (existing) {
      return res.status(409).json({ error: 'Email already in use' });
    }
    user.email = email;
  }
  if (password) {
    if (password.length < 6) {
      return res
        .status(400)
        .json({ error: 'Password must be at least 6 characters' });
    }
    user.password = await bcrypt.hash(password, 10);
  }

  res.json({
    id: user.id,
    name: user.name,
    email: user.email,
    role: user.role,
  });
});

/**
 * GET /api/users/:id/orders — get user's orders
 */
router.get('/:id/orders', authenticate, (req, res) => {
  // Ownership check
  if (
    !isBugEnabled('sec-idor') &&
    req.user.userId !== req.params.id &&
    req.user.role !== 'admin'
  ) {
    return res.status(403).json({ error: 'Access denied' });
  }

  const userOrders = db.orders.filter((o) => o.userId === req.params.id);
  res.json({ orders: userOrders, total: userOrders.length });
});

module.exports = router;
