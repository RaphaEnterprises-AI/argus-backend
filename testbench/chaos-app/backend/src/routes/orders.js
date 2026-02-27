const express = require('express');
const db = require('../db');
const { authenticate } = require('../middleware/auth');

const router = express.Router();

/**
 * GET /api/orders — list current user's orders
 */
router.get('/', authenticate, (req, res) => {
  const userId = req.user.userId;
  const userOrders = db.orders.filter((o) => o.userId === userId);

  res.json({
    orders: userOrders,
    total: userOrders.length,
  });
});

/**
 * GET /api/orders/:id — get order detail
 */
router.get('/:id', authenticate, (req, res) => {
  const order = db.orders.find((o) => o.id === req.params.id);
  if (!order) {
    return res.status(404).json({ error: 'Order not found' });
  }

  // Ownership check
  if (order.userId !== req.user.userId && req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Access denied' });
  }

  res.json(order);
});

/**
 * POST /api/orders — create order directly (alternative to cart checkout)
 */
router.post('/', authenticate, (req, res) => {
  const userId = req.user.userId;
  const { items } = req.body;

  if (!items || !Array.isArray(items) || items.length === 0) {
    return res.status(400).json({ error: 'items array is required' });
  }

  // Validate and calculate total
  let total = 0;
  const orderItems = [];

  for (const item of items) {
    if (!item.productId || !item.quantity) {
      return res
        .status(400)
        .json({ error: 'Each item needs productId and quantity' });
    }

    const product = db.products.find((p) => p.id === item.productId);
    if (!product) {
      return res
        .status(404)
        .json({ error: `Product ${item.productId} not found` });
    }

    const qty = parseInt(item.quantity, 10);
    if (qty < 1) {
      return res.status(400).json({ error: 'Quantity must be at least 1' });
    }

    if (product.stock < qty) {
      return res
        .status(400)
        .json({ error: `Insufficient stock for ${product.name}` });
    }

    orderItems.push({
      productId: product.id,
      name: product.name,
      price: product.price,
      quantity: qty,
      image: product.image,
    });

    total += product.price * qty;
  }

  // Deduct stock
  for (const item of orderItems) {
    const product = db.products.find((p) => p.id === item.productId);
    product.stock -= item.quantity;
  }

  const order = {
    id: db.getNextOrderId(),
    userId,
    items: orderItems,
    total,
    status: 'confirmed',
    createdAt: new Date().toISOString(),
  };

  db.orders.push(order);

  res.status(201).json(order);
});

module.exports = router;
