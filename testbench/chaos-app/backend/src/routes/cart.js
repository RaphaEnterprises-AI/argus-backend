const express = require('express');
const db = require('../db');
const { authenticate } = require('../middleware/auth');
const { getStaleCart } = require('../bugs/flaky');

const router = express.Router();

/**
 * GET /api/cart — get current user's cart
 */
router.get('/', authenticate, (req, res) => {
  const userId = req.user.userId;
  const cart = db.carts[userId] || [];

  // flaky-race-condition: might return stale data
  const stale = getStaleCart(userId, cart);
  if (stale) {
    return res.json({ items: stale, total: calculateTotal(stale) });
  }

  res.json({ items: cart, total: calculateTotal(cart) });
});

/**
 * POST /api/cart/add — add item to cart
 */
router.post('/add', authenticate, (req, res) => {
  const userId = req.user.userId;
  const { productId, quantity } = req.body;

  if (!productId) {
    return res.status(400).json({ error: 'productId is required' });
  }

  const product = db.products.find((p) => p.id === productId);
  if (!product) {
    return res.status(404).json({ error: 'Product not found' });
  }

  const qty = Math.max(1, parseInt(quantity, 10) || 1);

  if (!db.carts[userId]) {
    db.carts[userId] = [];
  }

  // Check if product already in cart
  const existingItem = db.carts[userId].find(
    (item) => item.productId === productId
  );
  if (existingItem) {
    existingItem.quantity += qty;
  } else {
    db.carts[userId].push({
      productId,
      quantity: qty,
      name: product.name,
      price: product.price,
      image: product.image,
    });
  }

  const cart = db.carts[userId];
  res.status(201).json({ items: cart, total: calculateTotal(cart) });
});

/**
 * PUT /api/cart/:itemId — update item quantity
 * itemId here is the productId within the cart
 */
router.put('/:itemId', authenticate, (req, res) => {
  const userId = req.user.userId;
  const { quantity } = req.body;

  if (!db.carts[userId]) {
    return res.status(404).json({ error: 'Cart is empty' });
  }

  const item = db.carts[userId].find(
    (item) => item.productId === req.params.itemId
  );
  if (!item) {
    return res.status(404).json({ error: 'Item not in cart' });
  }

  const qty = parseInt(quantity, 10);
  if (!qty || qty < 1) {
    return res.status(400).json({ error: 'Quantity must be at least 1' });
  }

  item.quantity = qty;

  const cart = db.carts[userId];
  res.json({ items: cart, total: calculateTotal(cart) });
});

/**
 * DELETE /api/cart/:itemId — remove item from cart
 */
router.delete('/:itemId', authenticate, (req, res) => {
  const userId = req.user.userId;

  if (!db.carts[userId]) {
    return res.status(404).json({ error: 'Cart is empty' });
  }

  const index = db.carts[userId].findIndex(
    (item) => item.productId === req.params.itemId
  );
  if (index === -1) {
    return res.status(404).json({ error: 'Item not in cart' });
  }

  db.carts[userId].splice(index, 1);

  const cart = db.carts[userId];
  res.json({ items: cart, total: calculateTotal(cart) });
});

/**
 * POST /api/cart/checkout — place order from cart
 */
router.post('/checkout', authenticate, (req, res) => {
  const userId = req.user.userId;
  const cart = db.carts[userId];

  if (!cart || cart.length === 0) {
    return res.status(400).json({ error: 'Cart is empty' });
  }

  // Check stock for all items
  for (const item of cart) {
    const product = db.products.find((p) => p.id === item.productId);
    if (!product) {
      return res
        .status(400)
        .json({ error: `Product ${item.productId} no longer exists` });
    }
    if (product.stock < item.quantity) {
      return res
        .status(400)
        .json({ error: `Insufficient stock for ${product.name}` });
    }
  }

  // Deduct stock
  for (const item of cart) {
    const product = db.products.find((p) => p.id === item.productId);
    product.stock -= item.quantity;
  }

  const total = calculateTotal(cart);
  const order = {
    id: db.getNextOrderId(),
    userId,
    items: [...cart],
    total,
    status: 'confirmed',
    createdAt: new Date().toISOString(),
  };

  db.orders.push(order);

  // Clear cart
  db.carts[userId] = [];

  res.status(201).json(order);
});

/**
 * Calculate total price for cart items.
 */
function calculateTotal(items) {
  return items.reduce((sum, item) => sum + item.price * item.quantity, 0);
}

module.exports = router;
