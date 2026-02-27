const express = require('express');
const { v4: uuidv4 } = require('uuid');
const db = require('../db');
const { authenticate, requireAdmin } = require('../middleware/auth');
const { isBugEnabled } = require('../bugs/registry');

const router = express.Router();

const PAGE_SIZE = 10;

/**
 * GET /api/products — list all products with optional filters
 * Query params: ?category=, ?page= (or ?offset= when pagination bug is active)
 */
router.get('/', (req, res) => {
  let results = [...db.products];

  // Filter by category
  if (req.query.category) {
    results = results.filter((p) => p.category === req.query.category);
  }

  // Filter by search query
  if (req.query.q) {
    const query = req.query.q.toLowerCase();
    results = results.filter(
      (p) =>
        p.name.toLowerCase().includes(query) ||
        p.description.toLowerCase().includes(query)
    );
  }

  // Pagination
  let page = 1;
  let offset = 0;

  if (req._paginationBroken) {
    // api-breaking-pagination is active and client used ?page= — return all results
    // (client must use ?offset= instead)
  } else if (req.query.offset) {
    offset = parseInt(req.query.offset, 10) || 0;
  } else if (req.query.page) {
    page = parseInt(req.query.page, 10) || 1;
    offset = (page - 1) * PAGE_SIZE;
  }

  const total = results.length;
  const paginated = results.slice(offset, offset + PAGE_SIZE);

  res.json({
    products: paginated,
    total,
    page,
    pageSize: PAGE_SIZE,
    totalPages: Math.ceil(total / PAGE_SIZE),
  });
});

/**
 * GET /api/products/search — search products by query
 * Used specifically for XSS testing with sec-xss-reflected bug.
 */
router.get('/search', (req, res) => {
  const query = req.query.q || '';
  let results = db.products;

  if (query) {
    const lowerQuery = query.toLowerCase();
    results = results.filter(
      (p) =>
        p.name.toLowerCase().includes(lowerQuery) ||
        p.description.toLowerCase().includes(lowerQuery) ||
        p.category.toLowerCase().includes(lowerQuery)
    );
  }

  const response = {
    query,
    results,
    count: results.length,
  };

  // sec-xss-reflected: Raw query is already included via security.transformJson
  // in the chaos middleware, but we also include it here for direct JSON responses

  res.json(response);
});

/**
 * GET /api/products/:id — get product by ID
 */
router.get('/:id', (req, res) => {
  const product = db.products.find((p) => p.id === req.params.id);
  if (!product) {
    return res.status(404).json({ error: 'Product not found' });
  }
  res.json(product);
});

/**
 * POST /api/products — create product (admin only)
 */
router.post('/', authenticate, requireAdmin, (req, res) => {
  const { name, price, image, category, stock, description } = req.body;

  if (!name || price == null || !category) {
    return res
      .status(400)
      .json({ error: 'name, price, and category are required' });
  }

  const product = {
    id: uuidv4(),
    name,
    price: Number(price),
    image: image || '/images/placeholder.jpg',
    category,
    stock: stock != null ? Number(stock) : 0,
    description: description || '',
  };

  db.products.push(product);
  res.status(201).json(product);
});

/**
 * PUT /api/products/:id — update product (admin only)
 */
router.put('/:id', authenticate, requireAdmin, (req, res) => {
  const index = db.products.findIndex((p) => p.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Product not found' });
  }

  const updates = req.body;
  const product = db.products[index];

  if (updates.name) product.name = updates.name;
  if (updates.price != null) product.price = Number(updates.price);
  if (updates.image) product.image = updates.image;
  if (updates.category) product.category = updates.category;
  if (updates.stock != null) product.stock = Number(updates.stock);
  if (updates.description) product.description = updates.description;

  res.json(product);
});

/**
 * DELETE /api/products/:id — delete product (admin only)
 */
router.delete('/:id', authenticate, requireAdmin, (req, res) => {
  const index = db.products.findIndex((p) => p.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Product not found' });
  }

  db.products.splice(index, 1);
  res.status(204).send();
});

module.exports = router;
