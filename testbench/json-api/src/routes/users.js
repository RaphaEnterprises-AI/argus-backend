const { Router } = require('express');
const { v4: uuidv4 } = require('uuid');
const { users } = require('../store');

const router = Router();

// GET /api/v1/users — list with pagination
router.get('/', (req, res) => {
  const page = Math.max(1, parseInt(req.query.page, 10) || 1);
  const limit = Math.max(1, Math.min(100, parseInt(req.query.limit, 10) || 10));
  const start = (page - 1) * limit;
  const end = start + limit;
  const items = users.slice(start, end);

  res.json({
    data: items,
    pagination: {
      page,
      limit,
      total: users.length,
      totalPages: Math.ceil(users.length / limit),
    },
  });
});

// POST /api/v1/users — create
router.post('/', (req, res) => {
  const { name, email } = req.body;

  if (!name || typeof name !== 'string') {
    return res.status(422).json({ error: 'Field "name" is required and must be a string' });
  }
  if (!email || typeof email !== 'string') {
    return res.status(422).json({ error: 'Field "email" is required and must be a string' });
  }

  const user = {
    id: uuidv4(),
    name,
    email,
    createdAt: new Date().toISOString(),
  };
  users.push(user);
  res.status(201).json(user);
});

// GET /api/v1/users/:id — get by ID
router.get('/:id', (req, res) => {
  const user = users.find((u) => u.id === req.params.id);
  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }
  res.json(user);
});

// PUT /api/v1/users/:id — update
router.put('/:id', (req, res) => {
  const index = users.findIndex((u) => u.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'User not found' });
  }

  const { name, email } = req.body;
  if (name !== undefined) users[index].name = name;
  if (email !== undefined) users[index].email = email;

  res.json(users[index]);
});

// DELETE /api/v1/users/:id — delete
router.delete('/:id', (req, res) => {
  const index = users.findIndex((u) => u.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'User not found' });
  }
  users.splice(index, 1);
  res.status(204).send();
});

module.exports = router;
