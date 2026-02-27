const { Router } = require('express');
const { v4: uuidv4 } = require('uuid');
const { todos } = require('../store');

const router = Router();

// GET /api/v1/todos — list with filter by userId and completed
router.get('/', (req, res) => {
  let filtered = todos;

  if (req.query.userId) {
    filtered = filtered.filter((t) => t.userId === req.query.userId);
  }
  if (req.query.completed !== undefined) {
    const completed = req.query.completed === 'true';
    filtered = filtered.filter((t) => t.completed === completed);
  }

  const page = Math.max(1, parseInt(req.query.page, 10) || 1);
  const limit = Math.max(1, Math.min(100, parseInt(req.query.limit, 10) || 10));
  const start = (page - 1) * limit;
  const end = start + limit;
  const items = filtered.slice(start, end);

  res.json({
    data: items,
    pagination: {
      page,
      limit,
      total: filtered.length,
      totalPages: Math.ceil(filtered.length / limit),
    },
  });
});

// POST /api/v1/todos — create
router.post('/', (req, res) => {
  const { title, userId, completed } = req.body;

  if (!title || typeof title !== 'string') {
    return res.status(422).json({ error: 'Field "title" is required and must be a string' });
  }
  if (!userId || typeof userId !== 'string') {
    return res.status(422).json({ error: 'Field "userId" is required and must be a string' });
  }

  const todo = {
    id: uuidv4(),
    title,
    userId,
    completed: completed === true ? true : false,
    createdAt: new Date().toISOString(),
  };
  todos.push(todo);
  res.status(201).json(todo);
});

// GET /api/v1/todos/:id — get by ID
router.get('/:id', (req, res) => {
  const todo = todos.find((t) => t.id === req.params.id);
  if (!todo) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  res.json(todo);
});

// PATCH /api/v1/todos/:id — partial update
router.patch('/:id', (req, res) => {
  const index = todos.findIndex((t) => t.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }

  const { title, completed } = req.body;
  if (title !== undefined) todos[index].title = title;
  if (completed !== undefined) todos[index].completed = Boolean(completed);

  res.json(todos[index]);
});

// DELETE /api/v1/todos/:id — delete
router.delete('/:id', (req, res) => {
  const index = todos.findIndex((t) => t.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Todo not found' });
  }
  todos.splice(index, 1);
  res.status(204).send();
});

module.exports = router;
