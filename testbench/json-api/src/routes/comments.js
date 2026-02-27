const { Router } = require('express');
const { v4: uuidv4 } = require('uuid');
const { comments } = require('../store');

const router = Router();

// GET /api/v1/comments — list with filter by postId
router.get('/', (req, res) => {
  let filtered = comments;

  if (req.query.postId) {
    filtered = filtered.filter((c) => c.postId === req.query.postId);
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

// POST /api/v1/comments — create
router.post('/', (req, res) => {
  const { body, postId, userId } = req.body;

  if (!body || typeof body !== 'string') {
    return res.status(422).json({ error: 'Field "body" is required and must be a string' });
  }
  if (!postId || typeof postId !== 'string') {
    return res.status(422).json({ error: 'Field "postId" is required and must be a string' });
  }
  if (!userId || typeof userId !== 'string') {
    return res.status(422).json({ error: 'Field "userId" is required and must be a string' });
  }

  const comment = {
    id: uuidv4(),
    body,
    postId,
    userId,
    createdAt: new Date().toISOString(),
  };
  comments.push(comment);
  res.status(201).json(comment);
});

// GET /api/v1/comments/:id — get by ID
router.get('/:id', (req, res) => {
  const comment = comments.find((c) => c.id === req.params.id);
  if (!comment) {
    return res.status(404).json({ error: 'Comment not found' });
  }
  res.json(comment);
});

// PUT /api/v1/comments/:id — update
router.put('/:id', (req, res) => {
  const index = comments.findIndex((c) => c.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Comment not found' });
  }

  const { body, postId, userId } = req.body;
  if (body !== undefined) comments[index].body = body;
  if (postId !== undefined) comments[index].postId = postId;
  if (userId !== undefined) comments[index].userId = userId;

  res.json(comments[index]);
});

// DELETE /api/v1/comments/:id — delete
router.delete('/:id', (req, res) => {
  const index = comments.findIndex((c) => c.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Comment not found' });
  }
  comments.splice(index, 1);
  res.status(204).send();
});

module.exports = router;
