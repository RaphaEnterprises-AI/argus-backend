const { Router } = require('express');
const { v4: uuidv4 } = require('uuid');
const { posts, comments } = require('../store');

const router = Router();

// GET /api/v1/posts — list with pagination + filter by userId
router.get('/', (req, res) => {
  let filtered = posts;

  if (req.query.userId) {
    filtered = filtered.filter((p) => p.userId === req.query.userId);
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

// POST /api/v1/posts — create
router.post('/', (req, res) => {
  const { title, body, userId } = req.body;

  if (!title || typeof title !== 'string') {
    return res.status(422).json({ error: 'Field "title" is required and must be a string' });
  }
  if (!body || typeof body !== 'string') {
    return res.status(422).json({ error: 'Field "body" is required and must be a string' });
  }
  if (!userId || typeof userId !== 'string') {
    return res.status(422).json({ error: 'Field "userId" is required and must be a string' });
  }

  const post = {
    id: uuidv4(),
    title,
    body,
    userId,
    createdAt: new Date().toISOString(),
  };
  posts.push(post);
  res.status(201).json(post);
});

// GET /api/v1/posts/:id — get by ID
router.get('/:id', (req, res) => {
  const post = posts.find((p) => p.id === req.params.id);
  if (!post) {
    return res.status(404).json({ error: 'Post not found' });
  }
  res.json(post);
});

// PUT /api/v1/posts/:id — update
router.put('/:id', (req, res) => {
  const index = posts.findIndex((p) => p.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Post not found' });
  }

  const { title, body, userId } = req.body;
  if (title !== undefined) posts[index].title = title;
  if (body !== undefined) posts[index].body = body;
  if (userId !== undefined) posts[index].userId = userId;

  res.json(posts[index]);
});

// DELETE /api/v1/posts/:id — delete
router.delete('/:id', (req, res) => {
  const index = posts.findIndex((p) => p.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({ error: 'Post not found' });
  }
  posts.splice(index, 1);
  res.status(204).send();
});

// GET /api/v1/posts/:id/comments — get comments for a post
router.get('/:id/comments', (req, res) => {
  const post = posts.find((p) => p.id === req.params.id);
  if (!post) {
    return res.status(404).json({ error: 'Post not found' });
  }

  const postComments = comments.filter((c) => c.postId === req.params.id);
  res.json({ data: postComments });
});

module.exports = router;
