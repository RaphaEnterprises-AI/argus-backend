const express = require('express');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');
const store = require('./store');

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'secret';

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.use(cors());
app.use(express.json());

// ---------------------------------------------------------------------------
// Auth helpers
// ---------------------------------------------------------------------------

function generateToken(user) {
  return jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, {
    expiresIn: '7d',
  });
}

function userResponse(user, token) {
  return {
    user: {
      email: user.email,
      token: token || generateToken(user),
      username: user.username,
      bio: user.bio || '',
      image: user.image || null,
    },
  };
}

function profileResponse(user, currentUserId) {
  return {
    profile: {
      username: user.username,
      bio: user.bio || '',
      image: user.image || null,
      following: currentUserId
        ? store.follows.has(`${currentUserId}:${user.id}`)
        : false,
    },
  };
}

function articleResponse(article, currentUserId) {
  const author = store.users.find((u) => u.id === article.authorId);
  return {
    slug: article.slug,
    title: article.title,
    description: article.description,
    body: article.body,
    tagList: article.tagList || [],
    createdAt: article.createdAt,
    updatedAt: article.updatedAt,
    favorited: currentUserId
      ? store.favorites.has(`${currentUserId}:${article.id}`)
      : false,
    favoritesCount: [...store.favorites].filter((f) =>
      f.endsWith(`:${article.id}`)
    ).length,
    author: {
      username: author ? author.username : 'unknown',
      bio: author ? author.bio || '' : '',
      image: author ? author.image || null : null,
      following: currentUserId && author
        ? store.follows.has(`${currentUserId}:${author.id}`)
        : false,
    },
  };
}

function commentResponse(comment, currentUserId) {
  const author = store.users.find((u) => u.id === comment.authorId);
  return {
    id: comment.id,
    body: comment.body,
    createdAt: comment.createdAt,
    updatedAt: comment.updatedAt,
    author: {
      username: author ? author.username : 'unknown',
      bio: author ? author.bio || '' : '',
      image: author ? author.image || null : null,
      following: currentUserId && author
        ? store.follows.has(`${currentUserId}:${author.id}`)
        : false,
    },
  };
}

// Auth middleware — sets req.user if valid token present, otherwise null
function optionalAuth(req, _res, next) {
  req.user = null;
  const header = req.headers.authorization;
  if (header && header.startsWith('Token ')) {
    const token = header.slice(6);
    try {
      const decoded = jwt.verify(token, JWT_SECRET);
      req.user = store.users.find((u) => u.id === decoded.id) || null;
    } catch {
      // invalid token — treat as unauthenticated
    }
  }
  next();
}

function requiredAuth(req, res, next) {
  optionalAuth(req, res, () => {
    if (!req.user) {
      return res.status(401).json({ errors: { body: ['Unauthorized'] } });
    }
    next();
  });
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'conduit',
    uptime: process.uptime(),
    users: store.users.length,
    articles: store.articles.length,
  });
});

// ===========================================================================
// Authentication
// ===========================================================================

// POST /api/users/login — authenticate
app.post('/api/users/login', (req, res) => {
  const { user: body } = req.body || {};
  if (!body || !body.email || !body.password) {
    return res
      .status(422)
      .json({ errors: { body: ['Email and password are required'] } });
  }

  const user = store.users.find((u) => u.email === body.email);
  if (!user || !bcrypt.compareSync(body.password, user.password)) {
    return res
      .status(422)
      .json({ errors: { body: ['Invalid email or password'] } });
  }

  res.json(userResponse(user));
});

// POST /api/users — register
app.post('/api/users', (req, res) => {
  const { user: body } = req.body || {};
  if (!body || !body.email || !body.username || !body.password) {
    return res
      .status(422)
      .json({ errors: { body: ['Email, username, and password are required'] } });
  }

  if (store.users.find((u) => u.email === body.email)) {
    return res
      .status(422)
      .json({ errors: { email: ['has already been taken'] } });
  }
  if (store.users.find((u) => u.username === body.username)) {
    return res
      .status(422)
      .json({ errors: { username: ['has already been taken'] } });
  }

  const now = new Date().toISOString();
  const user = {
    id: uuidv4(),
    email: body.email,
    username: body.username,
    password: bcrypt.hashSync(body.password, 10),
    bio: body.bio || '',
    image: body.image || null,
    createdAt: now,
    updatedAt: now,
  };
  store.users.push(user);

  res.status(201).json(userResponse(user));
});

// GET /api/user — current user
app.get('/api/user', requiredAuth, (req, res) => {
  res.json(userResponse(req.user));
});

// PUT /api/user — update user
app.put('/api/user', requiredAuth, (req, res) => {
  const { user: body } = req.body || {};
  if (!body) {
    return res.status(422).json({ errors: { body: ['No data provided'] } });
  }

  const user = req.user;
  if (body.email !== undefined) {
    const existing = store.users.find(
      (u) => u.email === body.email && u.id !== user.id
    );
    if (existing) {
      return res
        .status(422)
        .json({ errors: { email: ['has already been taken'] } });
    }
    user.email = body.email;
  }
  if (body.username !== undefined) {
    const existing = store.users.find(
      (u) => u.username === body.username && u.id !== user.id
    );
    if (existing) {
      return res
        .status(422)
        .json({ errors: { username: ['has already been taken'] } });
    }
    user.username = body.username;
  }
  if (body.password !== undefined) {
    user.password = bcrypt.hashSync(body.password, 10);
  }
  if (body.bio !== undefined) user.bio = body.bio;
  if (body.image !== undefined) user.image = body.image;
  user.updatedAt = new Date().toISOString();

  res.json(userResponse(user));
});

// ===========================================================================
// Profiles
// ===========================================================================

// GET /api/profiles/:username
app.get('/api/profiles/:username', optionalAuth, (req, res) => {
  const user = store.users.find((u) => u.username === req.params.username);
  if (!user) {
    return res.status(404).json({ errors: { body: ['User not found'] } });
  }
  res.json(profileResponse(user, req.user ? req.user.id : null));
});

// POST /api/profiles/:username/follow
app.post('/api/profiles/:username/follow', requiredAuth, (req, res) => {
  const target = store.users.find((u) => u.username === req.params.username);
  if (!target) {
    return res.status(404).json({ errors: { body: ['User not found'] } });
  }
  store.follows.add(`${req.user.id}:${target.id}`);
  res.json(profileResponse(target, req.user.id));
});

// DELETE /api/profiles/:username/follow
app.delete('/api/profiles/:username/follow', requiredAuth, (req, res) => {
  const target = store.users.find((u) => u.username === req.params.username);
  if (!target) {
    return res.status(404).json({ errors: { body: ['User not found'] } });
  }
  store.follows.delete(`${req.user.id}:${target.id}`);
  res.json(profileResponse(target, req.user.id));
});

// ===========================================================================
// Articles
// ===========================================================================

// GET /api/articles — list articles
app.get('/api/articles', optionalAuth, (req, res) => {
  let filtered = [...store.articles];

  // Filter by tag
  if (req.query.tag) {
    filtered = filtered.filter(
      (a) => a.tagList && a.tagList.includes(req.query.tag)
    );
  }

  // Filter by author
  if (req.query.author) {
    const author = store.users.find((u) => u.username === req.query.author);
    if (author) {
      filtered = filtered.filter((a) => a.authorId === author.id);
    } else {
      filtered = [];
    }
  }

  // Filter by favorited
  if (req.query.favorited) {
    const favUser = store.users.find(
      (u) => u.username === req.query.favorited
    );
    if (favUser) {
      filtered = filtered.filter((a) =>
        store.favorites.has(`${favUser.id}:${a.id}`)
      );
    } else {
      filtered = [];
    }
  }

  // Sort by most recent
  filtered.sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );

  const total = filtered.length;
  const limit = parseInt(req.query.limit, 10) || 20;
  const offset = parseInt(req.query.offset, 10) || 0;
  filtered = filtered.slice(offset, offset + limit);

  const currentUserId = req.user ? req.user.id : null;
  res.json({
    articles: filtered.map((a) => articleResponse(a, currentUserId)),
    articlesCount: total,
  });
});

// GET /api/articles/feed — followed users' articles
app.get('/api/articles/feed', requiredAuth, (req, res) => {
  const followedIds = [...store.follows]
    .filter((f) => f.startsWith(`${req.user.id}:`))
    .map((f) => f.split(':')[1]);

  let filtered = store.articles.filter((a) => followedIds.includes(a.authorId));

  filtered.sort(
    (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  );

  const total = filtered.length;
  const limit = parseInt(req.query.limit, 10) || 20;
  const offset = parseInt(req.query.offset, 10) || 0;
  filtered = filtered.slice(offset, offset + limit);

  res.json({
    articles: filtered.map((a) => articleResponse(a, req.user.id)),
    articlesCount: total,
  });
});

// POST /api/articles — create article
app.post('/api/articles', requiredAuth, (req, res) => {
  const { article: body } = req.body || {};
  if (!body || !body.title || !body.description || !body.body) {
    return res.status(422).json({
      errors: { body: ['Title, description, and body are required'] },
    });
  }

  const now = new Date().toISOString();
  const article = {
    id: uuidv4(),
    slug: store.makeSlug(body.title),
    title: body.title,
    description: body.description,
    body: body.body,
    tagList: body.tagList || [],
    authorId: req.user.id,
    createdAt: now,
    updatedAt: now,
  };
  store.articles.push(article);

  res.status(201).json({ article: articleResponse(article, req.user.id) });
});

// GET /api/articles/:slug — get single article
app.get('/api/articles/:slug', optionalAuth, (req, res) => {
  const article = store.articles.find((a) => a.slug === req.params.slug);
  if (!article) {
    return res.status(404).json({ errors: { body: ['Article not found'] } });
  }
  const currentUserId = req.user ? req.user.id : null;
  res.json({ article: articleResponse(article, currentUserId) });
});

// PUT /api/articles/:slug — update article
app.put('/api/articles/:slug', requiredAuth, (req, res) => {
  const article = store.articles.find((a) => a.slug === req.params.slug);
  if (!article) {
    return res.status(404).json({ errors: { body: ['Article not found'] } });
  }
  if (article.authorId !== req.user.id) {
    return res.status(403).json({ errors: { body: ['Forbidden'] } });
  }

  const { article: body } = req.body || {};
  if (!body) {
    return res.status(422).json({ errors: { body: ['No data provided'] } });
  }

  if (body.title !== undefined) {
    article.title = body.title;
    article.slug = store.makeSlug(body.title);
  }
  if (body.description !== undefined) article.description = body.description;
  if (body.body !== undefined) article.body = body.body;
  if (body.tagList !== undefined) article.tagList = body.tagList;
  article.updatedAt = new Date().toISOString();

  res.json({ article: articleResponse(article, req.user.id) });
});

// DELETE /api/articles/:slug — delete article
app.delete('/api/articles/:slug', requiredAuth, (req, res) => {
  const idx = store.articles.findIndex((a) => a.slug === req.params.slug);
  if (idx === -1) {
    return res.status(404).json({ errors: { body: ['Article not found'] } });
  }
  if (store.articles[idx].authorId !== req.user.id) {
    return res.status(403).json({ errors: { body: ['Forbidden'] } });
  }

  const articleId = store.articles[idx].id;

  // Remove associated comments
  for (let i = store.comments.length - 1; i >= 0; i--) {
    if (store.comments[i].articleId === articleId) {
      store.comments.splice(i, 1);
    }
  }

  // Remove favorites for this article
  for (const fav of store.favorites) {
    if (fav.endsWith(`:${articleId}`)) {
      store.favorites.delete(fav);
    }
  }

  store.articles.splice(idx, 1);
  res.status(204).send();
});

// ===========================================================================
// Comments
// ===========================================================================

// POST /api/articles/:slug/comments — add comment
app.post('/api/articles/:slug/comments', requiredAuth, (req, res) => {
  const article = store.articles.find((a) => a.slug === req.params.slug);
  if (!article) {
    return res.status(404).json({ errors: { body: ['Article not found'] } });
  }

  const { comment: body } = req.body || {};
  if (!body || !body.body) {
    return res
      .status(422)
      .json({ errors: { body: ['Comment body is required'] } });
  }

  const now = new Date().toISOString();
  const comment = {
    id: store.getNextCommentId(),
    body: body.body,
    articleId: article.id,
    authorId: req.user.id,
    createdAt: now,
    updatedAt: now,
  };
  store.comments.push(comment);

  res.status(201).json({ comment: commentResponse(comment, req.user.id) });
});

// GET /api/articles/:slug/comments — get comments
app.get('/api/articles/:slug/comments', optionalAuth, (req, res) => {
  const article = store.articles.find((a) => a.slug === req.params.slug);
  if (!article) {
    return res.status(404).json({ errors: { body: ['Article not found'] } });
  }

  const articleComments = store.comments
    .filter((c) => c.articleId === article.id)
    .sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );

  const currentUserId = req.user ? req.user.id : null;
  res.json({
    comments: articleComments.map((c) => commentResponse(c, currentUserId)),
  });
});

// DELETE /api/articles/:slug/comments/:id — delete comment
app.delete(
  '/api/articles/:slug/comments/:id',
  requiredAuth,
  (req, res) => {
    const article = store.articles.find((a) => a.slug === req.params.slug);
    if (!article) {
      return res.status(404).json({ errors: { body: ['Article not found'] } });
    }

    const commentId = parseInt(req.params.id, 10);
    const idx = store.comments.findIndex(
      (c) => c.id === commentId && c.articleId === article.id
    );
    if (idx === -1) {
      return res
        .status(404)
        .json({ errors: { body: ['Comment not found'] } });
    }
    if (store.comments[idx].authorId !== req.user.id) {
      return res.status(403).json({ errors: { body: ['Forbidden'] } });
    }

    store.comments.splice(idx, 1);
    res.status(204).send();
  }
);

// ===========================================================================
// Favorites
// ===========================================================================

// POST /api/articles/:slug/favorite — favorite article
app.post('/api/articles/:slug/favorite', requiredAuth, (req, res) => {
  const article = store.articles.find((a) => a.slug === req.params.slug);
  if (!article) {
    return res.status(404).json({ errors: { body: ['Article not found'] } });
  }
  store.favorites.add(`${req.user.id}:${article.id}`);
  res.json({ article: articleResponse(article, req.user.id) });
});

// DELETE /api/articles/:slug/favorite — unfavorite article
app.delete('/api/articles/:slug/favorite', requiredAuth, (req, res) => {
  const article = store.articles.find((a) => a.slug === req.params.slug);
  if (!article) {
    return res.status(404).json({ errors: { body: ['Article not found'] } });
  }
  store.favorites.delete(`${req.user.id}:${article.id}`);
  res.json({ article: articleResponse(article, req.user.id) });
});

// ===========================================================================
// Tags
// ===========================================================================

// GET /api/tags — get all tags
app.get('/api/tags', (_req, res) => {
  const tagSet = new Set();
  store.articles.forEach((a) => {
    if (a.tagList) a.tagList.forEach((t) => tagSet.add(t));
  });
  res.json({ tags: [...tagSet].sort() });
});

// ===========================================================================
// Error handling
// ===========================================================================

// 404 handler
app.use((req, res) => {
  res.status(404).json({ errors: { body: [`Route ${req.method} ${req.path} not found`] } });
});

// Global error handler
app.use((err, _req, res, _next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ errors: { body: ['Internal server error'] } });
});

// ===========================================================================
// Start
// ===========================================================================
app.listen(PORT, () => {
  console.log(`Conduit (RealWorld) testbench running on port ${PORT}`);
  console.log(`  Health:     http://localhost:${PORT}/health`);
  console.log(`  API:        http://localhost:${PORT}/api`);
  console.log(`  Users:      ${store.users.length} seeded`);
  console.log(`  Articles:   ${store.articles.length} seeded`);
  console.log(`  Comments:   ${store.comments.length} seeded`);
});

module.exports = app;
