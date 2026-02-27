const express = require('express');
const cors = require('cors');
const path = require('path');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { v4: uuidv4 } = require('uuid');

const store = require('./store');

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'plane-testbench-secret-key';

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------
app.use(cors());
app.use(express.json());

// Serve static SPA
app.use(express.static(path.join(__dirname, 'public')));

// Auth middleware - extracts user from JWT if present
function authenticate(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Authentication required' });
  }
  try {
    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, JWT_SECRET);
    const user = store.users.find((u) => u.id === decoded.user_id);
    if (!user) return res.status(401).json({ error: 'User not found' });
    req.user = user;
    next();
  } catch (err) {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

// Helper: find workspace by slug & verify membership
function resolveWorkspace(req, res) {
  const ws = store.workspaces.find((w) => w.slug === req.params.slug);
  if (!ws) {
    res.status(404).json({ error: 'Workspace not found' });
    return null;
  }
  const member = store.workspaceMembers.find(
    (m) => m.workspace_id === ws.id && m.user_id === req.user.id
  );
  if (!member) {
    res.status(403).json({ error: 'Not a member of this workspace' });
    return null;
  }
  return ws;
}

// Helper: find project within a workspace
function resolveProject(ws, projectId, res) {
  const project = store.projects.find(
    (p) => p.id === projectId && p.workspace_id === ws.id
  );
  if (!project) {
    res.status(404).json({ error: 'Project not found' });
    return null;
  }
  return project;
}

// Strip password from user objects
function sanitizeUser(u) {
  const { password, ...rest } = u;
  return rest;
}

// ===================================================================
// HEALTH
// ===================================================================
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'plane-testbench',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
  });
});

// ===================================================================
// AUTH
// ===================================================================
app.post('/api/auth/register', (req, res) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name) {
    return res.status(400).json({ error: 'email, password, and name are required' });
  }
  if (store.users.find((u) => u.email === email)) {
    return res.status(409).json({ error: 'Email already registered' });
  }
  const user = {
    id: uuidv4(),
    email,
    password: bcrypt.hashSync(password, 10),
    name,
    role: 'member',
    avatar: null,
    created_at: new Date().toISOString(),
  };
  store.users.push(user);
  const token = jwt.sign({ user_id: user.id }, JWT_SECRET, { expiresIn: '24h' });
  res.status(201).json({ user: sanitizeUser(user), token });
});

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ error: 'email and password are required' });
  }
  const user = store.users.find((u) => u.email === email);
  if (!user || !bcrypt.compareSync(password, user.password)) {
    return res.status(401).json({ error: 'Invalid email or password' });
  }
  const token = jwt.sign({ user_id: user.id }, JWT_SECRET, { expiresIn: '24h' });
  res.json({ user: sanitizeUser(user), token });
});

app.get('/api/auth/me', authenticate, (req, res) => {
  res.json({ user: sanitizeUser(req.user) });
});

// ===================================================================
// WORKSPACES
// ===================================================================
app.post('/api/workspaces', authenticate, (req, res) => {
  const { name, slug } = req.body;
  if (!name || !slug) {
    return res.status(400).json({ error: 'name and slug are required' });
  }
  if (!/^[a-z0-9-]+$/.test(slug)) {
    return res.status(400).json({ error: 'slug must be lowercase alphanumeric with hyphens' });
  }
  if (store.workspaces.find((w) => w.slug === slug)) {
    return res.status(409).json({ error: 'Workspace slug already taken' });
  }
  const ws = {
    id: uuidv4(),
    name,
    slug,
    owner_id: req.user.id,
    created_at: new Date().toISOString(),
  };
  store.workspaces.push(ws);
  store.workspaceMembers.push({
    workspace_id: ws.id,
    user_id: req.user.id,
    role: 'admin',
    joined_at: ws.created_at,
  });
  res.status(201).json(ws);
});

app.get('/api/workspaces', authenticate, (req, res) => {
  const memberOf = store.workspaceMembers
    .filter((m) => m.user_id === req.user.id)
    .map((m) => m.workspace_id);
  const result = store.workspaces.filter((w) => memberOf.includes(w.id));
  res.json({ results: result });
});

app.get('/api/workspaces/:slug', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  res.json(ws);
});

app.put('/api/workspaces/:slug', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const { name } = req.body;
  if (name) ws.name = name;
  res.json(ws);
});

app.get('/api/workspaces/:slug/members', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const members = store.workspaceMembers
    .filter((m) => m.workspace_id === ws.id)
    .map((m) => {
      const user = store.users.find((u) => u.id === m.user_id);
      return { ...m, user: user ? sanitizeUser(user) : null };
    });
  res.json({ results: members });
});

app.post('/api/workspaces/:slug/members', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const { email, role } = req.body;
  if (!email) return res.status(400).json({ error: 'email is required' });
  const user = store.users.find((u) => u.email === email);
  if (!user) return res.status(404).json({ error: 'User not found' });
  const existing = store.workspaceMembers.find(
    (m) => m.workspace_id === ws.id && m.user_id === user.id
  );
  if (existing) return res.status(409).json({ error: 'Already a member' });
  const member = {
    workspace_id: ws.id,
    user_id: user.id,
    role: role || 'member',
    joined_at: new Date().toISOString(),
  };
  store.workspaceMembers.push(member);
  res.status(201).json({ ...member, user: sanitizeUser(user) });
});

// ===================================================================
// PROJECTS
// ===================================================================
app.post('/api/workspaces/:slug/projects', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const { name, identifier, description, network } = req.body;
  if (!name || !identifier) {
    return res.status(400).json({ error: 'name and identifier are required' });
  }
  const project = {
    id: uuidv4(),
    workspace_id: ws.id,
    name,
    identifier,
    description: description || '',
    network: network || 'public',
    created_by: req.user.id,
    created_at: new Date().toISOString(),
  };
  store.projects.push(project);
  store.sequenceCounters[project.id] = 0;
  res.status(201).json(project);
});

app.get('/api/workspaces/:slug/projects', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const result = store.projects.filter((p) => p.workspace_id === ws.id);
  res.json({ results: result });
});

app.get('/api/workspaces/:slug/projects/:id', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  res.json(project);
});

app.put('/api/workspaces/:slug/projects/:id', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const { name, description, network } = req.body;
  if (name) project.name = name;
  if (description !== undefined) project.description = description;
  if (network) project.network = network;
  res.json(project);
});

// ===================================================================
// LABELS
// ===================================================================
app.get('/api/workspaces/:slug/projects/:id/labels', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const result = store.labels.filter((l) => l.project_id === project.id);
  res.json({ results: result });
});

app.post('/api/workspaces/:slug/projects/:id/labels', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const { name, color } = req.body;
  if (!name) return res.status(400).json({ error: 'name is required' });
  const label = {
    id: uuidv4(),
    project_id: project.id,
    name,
    color: color || '#6b7280',
    created_at: new Date().toISOString(),
  };
  store.labels.push(label);
  res.status(201).json(label);
});

// ===================================================================
// ISSUES
// ===================================================================
app.post('/api/workspaces/:slug/projects/:id/issues', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;

  const { title, description, state, priority, assignee_id, labels: labelIds } = req.body;
  if (!title) return res.status(400).json({ error: 'title is required' });

  // Validate state and priority
  if (state && !store.states.includes(state)) {
    return res.status(400).json({ error: `Invalid state. Must be one of: ${store.states.join(', ')}` });
  }
  if (priority && !store.priorities.includes(priority)) {
    return res.status(400).json({ error: `Invalid priority. Must be one of: ${store.priorities.join(', ')}` });
  }

  const counter = (store.sequenceCounters[project.id] || 0) + 1;
  store.sequenceCounters[project.id] = counter;

  const issue = {
    id: uuidv4(),
    project_id: project.id,
    title,
    description: description || '',
    state: state || 'backlog',
    priority: priority || 'none',
    assignee_id: assignee_id || null,
    labels: labelIds || [],
    sequence_id: counter,
    created_by: req.user.id,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };
  store.issues.push(issue);
  res.status(201).json(issue);
});

app.get('/api/workspaces/:slug/projects/:id/issues', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;

  let result = store.issues.filter((i) => i.project_id === project.id);

  // Filters
  if (req.query.state) {
    const filterStates = req.query.state.split(',');
    result = result.filter((i) => filterStates.includes(i.state));
  }
  if (req.query.priority) {
    const filterPriorities = req.query.priority.split(',');
    result = result.filter((i) => filterPriorities.includes(i.priority));
  }
  if (req.query.assignee) {
    result = result.filter((i) => i.assignee_id === req.query.assignee);
  }
  if (req.query.label) {
    result = result.filter((i) => i.labels.includes(req.query.label));
  }
  if (req.query.search) {
    const q = req.query.search.toLowerCase();
    result = result.filter(
      (i) => i.title.toLowerCase().includes(q) || i.description.toLowerCase().includes(q)
    );
  }

  // Pagination
  const total = result.length;
  const limit = Math.min(parseInt(req.query.limit) || 50, 100);
  const offset = parseInt(req.query.offset) || 0;
  result = result.slice(offset, offset + limit);

  res.json({ results: result, total, limit, offset });
});

app.get('/api/workspaces/:slug/projects/:id/issues/:issueId', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const issue = store.issues.find(
    (i) => i.id === req.params.issueId && i.project_id === project.id
  );
  if (!issue) return res.status(404).json({ error: 'Issue not found' });
  res.json(issue);
});

app.put('/api/workspaces/:slug/projects/:id/issues/:issueId', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const issue = store.issues.find(
    (i) => i.id === req.params.issueId && i.project_id === project.id
  );
  if (!issue) return res.status(404).json({ error: 'Issue not found' });

  const { title, description, state, priority, assignee_id, labels: labelIds } = req.body;
  if (title !== undefined) issue.title = title;
  if (description !== undefined) issue.description = description;
  if (state !== undefined) {
    if (!store.states.includes(state)) {
      return res.status(400).json({ error: `Invalid state. Must be one of: ${store.states.join(', ')}` });
    }
    issue.state = state;
  }
  if (priority !== undefined) {
    if (!store.priorities.includes(priority)) {
      return res.status(400).json({ error: `Invalid priority. Must be one of: ${store.priorities.join(', ')}` });
    }
    issue.priority = priority;
  }
  if (assignee_id !== undefined) issue.assignee_id = assignee_id;
  if (labelIds !== undefined) issue.labels = labelIds;
  issue.updated_at = new Date().toISOString();

  res.json(issue);
});

app.delete('/api/workspaces/:slug/projects/:id/issues/:issueId', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const idx = store.issues.findIndex(
    (i) => i.id === req.params.issueId && i.project_id === project.id
  );
  if (idx === -1) return res.status(404).json({ error: 'Issue not found' });
  const [deleted] = store.issues.splice(idx, 1);
  // Also remove from cycles
  const cycleIdx = store.cycleIssues.findIndex((ci) => ci.issue_id === deleted.id);
  if (cycleIdx !== -1) store.cycleIssues.splice(cycleIdx, 1);
  // Also remove comments
  store.comments = store.comments.filter((c) => c.issue_id !== deleted.id);
  res.json({ deleted: true, id: deleted.id });
});

// ===================================================================
// ISSUE COMMENTS
// ===================================================================
app.post(
  '/api/workspaces/:slug/projects/:id/issues/:issueId/comments',
  authenticate,
  (req, res) => {
    const ws = resolveWorkspace(req, res);
    if (!ws) return;
    const project = resolveProject(ws, req.params.id, res);
    if (!project) return;
    const issue = store.issues.find(
      (i) => i.id === req.params.issueId && i.project_id === project.id
    );
    if (!issue) return res.status(404).json({ error: 'Issue not found' });
    const { body } = req.body;
    if (!body) return res.status(400).json({ error: 'body is required' });
    const comment = {
      id: uuidv4(),
      issue_id: issue.id,
      user_id: req.user.id,
      body,
      created_at: new Date().toISOString(),
    };
    store.comments.push(comment);
    res.status(201).json(comment);
  }
);

app.get(
  '/api/workspaces/:slug/projects/:id/issues/:issueId/comments',
  authenticate,
  (req, res) => {
    const ws = resolveWorkspace(req, res);
    if (!ws) return;
    const project = resolveProject(ws, req.params.id, res);
    if (!project) return;
    const issue = store.issues.find(
      (i) => i.id === req.params.issueId && i.project_id === project.id
    );
    if (!issue) return res.status(404).json({ error: 'Issue not found' });
    const result = store.comments
      .filter((c) => c.issue_id === issue.id)
      .map((c) => {
        const user = store.users.find((u) => u.id === c.user_id);
        return { ...c, user: user ? sanitizeUser(user) : null };
      });
    res.json({ results: result });
  }
);

// ===================================================================
// ISSUE LABELS (add/remove labels on an issue)
// ===================================================================
app.post(
  '/api/workspaces/:slug/projects/:id/issues/:issueId/labels',
  authenticate,
  (req, res) => {
    const ws = resolveWorkspace(req, res);
    if (!ws) return;
    const project = resolveProject(ws, req.params.id, res);
    if (!project) return;
    const issue = store.issues.find(
      (i) => i.id === req.params.issueId && i.project_id === project.id
    );
    if (!issue) return res.status(404).json({ error: 'Issue not found' });
    const { label_id } = req.body;
    if (!label_id) return res.status(400).json({ error: 'label_id is required' });
    const label = store.labels.find((l) => l.id === label_id && l.project_id === project.id);
    if (!label) return res.status(404).json({ error: 'Label not found in this project' });
    if (!issue.labels.includes(label_id)) {
      issue.labels.push(label_id);
      issue.updated_at = new Date().toISOString();
    }
    res.json(issue);
  }
);

// ===================================================================
// CYCLES (sprints)
// ===================================================================
app.post('/api/workspaces/:slug/projects/:id/cycles', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const { name, description, start_date, end_date } = req.body;
  if (!name) return res.status(400).json({ error: 'name is required' });
  const cycle = {
    id: uuidv4(),
    project_id: project.id,
    name,
    description: description || '',
    start_date: start_date || null,
    end_date: end_date || null,
    status: 'upcoming',
    created_by: req.user.id,
    created_at: new Date().toISOString(),
  };
  store.cycles.push(cycle);
  res.status(201).json(cycle);
});

app.get('/api/workspaces/:slug/projects/:id/cycles', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const result = store.cycles
    .filter((c) => c.project_id === project.id)
    .map((c) => {
      const issueIdsInCycle = store.cycleIssues
        .filter((ci) => ci.cycle_id === c.id)
        .map((ci) => ci.issue_id);
      return { ...c, issue_count: issueIdsInCycle.length };
    });
  res.json({ results: result });
});

app.post(
  '/api/workspaces/:slug/projects/:id/cycles/:cycleId/issues',
  authenticate,
  (req, res) => {
    const ws = resolveWorkspace(req, res);
    if (!ws) return;
    const project = resolveProject(ws, req.params.id, res);
    if (!project) return;
    const cycle = store.cycles.find(
      (c) => c.id === req.params.cycleId && c.project_id === project.id
    );
    if (!cycle) return res.status(404).json({ error: 'Cycle not found' });
    const { issue_id } = req.body;
    if (!issue_id) return res.status(400).json({ error: 'issue_id is required' });
    const issue = store.issues.find((i) => i.id === issue_id && i.project_id === project.id);
    if (!issue) return res.status(404).json({ error: 'Issue not found in this project' });
    const existing = store.cycleIssues.find(
      (ci) => ci.cycle_id === cycle.id && ci.issue_id === issue_id
    );
    if (existing) return res.status(409).json({ error: 'Issue already in this cycle' });
    store.cycleIssues.push({ cycle_id: cycle.id, issue_id });
    res.status(201).json({ cycle_id: cycle.id, issue_id });
  }
);

// ===================================================================
// VIEWS
// ===================================================================
app.get('/api/workspaces/:slug/projects/:id/views', authenticate, (req, res) => {
  const ws = resolveWorkspace(req, res);
  if (!ws) return;
  const project = resolveProject(ws, req.params.id, res);
  if (!project) return;
  const result = store.views.filter((v) => v.project_id === project.id);
  res.json({ results: result });
});

// ===================================================================
// SPA fallback — serve index.html for non-API routes
// ===================================================================
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ===================================================================
// 404 handler
// ===================================================================
app.use((req, res) => {
  res.status(404).json({ error: `Route ${req.method} ${req.path} not found` });
});

// Error handler
app.use((err, req, res, _next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ error: 'Internal server error' });
});

// ===================================================================
// Start server
// ===================================================================
app.listen(PORT, () => {
  console.log(`Plane Testbench running on port ${PORT}`);
  console.log(`  Health: http://localhost:${PORT}/health`);
  console.log(`  UI:     http://localhost:${PORT}/`);
  console.log(`  API:    http://localhost:${PORT}/api/`);
});

module.exports = app;
