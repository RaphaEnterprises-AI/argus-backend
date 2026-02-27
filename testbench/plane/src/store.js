const { v4: uuidv4 } = require('uuid');
const bcrypt = require('bcryptjs');

// ---------------------------------------------------------------------------
// Fixed IDs for deterministic seed data
// ---------------------------------------------------------------------------
const userIds = {
  admin: uuidv4(),
  dev: uuidv4(),
};

const workspaceId = uuidv4();
const projectIds = { backend: uuidv4(), mobile: uuidv4() };
const labelIds = { bug: uuidv4(), feature: uuidv4(), improvement: uuidv4() };
const cycleId = uuidv4();

// Pre-generate issue IDs
const issueIds = Array.from({ length: 10 }, () => uuidv4());
const commentIds = Array.from({ length: 4 }, () => uuidv4());

const now = new Date().toISOString();
const passwordHash = bcrypt.hashSync('password123', 10);

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------
const users = [
  {
    id: userIds.admin,
    email: 'admin@acme.com',
    password: passwordHash,
    name: 'Admin User',
    role: 'admin',
    avatar: null,
    created_at: now,
  },
  {
    id: userIds.dev,
    email: 'dev@acme.com',
    password: passwordHash,
    name: 'Dev User',
    role: 'member',
    avatar: null,
    created_at: now,
  },
];

// ---------------------------------------------------------------------------
// Workspaces
// ---------------------------------------------------------------------------
const workspaces = [
  {
    id: workspaceId,
    name: 'Acme Corp',
    slug: 'acme-corp',
    owner_id: userIds.admin,
    created_at: now,
  },
];

// ---------------------------------------------------------------------------
// Workspace members
// ---------------------------------------------------------------------------
const workspaceMembers = [
  { workspace_id: workspaceId, user_id: userIds.admin, role: 'admin', joined_at: now },
  { workspace_id: workspaceId, user_id: userIds.dev, role: 'member', joined_at: now },
];

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------
const projects = [
  {
    id: projectIds.backend,
    workspace_id: workspaceId,
    name: 'Backend API',
    identifier: 'BAC',
    description: 'Core backend REST API service',
    network: 'public',
    created_by: userIds.admin,
    created_at: now,
  },
  {
    id: projectIds.mobile,
    workspace_id: workspaceId,
    name: 'Mobile App',
    identifier: 'MOB',
    description: 'Cross-platform mobile application',
    network: 'public',
    created_by: userIds.admin,
    created_at: now,
  },
];

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------
const labels = [
  { id: labelIds.bug, project_id: projectIds.backend, name: 'bug', color: '#d73a4a', created_at: now },
  { id: labelIds.feature, project_id: projectIds.backend, name: 'feature', color: '#0075ca', created_at: now },
  { id: labelIds.improvement, project_id: projectIds.backend, name: 'improvement', color: '#a2eeef', created_at: now },
  { id: uuidv4(), project_id: projectIds.mobile, name: 'bug', color: '#d73a4a', created_at: now },
  { id: uuidv4(), project_id: projectIds.mobile, name: 'feature', color: '#0075ca', created_at: now },
  { id: uuidv4(), project_id: projectIds.mobile, name: 'improvement', color: '#a2eeef', created_at: now },
];

// ---------------------------------------------------------------------------
// Issues
// ---------------------------------------------------------------------------
const states = ['backlog', 'todo', 'in_progress', 'done', 'cancelled'];
const priorities = ['urgent', 'high', 'medium', 'low', 'none'];

const issues = [
  // Backend API issues
  {
    id: issueIds[0],
    project_id: projectIds.backend,
    title: 'Set up database migrations',
    description: 'Create initial PostgreSQL schema and migration scripts for all models.',
    state: 'done',
    priority: 'high',
    assignee_id: userIds.admin,
    labels: [labelIds.feature],
    sequence_id: 1,
    created_by: userIds.admin,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[1],
    project_id: projectIds.backend,
    title: 'Implement JWT authentication',
    description: 'Add login/register endpoints with JWT token generation and validation.',
    state: 'in_progress',
    priority: 'urgent',
    assignee_id: userIds.dev,
    labels: [labelIds.feature],
    sequence_id: 2,
    created_by: userIds.admin,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[2],
    project_id: projectIds.backend,
    title: 'Fix rate limiter bypass',
    description: 'Rate limiter can be bypassed by changing X-Forwarded-For header.',
    state: 'todo',
    priority: 'urgent',
    assignee_id: userIds.admin,
    labels: [labelIds.bug],
    sequence_id: 3,
    created_by: userIds.dev,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[3],
    project_id: projectIds.backend,
    title: 'Add pagination to list endpoints',
    description: 'All list endpoints should support limit/offset pagination.',
    state: 'backlog',
    priority: 'medium',
    assignee_id: null,
    labels: [labelIds.improvement],
    sequence_id: 4,
    created_by: userIds.admin,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[4],
    project_id: projectIds.backend,
    title: 'Write API integration tests',
    description: 'Create comprehensive test suite covering all API endpoints.',
    state: 'todo',
    priority: 'high',
    assignee_id: userIds.dev,
    labels: [labelIds.improvement],
    sequence_id: 5,
    created_by: userIds.admin,
    created_at: now,
    updated_at: now,
  },
  // Mobile App issues
  {
    id: issueIds[5],
    project_id: projectIds.mobile,
    title: 'Design login screen',
    description: 'Create mockups and implement the login screen with email/password fields.',
    state: 'done',
    priority: 'high',
    assignee_id: userIds.dev,
    labels: [],
    sequence_id: 1,
    created_by: userIds.admin,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[6],
    project_id: projectIds.mobile,
    title: 'Implement push notifications',
    description: 'Set up Firebase Cloud Messaging for push notification support.',
    state: 'in_progress',
    priority: 'medium',
    assignee_id: userIds.dev,
    labels: [],
    sequence_id: 2,
    created_by: userIds.admin,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[7],
    project_id: projectIds.mobile,
    title: 'App crashes on Android 12',
    description: 'Users report crash on launch with Android 12 devices. Stack trace attached.',
    state: 'todo',
    priority: 'urgent',
    assignee_id: userIds.admin,
    labels: [],
    sequence_id: 3,
    created_by: userIds.dev,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[8],
    project_id: projectIds.mobile,
    title: 'Add offline mode support',
    description: 'Cache data locally so the app works without internet connection.',
    state: 'backlog',
    priority: 'low',
    assignee_id: null,
    labels: [],
    sequence_id: 4,
    created_by: userIds.admin,
    created_at: now,
    updated_at: now,
  },
  {
    id: issueIds[9],
    project_id: projectIds.mobile,
    title: 'Optimize image loading',
    description: 'Images load slowly on poor network. Implement progressive loading and caching.',
    state: 'cancelled',
    priority: 'none',
    assignee_id: null,
    labels: [],
    sequence_id: 5,
    created_by: userIds.dev,
    created_at: now,
    updated_at: now,
  },
];

// ---------------------------------------------------------------------------
// Comments
// ---------------------------------------------------------------------------
const comments = [
  {
    id: commentIds[0],
    issue_id: issueIds[1],
    user_id: userIds.admin,
    body: 'Let us use RS256 for JWT signing. HS256 is fine for dev but not prod.',
    created_at: now,
  },
  {
    id: commentIds[1],
    issue_id: issueIds[1],
    user_id: userIds.dev,
    body: 'Agreed. I will update the implementation to support RS256.',
    created_at: now,
  },
  {
    id: commentIds[2],
    issue_id: issueIds[2],
    user_id: userIds.admin,
    body: 'This is a critical security issue. Prioritizing for this sprint.',
    created_at: now,
  },
  {
    id: commentIds[3],
    issue_id: issueIds[7],
    user_id: userIds.dev,
    body: 'Looks like a compatibility issue with the new Scoped Storage API.',
    created_at: now,
  },
];

// ---------------------------------------------------------------------------
// Cycles (sprints)
// ---------------------------------------------------------------------------
const cycles = [
  {
    id: cycleId,
    project_id: projectIds.backend,
    name: 'Sprint 1',
    description: 'Initial development sprint focusing on auth and core API.',
    start_date: '2026-02-24',
    end_date: '2026-03-07',
    status: 'current',
    created_by: userIds.admin,
    created_at: now,
  },
];

// Issues assigned to Sprint 1
const cycleIssues = [
  { cycle_id: cycleId, issue_id: issueIds[0] },
  { cycle_id: cycleId, issue_id: issueIds[1] },
  { cycle_id: cycleId, issue_id: issueIds[2] },
];

// ---------------------------------------------------------------------------
// Saved views
// ---------------------------------------------------------------------------
const views = [
  {
    id: uuidv4(),
    project_id: projectIds.backend,
    name: 'My Open Issues',
    filters: { state: ['todo', 'in_progress'], assignee_id: null },
    created_by: userIds.admin,
    created_at: now,
  },
  {
    id: uuidv4(),
    project_id: projectIds.backend,
    name: 'High Priority Bugs',
    filters: { priority: ['urgent', 'high'], labels: ['bug'] },
    created_by: userIds.admin,
    created_at: now,
  },
];

// ---------------------------------------------------------------------------
// Sequence counters (per project)
// ---------------------------------------------------------------------------
const sequenceCounters = {
  [projectIds.backend]: 5,
  [projectIds.mobile]: 5,
};

module.exports = {
  users,
  workspaces,
  workspaceMembers,
  projects,
  labels,
  issues,
  comments,
  cycles,
  cycleIssues,
  views,
  sequenceCounters,
  states,
  priorities,
};
