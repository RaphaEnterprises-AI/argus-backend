const { v4: uuidv4 } = require('uuid');
const bcrypt = require('bcryptjs');
const slugify = require('slugify');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const hashPassword = (plain) => bcrypt.hashSync(plain, 10);
const shortId = () => uuidv4().split('-')[0];

const now = new Date().toISOString();
const earlier = new Date(Date.now() - 86400000).toISOString(); // 1 day ago
const evenEarlier = new Date(Date.now() - 172800000).toISOString(); // 2 days ago

// ---------------------------------------------------------------------------
// Users
// ---------------------------------------------------------------------------
const users = [
  {
    id: uuidv4(),
    email: 'john@example.com',
    username: 'john',
    password: hashPassword('password123'),
    bio: 'Full-stack developer who loves testing.',
    image: 'https://api.dicebear.com/7.x/avataaars/svg?seed=john',
    createdAt: evenEarlier,
    updatedAt: evenEarlier,
  },
  {
    id: uuidv4(),
    email: 'jane@example.com',
    username: 'jane',
    password: hashPassword('password123'),
    bio: 'QA engineer and automation enthusiast.',
    image: 'https://api.dicebear.com/7.x/avataaars/svg?seed=jane',
    createdAt: evenEarlier,
    updatedAt: evenEarlier,
  },
];

// ---------------------------------------------------------------------------
// Follows  — Set of "followerId:followedId"
// ---------------------------------------------------------------------------
const follows = new Set([
  `${users[0].id}:${users[1].id}`, // john follows jane
]);

// ---------------------------------------------------------------------------
// Articles
// ---------------------------------------------------------------------------
const makeSlug = (title) => `${slugify(title, { lower: true, strict: true })}-${shortId()}`;

const articles = [
  {
    id: uuidv4(),
    slug: makeSlug('Getting Started with E2E Testing'),
    title: 'Getting Started with E2E Testing',
    description: 'A comprehensive guide to end-to-end testing for modern web apps.',
    body: 'End-to-end testing validates complete user workflows from start to finish. Unlike unit tests that check individual functions, E2E tests simulate real user interactions with your application.\n\n## Why E2E Testing?\n\n1. Catches integration bugs\n2. Validates user workflows\n3. Builds confidence in deployments',
    tagList: ['testing', 'e2e', 'automation'],
    authorId: users[0].id,
    createdAt: evenEarlier,
    updatedAt: evenEarlier,
  },
  {
    id: uuidv4(),
    slug: makeSlug('Self-Healing Tests with AI'),
    title: 'Self-Healing Tests with AI',
    description: 'How AI agents can automatically fix broken test selectors.',
    body: 'When UI elements change, tests break. AI-powered self-healing can detect selector changes and automatically update tests.\n\n## How It Works\n\n1. Detect failure pattern\n2. Analyze DOM changes\n3. Generate new selector\n4. Validate and update test',
    tagList: ['ai', 'testing', 'self-healing'],
    authorId: users[0].id,
    createdAt: earlier,
    updatedAt: earlier,
  },
  {
    id: uuidv4(),
    slug: makeSlug('CI/CD Pipeline Best Practices'),
    title: 'CI/CD Pipeline Best Practices',
    description: 'Setting up robust continuous integration and deployment pipelines.',
    body: 'A well-configured CI/CD pipeline is essential for modern software delivery. Here are the key practices every team should follow.\n\n## Key Practices\n\n- Run tests on every PR\n- Use parallel test execution\n- Implement deployment gates\n- Monitor test flakiness',
    tagList: ['cicd', 'devops', 'automation'],
    authorId: users[1].id,
    createdAt: earlier,
    updatedAt: earlier,
  },
  {
    id: uuidv4(),
    slug: makeSlug('Understanding Browser Automation'),
    title: 'Understanding Browser Automation',
    description: 'Deep dive into Playwright, Selenium, and modern browser automation.',
    body: 'Browser automation tools let you programmatically control web browsers. This article compares the major frameworks.\n\n## Playwright vs Selenium\n\nPlaywright offers auto-wait, better debugging, and multi-browser support out of the box. Selenium has a larger ecosystem and wider language support.',
    tagList: ['automation', 'playwright', 'selenium'],
    authorId: users[1].id,
    createdAt: now,
    updatedAt: now,
  },
  {
    id: uuidv4(),
    slug: makeSlug('Test Impact Analysis'),
    title: 'Test Impact Analysis',
    description: 'Run only the tests that matter for your code changes.',
    body: 'Test impact analysis maps code changes to affected tests, allowing you to skip irrelevant tests and get faster feedback.\n\n## Benefits\n\n- 60-80% reduction in test execution time\n- Faster PR feedback loops\n- Better developer experience',
    tagList: ['testing', 'performance', 'cicd'],
    authorId: users[0].id,
    createdAt: now,
    updatedAt: now,
  },
];

// ---------------------------------------------------------------------------
// Comments
// ---------------------------------------------------------------------------
const comments = [
  {
    id: 1,
    body: 'Great introduction! This helped me understand E2E testing much better.',
    articleId: articles[0].id,
    authorId: users[1].id,
    createdAt: earlier,
    updatedAt: earlier,
  },
  {
    id: 2,
    body: 'Self-healing tests are the future. We saved 40% of maintenance time after adopting this approach.',
    articleId: articles[1].id,
    authorId: users[1].id,
    createdAt: now,
    updatedAt: now,
  },
  {
    id: 3,
    body: 'Our team implemented these CI/CD practices and deployment frequency increased 3x.',
    articleId: articles[2].id,
    authorId: users[0].id,
    createdAt: now,
    updatedAt: now,
  },
];

let nextCommentId = 4;

// ---------------------------------------------------------------------------
// Favorites  — Set of "userId:articleId"
// ---------------------------------------------------------------------------
const favorites = new Set([
  `${users[1].id}:${articles[0].id}`, // jane favorited john's first article
  `${users[0].id}:${articles[2].id}`, // john favorited jane's first article
  `${users[1].id}:${articles[1].id}`, // jane favorited john's second article
]);

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
module.exports = {
  users,
  follows,
  articles,
  comments,
  favorites,
  getNextCommentId: () => nextCommentId++,
  makeSlug,
};
