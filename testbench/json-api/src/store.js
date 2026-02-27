const { v4: uuidv4 } = require('uuid');

// Generate fixed IDs at startup so relationships are consistent
const userIds = [uuidv4(), uuidv4(), uuidv4(), uuidv4(), uuidv4()];
const postIds = [uuidv4(), uuidv4(), uuidv4(), uuidv4(), uuidv4()];
const commentIds = [uuidv4(), uuidv4(), uuidv4(), uuidv4(), uuidv4()];
const todoIds = [uuidv4(), uuidv4(), uuidv4(), uuidv4(), uuidv4()];

const now = new Date().toISOString();

const users = [
  { id: userIds[0], name: 'Alice Johnson', email: 'alice@example.com', createdAt: now },
  { id: userIds[1], name: 'Bob Smith', email: 'bob@example.com', createdAt: now },
  { id: userIds[2], name: 'Charlie Davis', email: 'charlie@example.com', createdAt: now },
  { id: userIds[3], name: 'Diana Lee', email: 'diana@example.com', createdAt: now },
  { id: userIds[4], name: 'Eve Martinez', email: 'eve@example.com', createdAt: now },
];

const posts = [
  { id: postIds[0], title: 'Getting Started with Node.js', body: 'Node.js is a JavaScript runtime built on Chrome\'s V8 engine.', userId: userIds[0], createdAt: now },
  { id: postIds[1], title: 'Understanding REST APIs', body: 'REST stands for Representational State Transfer.', userId: userIds[0], createdAt: now },
  { id: postIds[2], title: 'Database Design Patterns', body: 'Good database design is crucial for application performance.', userId: userIds[1], createdAt: now },
  { id: postIds[3], title: 'Testing Best Practices', body: 'Automated testing saves time and catches bugs early.', userId: userIds[2], createdAt: now },
  { id: postIds[4], title: 'DevOps Pipeline Setup', body: 'CI/CD pipelines automate the software delivery process.', userId: userIds[3], createdAt: now },
];

const comments = [
  { id: commentIds[0], body: 'Great introduction to Node.js!', postId: postIds[0], userId: userIds[1], createdAt: now },
  { id: commentIds[1], body: 'Very helpful REST API explanation.', postId: postIds[1], userId: userIds[2], createdAt: now },
  { id: commentIds[2], body: 'I learned a lot about database design.', postId: postIds[2], userId: userIds[3], createdAt: now },
  { id: commentIds[3], body: 'Testing is indeed important.', postId: postIds[3], userId: userIds[4], createdAt: now },
  { id: commentIds[4], body: 'Thanks for the DevOps tips!', postId: postIds[4], userId: userIds[0], createdAt: now },
];

const todos = [
  { id: todoIds[0], title: 'Set up project repository', userId: userIds[0], completed: true, createdAt: now },
  { id: todoIds[1], title: 'Write unit tests', userId: userIds[0], completed: false, createdAt: now },
  { id: todoIds[2], title: 'Review pull requests', userId: userIds[1], completed: false, createdAt: now },
  { id: todoIds[3], title: 'Update documentation', userId: userIds[2], completed: true, createdAt: now },
  { id: todoIds[4], title: 'Deploy to staging', userId: userIds[3], completed: false, createdAt: now },
];

module.exports = { users, posts, comments, todos };
