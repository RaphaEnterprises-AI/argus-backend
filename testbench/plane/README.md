# Plane Testbench

A lightweight Plane-like project management app for the Argus testbench. Simulates workspaces, projects, issues (kanban board), cycles (sprints), labels, and comments.

## Quick Start

```bash
cd testbench/plane
npm install
npm run dev
```

Open http://localhost:3000 and sign in with `admin@acme.com` / `password123`.

## API

All endpoints (except auth and health) require `Authorization: Bearer <token>`.

| Group | Endpoints |
|-------|-----------|
| Auth | `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me` |
| Workspaces | CRUD at `/api/workspaces` and `/api/workspaces/:slug` |
| Projects | CRUD at `/api/workspaces/:slug/projects` |
| Issues | CRUD at `/api/workspaces/:slug/projects/:id/issues` with filters (state, priority, assignee, label, search) and pagination (limit, offset) |
| Comments | `/api/workspaces/:slug/projects/:id/issues/:issueId/comments` |
| Labels | `/api/workspaces/:slug/projects/:id/labels` |
| Cycles | `/api/workspaces/:slug/projects/:id/cycles` |
| Views | `GET /api/workspaces/:slug/projects/:id/views` |
| Health | `GET /health` |

## Seed Data

- 1 workspace: Acme Corp (`acme-corp`)
- 2 users: admin@acme.com (admin), dev@acme.com (member)
- 2 projects: Backend API, Mobile App
- 10 issues across projects
- 3 labels per project (bug, feature, improvement)
- 1 cycle: Sprint 1
- 4 comments

## Deploy to Railway

Uses the Dockerfile at `testbench/plane/Dockerfile`. Set `PORT` env var (Railway provides this automatically).
