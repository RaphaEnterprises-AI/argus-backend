# Conduit (RealWorld) - Argus Testbench

A lightweight [RealWorld](https://realworld-docs.netlify.app/) Conduit blog app for dogfooding the Argus E2E testing platform.

## What is this?

Conduit is the canonical "RealWorld" example app -- a medium.com clone with CRUD for articles, comments, user profiles, follows, and favorites. This implementation uses Express with in-memory storage, following the [RealWorld API spec](https://realworld-docs.netlify.app/specifications/backend/endpoints/).

## Seeded data

| Entity   | Count | Details                                              |
|----------|-------|------------------------------------------------------|
| Users    | 2     | john@example.com / password123, jane@example.com / password123 |
| Articles | 5     | Various testing and DevOps topics with tags           |
| Comments | 3     | Spread across articles                                |
| Follows  | 1     | john follows jane                                     |
| Favorites| 3     | Cross-user favorites                                  |

## Running locally

```bash
npm install
npm run dev     # starts with --watch for auto-reload
```

## API endpoints

| Method | Path                                    | Auth     | Description              |
|--------|-----------------------------------------|----------|--------------------------|
| POST   | /api/users/login                        | No       | Authenticate             |
| POST   | /api/users                              | No       | Register                 |
| GET    | /api/user                               | Required | Current user             |
| PUT    | /api/user                               | Required | Update user              |
| GET    | /api/profiles/:username                 | Optional | Get profile              |
| POST   | /api/profiles/:username/follow          | Required | Follow user              |
| DELETE | /api/profiles/:username/follow          | Required | Unfollow user            |
| GET    | /api/articles                           | Optional | List (tag/author/favorited/limit/offset) |
| GET    | /api/articles/feed                      | Required | Feed from followed users |
| POST   | /api/articles                           | Required | Create article           |
| GET    | /api/articles/:slug                     | Optional | Get article              |
| PUT    | /api/articles/:slug                     | Required | Update article           |
| DELETE | /api/articles/:slug                     | Required | Delete article           |
| POST   | /api/articles/:slug/comments            | Required | Add comment              |
| GET    | /api/articles/:slug/comments            | Optional | List comments            |
| DELETE | /api/articles/:slug/comments/:id        | Required | Delete comment           |
| POST   | /api/articles/:slug/favorite            | Required | Favorite article         |
| DELETE | /api/articles/:slug/favorite            | Required | Unfavorite article       |
| GET    | /api/tags                               | No       | List all tags            |
| GET    | /health                                 | No       | Health check             |

## Auth

Use `Authorization: Token <jwt>` header (RealWorld spec uses "Token", not "Bearer").

## Deploy to Railway

```bash
railway link
railway up --detach
```
