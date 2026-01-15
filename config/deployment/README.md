# Deployment Configuration

This directory contains deployment configuration files for various platforms.

## Files

- **docker-compose.yml** - Docker Compose configuration for local development
- **fly.toml** - Fly.io deployment configuration
- **railway.json** - Railway deployment configuration (v1)
- **railway.toml** - Railway deployment configuration (v2)
- **render.yaml** - Render.com deployment configuration
- **wrangler.toml** - Cloudflare Worker deployment configuration

## Usage

### Docker Compose (Local Development)
```bash
cd ../../
docker-compose up -d
```

### Fly.io
```bash
flyctl deploy --config config/deployment/fly.toml
```

### Railway
```bash
railway up
```

### Render
Connect this repository to Render and it will automatically use `render.yaml`.

### Cloudflare Worker
```bash
wrangler deploy --config config/deployment/wrangler.toml
```

## Environment Variables

For all deployments, ensure you have a `.env` file at the root with the required environment variables. See `.env.example` for reference.
