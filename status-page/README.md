# Argus Status Page (Uptime Kuma)

Self-hosted status page for monitoring Argus services.

## Deploy to Railway

### Option 1: One-Click Deploy
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/UH5MJd)

### Option 2: Manual Deploy

1. Install Railway CLI:
```bash
npm install -g @railway/cli
railway login
```

2. Create new project:
```bash
cd status-page
railway init
```

3. Add a volume for persistent data:
```bash
railway volume add
```

4. Deploy:
```bash
railway up
```

5. Get the deployment URL:
```bash
railway domain
```

## Configure Custom Domain

1. In Railway dashboard, go to your deployment
2. Settings → Domains → Add Custom Domain
3. Add: `status.heyargus.ai`
4. Add the CNAME record in Cloudflare:
   - Type: CNAME
   - Name: status
   - Target: (Railway provided domain)
   - Proxy: OFF (DNS only)

## Initial Setup

1. Visit your deployment URL
2. Create admin account (first user becomes admin)
3. Add monitors (see below)

## Recommended Monitors

### Core Services
| Name | Type | URL/Host |
|------|------|----------|
| Dashboard | HTTP(s) | https://heyargus.ai |
| Dashboard (www) | HTTP(s) | https://www.heyargus.ai |
| API Brain | HTTP(s) | https://argus-brain-production.up.railway.app/health |
| Browser Worker | HTTP(s) | https://argus-api.samuelvinay-kumar.workers.dev |
| Documentation | HTTP(s) | https://docs.heyargus.ai |

### Database & Infrastructure
| Name | Type | URL/Host |
|------|------|----------|
| Supabase | HTTP(s) | https://REDACTED_PROJECT_REF.supabase.co/rest/v1/ |

### External Dependencies (Status Pages)
| Name | Type | URL/Host |
|------|------|----------|
| Anthropic | HTTP(s) | https://status.anthropic.com |
| OpenAI | HTTP(s) | https://status.openai.com |
| Cloudflare | HTTP(s) | https://www.cloudflarestatus.com |
| Vercel | HTTP(s) | https://www.vercel-status.com |
| Supabase | HTTP(s) | https://status.supabase.com |
| Railway | HTTP(s) | https://status.railway.app |
| Clerk | HTTP(s) | https://status.clerk.com |
| GitHub | HTTP(s) | https://www.githubstatus.com |

## Status Page Configuration

1. Go to Status Pages → Add New Status Page
2. Configure:
   - Title: Argus Status
   - Slug: argus (URL will be /status/argus)
   - Description: Real-time status of Argus services
   - Theme: Auto (respects user preference)
   - Show Powered By: Off

3. Add monitor groups:
   - Core Services
   - Infrastructure
   - External Dependencies

## Notifications

Set up notifications for downtime alerts:

1. Settings → Notifications → Setup Notification
2. Recommended:
   - **Slack**: For team alerts
   - **Email (SMTP)**: For admin alerts
   - **Discord**: For community updates

## Environment Variables (Optional)

```bash
# Custom port (Railway sets this automatically)
UPTIME_KUMA_PORT=3001

# Disable new user registration after setup
UPTIME_KUMA_DISABLE_FRAME_SAMEORIGIN=0
```

## Backup

Data is stored in `/app/data`. Railway volumes persist this data.

To backup manually:
```bash
railway run tar -czf backup.tar.gz /app/data
```

## Links

- [Uptime Kuma GitHub](https://github.com/louislam/uptime-kuma)
- [Uptime Kuma Wiki](https://github.com/louislam/uptime-kuma/wiki)
- [Railway Documentation](https://docs.railway.app)
