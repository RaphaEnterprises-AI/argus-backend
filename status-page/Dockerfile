# Uptime Kuma for Railway - Multi-stage build to avoid VOLUME directive
# The base image has VOLUME which Railway bans, so we copy to a clean image

# Stage 1: Use uptime-kuma as source
FROM louislam/uptime-kuma:1 AS source

# Stage 2: Clean image without VOLUME directive
FROM node:18-alpine

# Install required dependencies
RUN apk add --no-cache \
    chromium \
    git \
    sqlite \
    iputils \
    curl \
    && rm -rf /var/cache/apk/*

# Set Chromium path for Uptime Kuma
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true \
    PUPPETEER_EXECUTABLE_PATH=/usr/bin/chromium-browser

WORKDIR /app

# Copy application from source image (avoids VOLUME inheritance)
COPY --from=source /app /app

# Railway provides PORT, Uptime Kuma needs it as UPTIME_KUMA_PORT
ENV UPTIME_KUMA_PORT=${PORT:-3001}
ENV NODE_ENV=production

# Data directory - attach Railway Volume at /app/data via dashboard
# https://docs.railway.com/reference/volumes

EXPOSE ${UPTIME_KUMA_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${UPTIME_KUMA_PORT}/api/status-page/heartbeat || exit 1

# Start Uptime Kuma
CMD ["node", "server/server.js"]
