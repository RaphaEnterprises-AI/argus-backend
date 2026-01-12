# Argus Crawlee Service Dockerfile
# Uses official Apify Playwright Chrome image for browser support

# Build stage - install all deps and compile TypeScript
FROM apify/actor-node-playwright-chrome:20 AS builder

WORKDIR /app

# Ensure dev dependencies are installed regardless of NODE_ENV
ENV NODE_ENV=development

# Copy package files
COPY package*.json ./

# Install ALL dependencies (including dev for TypeScript)
RUN npm ci --include=dev

# Copy TypeScript config and source
COPY tsconfig.json ./
COPY src ./src

# Build TypeScript using local binary
RUN ./node_modules/.bin/tsc

# Production stage
FROM apify/actor-node-playwright-chrome:20

WORKDIR /app

# Copy package files and install production deps only
COPY package*.json ./
RUN npm ci --omit=dev

# Copy built files from builder
COPY --from=builder /app/dist ./dist

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Expose port
EXPOSE 3000

# Run the service
CMD ["node", "dist/index.js"]
