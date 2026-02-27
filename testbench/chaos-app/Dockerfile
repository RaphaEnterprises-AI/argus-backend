# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production server
FROM node:20-alpine
WORKDIR /app
COPY backend/package.json backend/package-lock.json* ./
RUN npm ci --production
COPY backend/src/ ./src/
COPY --from=frontend-builder /app/frontend/dist ./public/
EXPOSE ${PORT:-3001}
CMD ["node", "src/server.js"]
