import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Enable experimental features for AI SDK
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
  // Exclude heavy files from serverless function traces (must be at root level)
  outputFileTracingExcludes: {
    '*': [
      'node_modules/@swc/core-linux-x64-gnu',
      'node_modules/@swc/core-linux-x64-musl',
      'node_modules/@esbuild/linux-x64',
      'node_modules/@esbuild',
      'node_modules/esbuild',
      'node_modules/terser',
      'node_modules/webpack',
    ],
  },
  // Externalize heavy packages to reduce serverless function size
  serverExternalPackages: [
    '@anthropic-ai/sdk',
    'anthropic',
  ],
  // Environment variables for client-side
  env: {
    NEXT_PUBLIC_E2E_WORKER_URL: process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev',
  },
};

export default nextConfig;
