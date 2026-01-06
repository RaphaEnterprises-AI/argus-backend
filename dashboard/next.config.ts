import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // Enable experimental features for AI SDK
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb',
    },
  },
  // Environment variables for client-side
  env: {
    NEXT_PUBLIC_E2E_WORKER_URL: process.env.NEXT_PUBLIC_E2E_WORKER_URL || 'https://e2e-testing-agent.samuelvinay-kumar.workers.dev',
  },
};

export default nextConfig;
