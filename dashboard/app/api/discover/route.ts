import { NextRequest, NextResponse } from 'next/server';

/**
 * API endpoint for connecting and syncing observability platforms.
 * Handles OAuth flows and API key-based authentication.
 */

export interface PlatformConnection {
  platform: string;
  apiKey?: string;
  appKey?: string;
  site?: string;
  authToken?: string;
  organization?: string;
  project?: string;
  accountId?: string;
  host?: string;
}

export interface SyncResult {
  platform: string;
  status: 'success' | 'error';
  sessionsCount?: number;
  errorsCount?: number;
  lastSync?: string;
  error?: string;
}

// In-memory storage for demo (use a real database in production)
const connectedPlatforms: Map<string, PlatformConnection> = new Map();

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, platform, credentials } = body;

    switch (action) {
      case 'connect':
        return handleConnect(platform, credentials);
      case 'disconnect':
        return handleDisconnect(platform);
      case 'sync':
        return handleSync(platform);
      case 'sync_all':
        return handleSyncAll();
      case 'status':
        return handleStatus();
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 });
    }
  } catch (error) {
    console.error('Platform API error:', error);
    return NextResponse.json(
      { error: 'Failed to process request' },
      { status: 500 }
    );
  }
}

async function handleConnect(platform: string, credentials: PlatformConnection) {
  // Validate credentials based on platform
  const validationResult = await validateCredentials(platform, credentials);

  if (!validationResult.valid) {
    return NextResponse.json(
      { error: validationResult.error },
      { status: 400 }
    );
  }

  // Store connection
  connectedPlatforms.set(platform, credentials);

  // Perform initial sync
  const syncResult = await syncPlatform(platform, credentials);

  return NextResponse.json({
    status: 'connected',
    platform,
    initialSync: syncResult,
    message: `Successfully connected to ${platform}`,
  });
}

async function handleDisconnect(platform: string) {
  if (!connectedPlatforms.has(platform)) {
    return NextResponse.json(
      { error: `Platform ${platform} is not connected` },
      { status: 400 }
    );
  }

  connectedPlatforms.delete(platform);

  return NextResponse.json({
    status: 'disconnected',
    platform,
    message: `Successfully disconnected from ${platform}`,
  });
}

async function handleSync(platform: string) {
  const credentials = connectedPlatforms.get(platform);

  if (!credentials) {
    return NextResponse.json(
      { error: `Platform ${platform} is not connected` },
      { status: 400 }
    );
  }

  const result = await syncPlatform(platform, credentials);

  return NextResponse.json({
    status: 'synced',
    platform,
    result,
  });
}

async function handleSyncAll() {
  const results: SyncResult[] = [];

  for (const [platform, credentials] of connectedPlatforms) {
    const result = await syncPlatform(platform, credentials);
    results.push(result);
  }

  return NextResponse.json({
    status: 'synced',
    results,
    totalPlatforms: results.length,
    successful: results.filter(r => r.status === 'success').length,
  });
}

async function handleStatus() {
  const platforms = Array.from(connectedPlatforms.keys());

  return NextResponse.json({
    connectedPlatforms: platforms,
    count: platforms.length,
  });
}

async function validateCredentials(
  platform: string,
  credentials: PlatformConnection
): Promise<{ valid: boolean; error?: string }> {
  switch (platform) {
    case 'datadog':
      if (!credentials.apiKey || !credentials.appKey) {
        return { valid: false, error: 'Datadog requires both API Key and Application Key' };
      }
      // In production, make a test API call to validate
      return { valid: true };

    case 'sentry':
      if (!credentials.authToken || !credentials.organization || !credentials.project) {
        return { valid: false, error: 'Sentry requires Auth Token, Organization, and Project' };
      }
      return { valid: true };

    case 'newrelic':
      if (!credentials.apiKey || !credentials.accountId) {
        return { valid: false, error: 'New Relic requires API Key and Account ID' };
      }
      return { valid: true };

    case 'fullstory':
      if (!credentials.apiKey) {
        return { valid: false, error: 'FullStory requires API Key' };
      }
      return { valid: true };

    case 'posthog':
      if (!credentials.apiKey) {
        return { valid: false, error: 'PostHog requires API Key' };
      }
      return { valid: true };

    case 'logrocket':
      if (!credentials.apiKey) {
        return { valid: false, error: 'LogRocket requires API Key' };
      }
      return { valid: true };

    case 'amplitude':
      if (!credentials.apiKey) {
        return { valid: false, error: 'Amplitude requires API Key' };
      }
      return { valid: true };

    case 'mixpanel':
      if (!credentials.apiKey) {
        return { valid: false, error: 'Mixpanel requires API Key' };
      }
      return { valid: true };

    default:
      return { valid: false, error: `Unknown platform: ${platform}` };
  }
}

async function syncPlatform(
  platform: string,
  credentials: PlatformConnection
): Promise<SyncResult> {
  // In production, this would call the actual platform APIs
  // For demo, we simulate the sync with mock data

  try {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Generate mock sync results
    const mockData: Record<string, { sessions: number; errors: number }> = {
      datadog: { sessions: 12847, errors: 342 },
      sentry: { sessions: 0, errors: 3421 },
      newrelic: { sessions: 8932, errors: 156 },
      fullstory: { sessions: 5632, errors: 89 },
      posthog: { sessions: 4521, errors: 67 },
      logrocket: { sessions: 3214, errors: 45 },
      amplitude: { sessions: 15234, errors: 0 },
      mixpanel: { sessions: 9876, errors: 0 },
    };

    const data = mockData[platform] || { sessions: 0, errors: 0 };

    return {
      platform,
      status: 'success',
      sessionsCount: data.sessions,
      errorsCount: data.errors,
      lastSync: new Date().toISOString(),
    };
  } catch (error) {
    return {
      platform,
      status: 'error',
      error: error instanceof Error ? error.message : 'Unknown error during sync',
    };
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const platform = searchParams.get('platform');

  if (platform) {
    const isConnected = connectedPlatforms.has(platform);
    return NextResponse.json({
      platform,
      connected: isConnected,
    });
  }

  return handleStatus();
}
