/**
 * Argus Realtime Session - Durable Object
 * Handles WebSocket connections for real-time dashboard updates
 */

interface WebSocketSession {
  webSocket: WebSocket;
  organizationId: string;
  projectId?: string;
  subscriptions: Set<string>;
  connectedAt: Date;
}

/**
 * Message types for WebSocket communication
 */
export type RealtimeMessage =
  | { type: 'subscribe'; channels: string[] }
  | { type: 'unsubscribe'; channels: string[] }
  | { type: 'ping' }
  | { type: 'event'; channel: string; data: unknown };

export type RealtimeEvent =
  | { type: 'error:new'; projectId: string; data: unknown }
  | { type: 'quality:updated'; projectId: string; data: unknown }
  | { type: 'test:completed'; projectId: string; data: unknown }
  | { type: 'coverage:uploaded'; projectId: string; data: unknown }
  | { type: 'ci:status'; projectId: string; data: unknown };

/**
 * RealtimeSession Durable Object
 * Maintains WebSocket connections and broadcasts events to subscribers
 */
export class RealtimeSession implements DurableObject {
  private sessions: Map<WebSocket, WebSocketSession> = new Map();
  private state: DurableObjectState;

  constructor(state: DurableObjectState, env: unknown) {
    this.state = state;
  }

  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    // Handle WebSocket upgrade
    if (request.headers.get('Upgrade') === 'websocket') {
      return this.handleWebSocket(request);
    }

    // Handle broadcast request (from other workers)
    if (url.pathname === '/broadcast' && request.method === 'POST') {
      return this.handleBroadcast(request);
    }

    // Handle status request
    if (url.pathname === '/status') {
      return Response.json({
        connections: this.sessions.size,
        channels: this.getActiveChannels(),
      });
    }

    return new Response('Not Found', { status: 404 });
  }

  private async handleWebSocket(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const organizationId = url.searchParams.get('org');
    const projectId = url.searchParams.get('project') || undefined;

    if (!organizationId) {
      return new Response('Missing organization ID', { status: 400 });
    }

    // Create WebSocket pair
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);

    // Accept the WebSocket
    this.state.acceptWebSocket(server);

    // Store session data
    const session: WebSocketSession = {
      webSocket: server,
      organizationId,
      projectId,
      subscriptions: new Set(),
      connectedAt: new Date(),
    };
    this.sessions.set(server, session);

    // Auto-subscribe to organization events
    session.subscriptions.add(`org:${organizationId}`);
    if (projectId) {
      session.subscriptions.add(`project:${projectId}`);
    }

    // Send welcome message
    server.send(JSON.stringify({
      type: 'connected',
      organizationId,
      projectId,
      subscriptions: Array.from(session.subscriptions),
    }));

    return new Response(null, { status: 101, webSocket: client });
  }

  webSocketMessage(ws: WebSocket, message: string | ArrayBuffer): void {
    const session = this.sessions.get(ws);
    if (!session) return;

    try {
      const data = JSON.parse(message as string) as RealtimeMessage;

      switch (data.type) {
        case 'subscribe':
          this.handleSubscribe(session, data.channels);
          break;
        case 'unsubscribe':
          this.handleUnsubscribe(session, data.channels);
          break;
        case 'ping':
          ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
          break;
      }
    } catch (error) {
      ws.send(JSON.stringify({ type: 'error', message: 'Invalid message format' }));
    }
  }

  webSocketClose(ws: WebSocket, code: number, reason: string): void {
    this.sessions.delete(ws);
  }

  webSocketError(ws: WebSocket, error: unknown): void {
    console.error('WebSocket error:', error);
    this.sessions.delete(ws);
  }

  private handleSubscribe(session: WebSocketSession, channels: string[]): void {
    for (const channel of channels) {
      // Validate channel access (organization-scoped)
      if (this.canAccessChannel(session, channel)) {
        session.subscriptions.add(channel);
      }
    }

    session.webSocket.send(JSON.stringify({
      type: 'subscribed',
      subscriptions: Array.from(session.subscriptions),
    }));
  }

  private handleUnsubscribe(session: WebSocketSession, channels: string[]): void {
    for (const channel of channels) {
      session.subscriptions.delete(channel);
    }

    session.webSocket.send(JSON.stringify({
      type: 'unsubscribed',
      subscriptions: Array.from(session.subscriptions),
    }));
  }

  private canAccessChannel(session: WebSocketSession, channel: string): boolean {
    // Organization-scoped access control
    if (channel.startsWith('org:')) {
      return channel === `org:${session.organizationId}`;
    }
    if (channel.startsWith('project:')) {
      // In production, would check if project belongs to organization
      return true;
    }
    return false;
  }

  private async handleBroadcast(request: Request): Promise<Response> {
    try {
      const body = await request.json() as { channel: string; event: RealtimeEvent };
      const { channel, event } = body;

      let broadcastCount = 0;

      for (const [ws, session] of this.sessions) {
        if (session.subscriptions.has(channel)) {
          try {
            ws.send(JSON.stringify({ type: 'event', channel, data: event }));
            broadcastCount++;
          } catch {
            // WebSocket might be closed
            this.sessions.delete(ws);
          }
        }
      }

      return Response.json({ success: true, broadcastCount });
    } catch (error) {
      return Response.json(
        { success: false, error: error instanceof Error ? error.message : 'Unknown error' },
        { status: 400 }
      );
    }
  }

  private getActiveChannels(): string[] {
    const channels = new Set<string>();
    for (const session of this.sessions.values()) {
      for (const channel of session.subscriptions) {
        channels.add(channel);
      }
    }
    return Array.from(channels);
  }
}

/**
 * Helper to broadcast an event to realtime sessions
 */
export async function broadcastEvent(
  realtimeStub: DurableObjectStub,
  channel: string,
  event: RealtimeEvent
): Promise<{ success: boolean; broadcastCount?: number; error?: string }> {
  try {
    const response = await realtimeStub.fetch('http://internal/broadcast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ channel, event }),
    });

    return await response.json();
  } catch (error) {
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Broadcast failed',
    };
  }
}
