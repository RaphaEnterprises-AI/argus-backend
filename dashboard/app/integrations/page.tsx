'use client';

import { useState } from 'react';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
  Github,
  GitlabIcon,
  Slack,
  Webhook,
  CheckCircle,
  Copy,
  ExternalLink,
  Zap,
  Activity,
  LineChart,
  Bug,
  Video,
  BarChart3,
  Database,
  AlertTriangle,
  Loader2,
  Key,
  Eye,
  EyeOff,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'framer-motion';

const GITHUB_ACTIONS_YAML = `name: Argus E2E Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  argus-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start Application
        run: |
          npm ci
          npm run build
          npm start &
          sleep 10

      - name: Run Argus Tests
        env:
          ARGUS_API_URL: \${{ secrets.ARGUS_API_URL }}
          ANTHROPIC_API_KEY: \${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          curl -X POST \\
            "\${ARGUS_API_URL}/api/v1/tests/run" \\
            -H "Content-Type: application/json" \\
            -d '{
              "app_url": "http://localhost:3000",
              "github_pr": "\${{ github.event.pull_request.number }}",
              "github_repo": "\${{ github.repository }}"
            }'

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: argus-results
          path: test-results/`;

const GITLAB_CI_YAML = `argus-tests:
  stage: test
  image: node:20
  services:
    - name: your-app:latest
      alias: app
  variables:
    ARGUS_API_URL: \${ARGUS_API_URL}
    ANTHROPIC_API_KEY: \${ANTHROPIC_API_KEY}
  script:
    - |
      curl -X POST \\
        "\${ARGUS_API_URL}/api/v1/tests/run" \\
        -H "Content-Type: application/json" \\
        -d '{
          "app_url": "http://app:3000",
          "gitlab_mr": "'\${CI_MERGE_REQUEST_IID}'"
        }'
  artifacts:
    reports:
      junit: test-results/junit.xml
  only:
    - merge_requests`;

const WEBHOOK_EXAMPLE = `// n8n Webhook Configuration
{
  "url": "https://your-n8n-instance.com/webhook/argus-trigger",
  "method": "POST",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "app_url": "{{ $json.preview_url }}",
    "pr_number": "{{ $json.pr_number }}",
    "notify_slack": true
  }
}`;

const SDK_EXAMPLE = `import { Argus } from '@heyargus/sdk';

const agent = new Argus({
  apiKey: process.env.ANTHROPIC_API_KEY,
  baseUrl: 'https://api.heyargus.ai',
});

// Create test from natural language
const test = await agent.createTest(
  'Login as admin@example.com and verify dashboard loads'
);

// Run the test
const result = await agent.runTest(test.id, {
  appUrl: 'http://localhost:3000',
});

console.log(\`Test \${result.status}: \${result.passed}/\${result.total} passed\`);

// Auto-discover app flows
const discovery = await agent.discover('http://localhost:3000');
console.log(\`Found \${discovery.flows.length} user flows\`);`;

interface Integration {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  connected: boolean;
  configExample: string;
}

const integrations: Integration[] = [
  {
    id: 'github',
    name: 'GitHub Actions',
    description: 'Run Argus tests on PRs and get results as check status',
    icon: Github,
    connected: true,
    configExample: GITHUB_ACTIONS_YAML,
  },
  {
    id: 'gitlab',
    name: 'GitLab CI/CD',
    description: 'Integrate Argus with GitLab merge request pipelines',
    icon: GitlabIcon,
    connected: false,
    configExample: GITLAB_CI_YAML,
  },
  {
    id: 'slack',
    name: 'Slack Notifications',
    description: 'Get Argus test results and alerts in Slack channels',
    icon: Slack,
    connected: true,
    configExample: '',
  },
  {
    id: 'webhook',
    name: 'Webhooks (n8n/Zapier)',
    description: 'Trigger Argus tests from any automation platform',
    icon: Webhook,
    connected: false,
    configExample: WEBHOOK_EXAMPLE,
  },
];

// Observability Platform Integrations
interface ObservabilityPlatform {
  id: string;
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  connected: boolean;
  lastSync?: string;
  dataPoints?: number;
  authType: 'oauth' | 'api_key';
  features: string[];
}

const observabilityPlatforms: ObservabilityPlatform[] = [
  {
    id: 'datadog',
    name: 'Datadog',
    description: 'RUM, APM, and Log Analytics',
    icon: Activity,
    color: 'purple',
    connected: true,
    lastSync: '2 min ago',
    dataPoints: 12847,
    authType: 'api_key',
    features: ['Session Replay', 'Core Web Vitals', 'Error Tracking', 'APM Traces'],
  },
  {
    id: 'sentry',
    name: 'Sentry',
    description: 'Error Tracking & Performance',
    icon: Bug,
    color: 'pink',
    connected: true,
    lastSync: '5 min ago',
    dataPoints: 3421,
    authType: 'oauth',
    features: ['Error Aggregation', 'Stack Traces', 'Release Tracking', 'Session Replay'],
  },
  {
    id: 'newrelic',
    name: 'New Relic',
    description: 'Full-Stack Observability',
    icon: LineChart,
    color: 'green',
    connected: false,
    authType: 'api_key',
    features: ['Browser Monitoring', 'APM', 'Infrastructure', 'Logs'],
  },
  {
    id: 'fullstory',
    name: 'FullStory',
    description: 'Digital Experience Intelligence',
    icon: Video,
    color: 'blue',
    connected: true,
    lastSync: '1 min ago',
    dataPoints: 5632,
    authType: 'api_key',
    features: ['Session Replay', 'Rage Clicks', 'Frustration Signals', 'Heatmaps'],
  },
  {
    id: 'posthog',
    name: 'PostHog',
    description: 'Product Analytics & Session Recording',
    icon: BarChart3,
    color: 'orange',
    connected: false,
    authType: 'api_key',
    features: ['Product Analytics', 'Session Recording', 'Feature Flags', 'Funnels'],
  },
  {
    id: 'logrocket',
    name: 'LogRocket',
    description: 'Session Replay & Error Tracking',
    icon: Video,
    color: 'indigo',
    connected: false,
    authType: 'api_key',
    features: ['Session Replay', 'Console Logs', 'Network Requests', 'Redux State'],
  },
  {
    id: 'amplitude',
    name: 'Amplitude',
    description: 'Product & Behavioral Analytics',
    icon: LineChart,
    color: 'blue',
    connected: false,
    authType: 'api_key',
    features: ['User Journeys', 'Cohort Analysis', 'Funnel Analytics', 'Retention'],
  },
  {
    id: 'mixpanel',
    name: 'Mixpanel',
    description: 'Product Analytics',
    icon: BarChart3,
    color: 'purple',
    connected: false,
    authType: 'api_key',
    features: ['Event Tracking', 'User Flows', 'A/B Testing', 'Retention Analysis'],
  },
];

export default function IntegrationsPage() {
  const [selectedIntegration, setSelectedIntegration] = useState<Integration>(integrations[0]);
  const [selectedPlatform, setSelectedPlatform] = useState<ObservabilityPlatform | null>(null);
  const [copied, setCopied] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [activeTab, setActiveTab] = useState<'cicd' | 'observability'>('observability');

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleConnect = (platform: ObservabilityPlatform) => {
    setIsConnecting(true);
    // Simulate OAuth/API key connection
    setTimeout(() => {
      setIsConnecting(false);
      setSelectedPlatform(null);
    }, 2000);
  };

  const handleSync = () => {
    setIsSyncing(true);
    setTimeout(() => setIsSyncing(false), 3000);
  };

  const connectedPlatforms = observabilityPlatforms.filter(p => p.connected);
  const totalDataPoints = connectedPlatforms.reduce((acc, p) => acc + (p.dataPoints || 0), 0);

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-64">
        {/* Header */}
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b bg-background/80 backdrop-blur-sm px-6">
          <div className="flex-1">
            <h1 className="text-xl font-semibold flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              Integrations
            </h1>
            <p className="text-sm text-muted-foreground">
              Connect observability platforms and CI/CD tools
            </p>
          </div>
          <Button onClick={handleSync} disabled={isSyncing}>
            {isSyncing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Syncing...
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" />
                Sync All
              </>
            )}
          </Button>
        </header>

        <div className="p-6 space-y-6">
          {/* Tab Navigation */}
          <div className="flex gap-2 border-b">
            <button
              onClick={() => setActiveTab('observability')}
              className={cn(
                'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
                activeTab === 'observability'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              <Sparkles className="inline-block mr-2 h-4 w-4" />
              Observability Platforms
            </button>
            <button
              onClick={() => setActiveTab('cicd')}
              className={cn(
                'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
                activeTab === 'cicd'
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              )}
            >
              <Github className="inline-block mr-2 h-4 w-4" />
              CI/CD & Notifications
            </button>
          </div>

          <AnimatePresence mode="wait">
            {activeTab === 'observability' ? (
              <motion.div
                key="observability"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-6"
              >
                {/* Connected Platforms Summary */}
                <Card className="bg-gradient-to-r from-primary/5 via-purple-500/5 to-blue-500/5 border-primary/20">
                  <CardContent className="pt-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                          <Sparkles className="h-5 w-5 text-primary" />
                          AI-Powered Production Intelligence
                        </h3>
                        <p className="text-sm text-muted-foreground mt-1">
                          Connect your observability stack. AI automatically learns from real user behavior.
                        </p>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold text-primary">{totalDataPoints.toLocaleString()}</div>
                        <div className="text-sm text-muted-foreground">data points synced</div>
                      </div>
                    </div>
                    <div className="flex gap-4 mt-4">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm">{connectedPlatforms.length} platforms connected</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4 text-blue-500" />
                        <span className="text-sm">Real-time sync enabled</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Observability Platforms Grid */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {observabilityPlatforms.map((platform, index) => (
                    <motion.button
                      key={platform.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      onClick={() => setSelectedPlatform(platform)}
                      className={cn(
                        'p-4 rounded-lg border text-left transition-all hover:shadow-md',
                        platform.connected
                          ? 'border-green-500/30 bg-green-500/5'
                          : 'hover:bg-accent/50'
                      )}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <div className={cn(
                          'p-2 rounded-lg',
                          platform.connected
                            ? `bg-${platform.color}-500/10`
                            : 'bg-muted'
                        )}>
                          <platform.icon className={cn(
                            'h-5 w-5',
                            platform.connected ? `text-${platform.color}-500` : 'text-muted-foreground'
                          )} />
                        </div>
                        {platform.connected ? (
                          <CheckCircle className="h-5 w-5 text-green-500" />
                        ) : (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                            Not connected
                          </span>
                        )}
                      </div>
                      <div className="font-medium">{platform.name}</div>
                      <div className="text-xs text-muted-foreground mt-1">{platform.description}</div>
                      {platform.connected && (
                        <div className="mt-3 pt-3 border-t flex items-center justify-between text-xs">
                          <span className="text-green-500">{platform.dataPoints?.toLocaleString()} records</span>
                          <span className="text-muted-foreground">{platform.lastSync}</span>
                        </div>
                      )}
                    </motion.button>
                  ))}
                </div>

                {/* Platform Connection Modal */}
                <AnimatePresence>
                  {selectedPlatform && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center"
                      onClick={() => setSelectedPlatform(null)}
                    >
                      <motion.div
                        initial={{ scale: 0.95, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.95, opacity: 0 }}
                        onClick={(e) => e.stopPropagation()}
                        className="bg-card border rounded-xl shadow-xl w-full max-w-lg p-6"
                      >
                        <div className="flex items-center gap-3 mb-6">
                          <div className={cn('p-3 rounded-lg', `bg-${selectedPlatform.color}-500/10`)}>
                            <selectedPlatform.icon className={cn('h-6 w-6', `text-${selectedPlatform.color}-500`)} />
                          </div>
                          <div>
                            <h3 className="text-lg font-semibold">{selectedPlatform.name}</h3>
                            <p className="text-sm text-muted-foreground">{selectedPlatform.description}</p>
                          </div>
                        </div>

                        {selectedPlatform.connected ? (
                          <div className="space-y-4">
                            <div className="p-4 rounded-lg bg-green-500/10 border border-green-500/30">
                              <div className="flex items-center gap-2 text-green-500 font-medium">
                                <CheckCircle className="h-4 w-4" />
                                Connected
                              </div>
                              <p className="text-sm text-muted-foreground mt-1">
                                Last synced: {selectedPlatform.lastSync}
                              </p>
                            </div>

                            <div>
                              <h4 className="font-medium mb-2">Synced Features</h4>
                              <div className="flex flex-wrap gap-2">
                                {selectedPlatform.features.map((feature) => (
                                  <span
                                    key={feature}
                                    className="px-2 py-1 text-xs rounded-full bg-primary/10 text-primary"
                                  >
                                    {feature}
                                  </span>
                                ))}
                              </div>
                            </div>

                            <div className="flex gap-2">
                              <Button variant="outline" className="flex-1" onClick={() => setSelectedPlatform(null)}>
                                Close
                              </Button>
                              <Button variant="destructive" className="flex-1">
                                Disconnect
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            {selectedPlatform.authType === 'oauth' ? (
                              <Button
                                className="w-full"
                                onClick={() => handleConnect(selectedPlatform)}
                                disabled={isConnecting}
                              >
                                {isConnecting ? (
                                  <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Connecting...
                                  </>
                                ) : (
                                  <>
                                    <ExternalLink className="mr-2 h-4 w-4" />
                                    Connect with OAuth
                                  </>
                                )}
                              </Button>
                            ) : (
                              <div className="space-y-3">
                                <div>
                                  <label className="text-sm font-medium">API Key</label>
                                  <div className="relative mt-1">
                                    <Input
                                      type={showApiKey ? 'text' : 'password'}
                                      placeholder="Enter your API key"
                                      className="pr-10"
                                    />
                                    <button
                                      onClick={() => setShowApiKey(!showApiKey)}
                                      className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                                    >
                                      {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                    </button>
                                  </div>
                                </div>
                                {selectedPlatform.id === 'datadog' && (
                                  <>
                                    <div>
                                      <label className="text-sm font-medium">Application Key</label>
                                      <Input
                                        type="password"
                                        placeholder="Enter your application key"
                                        className="mt-1"
                                      />
                                    </div>
                                    <div>
                                      <label className="text-sm font-medium">Site</label>
                                      <Input
                                        placeholder="datadoghq.com"
                                        className="mt-1"
                                      />
                                    </div>
                                  </>
                                )}
                                <Button
                                  className="w-full"
                                  onClick={() => handleConnect(selectedPlatform)}
                                  disabled={isConnecting}
                                >
                                  {isConnecting ? (
                                    <>
                                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                      Connecting...
                                    </>
                                  ) : (
                                    <>
                                      <Key className="mr-2 h-4 w-4" />
                                      Connect
                                    </>
                                  )}
                                </Button>
                              </div>
                            )}

                            <div>
                              <h4 className="font-medium mb-2">What we&apos;ll sync</h4>
                              <div className="flex flex-wrap gap-2">
                                {selectedPlatform.features.map((feature) => (
                                  <span
                                    key={feature}
                                    className="px-2 py-1 text-xs rounded-full bg-muted text-muted-foreground"
                                  >
                                    {feature}
                                  </span>
                                ))}
                              </div>
                            </div>

                            <p className="text-xs text-muted-foreground">
                              Your credentials are encrypted and stored securely. We only read data, never write.
                            </p>
                          </div>
                        )}
                      </motion.div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* AI Features Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-primary" />
                      What AI Does With Your Data
                    </CardTitle>
                    <CardDescription>
                      Zero configuration required - AI automatically learns from connected platforms
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                      <div className="p-4 rounded-lg border">
                        <Video className="h-6 w-6 text-blue-500 mb-2" />
                        <h4 className="font-medium">Session → Test</h4>
                        <p className="text-xs text-muted-foreground mt-1">
                          Converts real user sessions into automated tests
                        </p>
                      </div>
                      <div className="p-4 rounded-lg border">
                        <Bug className="h-6 w-6 text-red-500 mb-2" />
                        <h4 className="font-medium">Error → Regression</h4>
                        <p className="text-xs text-muted-foreground mt-1">
                          Auto-generates tests to prevent error recurrence
                        </p>
                      </div>
                      <div className="p-4 rounded-lg border">
                        <AlertTriangle className="h-6 w-6 text-yellow-500 mb-2" />
                        <h4 className="font-medium">Predict Failures</h4>
                        <p className="text-xs text-muted-foreground mt-1">
                          Detects patterns that indicate incoming issues
                        </p>
                      </div>
                      <div className="p-4 rounded-lg border">
                        <BarChart3 className="h-6 w-6 text-green-500 mb-2" />
                        <h4 className="font-medium">Coverage Gaps</h4>
                        <p className="text-xs text-muted-foreground mt-1">
                          Identifies untested areas based on real traffic
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ) : (
              <motion.div
                key="cicd"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="space-y-6"
              >
                {/* CI/CD Integration Cards */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  {integrations.map((integration) => (
                    <button
                      key={integration.id}
                      onClick={() => setSelectedIntegration(integration)}
                      className={cn(
                        'p-4 rounded-lg border text-left transition-colors',
                        selectedIntegration.id === integration.id
                          ? 'border-primary bg-primary/5'
                          : 'hover:bg-accent/50'
                      )}
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <div className={cn(
                          'p-2 rounded-lg',
                          integration.connected ? 'bg-green-500/10' : 'bg-muted'
                        )}>
                          <integration.icon className={cn(
                            'h-5 w-5',
                            integration.connected ? 'text-green-500' : 'text-muted-foreground'
                          )} />
                        </div>
                        {integration.connected && (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        )}
                      </div>
                      <div className="font-medium">{integration.name}</div>
                      <div className="text-sm text-muted-foreground line-clamp-2">
                        {integration.description}
                      </div>
                    </button>
                  ))}
                </div>

          {/* Configuration */}
          <div className="grid grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <selectedIntegration.icon className="h-5 w-5" />
                  {selectedIntegration.name} Setup
                </CardTitle>
                <CardDescription>{selectedIntegration.description}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedIntegration.id === 'slack' ? (
                  <div className="space-y-4">
                    <div>
                      <label className="text-sm font-medium">Webhook URL</label>
                      <Input
                        placeholder="https://hooks.slack.com/services/..."
                        className="mt-1"
                      />
                    </div>
                    <div>
                      <label className="text-sm font-medium">Channel</label>
                      <Input placeholder="#engineering-alerts" className="mt-1" />
                    </div>
                    <div className="flex gap-2">
                      <Button>Connect Slack</Button>
                      <Button variant="outline">Test Connection</Button>
                    </div>
                  </div>
                ) : (
                  selectedIntegration.configExample && (
                    <div className="relative">
                      <div className="absolute top-2 right-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => copyToClipboard(selectedIntegration.configExample)}
                        >
                          {copied ? (
                            <CheckCircle className="h-4 w-4 text-green-500" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                      <pre className="p-4 rounded-lg bg-muted overflow-x-auto text-xs">
                        <code>{selectedIntegration.configExample}</code>
                      </pre>
                    </div>
                  )
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  SDK Usage
                </CardTitle>
                <CardDescription>
                  Use the SDK to integrate programmatically
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <div className="absolute top-2 right-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(SDK_EXAMPLE)}
                    >
                      {copied ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  <pre className="p-4 rounded-lg bg-muted overflow-x-auto text-xs">
                    <code>{SDK_EXAMPLE}</code>
                  </pre>
                </div>
                <div className="mt-4 flex gap-2">
                  <Button variant="outline" size="sm">
                    <ExternalLink className="mr-2 h-4 w-4" />
                    View Documentation
                  </Button>
                  <Button variant="outline" size="sm">
                    <Github className="mr-2 h-4 w-4" />
                    Example Project
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* API Endpoint Reference */}
          <Card>
            <CardHeader>
              <CardTitle>API Endpoints</CardTitle>
              <CardDescription>Direct API access for custom integrations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { method: 'POST', endpoint: '/api/v1/tests/run', desc: 'Run test suite' },
                  { method: 'POST', endpoint: '/api/v1/tests/create', desc: 'Create test from NLP' },
                  { method: 'POST', endpoint: '/api/v1/discover', desc: 'Auto-discover app' },
                  { method: 'POST', endpoint: '/api/v1/visual/compare', desc: 'Visual comparison' },
                  { method: 'GET', endpoint: '/api/v1/jobs/{id}', desc: 'Get job status' },
                  { method: 'POST', endpoint: '/api/v1/webhooks/github', desc: 'GitHub webhook' },
                ].map((api) => (
                  <div
                    key={api.endpoint}
                    className="flex items-center gap-3 p-3 rounded-lg border"
                  >
                    <span className={cn(
                      'px-2 py-0.5 rounded text-xs font-mono',
                      api.method === 'GET' ? 'bg-blue-500/10 text-blue-500' : 'bg-green-500/10 text-green-500'
                    )}>
                      {api.method}
                    </span>
                    <div className="flex-1">
                      <code className="text-sm">{api.endpoint}</code>
                      <div className="text-xs text-muted-foreground">{api.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
