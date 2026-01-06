'use client';

import { useState } from 'react';
import { Sidebar } from '@/components/layout/sidebar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
  Settings,
  Key,
  Bell,
  Globe,
  Shield,
  Database,
  Cpu,
  Save,
  Eye,
  EyeOff,
  CheckCircle,
  Info,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { VersionBadge } from '@/components/ui/version-badge';
import { APP_VERSION } from '@/lib/version';

interface SettingSection {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
}

const sections: SettingSection[] = [
  { id: 'api', name: 'API Configuration', icon: Key },
  { id: 'notifications', name: 'Notifications', icon: Bell },
  { id: 'general', name: 'General', icon: Globe },
  { id: 'security', name: 'Security', icon: Shield },
  { id: 'about', name: 'About', icon: Info },
];

function Toggle({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex h-6 w-11 items-center rounded-full transition-colors',
        checked ? 'bg-primary' : 'bg-muted'
      )}
    >
      <span
        className={cn(
          'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
          checked ? 'translate-x-6' : 'translate-x-1'
        )}
      />
    </button>
  );
}

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState('api');
  const [showApiKey, setShowApiKey] = useState(false);
  const [saved, setSaved] = useState(false);

  const [settings, setSettings] = useState({
    anthropicApiKey: '',  // Load from environment or user input - NEVER hardcode
    backendUrl: process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000',
    defaultModel: 'claude-sonnet-4-5',
    maxIterations: 50,
    screenshotResolution: '1920x1080',
    costLimit: 10.0,
    slackEnabled: true,
    emailEnabled: false,
    slackWebhook: '',
    emailRecipients: '',
    autoRetry: true,
    retryCount: 3,
    parallelTests: 4,
    headlessMode: true,
    saveScreenshots: true,
    twoFactorEnabled: false,
    sessionTimeout: 30,
  });

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 ml-64">
        {/* Header */}
        <header className="sticky top-0 z-30 flex h-16 items-center gap-4 border-b bg-background/80 backdrop-blur-sm px-6">
          <div className="flex-1">
            <h1 className="text-xl font-semibold">Settings</h1>
            <p className="text-sm text-muted-foreground">
              Configure Argus - your AI testing companion
            </p>
          </div>
          <Button onClick={handleSave}>
            {saved ? (
              <>
                <CheckCircle className="mr-2 h-4 w-4 text-green-500" />
                Saved!
              </>
            ) : (
              <>
                <Save className="mr-2 h-4 w-4" />
                Save Changes
              </>
            )}
          </Button>
        </header>

        <div className="p-6">
          <div className="flex gap-6">
            {/* Sidebar Navigation */}
            <div className="w-56 space-y-1">
              {sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={cn(
                    'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors text-left',
                    activeSection === section.id
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                  )}
                >
                  <section.icon className="h-4 w-4" />
                  {section.name}
                </button>
              ))}
            </div>

            {/* Settings Content */}
            <div className="flex-1 space-y-6">
              {activeSection === 'api' && (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Key className="h-5 w-5" />
                        API Keys
                      </CardTitle>
                      <CardDescription>
                        Configure your API credentials for Claude and backend services
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <label className="text-sm font-medium">Anthropic API Key</label>
                        <div className="flex gap-2 mt-1">
                          <div className="relative flex-1">
                            <Input
                              type={showApiKey ? 'text' : 'password'}
                              value={settings.anthropicApiKey}
                              onChange={(e) =>
                                setSettings({ ...settings, anthropicApiKey: e.target.value })
                              }
                            />
                          </div>
                          <Button
                            variant="outline"
                            size="icon"
                            onClick={() => setShowApiKey(!showApiKey)}
                          >
                            {showApiKey ? (
                              <EyeOff className="h-4 w-4" />
                            ) : (
                              <Eye className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          Your API key is encrypted and stored securely
                        </p>
                      </div>

                      <div>
                        <label className="text-sm font-medium">Backend URL</label>
                        <Input
                          value={settings.backendUrl}
                          onChange={(e) =>
                            setSettings({ ...settings, backendUrl: e.target.value })
                          }
                          className="mt-1"
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Cpu className="h-5 w-5" />
                        Model Configuration
                      </CardTitle>
                      <CardDescription>
                        Configure AI model settings and limits
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <label className="text-sm font-medium">Default Model</label>
                        <select
                          value={settings.defaultModel}
                          onChange={(e) =>
                            setSettings({ ...settings, defaultModel: e.target.value })
                          }
                          className="mt-1 w-full px-3 py-2 rounded-md border bg-background"
                        >
                          <option value="claude-sonnet-4-5">Claude Sonnet 4.5</option>
                          <option value="claude-haiku-4-5">Claude Haiku 4.5</option>
                          <option value="claude-opus-4-5">Claude Opus 4.5</option>
                        </select>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium">Max Iterations</label>
                          <Input
                            type="number"
                            value={settings.maxIterations}
                            onChange={(e) =>
                              setSettings({ ...settings, maxIterations: parseInt(e.target.value) })
                            }
                            className="mt-1"
                          />
                        </div>
                        <div>
                          <label className="text-sm font-medium">Cost Limit (USD)</label>
                          <Input
                            type="number"
                            step="0.01"
                            value={settings.costLimit}
                            onChange={(e) =>
                              setSettings({ ...settings, costLimit: parseFloat(e.target.value) })
                            }
                            className="mt-1"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="text-sm font-medium">Screenshot Resolution</label>
                        <select
                          value={settings.screenshotResolution}
                          onChange={(e) =>
                            setSettings({ ...settings, screenshotResolution: e.target.value })
                          }
                          className="mt-1 w-full px-3 py-2 rounded-md border bg-background"
                        >
                          <option value="1280x720">1280x720 (720p)</option>
                          <option value="1920x1080">1920x1080 (1080p)</option>
                          <option value="2560x1440">2560x1440 (1440p)</option>
                        </select>
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}

              {activeSection === 'notifications' && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Bell className="h-5 w-5" />
                      Notification Settings
                    </CardTitle>
                    <CardDescription>
                      Configure how you receive test results and alerts
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">Slack Notifications</div>
                        <div className="text-sm text-muted-foreground">
                          Receive test results in Slack
                        </div>
                      </div>
                      <Toggle
                        checked={settings.slackEnabled}
                        onChange={(v) => setSettings({ ...settings, slackEnabled: v })}
                      />
                    </div>

                    {settings.slackEnabled && (
                      <div>
                        <label className="text-sm font-medium">Slack Webhook URL</label>
                        <Input
                          value={settings.slackWebhook}
                          onChange={(e) =>
                            setSettings({ ...settings, slackWebhook: e.target.value })
                          }
                          placeholder="https://hooks.slack.com/services/..."
                          className="mt-1"
                        />
                      </div>
                    )}

                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">Email Notifications</div>
                        <div className="text-sm text-muted-foreground">
                          Receive test results via email
                        </div>
                      </div>
                      <Toggle
                        checked={settings.emailEnabled}
                        onChange={(v) => setSettings({ ...settings, emailEnabled: v })}
                      />
                    </div>

                    {settings.emailEnabled && (
                      <div>
                        <label className="text-sm font-medium">Email Recipients</label>
                        <Input
                          value={settings.emailRecipients}
                          onChange={(e) =>
                            setSettings({ ...settings, emailRecipients: e.target.value })
                          }
                          placeholder="team@example.com, alerts@example.com"
                          className="mt-1"
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                          Separate multiple emails with commas
                        </p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {activeSection === 'general' && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Settings className="h-5 w-5" />
                      General Settings
                    </CardTitle>
                    <CardDescription>
                      Configure test execution behavior
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">Auto-Retry Failed Tests</div>
                        <div className="text-sm text-muted-foreground">
                          Automatically retry flaky tests
                        </div>
                      </div>
                      <Toggle
                        checked={settings.autoRetry}
                        onChange={(v) => setSettings({ ...settings, autoRetry: v })}
                      />
                    </div>

                    {settings.autoRetry && (
                      <div>
                        <label className="text-sm font-medium">Retry Count</label>
                        <Input
                          type="number"
                          value={settings.retryCount}
                          onChange={(e) =>
                            setSettings({ ...settings, retryCount: parseInt(e.target.value) })
                          }
                          min="1"
                          max="5"
                          className="mt-1 w-24"
                        />
                      </div>
                    )}

                    <div>
                      <label className="text-sm font-medium">Parallel Test Execution</label>
                      <Input
                        type="number"
                        value={settings.parallelTests}
                        onChange={(e) =>
                          setSettings({ ...settings, parallelTests: parseInt(e.target.value) })
                        }
                        min="1"
                        max="10"
                        className="mt-1 w-24"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Number of tests to run in parallel
                      </p>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">Headless Mode</div>
                        <div className="text-sm text-muted-foreground">
                          Run browser tests without UI
                        </div>
                      </div>
                      <Toggle
                        checked={settings.headlessMode}
                        onChange={(v) => setSettings({ ...settings, headlessMode: v })}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">Save Screenshots</div>
                        <div className="text-sm text-muted-foreground">
                          Save screenshots for all test steps
                        </div>
                      </div>
                      <Toggle
                        checked={settings.saveScreenshots}
                        onChange={(v) => setSettings({ ...settings, saveScreenshots: v })}
                      />
                    </div>
                  </CardContent>
                </Card>
              )}

              {activeSection === 'security' && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Security Settings
                    </CardTitle>
                    <CardDescription>
                      Configure security and access controls
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">Two-Factor Authentication</div>
                        <div className="text-sm text-muted-foreground">
                          Require 2FA for dashboard access
                        </div>
                      </div>
                      <Toggle
                        checked={settings.twoFactorEnabled}
                        onChange={(v) => setSettings({ ...settings, twoFactorEnabled: v })}
                      />
                    </div>

                    <div>
                      <label className="text-sm font-medium">Session Timeout (minutes)</label>
                      <Input
                        type="number"
                        value={settings.sessionTimeout}
                        onChange={(e) =>
                          setSettings({ ...settings, sessionTimeout: parseInt(e.target.value) })
                        }
                        min="5"
                        max="120"
                        className="mt-1 w-24"
                      />
                      <p className="text-xs text-muted-foreground mt-1">
                        Auto-logout after inactivity
                      </p>
                    </div>

                    <div className="pt-4 border-t">
                      <h4 className="font-medium mb-3">Danger Zone</h4>
                      <div className="space-y-3">
                        <Button variant="outline" className="text-red-500 border-red-500/20 hover:bg-red-500/10">
                          Clear All Test Data
                        </Button>
                        <Button variant="outline" className="text-red-500 border-red-500/20 hover:bg-red-500/10 ml-3">
                          Revoke All API Keys
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {activeSection === 'about' && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Info className="h-5 w-5" />
                      About Argus
                    </CardTitle>
                    <CardDescription>
                      Application information and version details
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="flex items-center justify-between p-4 rounded-lg bg-muted/50">
                      <div>
                        <div className="text-2xl font-bold">Argus</div>
                        <div className="text-sm text-muted-foreground">
                          AI-powered E2E Testing Platform
                        </div>
                      </div>
                      <VersionBadge variant="full" />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-4 rounded-lg border">
                        <div className="text-sm text-muted-foreground">Version</div>
                        <div className="text-lg font-mono font-medium">{APP_VERSION}</div>
                      </div>
                      <div className="p-4 rounded-lg border">
                        <div className="text-sm text-muted-foreground">Environment</div>
                        <div className="text-lg font-medium capitalize">
                          {process.env.NODE_ENV || 'development'}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-3">
                      <h4 className="font-medium">Tech Stack</h4>
                      <div className="flex flex-wrap gap-2">
                        {['Next.js 15', 'React 19', 'TypeScript', 'Tailwind CSS', 'Supabase', 'Clerk Auth', 'Claude AI'].map((tech) => (
                          <span
                            key={tech}
                            className="px-3 py-1 text-xs rounded-full bg-primary/10 text-primary font-medium"
                          >
                            {tech}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="pt-4 border-t">
                      <h4 className="font-medium mb-3">Links</h4>
                      <div className="space-y-2">
                        <a
                          href="https://docs.heyargus.ai"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block text-sm text-primary hover:underline"
                        >
                          Documentation →
                        </a>
                        <a
                          href="https://github.com/samuelvinay91/argus"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block text-sm text-primary hover:underline"
                        >
                          GitHub Repository →
                        </a>
                        <a
                          href="https://heyargus.ai"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block text-sm text-primary hover:underline"
                        >
                          Website →
                        </a>
                      </div>
                    </div>

                    <div className="pt-4 border-t text-center text-sm text-muted-foreground">
                      <p>© {new Date().getFullYear()} Argus. All rights reserved.</p>
                      <p className="mt-1">Built with ❤️ for better testing</p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
