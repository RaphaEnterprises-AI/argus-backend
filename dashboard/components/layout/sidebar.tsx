'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { UserButton } from '@clerk/nextjs';
import {
  MessageSquare,
  TestTube,
  Eye,
  Compass,
  BarChart3,
  Settings,
  Github,
  Zap,
  Brain,
  Globe,
  Shield,
  AlertCircle,
  FileText,
  HelpCircle,
  ExternalLink,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { VersionBadge } from '@/components/ui/version-badge';

const navigation = [
  { name: 'Chat', href: '/', icon: MessageSquare, description: 'AI Assistant' },
  { name: 'Test Runner', href: '/tests', icon: TestTube, description: 'Execute tests' },
  { name: 'Discovery', href: '/discovery', icon: Compass, description: 'Find testable surfaces' },
  { name: 'Visual AI', href: '/visual', icon: Eye, description: 'Visual regression' },
  { name: 'AI Insights', href: '/insights', icon: Brain, description: 'Pattern analysis' },
  { name: 'Global Testing', href: '/global', icon: Globe, description: 'Cross-browser' },
  { name: 'Quality Audit', href: '/quality', icon: Shield, description: 'Code health' },
  { name: 'Intelligence', href: '/intelligence', icon: AlertCircle, description: 'Quality score' },
  { name: 'Reports', href: '/reports', icon: BarChart3, description: 'Analytics' },
];

const bottomNavigation = [
  { name: 'Integrations', href: '/integrations', icon: Zap },
  { name: 'Settings', href: '/settings', icon: Settings },
];

const legalLinks = [
  { name: 'Legal', href: '/legal' },
  { name: 'Privacy', href: '/legal/privacy' },
  { name: 'Terms', href: '/legal/terms' },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-border bg-card flex flex-col">
      {/* Logo */}
      <div className="flex h-16 items-center gap-3 border-b border-border px-6">
        <div className="relative">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg gradient-primary">
            <Eye className="h-5 w-5 text-white" />
          </div>
          <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-amber-500 rounded-full border-2 border-card" />
        </div>
        <span className="font-bold text-lg tracking-tight">Argus</span>
      </div>

      {/* Main Navigation */}
      <nav className="flex-1 overflow-y-auto px-3 py-4">
        <div className="space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200',
                  isActive
                    ? 'bg-primary text-primary-foreground shadow-sm'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                )}
              >
                <item.icon className={cn(
                  'h-5 w-5 flex-shrink-0 transition-colors',
                  isActive ? 'text-primary-foreground' : 'text-muted-foreground group-hover:text-foreground'
                )} />
                <span className="truncate">{item.name}</span>
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Bottom Section */}
      <div className="border-t border-border px-3 py-4 space-y-1">
        {bottomNavigation.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )}
            >
              <item.icon className="h-5 w-5 flex-shrink-0" />
              {item.name}
            </Link>
          );
        })}

        {/* External Links */}
        <a
          href="https://docs.heyargus.ai"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-all duration-200"
        >
          <HelpCircle className="h-5 w-5 flex-shrink-0" />
          <span>Docs</span>
          <ExternalLink className="h-3 w-3 ml-auto opacity-50" />
        </a>

        <a
          href="https://github.com/heyargus"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-all duration-200"
        >
          <Github className="h-5 w-5 flex-shrink-0" />
          <span>GitHub</span>
          <ExternalLink className="h-3 w-3 ml-auto opacity-50" />
        </a>
      </div>

      {/* User Profile */}
      <div className="border-t border-border px-4 py-4">
        <div className="flex items-center gap-3">
          <UserButton
            appearance={{
              elements: {
                avatarBox: "h-9 w-9 rounded-lg",
              },
            }}
            afterSignOutUrl="/"
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">Account</p>
            <p className="text-xs text-muted-foreground truncate">Manage settings</p>
          </div>
        </div>
      </div>

      {/* Footer Links */}
      <div className="border-t border-border px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            {legalLinks.map((link, i) => (
              <span key={link.name} className="flex items-center gap-3">
                <Link href={link.href} className="hover:text-foreground transition-colors">
                  {link.name}
                </Link>
                {i < legalLinks.length - 1 && <span className="text-border">·</span>}
              </span>
            ))}
          </div>
          <VersionBadge variant="minimal" />
        </div>
      </div>
    </aside>
  );
}
