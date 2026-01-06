'use client';

import Link from 'next/link';
import { Eye, Github, Twitter, Linkedin, ExternalLink } from 'lucide-react';

export function AppFooter() {
  return (
    <footer className="border-t border-border bg-card/50 mt-auto">
      <div className="px-6 py-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Left - Brand & Copyright */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded gradient-primary flex items-center justify-center">
                <Eye className="w-3.5 h-3.5 text-white" />
              </div>
              <span className="text-sm font-semibold">Argus</span>
            </div>
            <span className="text-xs text-muted-foreground">
              © {new Date().getFullYear()} Argus. All rights reserved.
            </span>
          </div>

          {/* Center - Links */}
          <div className="flex items-center gap-6 text-xs">
            <Link href="/legal" className="text-muted-foreground hover:text-foreground transition-colors">
              Legal
            </Link>
            <Link href="/legal/privacy" className="text-muted-foreground hover:text-foreground transition-colors">
              Privacy
            </Link>
            <Link href="/legal/terms" className="text-muted-foreground hover:text-foreground transition-colors">
              Terms
            </Link>
            <Link href="/legal/security" className="text-muted-foreground hover:text-foreground transition-colors">
              Security
            </Link>
            <a
              href="https://docs.heyargus.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1"
            >
              Docs
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          {/* Right - Social */}
          <div className="flex items-center gap-3">
            <a
              href="https://github.com/heyargus"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-all"
            >
              <Github className="w-4 h-4" />
            </a>
            <a
              href="https://twitter.com/heyargus"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-all"
            >
              <Twitter className="w-4 h-4" />
            </a>
            <a
              href="https://linkedin.com/company/heyargus"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-lg transition-all"
            >
              <Linkedin className="w-4 h-4" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
