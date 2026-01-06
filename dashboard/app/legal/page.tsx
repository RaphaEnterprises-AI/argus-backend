'use client';

import Link from 'next/link';
import { ArrowLeft, FileText, Shield, Lock, Code, Globe, ArrowRight } from 'lucide-react';

const legalPages = [
  {
    title: 'Terms of Service',
    description: 'The legal agreement governing your use of Argus',
    href: '/legal/terms',
    icon: FileText,
    lastUpdated: 'December 30, 2024',
  },
  {
    title: 'Privacy Policy',
    description: 'How we collect, use, and protect your personal information',
    href: '/legal/privacy',
    icon: Lock,
    lastUpdated: 'December 30, 2024',
  },
  {
    title: 'Security & Compliance',
    description: 'Our security practices, certifications, and compliance commitments',
    href: '/legal/security',
    icon: Shield,
    lastUpdated: 'December 30, 2024',
  },
  {
    title: 'GDPR Data Processing Agreement',
    description: 'DPA for customers subject to GDPR requirements',
    href: '/legal/gdpr',
    icon: Globe,
    lastUpdated: 'December 30, 2024',
  },
  {
    title: 'Open Source Licenses',
    description: 'Acknowledgment of open source software used in Argus',
    href: '/legal/licenses',
    icon: Code,
    lastUpdated: 'December 30, 2024',
  },
];

export default function LegalIndexPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-4xl mx-auto px-6 py-12">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>

        <h1 className="text-4xl font-bold mb-2">Legal</h1>
        <p className="text-muted-foreground mb-8">
          Legal documents, policies, and compliance information for Argus
        </p>

        <div className="grid gap-4">
          {legalPages.map((page) => (
            <Link
              key={page.href}
              href={page.href}
              className="group p-6 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
            >
              <div className="flex items-start gap-4">
                <div className="p-2 rounded-lg bg-primary/10 text-primary">
                  <page.icon className="h-6 w-6" />
                </div>
                <div className="flex-1">
                  <h2 className="text-xl font-semibold group-hover:text-primary transition-colors">
                    {page.title}
                  </h2>
                  <p className="text-muted-foreground mt-1">{page.description}</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Last updated: {page.lastUpdated}
                  </p>
                </div>
                <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
              </div>
            </Link>
          ))}
        </div>

        <div className="mt-12 p-6 rounded-lg border bg-muted/30">
          <h2 className="text-lg font-semibold mb-2">Questions?</h2>
          <p className="text-muted-foreground">
            If you have any questions about our legal documents or policies, please contact us at{' '}
            <a href="mailto:legal@heyargus.ai" className="text-primary hover:underline">
              legal@heyargus.ai
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
