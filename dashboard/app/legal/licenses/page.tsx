'use client';

import Link from 'next/link';
import { ArrowLeft, Code, ExternalLink } from 'lucide-react';

interface License {
  name: string;
  version: string;
  license: string;
  repository?: string;
  description: string;
}

const frontendDependencies: License[] = [
  { name: 'Next.js', version: '15.0.0', license: 'MIT', repository: 'https://github.com/vercel/next.js', description: 'React framework for production' },
  { name: 'React', version: '19.0.0', license: 'MIT', repository: 'https://github.com/facebook/react', description: 'JavaScript library for building user interfaces' },
  { name: 'Tailwind CSS', version: '3.4.0', license: 'MIT', repository: 'https://github.com/tailwindlabs/tailwindcss', description: 'Utility-first CSS framework' },
  { name: '@clerk/nextjs', version: '6.36.5', license: 'MIT', repository: 'https://github.com/clerk/javascript', description: 'Authentication for Next.js' },
  { name: '@tanstack/react-query', version: '5.90.15', license: 'MIT', repository: 'https://github.com/TanStack/query', description: 'Powerful async state management' },
  { name: '@tanstack/react-table', version: '8.21.3', license: 'MIT', repository: 'https://github.com/TanStack/table', description: 'Headless UI for tables' },
  { name: '@supabase/supabase-js', version: '2.89.0', license: 'MIT', repository: 'https://github.com/supabase/supabase-js', description: 'Supabase client library' },
  { name: 'Framer Motion', version: '11.11.0', license: 'MIT', repository: 'https://github.com/framer/motion', description: 'Animation library for React' },
  { name: 'Lucide React', version: '0.460.0', license: 'ISC', repository: 'https://github.com/lucide-icons/lucide', description: 'Beautiful & consistent icons' },
  { name: 'Radix UI', version: 'various', license: 'MIT', repository: 'https://github.com/radix-ui/primitives', description: 'Unstyled, accessible UI primitives' },
  { name: 'Zustand', version: '5.0.0', license: 'MIT', repository: 'https://github.com/pmndrs/zustand', description: 'State management' },
  { name: 'date-fns', version: '4.1.0', license: 'MIT', repository: 'https://github.com/date-fns/date-fns', description: 'Modern date utility library' },
  { name: 'Recharts', version: '2.13.0', license: 'MIT', repository: 'https://github.com/recharts/recharts', description: 'Composable charting library' },
  { name: 'cmdk', version: '1.1.1', license: 'MIT', repository: 'https://github.com/pacocoursey/cmdk', description: 'Command menu component' },
  { name: 'class-variance-authority', version: '0.7.0', license: 'Apache-2.0', repository: 'https://github.com/joe-bell/cva', description: 'Variant-based styling' },
  { name: 'clsx', version: '2.1.0', license: 'MIT', repository: 'https://github.com/lukeed/clsx', description: 'Classname utility' },
  { name: 'tailwind-merge', version: '2.5.0', license: 'MIT', repository: 'https://github.com/dcastil/tailwind-merge', description: 'Merge Tailwind classes' },
];

const backendDependencies: License[] = [
  { name: '@cloudflare/playwright', version: '1.0.0', license: 'Apache-2.0', repository: 'https://github.com/nicholascelestin/puppeteer', description: 'Browser automation for Cloudflare Workers' },
  { name: 'Zod', version: '3.25.76', license: 'MIT', repository: 'https://github.com/colinhacks/zod', description: 'TypeScript-first schema validation' },
  { name: 'Wrangler', version: '4.0.0', license: 'MIT', repository: 'https://github.com/cloudflare/workers-sdk', description: 'Cloudflare Workers CLI' },
];

const aiDependencies: License[] = [
  { name: 'Vercel AI SDK', version: '4.0.0', license: 'Apache-2.0', repository: 'https://github.com/vercel/ai', description: 'Build AI-powered applications' },
  { name: '@ai-sdk/anthropic', version: '1.0.0', license: 'Apache-2.0', repository: 'https://github.com/vercel/ai', description: 'Anthropic provider for AI SDK' },
];

function LicenseTable({ licenses, title }: { licenses: License[]; title: string }) {
  return (
    <div className="mb-8">
      <h3 className="text-xl font-semibold mb-4">{title}</h3>
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              <th className="text-left py-3 px-4 font-medium">Package</th>
              <th className="text-left py-3 px-4 font-medium">Version</th>
              <th className="text-left py-3 px-4 font-medium">License</th>
              <th className="text-left py-3 px-4 font-medium">Description</th>
            </tr>
          </thead>
          <tbody className="divide-y">
            {licenses.map((dep) => (
              <tr key={dep.name} className="hover:bg-muted/30">
                <td className="py-3 px-4">
                  {dep.repository ? (
                    <a
                      href={dep.repository}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:underline inline-flex items-center gap-1"
                    >
                      {dep.name}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  ) : (
                    dep.name
                  )}
                </td>
                <td className="py-3 px-4 text-muted-foreground font-mono text-xs">{dep.version}</td>
                <td className="py-3 px-4">
                  <span className="inline-flex px-2 py-0.5 rounded-full text-xs bg-primary/10 text-primary">
                    {dep.license}
                  </span>
                </td>
                <td className="py-3 px-4 text-muted-foreground">{dep.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function LicensesPage() {
  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <Link
          href="/"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Dashboard
        </Link>

        <div className="flex items-center gap-3 mb-2">
          <Code className="h-10 w-10 text-primary" />
          <h1 className="text-4xl font-bold">Open Source Licenses</h1>
        </div>
        <p className="text-muted-foreground mb-8">
          Argus is built on the shoulders of giants. We are grateful to the open source community
          for the amazing tools and libraries that make our product possible.
        </p>

        <div className="prose prose-invert max-w-none space-y-8">
          <section className="p-6 rounded-lg border bg-card">
            <h2 className="text-2xl font-semibold mb-4">Our Commitment to Open Source</h2>
            <p className="text-muted-foreground leading-relaxed">
              At Argus, we believe in the power of open source software. We use open source libraries
              in our products and contribute back to the community. All the libraries listed below
              are used in accordance with their respective licenses.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              <strong>Note:</strong> Argus itself is proprietary software. However, the underlying
              infrastructure and many of the tools we use are open source. No proprietary code from
              other companies is used in our products.
            </p>
          </section>

          <LicenseTable licenses={frontendDependencies} title="Frontend Dependencies" />
          <LicenseTable licenses={backendDependencies} title="Backend & Worker Dependencies" />
          <LicenseTable licenses={aiDependencies} title="AI & Machine Learning" />

          <section>
            <h2 className="text-2xl font-semibold mb-4">License Summaries</h2>

            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg border">
                <h3 className="font-semibold mb-2">MIT License</h3>
                <p className="text-sm text-muted-foreground">
                  A permissive license that allows commercial use, modification, distribution, and private use.
                  Only requirement is to include the original copyright notice.
                </p>
              </div>

              <div className="p-4 rounded-lg border">
                <h3 className="font-semibold mb-2">Apache License 2.0</h3>
                <p className="text-sm text-muted-foreground">
                  A permissive license similar to MIT but also provides an express grant of patent rights
                  and requires preservation of copyright and license notices.
                </p>
              </div>

              <div className="p-4 rounded-lg border">
                <h3 className="font-semibold mb-2">ISC License</h3>
                <p className="text-sm text-muted-foreground">
                  A permissive license functionally equivalent to MIT, allowing commercial use, modification,
                  and distribution.
                </p>
              </div>

              <div className="p-4 rounded-lg border">
                <h3 className="font-semibold mb-2">BSD Licenses</h3>
                <p className="text-sm text-muted-foreground">
                  Family of permissive licenses with minimal restrictions on redistribution, both commercially
                  and non-commercially.
                </p>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Infrastructure & Services</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              In addition to open source libraries, Argus uses the following third-party services.
              These are commercial services and are not open source:
            </p>
            <ul className="space-y-2 text-muted-foreground">
              <li>
                <strong>Cloudflare Workers:</strong> Serverless compute platform (proprietary)
              </li>
              <li>
                <strong>Cloudflare Browser Rendering:</strong> Headless browser service (proprietary)
              </li>
              <li>
                <strong>Supabase:</strong> Open source database platform (Apache-2.0, with hosted service)
              </li>
              <li>
                <strong>Clerk:</strong> Authentication platform (proprietary)
              </li>
              <li>
                <strong>Anthropic Claude API:</strong> AI model service (proprietary)
              </li>
              <li>
                <strong>TestingBot:</strong> Cross-browser testing service (proprietary)
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Reporting License Issues</h2>
            <p className="text-muted-foreground leading-relaxed">
              If you believe we are using a library incorrectly or have questions about our use
              of open source software, please contact us at{' '}
              <a href="mailto:legal@heyargus.ai" className="text-primary hover:underline">
                legal@heyargus.ai
              </a>.
            </p>
          </section>

          <section className="p-6 rounded-lg border bg-muted/30">
            <h2 className="text-xl font-semibold mb-2">Full License Texts</h2>
            <p className="text-sm text-muted-foreground">
              Complete license texts for all dependencies can be found in the LICENSE files of their
              respective repositories linked above. You can also request a copy of all licenses by
              contacting us at legal@heyargus.ai.
            </p>
          </section>
        </div>

        <div className="mt-12 pt-8 border-t">
          <div className="flex gap-4 text-sm text-muted-foreground">
            <Link href="/legal/terms" className="hover:text-foreground">Terms of Service</Link>
            <Link href="/legal/privacy" className="hover:text-foreground">Privacy Policy</Link>
            <Link href="/legal/security" className="hover:text-foreground">Security</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
