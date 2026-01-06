'use client';

import Link from 'next/link';
import { ArrowLeft, Shield, Lock, Server, Eye, CheckCircle, AlertTriangle } from 'lucide-react';

export default function SecurityPage() {
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

        <div className="flex items-center gap-3 mb-2">
          <Shield className="h-10 w-10 text-primary" />
          <h1 className="text-4xl font-bold">Security & Compliance</h1>
        </div>
        <p className="text-muted-foreground mb-8">Last updated: December 30, 2024</p>

        <div className="prose prose-invert max-w-none space-y-8">
          <section className="p-6 rounded-lg border bg-card">
            <h2 className="text-2xl font-semibold mb-4">Our Security Commitment</h2>
            <p className="text-muted-foreground leading-relaxed">
              At Argus, security is not an afterthought - it&apos;s foundational to everything we build.
              We understand that you&apos;re trusting us with access to your applications and test data.
              This document outlines our security practices, compliance certifications, and commitments.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <Lock className="h-6 w-6 text-primary" />
              Data Encryption
            </h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg border">
                <h3 className="font-medium mb-2">In Transit</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    TLS 1.3 encryption for all connections
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    HSTS enabled with preload
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Perfect forward secrecy
                  </li>
                </ul>
              </div>
              <div className="p-4 rounded-lg border">
                <h3 className="font-medium mb-2">At Rest</h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    AES-256 encryption for all data
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Encrypted backups
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                    Key management via cloud HSM
                  </li>
                </ul>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <Server className="h-6 w-6 text-primary" />
              Infrastructure Security
            </h2>
            <ul className="space-y-3 text-muted-foreground">
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">Cloud Infrastructure:</strong> Hosted on Cloudflare Workers
                  with global edge deployment, automatic DDoS protection, and WAF
                </div>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">Isolated Execution:</strong> Each browser session runs in
                  an isolated environment with no persistence between tests
                </div>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">Network Segmentation:</strong> Production systems are isolated
                  from development and testing environments
                </div>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">Regular Patching:</strong> Systems are automatically updated
                  with security patches within 24-48 hours of release
                </div>
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
              <Eye className="h-6 w-6 text-primary" />
              Access Controls
            </h2>
            <ul className="space-y-3 text-muted-foreground">
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">Authentication:</strong> Secure authentication via Clerk
                  with support for SSO, MFA, and social login
                </div>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">Role-Based Access:</strong> Granular permissions for teams
                  with owner, admin, and member roles
                </div>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">API Security:</strong> API keys with scoped permissions,
                  rate limiting, and audit logging
                </div>
              </li>
              <li className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5" />
                <div>
                  <strong className="text-foreground">Employee Access:</strong> Least-privilege access for employees,
                  with access reviews every 90 days
                </div>
              </li>
            </ul>
          </section>

          <section className="p-6 rounded-lg border bg-primary/5">
            <h2 className="text-2xl font-semibold mb-4">Compliance Certifications</h2>

            <div className="grid md:grid-cols-2 gap-6">
              <div className="p-4 rounded-lg bg-background border">
                <h3 className="font-semibold text-lg mb-2">SOC 2 Type II</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  We are pursuing SOC 2 Type II certification covering Security, Availability,
                  and Confidentiality trust principles.
                </p>
                <span className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-yellow-500/10 text-yellow-500">
                  <AlertTriangle className="h-3 w-3" />
                  In Progress - Q2 2025
                </span>
              </div>

              <div className="p-4 rounded-lg bg-background border">
                <h3 className="font-semibold text-lg mb-2">GDPR Compliant</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  Fully compliant with the EU General Data Protection Regulation. We offer
                  Data Processing Agreements (DPA) for enterprise customers.
                </p>
                <span className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-green-500/10 text-green-500">
                  <CheckCircle className="h-3 w-3" />
                  Compliant
                </span>
              </div>

              <div className="p-4 rounded-lg bg-background border">
                <h3 className="font-semibold text-lg mb-2">CCPA Compliant</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  Compliant with the California Consumer Privacy Act. California residents
                  can exercise their rights through our privacy settings.
                </p>
                <span className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-green-500/10 text-green-500">
                  <CheckCircle className="h-3 w-3" />
                  Compliant
                </span>
              </div>

              <div className="p-4 rounded-lg bg-background border">
                <h3 className="font-semibold text-lg mb-2">HIPAA Ready</h3>
                <p className="text-sm text-muted-foreground mb-3">
                  Our infrastructure supports HIPAA compliance. Business Associate Agreements (BAA)
                  available for healthcare customers on Enterprise plans.
                </p>
                <span className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-full bg-blue-500/10 text-blue-500">
                  <CheckCircle className="h-3 w-3" />
                  Available on Enterprise
                </span>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Security Practices</h2>

            <div className="space-y-4">
              <div className="p-4 rounded-lg border">
                <h3 className="font-medium mb-2">Secure Development Lifecycle</h3>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>- Code review required for all changes</li>
                  <li>- Automated security scanning in CI/CD pipeline</li>
                  <li>- Dependency vulnerability monitoring</li>
                  <li>- Regular security training for developers</li>
                </ul>
              </div>

              <div className="p-4 rounded-lg border">
                <h3 className="font-medium mb-2">Penetration Testing</h3>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>- Annual third-party penetration testing</li>
                  <li>- Continuous vulnerability scanning</li>
                  <li>- Bug bounty program (coming soon)</li>
                </ul>
              </div>

              <div className="p-4 rounded-lg border">
                <h3 className="font-medium mb-2">Incident Response</h3>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>- 24/7 security monitoring and alerting</li>
                  <li>- Documented incident response procedures</li>
                  <li>- Customer notification within 72 hours of confirmed breach</li>
                  <li>- Post-incident review and improvement process</li>
                </ul>
              </div>

              <div className="p-4 rounded-lg border">
                <h3 className="font-medium mb-2">Business Continuity</h3>
                <ul className="space-y-1 text-sm text-muted-foreground">
                  <li>- Multi-region deployment for high availability</li>
                  <li>- Automated backups with point-in-time recovery</li>
                  <li>- 99.9% uptime SLA for Enterprise plans</li>
                  <li>- Disaster recovery plan with annual testing</li>
                </ul>
              </div>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Vendor Security</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              We carefully vet all third-party vendors and require them to meet our security standards:
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-4">Vendor</th>
                    <th className="text-left py-3 px-4">Purpose</th>
                    <th className="text-left py-3 px-4">Certifications</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b">
                    <td className="py-3 px-4">Cloudflare</td>
                    <td className="py-3 px-4">Hosting, CDN, Security</td>
                    <td className="py-3 px-4">SOC 2, ISO 27001, PCI DSS</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Supabase</td>
                    <td className="py-3 px-4">Database</td>
                    <td className="py-3 px-4">SOC 2 Type II, HIPAA</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Clerk</td>
                    <td className="py-3 px-4">Authentication</td>
                    <td className="py-3 px-4">SOC 2 Type II</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Anthropic</td>
                    <td className="py-3 px-4">AI Processing</td>
                    <td className="py-3 px-4">SOC 2 Type II</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Stripe</td>
                    <td className="py-3 px-4">Payment Processing</td>
                    <td className="py-3 px-4">PCI DSS Level 1</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Enterprise Security Features</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              Enterprise plans include additional security features:
            </p>
            <ul className="space-y-2 text-muted-foreground">
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-primary" />
                Single Sign-On (SSO) with SAML 2.0 and OIDC
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-primary" />
                Custom data retention policies
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-primary" />
                IP allowlisting
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-primary" />
                Audit logs with SIEM integration
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-primary" />
                Dedicated infrastructure options
              </li>
              <li className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-primary" />
                Custom DPA and BAA agreements
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Report a Vulnerability</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              We take security vulnerabilities seriously. If you discover a security issue,
              please report it responsibly:
            </p>
            <div className="p-4 rounded-lg bg-muted/30 border">
              <p className="font-medium">Security Team</p>
              <p className="text-muted-foreground">Email: security@heyargus.ai</p>
              <p className="text-muted-foreground mt-2 text-sm">
                Please include a detailed description of the vulnerability, steps to reproduce,
                and any relevant evidence. We aim to respond within 24 hours.
              </p>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">Request Security Documentation</h2>
            <p className="text-muted-foreground leading-relaxed">
              Enterprise customers can request additional security documentation including:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Security questionnaire responses</li>
              <li>Penetration test reports (summary)</li>
              <li>SOC 2 report (when available)</li>
              <li>Insurance certificates</li>
              <li>Data Processing Agreement (DPA)</li>
              <li>Business Associate Agreement (BAA)</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              Contact <a href="mailto:security@heyargus.ai" className="text-primary hover:underline">security@heyargus.ai</a> to
              request documentation.
            </p>
          </section>
        </div>

        <div className="mt-12 pt-8 border-t">
          <div className="flex gap-4 text-sm text-muted-foreground">
            <Link href="/legal/terms" className="hover:text-foreground">Terms of Service</Link>
            <Link href="/legal/privacy" className="hover:text-foreground">Privacy Policy</Link>
            <Link href="/legal/gdpr" className="hover:text-foreground">GDPR DPA</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
