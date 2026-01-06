'use client';

import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

export default function PrivacyPolicyPage() {
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

        <h1 className="text-4xl font-bold mb-2">Privacy Policy</h1>
        <p className="text-muted-foreground mb-8">Last updated: December 30, 2024</p>

        <div className="prose prose-invert max-w-none space-y-8">
          <section>
            <h2 className="text-2xl font-semibold mb-4">1. Introduction</h2>
            <p className="text-muted-foreground leading-relaxed">
              Argus Technologies (&quot;Company,&quot; &quot;we,&quot; &quot;us,&quot; or &quot;our&quot;) respects your privacy and is committed
              to protecting your personal data. This Privacy Policy explains how we collect, use, disclose,
              and safeguard your information when you use the Argus platform (&quot;Service&quot;) at heyargus.ai.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              This policy applies to information collected through our website, API, and related services.
              Please read this policy carefully. By using our Service, you consent to the practices described herein.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">2. Information We Collect</h2>

            <h3 className="text-xl font-medium mb-3 mt-6">2.1 Information You Provide</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>Account Information:</strong> Name, email address, company name, and password when you register</li>
              <li><strong>Billing Information:</strong> Payment card details (processed by our payment provider), billing address</li>
              <li><strong>Test Configuration:</strong> URLs, test scripts, and configurations you create</li>
              <li><strong>Support Communications:</strong> Information you provide when contacting support</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">2.2 Information Collected Automatically</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>Usage Data:</strong> Pages visited, features used, test execution metrics</li>
              <li><strong>Device Information:</strong> Browser type, operating system, IP address</li>
              <li><strong>Log Data:</strong> Access times, pages viewed, referring URLs</li>
              <li><strong>Cookies:</strong> Session identifiers, preferences, authentication tokens</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">2.3 Test Execution Data</h3>
            <p className="text-muted-foreground leading-relaxed">
              When you run tests through Argus, we may temporarily process:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Screenshots and recordings of test executions</li>
              <li>Page content and DOM structure of tested pages</li>
              <li>Network requests made during testing</li>
              <li>Test results and error logs</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-2">
              This data is processed ephemerally and retained according to your plan settings (default: 30 days).
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">3. How We Use Your Information</h2>
            <p className="text-muted-foreground leading-relaxed">
              We use collected information for the following purposes:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li><strong>Service Delivery:</strong> To provide, operate, and maintain the Argus platform</li>
              <li><strong>Account Management:</strong> To create and manage your user account</li>
              <li><strong>Communication:</strong> To send service updates, security alerts, and support messages</li>
              <li><strong>Improvement:</strong> To analyze usage patterns and improve our Service</li>
              <li><strong>Security:</strong> To detect, prevent, and address technical issues and security threats</li>
              <li><strong>Compliance:</strong> To comply with legal obligations</li>
              <li><strong>AI Training:</strong> Aggregated, anonymized data may be used to improve our AI models (opt-out available)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">4. Legal Bases for Processing (GDPR)</h2>
            <p className="text-muted-foreground leading-relaxed">
              If you are in the European Economic Area (EEA), we process your personal data based on:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li><strong>Contract Performance:</strong> Processing necessary to provide our Service to you</li>
              <li><strong>Legitimate Interests:</strong> Improving our services, security, and fraud prevention</li>
              <li><strong>Consent:</strong> For marketing communications and optional data processing</li>
              <li><strong>Legal Obligation:</strong> To comply with applicable laws</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">5. Data Sharing and Disclosure</h2>
            <p className="text-muted-foreground leading-relaxed">
              We may share your information with:
            </p>

            <h3 className="text-xl font-medium mb-3 mt-6">5.1 Service Providers</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>Cloud Infrastructure:</strong> Cloudflare, AWS (hosting and CDN)</li>
              <li><strong>Authentication:</strong> Clerk (user authentication)</li>
              <li><strong>Database:</strong> Supabase (data storage)</li>
              <li><strong>AI Providers:</strong> Anthropic, OpenAI (test automation)</li>
              <li><strong>Browser Testing:</strong> TestingBot (cross-browser testing)</li>
              <li><strong>Payment Processing:</strong> Stripe (payment handling)</li>
              <li><strong>Analytics:</strong> Privacy-focused analytics tools</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">5.2 Legal Requirements</h3>
            <p className="text-muted-foreground leading-relaxed">
              We may disclose information if required by law, court order, or governmental authority,
              or if we believe disclosure is necessary to protect our rights, your safety, or the safety of others.
            </p>

            <h3 className="text-xl font-medium mb-3 mt-6">5.3 Business Transfers</h3>
            <p className="text-muted-foreground leading-relaxed">
              In the event of a merger, acquisition, or sale of assets, your information may be transferred.
              We will notify you of any such change in ownership.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">6. International Data Transfers</h2>
            <p className="text-muted-foreground leading-relaxed">
              Your data may be transferred to and processed in countries outside your residence. We ensure
              appropriate safeguards are in place:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Standard Contractual Clauses (SCCs) approved by the European Commission</li>
              <li>Data processing agreements with all service providers</li>
              <li>Adequacy decisions where applicable</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">7. Data Retention</h2>
            <p className="text-muted-foreground leading-relaxed">
              We retain your data for as long as necessary to fulfill the purposes outlined in this policy:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li><strong>Account Data:</strong> Until account deletion plus 30 days</li>
              <li><strong>Test Results:</strong> According to your plan settings (7-90 days)</li>
              <li><strong>Screenshots:</strong> According to your plan settings (7-30 days)</li>
              <li><strong>Logs:</strong> Up to 90 days for operational purposes</li>
              <li><strong>Billing Records:</strong> 7 years for tax and legal compliance</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">8. Your Rights</h2>

            <h3 className="text-xl font-medium mb-3 mt-6">8.1 GDPR Rights (EEA Residents)</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>Access:</strong> Request a copy of your personal data</li>
              <li><strong>Rectification:</strong> Request correction of inaccurate data</li>
              <li><strong>Erasure:</strong> Request deletion of your data (&quot;right to be forgotten&quot;)</li>
              <li><strong>Portability:</strong> Receive your data in a structured, machine-readable format</li>
              <li><strong>Restriction:</strong> Request restriction of processing</li>
              <li><strong>Objection:</strong> Object to processing based on legitimate interests</li>
              <li><strong>Withdraw Consent:</strong> Withdraw consent at any time</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">8.2 CCPA Rights (California Residents)</h3>
            <ul className="list-disc pl-6 space-y-2 text-muted-foreground">
              <li><strong>Know:</strong> Right to know what personal information is collected</li>
              <li><strong>Delete:</strong> Right to request deletion of personal information</li>
              <li><strong>Opt-Out:</strong> Right to opt-out of sale of personal information (we do not sell data)</li>
              <li><strong>Non-Discrimination:</strong> Right to not be discriminated against for exercising rights</li>
            </ul>

            <p className="text-muted-foreground leading-relaxed mt-4">
              To exercise your rights, contact us at privacy@heyargus.ai or use the settings in your account dashboard.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">9. Cookies and Tracking</h2>
            <p className="text-muted-foreground leading-relaxed">
              We use cookies and similar technologies to:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li><strong>Essential Cookies:</strong> Required for authentication and security</li>
              <li><strong>Functional Cookies:</strong> Remember your preferences and settings</li>
              <li><strong>Analytics Cookies:</strong> Understand how you use our Service (anonymized)</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              You can control cookies through your browser settings. Disabling essential cookies may affect functionality.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">10. Security</h2>
            <p className="text-muted-foreground leading-relaxed">
              We implement appropriate technical and organizational measures to protect your data:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Encryption in transit (TLS 1.3) and at rest (AES-256)</li>
              <li>Regular security assessments and penetration testing</li>
              <li>Access controls and audit logging</li>
              <li>Employee security training</li>
              <li>Incident response procedures</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              See our <Link href="/legal/security" className="text-primary hover:underline">Security page</Link> for more details.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">11. Children&apos;s Privacy</h2>
            <p className="text-muted-foreground leading-relaxed">
              Our Service is not intended for children under 16. We do not knowingly collect personal
              information from children. If you believe we have collected data from a child, please
              contact us immediately.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">12. Changes to This Policy</h2>
            <p className="text-muted-foreground leading-relaxed">
              We may update this Privacy Policy from time to time. We will notify you of any material
              changes by posting the new policy on this page and updating the &quot;Last updated&quot; date.
              For significant changes, we will provide additional notice (e.g., email notification).
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">13. Contact Us</h2>
            <p className="text-muted-foreground leading-relaxed">
              For questions, concerns, or to exercise your rights, contact our Data Protection team:
            </p>
            <div className="mt-4 p-4 rounded-lg bg-muted/30 border">
              <p className="font-medium">Argus Technologies - Data Protection</p>
              <p className="text-muted-foreground">Email: privacy@heyargus.ai</p>
              <p className="text-muted-foreground">Website: https://heyargus.ai</p>
            </div>
            <p className="text-muted-foreground leading-relaxed mt-4">
              EEA residents may also contact their local data protection authority.
            </p>
          </section>
        </div>

        <div className="mt-12 pt-8 border-t">
          <div className="flex gap-4 text-sm text-muted-foreground">
            <Link href="/legal/terms" className="hover:text-foreground">Terms of Service</Link>
            <Link href="/legal/security" className="hover:text-foreground">Security</Link>
            <Link href="/legal/gdpr" className="hover:text-foreground">GDPR DPA</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
