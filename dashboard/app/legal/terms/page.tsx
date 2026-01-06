'use client';

import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';

export default function TermsOfServicePage() {
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

        <h1 className="text-4xl font-bold mb-2">Terms of Service</h1>
        <p className="text-muted-foreground mb-8">Last updated: December 30, 2024</p>

        <div className="prose prose-invert max-w-none space-y-8">
          <section>
            <h2 className="text-2xl font-semibold mb-4">1. Agreement to Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              By accessing or using Argus (&quot;Service&quot;), operated by Argus Technologies (&quot;Company,&quot; &quot;we,&quot; &quot;us,&quot; or &quot;our&quot;),
              you agree to be bound by these Terms of Service (&quot;Terms&quot;). If you disagree with any part of the terms,
              you may not access the Service.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              The Service is available at heyargus.ai and related domains. These Terms apply to all visitors,
              users, and others who access or use the Service.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">2. Description of Service</h2>
            <p className="text-muted-foreground leading-relaxed">
              Argus is an AI-powered end-to-end testing platform that provides:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Autonomous browser automation and testing</li>
              <li>Natural language test creation and execution</li>
              <li>Cross-browser and cross-device testing capabilities</li>
              <li>Visual regression testing and element discovery</li>
              <li>Integration with CI/CD pipelines and observability platforms</li>
              <li>API access for programmatic test management</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">3. Account Registration</h2>
            <p className="text-muted-foreground leading-relaxed">
              To use certain features of the Service, you must register for an account. When you register:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>You must provide accurate, current, and complete information</li>
              <li>You must maintain the security of your account credentials</li>
              <li>You must immediately notify us of any unauthorized access</li>
              <li>You are responsible for all activities under your account</li>
              <li>You must be at least 18 years old or the age of majority in your jurisdiction</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">4. Acceptable Use</h2>
            <p className="text-muted-foreground leading-relaxed">
              You agree to use the Service only for lawful purposes. You shall not:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Use the Service to test websites or applications you do not own or have authorization to test</li>
              <li>Attempt to disrupt, overload, or impair the Service or its infrastructure</li>
              <li>Use the Service to engage in any form of hacking, penetration testing, or security testing on unauthorized systems</li>
              <li>Transmit any malware, viruses, or harmful code through the Service</li>
              <li>Use the Service to harvest data from websites without proper authorization</li>
              <li>Violate any applicable laws or regulations</li>
              <li>Infringe upon the intellectual property rights of others</li>
              <li>Resell or redistribute the Service without authorization</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">5. API Usage and Rate Limits</h2>
            <p className="text-muted-foreground leading-relaxed">
              Access to the Argus API is subject to the following terms:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>API keys are confidential and must not be shared or exposed publicly</li>
              <li>Rate limits apply based on your subscription tier</li>
              <li>We reserve the right to throttle or suspend API access for abuse</li>
              <li>API responses and data remain our intellectual property</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">6. Subscription and Payments</h2>
            <p className="text-muted-foreground leading-relaxed">
              Certain features of the Service require a paid subscription:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Subscriptions are billed in advance on a monthly or annual basis</li>
              <li>Prices are subject to change with 30 days advance notice</li>
              <li>All fees are non-refundable unless otherwise stated</li>
              <li>You may cancel your subscription at any time; access continues until the end of the billing period</li>
              <li>We may offer promotional pricing which may be subject to additional terms</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">7. Intellectual Property</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Service and its original content, features, and functionality are owned by Argus Technologies
              and are protected by international copyright, trademark, and other intellectual property laws.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              You retain all rights to your data and content uploaded to the Service. By using the Service,
              you grant us a limited license to process your data solely for providing the Service.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">8. Third-Party Services</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Service may integrate with third-party services including:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Browser rendering providers (Cloudflare, TestingBot)</li>
              <li>AI/ML providers (Anthropic, OpenAI)</li>
              <li>Observability platforms (Datadog, Sentry, etc.)</li>
              <li>CI/CD platforms (GitHub Actions, GitLab CI)</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              Your use of these integrations is subject to their respective terms of service.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">9. Data Processing and Security</h2>
            <p className="text-muted-foreground leading-relaxed">
              We take data security seriously:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>All data is encrypted in transit and at rest</li>
              <li>We maintain industry-standard security practices</li>
              <li>Browser sessions are isolated and ephemeral</li>
              <li>Test results and screenshots are retained according to your plan settings</li>
              <li>For enterprise customers, we offer data processing agreements (DPA)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">10. Disclaimers</h2>
            <p className="text-muted-foreground leading-relaxed">
              THE SERVICE IS PROVIDED &quot;AS IS&quot; AND &quot;AS AVAILABLE&quot; WITHOUT WARRANTIES OF ANY KIND,
              EITHER EXPRESS OR IMPLIED. WE DO NOT WARRANT THAT THE SERVICE WILL BE UNINTERRUPTED,
              SECURE, OR ERROR-FREE.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              AI-generated test results and recommendations should be reviewed by qualified personnel
              before being used in production environments.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">11. Limitation of Liability</h2>
            <p className="text-muted-foreground leading-relaxed">
              TO THE MAXIMUM EXTENT PERMITTED BY LAW, ARGUS TECHNOLOGIES SHALL NOT BE LIABLE FOR ANY
              INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, INCLUDING BUT NOT
              LIMITED TO LOSS OF PROFITS, DATA, OR BUSINESS OPPORTUNITIES.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              Our total liability shall not exceed the amount you paid for the Service in the twelve
              months preceding the claim.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">12. Indemnification</h2>
            <p className="text-muted-foreground leading-relaxed">
              You agree to defend, indemnify, and hold harmless Argus Technologies from any claims,
              damages, or expenses arising from your use of the Service or violation of these Terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">13. Termination</h2>
            <p className="text-muted-foreground leading-relaxed">
              We may terminate or suspend your account immediately, without prior notice, for conduct
              that we believe violates these Terms or is harmful to other users, us, or third parties,
              or for any other reason at our discretion.
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              Upon termination, your right to use the Service will immediately cease. You may request
              export of your data within 30 days of termination.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">14. Changes to Terms</h2>
            <p className="text-muted-foreground leading-relaxed">
              We reserve the right to modify or replace these Terms at any time. We will provide at
              least 30 days notice before any material changes take effect. Continued use of the
              Service after changes constitutes acceptance of the new Terms.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">15. Governing Law</h2>
            <p className="text-muted-foreground leading-relaxed">
              These Terms shall be governed by the laws of the State of Delaware, United States,
              without regard to its conflict of law provisions. Any disputes shall be resolved in
              the courts of Delaware.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">16. Contact Information</h2>
            <p className="text-muted-foreground leading-relaxed">
              For questions about these Terms, please contact us at:
            </p>
            <div className="mt-4 p-4 rounded-lg bg-muted/30 border">
              <p className="font-medium">Argus Technologies</p>
              <p className="text-muted-foreground">Email: legal@heyargus.ai</p>
              <p className="text-muted-foreground">Website: https://heyargus.ai</p>
            </div>
          </section>
        </div>

        <div className="mt-12 pt-8 border-t">
          <div className="flex gap-4 text-sm text-muted-foreground">
            <Link href="/legal/privacy" className="hover:text-foreground">Privacy Policy</Link>
            <Link href="/legal/security" className="hover:text-foreground">Security</Link>
            <Link href="/legal/licenses" className="hover:text-foreground">Open Source Licenses</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
