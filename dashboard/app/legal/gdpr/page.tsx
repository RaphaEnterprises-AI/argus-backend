'use client';

import Link from 'next/link';
import { ArrowLeft, FileText, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';

export default function GDPRPage() {
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
          <FileText className="h-10 w-10 text-primary" />
          <h1 className="text-4xl font-bold">GDPR Data Processing Agreement</h1>
        </div>
        <p className="text-muted-foreground mb-8">Last updated: December 30, 2024</p>

        <div className="prose prose-invert max-w-none space-y-8">
          <section className="p-6 rounded-lg border bg-primary/5">
            <h2 className="text-xl font-semibold mb-4">Request a DPA</h2>
            <p className="text-muted-foreground leading-relaxed mb-4">
              Enterprise customers requiring a signed Data Processing Agreement (DPA) can request one
              by contacting our legal team. We provide standard DPAs that include the EU Standard
              Contractual Clauses (SCCs).
            </p>
            <div className="flex gap-4">
              <Button asChild>
                <a href="mailto:legal@heyargus.ai?subject=DPA Request">
                  <FileText className="mr-2 h-4 w-4" />
                  Request DPA
                </a>
              </Button>
              <Button variant="outline" disabled>
                <Download className="mr-2 h-4 w-4" />
                Download Template (Coming Soon)
              </Button>
            </div>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">1. Introduction</h2>
            <p className="text-muted-foreground leading-relaxed">
              This Data Processing Agreement (&quot;DPA&quot;) forms part of the Terms of Service (&quot;Agreement&quot;)
              between Argus Technologies (&quot;Processor&quot;) and you (&quot;Controller&quot;) for the use of the
              Argus platform (&quot;Service&quot;).
            </p>
            <p className="text-muted-foreground leading-relaxed mt-4">
              This DPA reflects the parties&apos; agreement with respect to the processing of Personal Data
              by the Processor on behalf of the Controller in accordance with the requirements of
              the General Data Protection Regulation (EU) 2016/679 (&quot;GDPR&quot;).
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">2. Definitions</h2>
            <ul className="space-y-3 text-muted-foreground">
              <li>
                <strong className="text-foreground">&quot;Personal Data&quot;</strong> means any information relating to
                an identified or identifiable natural person as defined in Article 4(1) of the GDPR.
              </li>
              <li>
                <strong className="text-foreground">&quot;Processing&quot;</strong> means any operation performed on
                Personal Data, including collection, storage, use, disclosure, or deletion.
              </li>
              <li>
                <strong className="text-foreground">&quot;Data Subject&quot;</strong> means the individual to whom
                Personal Data relates.
              </li>
              <li>
                <strong className="text-foreground">&quot;Sub-processor&quot;</strong> means any third party engaged
                by the Processor to process Personal Data on behalf of the Controller.
              </li>
              <li>
                <strong className="text-foreground">&quot;Standard Contractual Clauses&quot; or &quot;SCCs&quot;</strong> means
                the contractual clauses adopted by the European Commission for international data transfers.
              </li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">3. Scope of Processing</h2>

            <h3 className="text-xl font-medium mb-3 mt-6">3.1 Subject Matter</h3>
            <p className="text-muted-foreground leading-relaxed">
              The Processor will process Personal Data as necessary to provide the Service, which includes
              AI-powered end-to-end testing, browser automation, and related functionality.
            </p>

            <h3 className="text-xl font-medium mb-3 mt-6">3.2 Duration</h3>
            <p className="text-muted-foreground leading-relaxed">
              Processing will continue for the duration of the Agreement, unless otherwise agreed in writing.
            </p>

            <h3 className="text-xl font-medium mb-3 mt-6">3.3 Nature and Purpose</h3>
            <p className="text-muted-foreground leading-relaxed">
              The purpose of processing is to:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Execute browser-based tests on Controller&apos;s applications</li>
              <li>Capture screenshots and recordings of test executions</li>
              <li>Analyze page content for test automation purposes</li>
              <li>Generate test reports and analytics</li>
              <li>Provide support and maintain the Service</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">3.4 Categories of Data Subjects</h3>
            <p className="text-muted-foreground leading-relaxed">
              Data Subjects may include:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>End users of Controller&apos;s applications being tested</li>
              <li>Controller&apos;s employees who use the Service</li>
              <li>Individuals whose data appears on pages being tested</li>
            </ul>

            <h3 className="text-xl font-medium mb-3 mt-6">3.5 Types of Personal Data</h3>
            <p className="text-muted-foreground leading-relaxed">
              Personal Data processed may include:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Names, email addresses, and other identifiers visible on tested pages</li>
              <li>User-generated content captured in screenshots</li>
              <li>IP addresses and device information from test executions</li>
              <li>Login credentials used for test authentication (encrypted)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">4. Controller Obligations</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Controller shall:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Ensure lawful basis for processing Personal Data through the Service</li>
              <li>Provide clear instructions regarding the processing of Personal Data</li>
              <li>Obtain necessary consents from Data Subjects where required</li>
              <li>Avoid testing pages containing sensitive personal data unless necessary</li>
              <li>Configure appropriate data retention settings</li>
              <li>Notify Processor promptly of any data subject requests</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">5. Processor Obligations</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Processor shall:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Process Personal Data only on documented instructions from the Controller</li>
              <li>Ensure persons authorized to process Personal Data are bound by confidentiality</li>
              <li>Implement appropriate technical and organizational security measures</li>
              <li>Not engage Sub-processors without prior written authorization</li>
              <li>Assist the Controller in responding to Data Subject requests</li>
              <li>Delete or return all Personal Data at the end of the Agreement</li>
              <li>Make available all information necessary to demonstrate compliance</li>
              <li>Allow and contribute to audits conducted by the Controller</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">6. Security Measures</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Processor implements the following security measures:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Encryption of Personal Data in transit (TLS 1.3) and at rest (AES-256)</li>
              <li>Access controls and authentication mechanisms</li>
              <li>Regular security testing and vulnerability assessments</li>
              <li>Incident detection and response procedures</li>
              <li>Employee security training and awareness programs</li>
              <li>Physical security controls at data center facilities</li>
              <li>Business continuity and disaster recovery plans</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              See our <Link href="/legal/security" className="text-primary hover:underline">Security page</Link> for
              detailed information about our security practices.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">7. Sub-processors</h2>

            <h3 className="text-xl font-medium mb-3 mt-6">7.1 Authorized Sub-processors</h3>
            <p className="text-muted-foreground leading-relaxed mb-4">
              The Controller hereby authorizes the engagement of the following Sub-processors:
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-4">Sub-processor</th>
                    <th className="text-left py-3 px-4">Purpose</th>
                    <th className="text-left py-3 px-4">Location</th>
                  </tr>
                </thead>
                <tbody className="text-muted-foreground">
                  <tr className="border-b">
                    <td className="py-3 px-4">Cloudflare, Inc.</td>
                    <td className="py-3 px-4">Hosting, CDN, Browser Rendering</td>
                    <td className="py-3 px-4">USA (Global Edge)</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Supabase, Inc.</td>
                    <td className="py-3 px-4">Database Storage</td>
                    <td className="py-3 px-4">USA (AWS)</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Clerk, Inc.</td>
                    <td className="py-3 px-4">Authentication</td>
                    <td className="py-3 px-4">USA</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Anthropic, PBC</td>
                    <td className="py-3 px-4">AI Processing</td>
                    <td className="py-3 px-4">USA</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">TestingBot BVBA</td>
                    <td className="py-3 px-4">Cross-browser Testing</td>
                    <td className="py-3 px-4">Belgium (EU)</td>
                  </tr>
                  <tr className="border-b">
                    <td className="py-3 px-4">Stripe, Inc.</td>
                    <td className="py-3 px-4">Payment Processing</td>
                    <td className="py-3 px-4">USA</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h3 className="text-xl font-medium mb-3 mt-6">7.2 Changes to Sub-processors</h3>
            <p className="text-muted-foreground leading-relaxed">
              The Processor will notify the Controller of any intended changes to Sub-processors,
              giving the Controller an opportunity to object. Updates will be posted to this page.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">8. International Transfers</h2>
            <p className="text-muted-foreground leading-relaxed">
              For transfers of Personal Data outside the EEA, the Processor relies on:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>European Commission Standard Contractual Clauses (Module 2: Controller to Processor)</li>
              <li>Additional safeguards including encryption and access controls</li>
              <li>Data processing agreements with all Sub-processors</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              The SCCs are incorporated into this DPA by reference. A copy of the signed SCCs
              can be requested from legal@heyargus.ai.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">9. Data Subject Rights</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Processor shall assist the Controller in responding to requests from Data Subjects
              exercising their rights under GDPR, including:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Right of access (Article 15)</li>
              <li>Right to rectification (Article 16)</li>
              <li>Right to erasure (Article 17)</li>
              <li>Right to restriction (Article 18)</li>
              <li>Right to data portability (Article 20)</li>
              <li>Right to object (Article 21)</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              The Processor will notify the Controller within 48 hours of receiving any Data Subject
              request related to Personal Data processed under this DPA.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">10. Data Breach Notification</h2>
            <p className="text-muted-foreground leading-relaxed">
              In the event of a Personal Data breach, the Processor shall:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Notify the Controller without undue delay (within 48 hours) of becoming aware of a breach</li>
              <li>Provide sufficient information for the Controller to meet its notification obligations</li>
              <li>Cooperate with the Controller in investigating and mitigating the breach</li>
              <li>Take reasonable steps to contain and remediate the breach</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">11. Data Retention and Deletion</h2>
            <p className="text-muted-foreground leading-relaxed">
              Upon termination of the Agreement or upon Controller&apos;s request:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>The Processor will delete all Personal Data within 30 days</li>
              <li>Controller may request export of data before deletion</li>
              <li>Deletion will be certified upon request</li>
              <li>Backups will be deleted according to the backup retention schedule (max 90 days)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">12. Audits</h2>
            <p className="text-muted-foreground leading-relaxed">
              The Processor shall make available to the Controller all information necessary to
              demonstrate compliance with this DPA. The Controller may conduct audits or inspections,
              subject to:
            </p>
            <ul className="list-disc pl-6 mt-2 space-y-2 text-muted-foreground">
              <li>Reasonable advance notice (minimum 30 days)</li>
              <li>Reasonable scope and duration</li>
              <li>Confidentiality obligations</li>
              <li>Non-interference with operations</li>
            </ul>
            <p className="text-muted-foreground leading-relaxed mt-4">
              Alternatively, the Processor may provide third-party audit reports (e.g., SOC 2)
              to satisfy audit requirements.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4">13. Contact</h2>
            <p className="text-muted-foreground leading-relaxed">
              For questions about this DPA or to request a signed copy:
            </p>
            <div className="mt-4 p-4 rounded-lg bg-muted/30 border">
              <p className="font-medium">Argus Technologies - Data Protection</p>
              <p className="text-muted-foreground">Email: dpa@heyargus.ai</p>
              <p className="text-muted-foreground">Legal: legal@heyargus.ai</p>
            </div>
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
