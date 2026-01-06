'use client';

import Link from 'next/link';
import {
  ArrowRight,
  CheckCircle2,
  Zap,
  Shield,
  Brain,
  GitBranch,
  BarChart3,
  Sparkles,
  Play,
  Globe,
  Eye,
  ChevronRight,
  Github,
  Twitter,
  Linkedin,
  ArrowUpRight,
  Layers,
  RefreshCw,
  FileSearch,
  TestTube,
} from 'lucide-react';
import { useState, useEffect } from 'react';
import { AuthButtons } from './auth-buttons';

const features = [
  {
    icon: Brain,
    title: 'AI Test Generation',
    description: 'Transform production errors into comprehensive tests automatically. No manual writing required.',
    color: 'from-teal-500 to-cyan-500',
  },
  {
    icon: RefreshCw,
    title: 'Self-Healing Tests',
    description: 'Tests automatically adapt when your UI changes. Say goodbye to flaky selectors forever.',
    color: 'from-cyan-500 to-blue-500',
  },
  {
    icon: Shield,
    title: 'Predictive Quality',
    description: 'AI identifies potential bugs before they reach production using pattern analysis.',
    color: 'from-emerald-500 to-teal-500',
  },
  {
    icon: GitBranch,
    title: 'CI/CD Native',
    description: 'Seamless integration with GitHub Actions, GitLab CI, CircleCI, and Jenkins.',
    color: 'from-amber-500 to-orange-500',
  },
  {
    icon: BarChart3,
    title: 'Quality Intelligence',
    description: 'Real-time dashboards with AI Quality Score, risk analysis, and trend insights.',
    color: 'from-violet-500 to-purple-500',
  },
  {
    icon: Globe,
    title: 'Cross-Browser',
    description: 'Test on Chrome, Firefox, Safari, Edge, and real iOS/Android devices.',
    color: 'from-pink-500 to-rose-500',
  },
];

const integrations = [
  { name: 'Sentry', desc: 'Error tracking' },
  { name: 'Datadog', desc: 'Monitoring' },
  { name: 'GitHub', desc: 'Source control' },
  { name: 'Slack', desc: 'Notifications' },
  { name: 'Linear', desc: 'Issue tracking' },
  { name: 'Jira', desc: 'Project mgmt' },
  { name: 'PagerDuty', desc: 'Incidents' },
  { name: 'Vercel', desc: 'Deployment' },
];

const pricingPlans = [
  {
    name: 'Free',
    price: '$0',
    period: 'forever',
    description: 'Get started with AI testing',
    features: [
      '100 test runs/month',
      '1 project',
      '3 integrations',
      'Community support',
      'Basic analytics',
    ],
    cta: 'Start Free',
    popular: false,
  },
  {
    name: 'Pro',
    price: '$49',
    period: '/month',
    description: 'For professional developers',
    features: [
      '1,000 test runs/month',
      '5 projects',
      'All integrations',
      'Email support',
      'AI Quality Score',
      'Predictive alerts',
    ],
    cta: 'Start Trial',
    popular: true,
  },
  {
    name: 'Team',
    price: '$199',
    period: '/month',
    description: 'For growing teams',
    features: [
      '10,000 test runs/month',
      'Unlimited projects',
      'Priority support',
      'API access',
      'Custom webhooks',
      'Team analytics',
    ],
    cta: 'Start Trial',
    popular: false,
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'For large organizations',
    features: [
      'Unlimited everything',
      'Dedicated support',
      'SLA guarantees',
      'On-premise option',
      'SSO/SAML',
      'Custom AI models',
    ],
    cta: 'Contact Sales',
    popular: false,
  },
];

const stats = [
  { value: '10x', label: 'Faster Testing', desc: 'vs manual test writing' },
  { value: '70%', label: 'Less Maintenance', desc: 'with self-healing' },
  { value: '99.9%', label: 'Uptime', desc: 'edge-powered reliability' },
  { value: '<50ms', label: 'Response', desc: 'global edge network' },
];

const steps = [
  {
    num: '01',
    title: 'Connect',
    desc: 'Link your error monitoring tools like Sentry or Datadog',
    icon: Layers,
  },
  {
    num: '02',
    title: 'Analyze',
    desc: 'AI identifies patterns and prioritizes test coverage gaps',
    icon: FileSearch,
  },
  {
    num: '03',
    title: 'Generate',
    desc: 'Tests are automatically created and validated',
    icon: TestTube,
  },
  {
    num: '04',
    title: 'Protect',
    desc: 'Continuous monitoring prevents regressions',
    icon: Shield,
  },
];

export function LandingPage() {
  const [activeFeature, setActiveFeature] = useState(0);

  // Auto-rotate features
  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % features.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-3">
              <div className="relative">
                <div className="w-9 h-9 rounded-lg gradient-primary flex items-center justify-center">
                  <Eye className="w-5 h-5 text-white" />
                </div>
                <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-amber-500 rounded-full border-2 border-background" />
              </div>
              <span className="text-xl font-bold tracking-tight">Argus</span>
            </Link>

            {/* Nav Links */}
            <div className="hidden md:flex items-center gap-8">
              <a href="#features" className="text-sm link-subtle">Features</a>
              <a href="#how-it-works" className="text-sm link-subtle">How it Works</a>
              <a href="#integrations" className="text-sm link-subtle">Integrations</a>
              <a href="#pricing" className="text-sm link-subtle">Pricing</a>
              <Link href="/legal" className="text-sm link-subtle">Legal</Link>
            </div>

            {/* Auth Buttons */}
            <AuthButtons variant="nav" />
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-24 px-6 lg:px-8 relative overflow-hidden">
        {/* Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-20 left-1/4 w-[500px] h-[500px] bg-primary/10 rounded-full blur-[120px]" />
          <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-cyan-500/10 rounded-full blur-[100px]" />
          <div className="absolute top-1/2 right-10 w-[200px] h-[200px] bg-amber-500/5 rounded-full blur-[80px]" />
          {/* Grid pattern */}
          <div
            className="absolute inset-0 opacity-[0.02]"
            style={{
              backgroundImage: `linear-gradient(hsl(var(--foreground)) 1px, transparent 1px),
                               linear-gradient(90deg, hsl(var(--foreground)) 1px, transparent 1px)`,
              backgroundSize: '60px 60px',
            }}
          />
        </div>

        <div className="max-w-5xl mx-auto text-center relative">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-8">
            <Sparkles className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-primary">Powered by Advanced AI Models</span>
          </div>

          {/* Headline */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold tracking-tight mb-6 leading-[1.1]">
            <span className="text-foreground">E2E Testing That</span>
            <br />
            <span className="gradient-text">Writes Itself</span>
          </h1>

          {/* Subheadline */}
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
            Generate tests from production errors. Predict bugs before they ship.
            Self-healing tests that adapt to your UI changes automatically.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-20">
            <AuthButtons variant="hero" />
            <button className="group w-full sm:w-auto btn-secondary px-8 py-4 text-base rounded-xl hover-lift">
              <Play className="w-5 h-5" />
              Watch Demo
            </button>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 max-w-4xl mx-auto">
            {stats.map((stat, i) => (
              <div
                key={i}
                className="card-elevated p-6 text-center hover-lift"
              >
                <div className="text-3xl font-bold text-primary mb-1">{stat.value}</div>
                <div className="text-sm font-medium text-foreground mb-1">{stat.label}</div>
                <div className="text-xs text-muted-foreground">{stat.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="section-padding bg-muted/30">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">HOW IT WORKS</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">From Error to Test in Minutes</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Argus automates the entire testing workflow, from error detection to test generation
            </p>
          </div>

          <div className="grid md:grid-cols-4 gap-8">
            {steps.map((step, i) => (
              <div key={i} className="relative">
                <div className="card-elevated-hover p-8 h-full">
                  <div className="text-5xl font-bold text-primary/20 mb-4">{step.num}</div>
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                    <step.icon className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                  <p className="text-sm text-muted-foreground">{step.desc}</p>
                </div>
                {i < 3 && (
                  <ChevronRight className="hidden md:block absolute top-1/2 -right-4 w-8 h-8 text-border -translate-y-1/2 z-10" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="section-padding">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">FEATURES</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Built for Modern Development</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Everything you need to ship quality software faster
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <div
                key={i}
                className={`card-elevated-hover p-8 cursor-default transition-all duration-300 ${
                  activeFeature === i ? 'border-primary/50 shadow-lg shadow-primary/5' : ''
                }`}
                onMouseEnter={() => setActiveFeature(i)}
              >
                <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6`}>
                  <feature.icon className="w-7 h-7 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* AI Quality Score Section */}
      <section className="section-padding bg-muted/30">
        <div className="container-wide px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div>
              <h6 className="text-primary mb-3">AI QUALITY INTELLIGENCE</h6>
              <h2 className="text-3xl sm:text-4xl font-bold mb-6">
                Your Codebase's
                <br />
                <span className="gradient-text">Health Score</span>
              </h2>
              <p className="text-muted-foreground mb-8 leading-relaxed text-lg">
                Get a comprehensive view of your software quality with AI-powered metrics,
                predictive insights, and actionable recommendations.
              </p>
              <ul className="space-y-4">
                {[
                  'Real-time quality monitoring across all projects',
                  'Predictive bug detection using pattern analysis',
                  'Risk-prioritized test recommendations',
                  'Trend analysis and team performance insights',
                ].map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500 flex-shrink-0 mt-0.5" />
                    <span className="text-foreground">{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Quality Score Card */}
            <div className="card-elevated p-8 lg:p-10">
              <div className="text-center mb-10">
                <div className="inline-flex items-center justify-center w-32 h-32 rounded-full bg-gradient-to-br from-emerald-500/20 to-teal-500/20 border-4 border-emerald-500/30 mb-4">
                  <span className="text-5xl font-bold text-emerald-500">A+</span>
                </div>
                <div className="text-lg font-medium">AI Quality Score</div>
                <div className="text-sm text-muted-foreground">Updated 2 min ago</div>
              </div>

              <div className="space-y-6">
                {[
                  { label: 'Test Coverage', value: 94, color: 'bg-emerald-500' },
                  { label: 'Error Resolution', value: 88, color: 'bg-teal-500' },
                  { label: 'Code Health', value: 96, color: 'bg-cyan-500' },
                  { label: 'Risk Score', value: 92, color: 'bg-primary' },
                ].map((item, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-muted-foreground">{item.label}</span>
                      <span className="font-semibold">{item.value}%</span>
                    </div>
                    <div className="h-2 bg-muted rounded-full overflow-hidden">
                      <div
                        className={`h-full ${item.color} rounded-full transition-all duration-1000`}
                        style={{ width: `${item.value}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Integrations */}
      <section id="integrations" className="section-padding">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">INTEGRATIONS</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Works With Your Stack</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Connect your existing tools in minutes with our native integrations
            </p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mb-8">
            {integrations.map((integration, i) => (
              <div
                key={i}
                className="card-elevated-hover p-6 text-center"
              >
                <div className="w-12 h-12 mx-auto mb-3 rounded-xl bg-primary/10 flex items-center justify-center">
                  <span className="text-xl font-bold text-primary">{integration.name[0]}</span>
                </div>
                <div className="font-medium mb-1">{integration.name}</div>
                <div className="text-xs text-muted-foreground">{integration.desc}</div>
              </div>
            ))}
          </div>

          <p className="text-center text-sm text-muted-foreground">
            + FullStory, LogRocket, NewRelic, Bugsnag, Rollbar, and 15 more
          </p>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="section-padding bg-muted/30">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">PRICING</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Simple, Transparent Pricing</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Start free, scale as you grow. No hidden fees.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            {pricingPlans.map((plan, i) => (
              <div
                key={i}
                className={`relative card-elevated p-8 ${
                  plan.popular ? 'border-primary ring-1 ring-primary' : ''
                }`}
              >
                {plan.popular && (
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                    <span className="badge-primary px-4 py-1">Most Popular</span>
                  </div>
                )}

                <div className="mb-8">
                  <h3 className="text-lg font-semibold mb-2">{plan.name}</h3>
                  <div className="flex items-baseline gap-1 mb-2">
                    <span className="text-4xl font-bold">{plan.price}</span>
                    {plan.period && <span className="text-muted-foreground">{plan.period}</span>}
                  </div>
                  <p className="text-sm text-muted-foreground">{plan.description}</p>
                </div>

                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature, j) => (
                    <li key={j} className="flex items-center gap-2 text-sm">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>

                <AuthButtons variant="pricing" ctaText={plan.cta} popular={plan.popular} />
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section-padding">
        <div className="container-tight px-6 lg:px-8">
          <div className="card-elevated p-12 lg:p-16 text-center relative overflow-hidden">
            {/* Background decoration */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="absolute -top-20 -right-20 w-64 h-64 bg-primary/10 rounded-full blur-[80px]" />
              <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-cyan-500/10 rounded-full blur-[80px]" />
            </div>

            <div className="relative">
              <h2 className="text-3xl sm:text-4xl font-bold mb-6">
                Ready to Ship Better Software?
              </h2>
              <p className="text-xl text-muted-foreground mb-10 max-w-xl mx-auto">
                Join developers who trust Argus to catch bugs before users do.
              </p>
              <AuthButtons variant="cta" />
              <p className="text-sm text-muted-foreground mt-4">
                No credit card required
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-16 px-6 lg:px-8 border-t border-border">
        <div className="container-wide">
          <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-12 mb-12">
            {/* Brand */}
            <div className="lg:col-span-2">
              <Link href="/" className="flex items-center gap-3 mb-4">
                <div className="w-9 h-9 rounded-lg gradient-primary flex items-center justify-center">
                  <Eye className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold">Argus</span>
              </Link>
              <p className="text-muted-foreground mb-6 max-w-sm">
                AI-powered E2E testing that writes itself. Ship quality software faster with predictive quality intelligence.
              </p>
              <div className="flex items-center gap-4">
                <a href="https://github.com/heyargus" target="_blank" rel="noopener noreferrer" className="link-subtle p-2 hover:bg-muted rounded-lg transition-colors">
                  <Github className="w-5 h-5" />
                </a>
                <a href="https://twitter.com/heyargus" target="_blank" rel="noopener noreferrer" className="link-subtle p-2 hover:bg-muted rounded-lg transition-colors">
                  <Twitter className="w-5 h-5" />
                </a>
                <a href="https://linkedin.com/company/heyargus" target="_blank" rel="noopener noreferrer" className="link-subtle p-2 hover:bg-muted rounded-lg transition-colors">
                  <Linkedin className="w-5 h-5" />
                </a>
              </div>
            </div>

            {/* Product Links */}
            <div>
              <h6 className="font-semibold mb-4">Product</h6>
              <ul className="space-y-3">
                <li><a href="#features" className="text-sm link-subtle">Features</a></li>
                <li><a href="#pricing" className="text-sm link-subtle">Pricing</a></li>
                <li><a href="#integrations" className="text-sm link-subtle">Integrations</a></li>
                <li><Link href="/legal" className="text-sm link-subtle">Changelog</Link></li>
              </ul>
            </div>

            {/* Resources */}
            <div>
              <h6 className="font-semibold mb-4">Resources</h6>
              <ul className="space-y-3">
                <li>
                  <a href="https://docs.heyargus.ai" target="_blank" rel="noopener noreferrer" className="text-sm link-subtle inline-flex items-center gap-1">
                    Documentation <ArrowUpRight className="w-3 h-3" />
                  </a>
                </li>
                <li><Link href="/legal" className="text-sm link-subtle">API Reference</Link></li>
                <li><Link href="/legal" className="text-sm link-subtle">Blog</Link></li>
                <li><Link href="/legal" className="text-sm link-subtle">Status</Link></li>
              </ul>
            </div>

            {/* Legal */}
            <div>
              <h6 className="font-semibold mb-4">Legal</h6>
              <ul className="space-y-3">
                <li><Link href="/legal/privacy" className="text-sm link-subtle">Privacy Policy</Link></li>
                <li><Link href="/legal/terms" className="text-sm link-subtle">Terms of Service</Link></li>
                <li><Link href="/legal/security" className="text-sm link-subtle">Security</Link></li>
                <li><Link href="/legal/gdpr" className="text-sm link-subtle">GDPR</Link></li>
              </ul>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="pt-8 border-t border-border flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-sm text-muted-foreground">
              © {new Date().getFullYear()} Argus. All rights reserved.
            </p>
            <p className="text-sm text-muted-foreground">
              Built with Cloudflare Workers, Vercel, and AI
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
