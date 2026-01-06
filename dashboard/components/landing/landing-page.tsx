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
  Code2,
  Terminal,
  MessageSquare,
  ChevronDown,
  Quote,
  Star,
  TrendingUp,
  Search,
  Lightbulb,
  Target,
} from 'lucide-react';
import Image from 'next/image';
import { useState, useEffect } from 'react';
import { AuthButtons } from './auth-buttons';

// IDE Integrations
const ideIntegrations = [
  { name: 'Claude Code', desc: 'MCP Tools', highlight: true },
  { name: 'Cursor', desc: 'AI Editor' },
  { name: 'VS Code', desc: 'Extensions' },
  { name: 'Windsurf', desc: 'AI IDE' },
  { name: 'JetBrains', desc: 'Plugin' },
  { name: 'Zed', desc: 'Coming Soon' },
];

// Trusted companies
const trustedCompanies = [
  'Vercel', 'Supabase', 'Stripe', 'Linear', 'Notion', 'Figma', 'Slack', 'Discord'
];

// Workflow steps (TestSprite-inspired)
const workflowSteps = [
  {
    num: '01',
    title: 'Understand',
    subtitle: 'What You Want',
    desc: 'Instantly parses your codebase, PRDs, and production errors to understand what you\'re actually building.',
    icon: Search,
    color: 'from-violet-500 to-purple-500',
  },
  {
    num: '02',
    title: 'Validate',
    subtitle: 'What You Have',
    desc: 'Generates and runs tests across UI, API, and edge cases to prove everything works as intended.',
    icon: CheckCircle2,
    color: 'from-teal-500 to-cyan-500',
  },
  {
    num: '03',
    title: 'Suggest',
    subtitle: 'What You Need',
    desc: 'Delivers pinpoint feedback and fix recommendations via MCP, so code self-repairs automatically.',
    icon: Lightbulb,
    color: 'from-amber-500 to-orange-500',
  },
  {
    num: '04',
    title: 'Deliver',
    subtitle: 'What You Planned',
    desc: 'Takes AI-generated code from "almost there" to production-ready with full automation.',
    icon: Target,
    color: 'from-emerald-500 to-teal-500',
  },
];

// Testimonials
const testimonials = [
  {
    quote: "Argus transformed how we approach quality. The AI Quality Score gives us confidence before every release.",
    author: "Sarah Chen",
    role: "Engineering Lead",
    company: "TechCorp",
  },
  {
    quote: "The MCP integration with Claude Code is game-changing. I can ask about quality risks right in my editor.",
    author: "Marcus Johnson",
    role: "Senior Developer",
    company: "StartupXYZ",
  },
  {
    quote: "We reduced our QA cycle from days to hours. The self-healing tests alone saved us 20 hours a week.",
    author: "Emily Rodriguez",
    role: "QA Manager",
    company: "Enterprise Co",
  },
  {
    quote: "Finally, a tool that doesn't try to replace our existing stack. It just makes everything smarter.",
    author: "David Kim",
    role: "CTO",
    company: "InnovateLabs",
  },
];

// FAQ items
const faqItems = [
  {
    question: "How does Argus integrate with my existing tools?",
    answer: "Argus deploys 'probes' that connect to your existing tools (Sentry, GitHub, Jest, Playwright, etc.) via webhooks and APIs. We don't replace anything—we correlate data from all your tools into one unified quality intelligence brain."
  },
  {
    question: "What is the AI Quality Score?",
    answer: "The AI Quality Score is a 0-100 metric that combines test coverage, production error rates, CI/CD success rates, and flaky test detection into a single, actionable number. It updates in real-time as your codebase changes."
  },
  {
    question: "How do MCP tools work with Claude Code?",
    answer: "Our MCP server exposes tools like get_quality_score, get_recommendations, and get_risk_files that Claude Code can call directly. This means you can ask Claude about quality risks, get test suggestions, and understand coverage gaps without leaving your editor."
  },
  {
    question: "Is there a free tier?",
    answer: "Yes! Argus Core is free forever. You get the AI Quality Score dashboard, 3 integrations, production error monitoring, and basic risk analysis. Upgrade to Pro for AI test generation and unlimited integrations."
  },
  {
    question: "How does self-healing work?",
    answer: "When a test fails due to selector changes (not actual bugs), Argus analyzes the DOM diff and automatically suggests or applies fixes. This eliminates the #1 cause of flaky tests—stale selectors."
  },
  {
    question: "Can I use Argus with my CI/CD pipeline?",
    answer: "Absolutely. Argus has native integrations with GitHub Actions, GitLab CI, CircleCI, and Jenkins. We receive webhooks on every CI run and correlate test results with code changes automatically."
  },
];

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
    name: 'Core',
    price: '$0',
    period: 'forever',
    description: 'Quality intelligence basics',
    features: [
      'AI Quality Score dashboard',
      '3 integrations (Sentry, GitHub, etc.)',
      '1 project',
      'Production error monitoring',
      'Basic risk analysis',
      'Community support',
    ],
    cta: 'Start Free',
    popular: false,
    badge: null,
  },
  {
    name: 'Pro',
    price: '$49',
    period: '/month',
    description: 'Core + AI Test Generation',
    features: [
      'Everything in Core',
      'AI Test Generation from errors',
      'Unlimited integrations',
      '5 projects',
      'MCP tools for Claude Code',
      'Email support',
    ],
    cta: 'Start Trial',
    popular: true,
    badge: 'Core + Generate',
  },
  {
    name: 'Team',
    price: '$149',
    period: '/month',
    description: 'Pro + Self-Healing Tests',
    features: [
      'Everything in Pro',
      'Self-healing test maintenance',
      'Flaky test detection',
      'Unlimited projects',
      'CI/CD deep integration',
      'Priority support',
    ],
    cta: 'Start Trial',
    popular: false,
    badge: 'Core + Generate + Heal',
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    period: '',
    description: 'Full platform access',
    features: [
      'Everything in Team',
      'Visual regression testing',
      'Edge case generation',
      'Custom AI models',
      'SSO/SAML & SLA',
      'Dedicated support',
    ],
    cta: 'Contact Sales',
    popular: false,
    badge: 'All Products',
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
              <a href="#products" className="text-sm link-subtle">Products</a>
              <a href="#features" className="text-sm link-subtle">Features</a>
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
            <span className="text-foreground">See Everything.</span>
            <br />
            <span className="gradient-text">Ship with Confidence.</span>
          </h1>

          {/* Subheadline */}
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed">
            The all-seeing quality intelligence platform. We deploy probes at every layer of your SDLC,
            correlating data and predicting issues before they reach production.
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

      {/* IDE Integrations - TestSprite inspired */}
      <section className="py-16 px-6 lg:px-8 border-y border-border/50">
        <div className="max-w-5xl mx-auto">
          <p className="text-center text-sm text-muted-foreground mb-8">
            Seamlessly integrates with your favorite AI-powered editors
          </p>
          <div className="flex flex-wrap items-center justify-center gap-6 md:gap-10">
            {ideIntegrations.map((ide, i) => (
              <div
                key={i}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                  ide.highlight
                    ? 'bg-primary/10 border border-primary/30'
                    : 'hover:bg-muted'
                }`}
              >
                <Code2 className={`w-5 h-5 ${ide.highlight ? 'text-primary' : 'text-muted-foreground'}`} />
                <div>
                  <span className={`font-medium ${ide.highlight ? 'text-primary' : ''}`}>{ide.name}</span>
                  <span className="text-xs text-muted-foreground ml-2">{ide.desc}</span>
                </div>
                {ide.highlight && (
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-primary text-primary-foreground ml-1">NEW</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Social Proof - Trusted By */}
      <section className="py-12 px-6 lg:px-8 bg-muted/20">
        <div className="max-w-5xl mx-auto text-center">
          <p className="text-sm text-muted-foreground mb-6">
            Trusted by developers at companies worldwide
          </p>
          <div className="flex flex-wrap items-center justify-center gap-8 md:gap-12 opacity-60">
            {trustedCompanies.map((company, i) => (
              <span key={i} className="text-lg font-semibold text-muted-foreground">
                {company}
              </span>
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

      {/* Products */}
      <section id="products" className="section-padding">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">THE ARGUS SUITE</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Probes for Every Layer</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Modular products that integrate with your existing tools, not replace them
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-4 max-w-6xl mx-auto">
            {[
              { name: 'Core', icon: Eye, desc: 'Quality Intelligence Brain - AI Quality Score, risk analysis, unified dashboard', color: 'from-teal-500 to-cyan-500', free: true },
              { name: 'Generate', icon: Sparkles, desc: 'AI Test Generation - Convert production errors into tests automatically', color: 'from-violet-500 to-purple-500', free: false },
              { name: 'Heal', icon: RefreshCw, desc: 'Self-Healing Tests - Automatic selector and assertion repair', color: 'from-amber-500 to-orange-500', free: false },
              { name: 'Visual', icon: Eye, desc: 'Visual Regression - AI-powered screenshot comparison', color: 'from-pink-500 to-rose-500', free: false },
              { name: 'Edge', icon: Zap, desc: 'Edge Testing - Performance, chaos, and boundary testing', color: 'from-emerald-500 to-teal-500', free: false },
            ].map((product, i) => (
              <div key={i} className="card-elevated-hover p-6 text-center relative">
                {product.free && (
                  <div className="absolute -top-2 -right-2">
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-500 border border-emerald-500/30">FREE</span>
                  </div>
                )}
                <div className={`w-12 h-12 mx-auto rounded-xl bg-gradient-to-br ${product.color} flex items-center justify-center mb-4`}>
                  <product.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-semibold mb-2">Argus {product.name}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">{product.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="section-padding bg-muted/30">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">CAPABILITIES</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Intelligence at Every Layer</h2>
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

      {/* Workflow Section - TestSprite inspired */}
      <section className="section-padding">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">THE ARGUS WAY</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">
              From Broken Code to<br />
              <span className="gradient-text">Working Software</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Argus&apos;s automated testing and feedback loop turns even the most broken code into fully working, release-ready software.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            {workflowSteps.map((step, i) => (
              <div key={i} className="card-elevated-hover p-8 text-center relative group">
                <div className={`w-14 h-14 mx-auto rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <step.icon className="w-7 h-7 text-white" />
                </div>
                <div className="text-xs font-bold text-primary/60 mb-2">{step.num}</div>
                <h3 className="text-xl font-bold mb-1">{step.title}</h3>
                <p className="text-sm text-primary font-medium mb-3">{step.subtitle}</p>
                <p className="text-sm text-muted-foreground leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Comparison - TestSprite inspired */}
      <section className="section-padding bg-muted/30">
        <div className="container-tight px-6 lg:px-8">
          <div className="card-elevated p-12 lg:p-16 relative overflow-hidden">
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="absolute -top-20 -right-20 w-64 h-64 bg-emerald-500/10 rounded-full blur-[80px]" />
              <div className="absolute -bottom-20 -left-20 w-64 h-64 bg-primary/10 rounded-full blur-[80px]" />
            </div>

            <div className="relative grid lg:grid-cols-2 gap-12 items-center">
              <div>
                <h6 className="text-primary mb-3">BOOST ACCURACY</h6>
                <h2 className="text-3xl sm:text-4xl font-bold mb-6">
                  Deliver What<br />
                  <span className="gradient-text">You Planned</span>
                </h2>
                <p className="text-muted-foreground mb-8 leading-relaxed">
                  Take AI-generated code from &quot;almost there&quot; to producing reliable, feature-complete results with full automation.
                </p>
                <AuthButtons variant="hero" />
              </div>

              <div className="space-y-6">
                <div className="flex items-center gap-6">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-muted-foreground">With Coding Agent Only</span>
                      <span className="text-2xl font-bold text-muted-foreground">42%</span>
                    </div>
                    <div className="h-4 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-muted-foreground/30 rounded-full" style={{ width: '42%' }} />
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-emerald-500">With Argus</span>
                      <span className="text-2xl font-bold text-emerald-500">93%</span>
                    </div>
                    <div className="h-4 bg-muted rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full transition-all duration-1000" style={{ width: '93%' }} />
                    </div>
                  </div>
                </div>

                <div className="text-center pt-4">
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                    <TrendingUp className="w-4 h-4 text-emerald-500" />
                    <span className="text-sm font-medium text-emerald-500">+121% Features Delivered</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="section-padding">
        <div className="container-wide px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">TESTIMONIALS</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">The New Way to Validate Software</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Organizations and developers worldwide use Argus to ship faster and focus on what matters most.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
            {testimonials.map((testimonial, i) => (
              <div key={i} className="card-elevated-hover p-6">
                <Quote className="w-8 h-8 text-primary/20 mb-4" />
                <p className="text-sm text-muted-foreground mb-6 leading-relaxed italic">
                  &quot;{testimonial.quote}&quot;
                </p>
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                    <span className="text-sm font-bold text-primary">{testimonial.author[0]}</span>
                  </div>
                  <div>
                    <div className="font-medium text-sm">{testimonial.author}</div>
                    <div className="text-xs text-muted-foreground">{testimonial.role} · {testimonial.company}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
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
                  <h3 className="text-lg font-semibold mb-2">Argus {plan.name}</h3>
                  <div className="flex items-baseline gap-1 mb-2">
                    <span className="text-4xl font-bold">{plan.price}</span>
                    {plan.period && <span className="text-muted-foreground">{plan.period}</span>}
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">{plan.description}</p>
                  {plan.badge && (
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
                      {plan.badge}
                    </span>
                  )}
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
                Ship With Confidence.<br />
                <span className="gradient-text">Automate Your Testing With AI.</span>
              </h2>
              <p className="text-xl text-muted-foreground mb-10 max-w-xl mx-auto">
                Join developers who trust Argus to see everything and catch bugs before users do.
              </p>
              <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                <AuthButtons variant="cta" />
                <a
                  href="https://discord.gg/argus"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-secondary px-6 py-3 text-base rounded-xl hover-lift"
                >
                  <MessageSquare className="w-5 h-5" />
                  Join Discord
                </a>
              </div>
              <p className="text-sm text-muted-foreground mt-4">
                No credit card required · Free forever tier
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="section-padding bg-muted/30">
        <div className="container-tight px-6 lg:px-8">
          <div className="text-center mb-16">
            <h6 className="text-primary mb-3">FAQ</h6>
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Frequently Asked Questions</h2>
          </div>

          <div className="space-y-4 max-w-3xl mx-auto">
            {faqItems.map((item, i) => (
              <details key={i} className="group card-elevated">
                <summary className="flex items-center justify-between p-6 cursor-pointer list-none">
                  <span className="font-medium pr-4">{item.question}</span>
                  <ChevronDown className="w-5 h-5 text-muted-foreground group-open:rotate-180 transition-transform flex-shrink-0" />
                </summary>
                <div className="px-6 pb-6 pt-0">
                  <p className="text-muted-foreground leading-relaxed">{item.answer}</p>
                </div>
              </details>
            ))}
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
                The all-seeing quality intelligence platform. Probes at every layer of your SDLC, unified in one AI-powered brain.
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
