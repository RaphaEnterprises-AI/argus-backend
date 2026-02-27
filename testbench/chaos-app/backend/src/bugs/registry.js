const BUGS = {
  // Category 1: Selector Drift (5 bugs)
  'selector-login-id': {
    category: 'selectors',
    description: 'Login button #login-btn -> #sign-in-button',
    enabled: false,
  },
  'selector-class-rename': {
    category: 'selectors',
    description: '.add-to-cart -> .btn-add-cart',
    enabled: false,
  },
  'selector-nesting': {
    category: 'selectors',
    description: 'Button moves deeper in DOM',
    enabled: false,
  },
  'selector-dynamic-id': {
    category: 'selectors',
    description: 'Element ID becomes random',
    enabled: false,
  },
  'selector-removed': {
    category: 'selectors',
    description: 'Checkout button deleted',
    enabled: false,
  },

  // Category 2: Performance (4 bugs)
  'perf-slow-api': {
    category: 'performance',
    description: '5s delay on /api/products',
    enabled: false,
  },
  'perf-memory-leak': {
    category: 'performance',
    description: 'Leak DOM nodes in response',
    enabled: false,
  },
  'perf-large-payload': {
    category: 'performance',
    description: '10MB JSON response',
    enabled: false,
  },
  'perf-blocking-render': {
    category: 'performance',
    description: 'Inject 3s blocking script',
    enabled: false,
  },

  // Category 3: Accessibility (5 bugs)
  'a11y-no-alt': {
    category: 'a11y',
    description: 'Remove image alt attributes',
    enabled: false,
  },
  'a11y-low-contrast': {
    category: 'a11y',
    description: 'Text #ccc on #fff',
    enabled: false,
  },
  'a11y-no-labels': {
    category: 'a11y',
    description: 'Remove form labels',
    enabled: false,
  },
  'a11y-no-keyboard': {
    category: 'a11y',
    description: 'Disable keyboard focus',
    enabled: false,
  },
  'a11y-missing-aria': {
    category: 'a11y',
    description: 'Remove ARIA landmarks',
    enabled: false,
  },

  // Category 4: Security (6 bugs)
  'sec-xss-reflected': {
    category: 'security',
    description: 'Search reflects query unescaped',
    enabled: false,
  },
  'sec-sqli': {
    category: 'security',
    description: 'Login concatenates SQL-like input',
    enabled: false,
  },
  'sec-broken-auth': {
    category: 'security',
    description: 'Accept expired JWT',
    enabled: false,
  },
  'sec-idor': {
    category: 'security',
    description: 'No ownership check on /api/users/:id',
    enabled: false,
  },
  'sec-ssrf': {
    category: 'security',
    description: 'Image URL fetcher follows internal URLs',
    enabled: false,
  },
  'sec-secrets-in-code': {
    category: 'security',
    description: 'API key in frontend bundle',
    enabled: false,
  },

  // Category 5: API Contract (5 bugs)
  'api-schema-drift': {
    category: 'api',
    description: 'price changes number->string',
    enabled: false,
  },
  'api-missing-field': {
    category: 'api',
    description: 'Remove email from user response',
    enabled: false,
  },
  'api-wrong-status': {
    category: 'api',
    description: 'POST returns 200 instead of 201',
    enabled: false,
  },
  'api-breaking-pagination': {
    category: 'api',
    description: '?page= -> ?offset=',
    enabled: false,
  },
  'api-500-random': {
    category: 'api',
    description: '10% of requests return 500',
    enabled: false,
  },

  // Category 6: Flaky (5 bugs)
  'flaky-timing': {
    category: 'flaky',
    description: 'Random 0-3s delay',
    enabled: false,
  },
  'flaky-animation': {
    category: 'flaky',
    description: 'Button clickable after random animation delay',
    enabled: false,
  },
  'flaky-race-condition': {
    category: 'flaky',
    description: 'Cart count sometimes stale',
    enabled: false,
  },
  'flaky-network': {
    category: 'flaky',
    description: '20% of API calls drop',
    enabled: false,
  },
  'flaky-stale-dom': {
    category: 'flaky',
    description: 'DOM text not updated after state change',
    enabled: false,
  },
};

const SCENARIOS = {
  'selector-drift': [
    'selector-login-id',
    'selector-class-rename',
    'selector-nesting',
    'selector-dynamic-id',
    'selector-removed',
  ],
  'perf-degradation': [
    'perf-slow-api',
    'perf-memory-leak',
    'perf-large-payload',
    'perf-blocking-render',
  ],
  'a11y-violations': [
    'a11y-no-alt',
    'a11y-low-contrast',
    'a11y-no-labels',
    'a11y-no-keyboard',
    'a11y-missing-aria',
  ],
  'security-vulns': [
    'sec-xss-reflected',
    'sec-sqli',
    'sec-broken-auth',
    'sec-idor',
    'sec-ssrf',
    'sec-secrets-in-code',
  ],
  'api-chaos': [
    'api-schema-drift',
    'api-missing-field',
    'api-wrong-status',
    'api-breaking-pagination',
    'api-500-random',
  ],
  'flaky-everything': [
    'flaky-timing',
    'flaky-animation',
    'flaky-race-condition',
    'flaky-network',
    'flaky-stale-dom',
  ],
  'full-chaos': Object.keys(BUGS),
};

module.exports = {
  getBug: (id) => BUGS[id] || null,
  isBugEnabled: (id) => BUGS[id]?.enabled || false,
  enableBug: (id) => {
    if (BUGS[id]) BUGS[id].enabled = true;
  },
  disableBug: (id) => {
    if (BUGS[id]) BUGS[id].enabled = false;
  },
  resetAll: () => {
    Object.values(BUGS).forEach((b) => (b.enabled = false));
  },
  enableScenario: (name) => {
    (SCENARIOS[name] || []).forEach((id) => {
      if (BUGS[id]) BUGS[id].enabled = true;
    });
  },
  listBugs: () => Object.entries(BUGS).map(([id, bug]) => ({ id, ...bug })),
  getActiveBugCount: () => Object.values(BUGS).filter((b) => b.enabled).length,
  SCENARIOS,
};
