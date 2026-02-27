const { isBugEnabled } = require('../bugs/registry');
const performance = require('../bugs/performance');
const security = require('../bugs/security');
const api = require('../bugs/api');
const flaky = require('../bugs/flaky');
const selectors = require('../bugs/selectors');
const a11y = require('../bugs/a11y');

/**
 * Central chaos middleware that orchestrates all bug modules.
 * Must be mounted BEFORE route handlers.
 */
module.exports = function chaosMiddleware(req, res, next) {
  // 1. Flaky bugs (may kill request entirely)
  if (flaky.apply(req, res)) return; // returns true if request was killed

  // 2. Check for random 500 error
  if (api.checkRandom500(req, res)) return;

  // 3. Build chain of async operations (delays)
  const delays = [];

  // Flaky timing delay
  const flakyDelay = flaky.applyDelay(req);
  if (flakyDelay) delays.push(flakyDelay);

  // Performance slow API delay
  const perfDelay = performance.applyPreRoute(req, res);
  if (perfDelay) delays.push(perfDelay);

  // 4. Apply pagination transformation
  api.transformPagination(req);

  // 5. Security middleware (pre-route modifications)
  security.apply(req, res);

  // 6. Intercept response for HTML/JSON transformations
  const originalSend = res.send.bind(res);
  const originalJson = res.json.bind(res);

  res.json = function (data) {
    // Apply API contract bugs
    data = api.transformJson(req, res, data);
    // Apply performance bugs to JSON
    data = performance.transformJson(req, res, data);
    // Apply security bugs to JSON
    data = security.transformJson(req, res, data);
    return originalJson(data);
  };

  res.send = function (body) {
    if (typeof body === 'string' && body.includes('<!DOCTYPE')) {
      // Apply HTML transformations
      body = selectors.transformHtml(body);
      body = a11y.transformHtml(body);
      body = performance.injectScripts(body);
      body = security.injectScripts(body);
      body = flaky.injectScripts(body);
    }
    return originalSend(body);
  };

  // 7. API status code bugs
  api.interceptStatus(req, res);

  // 8. Wait for any delays, then continue
  if (delays.length > 0) {
    Promise.all(delays).then(() => next());
  } else {
    next();
  }
};
