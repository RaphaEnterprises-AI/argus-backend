const { isBugEnabled } = require('./registry');

// Stale cart cache for race condition bug
let staleCartCache = null;
let staleCacheTime = 0;

/**
 * Apply flaky bugs. Returns true if request was terminated (killed).
 */
function apply(req, res) {
  // flaky-network: 20% chance to destroy the socket
  if (isBugEnabled('flaky-network') && req.path.startsWith('/api/')) {
    if (Math.random() < 0.2) {
      req.destroy();
      return true;
    }
  }

  return false;
}

/**
 * Apply timing delay. Returns a promise if delay is needed, null otherwise.
 */
function applyDelay(req) {
  // flaky-timing: Random 0-3s delay on every request
  if (isBugEnabled('flaky-timing')) {
    const delay = Math.floor(Math.random() * 3000);
    if (delay > 0) {
      return new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  return null;
}

/**
 * Inject flaky animation scripts into HTML.
 */
function injectScripts(html) {
  // flaky-stale-dom: Inject script that delays DOM text updates
  if (isBugEnabled('flaky-stale-dom')) {
    const staleDomScript = `<script>
(function(){
  var origSet = Object.getOwnPropertyDescriptor(Node.prototype,'textContent').set;
  Object.defineProperty(Node.prototype,'textContent',{
    set:function(v){
      var node=this;
      setTimeout(function(){origSet.call(node,v);},Math.random()*2000+200);
    },
    get:Object.getOwnPropertyDescriptor(Node.prototype,'textContent').get
  });
})();
</script>`;
    html = html.replace('</head>', staleDomScript + '</head>');
  }

  // flaky-animation: Delay button visibility with random animation
  if (isBugEnabled('flaky-animation')) {
    const animationDelay = Math.floor(Math.random() * 2000) + 500; // 500-2500ms
    const animStyle = `<style>
button, [type="submit"], .btn {
  opacity: 0;
  animation: chaosAppear ${animationDelay}ms ease-in forwards;
  pointer-events: none;
  animation-fill-mode: forwards;
}
@keyframes chaosAppear {
  0% { opacity: 0; pointer-events: none; }
  99% { opacity: 0; pointer-events: none; }
  100% { opacity: 1; pointer-events: auto; }
}
</style>`;
    html = html.replace('</head>', animStyle + '</head>');
  }
  return html;
}

/**
 * Get potentially stale cart data for race condition bug.
 * Returns stale data 30% of the time, or null to use fresh data.
 */
function getStaleCart(userId, freshCart) {
  if (!isBugEnabled('flaky-race-condition')) return null;

  const now = Date.now();

  // 30% chance to return stale data
  if (staleCartCache && Math.random() < 0.3) {
    return staleCartCache;
  }

  // Update stale cache with current data (will be stale next time)
  staleCartCache = JSON.parse(JSON.stringify(freshCart || []));
  staleCacheTime = now;

  return null; // Use fresh data
}

module.exports = {
  apply,
  applyDelay,
  injectScripts,
  getStaleCart,
};
