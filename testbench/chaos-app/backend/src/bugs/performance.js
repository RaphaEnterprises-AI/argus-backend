const { isBugEnabled } = require('./registry');

/**
 * Pre-route performance bugs (delays).
 * Called before route handlers execute.
 */
function applyPreRoute(req, res) {
  // perf-slow-api: 5s delay on /api/products
  if (isBugEnabled('perf-slow-api') && req.path.startsWith('/api/products')) {
    return new Promise((resolve) => {
      setTimeout(resolve, 5000);
    });
  }
  return null;
}

/**
 * Transform JSON responses for performance bugs.
 */
function transformJson(req, res, data) {
  // perf-large-payload: Pad JSON response to ~10MB
  if (isBugEnabled('perf-large-payload')) {
    const padding = 'X'.repeat(10 * 1024 * 1024); // 10MB of padding
    if (typeof data === 'object' && data !== null) {
      data._padding = padding;
    }
  }
  return data;
}

/**
 * Inject scripts into HTML responses for performance bugs.
 * NOTE: This is an intentional chaos/testing app. The injected scripts
 * are deliberately buggy to test Argus's detection capabilities.
 * No real user data is involved — this is a synthetic test target.
 */
function injectScripts(html) {
  // perf-blocking-render: Inject 3s blocking script
  if (isBugEnabled('perf-blocking-render')) {
    const blockingScript =
      '<script>const __s=Date.now();while(Date.now()-__s<3000){}</script>';
    html = html.replace('</head>', blockingScript + '</head>');
  }

  // perf-memory-leak: Inject script that creates 1000 detached DOM nodes
  if (isBugEnabled('perf-memory-leak')) {
    const leakScript = `<script>
(function(){
  var leaked=[];
  for(var i=0;i<1000;i++){
    var div=document.createElement('div');
    div.textContent='leaked node '+i+' memory leak padding data';
    leaked.push(div);
  }
  window.__leakedNodes=(window.__leakedNodes||[]).concat(leaked);
})();
</script>`;
    html = html.replace('</body>', leakScript + '</body>');
  }

  return html;
}

module.exports = {
  applyPreRoute,
  transformJson,
  injectScripts,
};
