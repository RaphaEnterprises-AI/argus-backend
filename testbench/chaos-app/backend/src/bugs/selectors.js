const { isBugEnabled } = require('./registry');

/**
 * Transform HTML to introduce selector drift bugs.
 * Uses simple string replacement with regex — no DOM parser needed.
 */
function transformHtml(html) {
  // selector-login-id: Replace id="login-btn" with id="sign-in-button"
  if (isBugEnabled('selector-login-id')) {
    html = html.replace(/id="login-btn"/g, 'id="sign-in-button"');
    html = html.replace(/id='login-btn'/g, "id='sign-in-button'");
    // Also update any JS references
    html = html.replace(/#login-btn/g, '#sign-in-button');
  }

  // selector-class-rename: Replace class="add-to-cart" with class="btn-add-cart"
  if (isBugEnabled('selector-class-rename')) {
    html = html.replace(
      /class="([^"]*\b)add-to-cart(\b[^"]*)"/g,
      'class="$1btn-add-cart$2"'
    );
    html = html.replace(
      /class='([^']*\b)add-to-cart(\b[^']*)'/g,
      "class='$1btn-add-cart$2'"
    );
    html = html.replace(/\.add-to-cart/g, '.btn-add-cart');
  }

  // selector-nesting: Wrap targeted buttons in extra DOM layers
  if (isBugEnabled('selector-nesting')) {
    // Wrap buttons with data-testid in extra nesting
    html = html.replace(
      /(<button[^>]*data-testid="([^"]*)"[^>]*>)(.*?)(<\/button>)/g,
      '<div class="chaos-wrapper"><section class="chaos-inner">$1$3$4</section></div>'
    );
    // Also wrap submit buttons
    html = html.replace(
      /(<button[^>]*type="submit"[^>]*>)(.*?)(<\/button>)/g,
      '<div class="chaos-wrapper"><section class="chaos-inner">$1$2$3</section></div>'
    );
  }

  // selector-dynamic-id: Replace id="item-" prefixed IDs with random suffixes
  if (isBugEnabled('selector-dynamic-id')) {
    html = html.replace(/id="item-([^"]*)"/g, () => {
      const randomId = Math.random().toString(36).slice(2, 10);
      return `id="item-${randomId}"`;
    });
    html = html.replace(/id='item-([^']*)'/g, () => {
      const randomId = Math.random().toString(36).slice(2, 10);
      return `id='item-${randomId}'`;
    });
  }

  // selector-removed: Remove checkout button elements entirely
  if (isBugEnabled('selector-removed')) {
    // Remove elements with id="checkout-btn"
    html = html.replace(
      /<button[^>]*id="checkout-btn"[^>]*>.*?<\/button>/g,
      '<!-- checkout button removed by chaos -->'
    );
    // Remove elements with class="checkout-submit"
    html = html.replace(
      /<button[^>]*class="[^"]*checkout-submit[^"]*"[^>]*>.*?<\/button>/g,
      '<!-- checkout submit removed by chaos -->'
    );
    // Also handle self-closing or input variants
    html = html.replace(
      /<input[^>]*id="checkout-btn"[^>]*\/?>/g,
      '<!-- checkout input removed by chaos -->'
    );
  }

  return html;
}

module.exports = {
  transformHtml,
};
