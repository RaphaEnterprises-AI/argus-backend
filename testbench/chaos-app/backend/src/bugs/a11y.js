const { isBugEnabled } = require('./registry');

/**
 * Transform HTML to introduce accessibility violations.
 * Uses simple string replacement with regex.
 */
function transformHtml(html) {
  // a11y-no-alt: Remove alt attributes from all <img> tags
  if (isBugEnabled('a11y-no-alt')) {
    html = html.replace(/(<img\b[^>]*)\s+alt="[^"]*"/g, '$1');
    html = html.replace(/(<img\b[^>]*)\s+alt='[^']*'/g, '$1');
  }

  // a11y-low-contrast: Inject styles that make text nearly invisible on white bg
  if (isBugEnabled('a11y-low-contrast')) {
    const lowContrastStyle =
      '<style>body{color:#ccc!important;background:#fff!important}a{color:#ddd!important}p,span,div,li,td,th{color:#ccc!important}h1,h2,h3,h4,h5,h6{color:#bbb!important}</style>';
    html = html.replace('</head>', lowContrastStyle + '</head>');
  }

  // a11y-no-labels: Remove all <label> elements
  if (isBugEnabled('a11y-no-labels')) {
    // Remove label tags but keep their content (children)
    html = html.replace(/<label\b[^>]*>(.*?)<\/label>/gs, '$1');
  }

  // a11y-no-keyboard: Add tabindex="-1" to interactive elements
  if (isBugEnabled('a11y-no-keyboard')) {
    // Add tabindex="-1" to <a> tags
    html = html.replace(/<a\b(?![^>]*tabindex)/g, '<a tabindex="-1"');
    // Add tabindex="-1" to <button> tags
    html = html.replace(
      /<button\b(?![^>]*tabindex)/g,
      '<button tabindex="-1"'
    );
    // Add tabindex="-1" to <input> tags
    html = html.replace(/<input\b(?![^>]*tabindex)/g, '<input tabindex="-1"');
    // Add tabindex="-1" to <select> tags
    html = html.replace(
      /<select\b(?![^>]*tabindex)/g,
      '<select tabindex="-1"'
    );
    // Add tabindex="-1" to <textarea> tags
    html = html.replace(
      /<textarea\b(?![^>]*tabindex)/g,
      '<textarea tabindex="-1"'
    );
  }

  // a11y-missing-aria: Remove ARIA attributes and role
  if (isBugEnabled('a11y-missing-aria')) {
    // Remove role="..." attributes
    html = html.replace(/\s+role="[^"]*"/g, '');
    html = html.replace(/\s+role='[^']*'/g, '');
    // Remove aria-label="..." attributes
    html = html.replace(/\s+aria-label="[^"]*"/g, '');
    html = html.replace(/\s+aria-label='[^']*'/g, '');
    // Remove all aria-* attributes
    html = html.replace(/\s+aria-[a-z][-a-z]*="[^"]*"/g, '');
    html = html.replace(/\s+aria-[a-z][-a-z]*='[^']*'/g, '');
  }

  return html;
}

module.exports = {
  transformHtml,
};
