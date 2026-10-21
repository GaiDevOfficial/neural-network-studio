/**
 * Small, safe inline formatter for lesson copy.
 *
 * Curriculum text is authored with `**bold**` emphasis and occasional `code` spans. Tour cards
 * render through `innerHTML`; mission cards previously used `textContent`, which showed the
 * literal asterisks. This helper escapes untrusted characters first, then restores the two
 * intentional inline forms, so both presenters can share identical copy without opening an
 * HTML-injection path.
 *
 * File: src/tutorial/inline_text.js
 */

const HTML_ESCAPES = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;'
};

export function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => HTML_ESCAPES[char]);
}

/** Escape HTML, then render `**bold**`, `` `code` `` and line breaks. */
export function inlineMarkdown(value) {
  const safe = escapeHtml(value).replace(/\r\n?/g, '\n');
  return safe
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}
