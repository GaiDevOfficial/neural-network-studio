/**
 * Universal menu / modal lifecycle manager.
 *
 * One registry owns every popup, dropdown, modal and overlay on a page:
 *   - Escape closes the topmost open entry (a real stack, so nested overlays unwind in order),
 *   - an outside pointerdown closes a popover,
 *   - every popover gets an explicit `×` close button if it does not declare one,
 *   - modals/overlays register a `requestClose` callback instead of a panel to hide.
 *
 * No DOM is touched at import time (Node-safe). `destroy()` removes every listener it bound.
 *
 * File: src/ui/menu_manager.js
 */

let sequence = 0;

export class MenuManager {
  constructor(options = {}) {
    this.doc = options.document || (typeof document !== 'undefined' ? document : null);
    this.win = options.window || (typeof window !== 'undefined' ? window : null);
    this.entries = new Map();
    this.openStack = [];
    this.onError = options.onError || null;
    this._boundKeyDown = (event) => this._handleKeyDown(event);
    this._boundPointerDown = (event) => this._handlePointerDown(event);
    if (this.doc) {
      this.doc.addEventListener('keydown', this._boundKeyDown, true);
      this.doc.addEventListener('pointerdown', this._boundPointerDown, true);
    }
  }

  /**
   * @param {object} config
   *   id          unique id
   *   trigger     element that opens/toggles a popover (optional)
   *   panel       element to show/hide (popovers and panel modals)
   *   kind        'popover' | 'modal' | 'overlay'
   *   closeButton explicit close button/selector (optional; popovers get one injected)
   *   isOpen      () => boolean   (required for overlays)
   *   requestClose () => void     (required for overlays; optional for panels)
   *   onOpen / onClose callbacks
   *   toggleOnTrigger default true
   */
  register(config = {}) {
    if (!config.id) throw new Error('MenuManager.register: id is required.');
    const previous = this.entries.get(config.id);
    if (previous) this.unregister(config.id);

    const entry = {
      id: config.id,
      kind: config.kind || 'popover',
      trigger: config.trigger || null,
      panel: config.panel || null,
      isOpenFn: typeof config.isOpen === 'function' ? config.isOpen : null,
      requestClose: typeof config.requestClose === 'function' ? config.requestClose : null,
      onOpen: config.onOpen || null,
      onClose: config.onClose || null,
      toggleOnTrigger: config.toggleOnTrigger !== false,
      explicitCloseButton: this._resolveElement(config.closeButton),
      injectedCloseButton: null,
      cleanup: []
    };

    if (entry.trigger) {
      const handler = (event) => {
        if (event) {
          event.preventDefault();
          event.stopPropagation();
        }
        if (entry.toggleOnTrigger) this.toggle(entry.id);
        else this.open(entry.id);
      };
      entry.trigger.addEventListener('click', handler);
      entry.cleanup.push(() => entry.trigger.removeEventListener('click', handler));
      if (entry.trigger.setAttribute) entry.trigger.setAttribute('aria-haspopup', entry.kind === 'popover' ? 'menu' : 'dialog');
    }

    if (entry.explicitCloseButton) {
      const handler = (event) => {
        if (event) event.preventDefault();
        this.close(entry.id, { reason: 'close-button' });
      };
      entry.explicitCloseButton.addEventListener('click', handler);
      entry.cleanup.push(() => entry.explicitCloseButton.removeEventListener('click', handler));
    }

    // Guarantee an explicit close affordance on every popover/modal panel.
    if (entry.panel && (entry.kind === 'popover' || entry.kind === 'modal')) {
      if (!entry.explicitCloseButton) {
        const existing = entry.panel.querySelector ? entry.panel.querySelector('[data-menu-close]') : null;
        if (existing) {
          const handler = (event) => {
            if (event) event.preventDefault();
            this.close(entry.id, { reason: 'close-button' });
          };
          existing.addEventListener('click', handler);
          entry.explicitCloseButton = existing;
          entry.cleanup.push(() => existing.removeEventListener('click', handler));
        } else {
          entry.injectedCloseButton = this._injectCloseButton(entry.panel);
          if (entry.injectedCloseButton) {
            const handler = (event) => {
              if (event) event.preventDefault();
              this.close(entry.id, { reason: 'close-button' });
            };
            entry.injectedCloseButton.addEventListener('click', handler);
            entry.cleanup.push(() => entry.injectedCloseButton.removeEventListener('click', handler));
          }
        }
      }
      // A popover with no X written by the page still gets one; panels start hidden.
      if (entry.panel.hidden === undefined) entry.panel.hidden = true;
    }

    this.entries.set(entry.id, entry);
    return entry;
  }

  /** Register a non-panel overlay (tour, story player, …) so Escape closes it too. */
  registerOverlay(config = {}) {
    if (typeof config.isOpen !== 'function' || typeof config.requestClose !== 'function') {
      throw new Error('MenuManager.registerOverlay: isOpen() and requestClose() are required.');
    }
    return this.register({ ...config, kind: 'overlay', panel: null, trigger: null });
  }

  unregister(id) {
    const entry = this.entries.get(id);
    if (!entry) return false;
    this.close(id, { reason: 'unregister', silent: true });
    for (const cleanup of entry.cleanup) cleanup();
    if (entry.injectedCloseButton && entry.injectedCloseButton.parentNode) {
      entry.injectedCloseButton.parentNode.removeChild(entry.injectedCloseButton);
    }
    this.entries.delete(id);
    return true;
  }

  get(id) {
    return this.entries.get(id) || null;
  }

  isOpen(id) {
    const entry = this.entries.get(id);
    if (!entry) return false;
    if (entry.isOpenFn) return Boolean(entry.isOpenFn());
    if (entry.kind === 'overlay') return false;
    return Boolean(entry.panel && !entry.panel.hidden);
  }

  open(id) {
    const entry = this.entries.get(id);
    if (!entry || entry.kind === 'overlay') return false;
    if (this.isOpen(id)) return true;

    // Only one popover can be open at a time.
    for (const other of this.entries.values()) {
      if (other.id !== id && other.kind === 'popover' && this.isOpen(other.id)) {
        this.close(other.id, { reason: 'switch' });
      }
    }

    this._setPanelOpen(entry, true);
    this.openStack = this.openStack.filter((openId) => openId !== id);
    this.openStack.push(id);
    if (entry.trigger && entry.trigger.setAttribute) entry.trigger.setAttribute('aria-expanded', 'true');
    if (entry.onOpen) entry.onOpen();
    return true;
  }

  close(id, options = {}) {
    const entry = this.entries.get(id);
    if (!entry) return false;
    const wasOpen = this.isOpen(id);
    if (!wasOpen && !options.force) return false;

    if (entry.kind === 'overlay') {
      if (entry.requestClose) entry.requestClose();
    } else {
      if (entry.kind === 'modal' && entry.requestClose) entry.requestClose();
      this._setPanelOpen(entry, false);
    }
    this.openStack = this.openStack.filter((openId) => openId !== id);
    if (entry.trigger && entry.trigger.setAttribute) entry.trigger.setAttribute('aria-expanded', 'false');
    if (!options.silent && entry.onClose) entry.onClose(options.reason || 'close');
    return true;
  }

  toggle(id) {
    return this.isOpen(id) ? this.close(id, { reason: 'trigger' }) : this.open(id);
  }

  closeAll(options = {}) {
    const ids = Array.from(this.entries.keys());
    let closed = 0;
    for (const id of ids) {
      if (this.close(id, options)) closed++;
    }
    this.openStack = [];
    return closed;
  }

  destroy() {
    if (this.doc) {
      this.doc.removeEventListener('keydown', this._boundKeyDown, true);
      this.doc.removeEventListener('pointerdown', this._boundPointerDown, true);
    }
    for (const id of Array.from(this.entries.keys())) this.unregister(id);
    this.openStack = [];
  }

  /* --------------------------------------------------------------- internals */

  _setPanelOpen(entry, open) {
    if (entry.panel && 'hidden' in entry.panel) entry.panel.hidden = !open;
    if (entry.panel && entry.panel.classList) entry.panel.classList.toggle('is-open', open);
    if (entry.trigger && entry.trigger.classList) entry.trigger.classList.toggle('is-open', open);
  }

  _handleKeyDown(event) {
    if (event.key !== 'Escape' && event.key !== 'Esc') return;

    // 1. Topmost open panel (a menu opened above a tour closes first).
    while (this.openStack.length > 0) {
      const id = this.openStack[this.openStack.length - 1];
      if (this.isOpen(id)) {
        this.close(id, { reason: 'escape' });
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      this.openStack.pop();
    }

    // 2. Registered overlay that is currently active (tours, story players).
    const overlays = Array.from(this.entries.values())
      .filter((entry) => entry.kind === 'overlay' && entry.isOpenFn && entry.isOpenFn());
    if (overlays.length > 0) {
      const overlay = overlays[overlays.length - 1];
      if (overlay.requestClose) overlay.requestClose();
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    // 3. Registered panel that is open but was never opened through this manager.
    for (const entry of this.entries.values()) {
      if (entry.kind !== 'overlay' && this.isOpen(entry.id)) {
        this.close(entry.id, { reason: 'escape' });
        event.preventDefault();
        event.stopPropagation();
        return;
      }
    }
  }

  _handlePointerDown(event) {
    const target = event.target;
    if (!target) return;

    for (const entry of this.entries.values()) {
      if (entry.kind !== 'popover' || !this.isOpen(entry.id)) continue;
      const insidePanel = entry.panel && (entry.panel === target || (entry.panel.contains && entry.panel.contains(target)));
      const insideTrigger = entry.trigger && (entry.trigger === target || (entry.trigger.contains && entry.trigger.contains(target)));
      if (!insidePanel && !insideTrigger && !this._isInjectedCloseClick(entry, target)) {
        this.close(entry.id, { reason: 'outside' });
      }
    }
  }

  _isInjectedCloseClick(entry, target) {
    if (!entry.injectedCloseButton || !target) return false;
    return entry.injectedCloseButton === target || (entry.injectedCloseButton.contains && entry.injectedCloseButton.contains(target));
  }

  _resolveElement(value) {
    if (!value) return null;
    if (typeof value === 'string') {
      if (!this.doc || !this.doc.querySelector) return null;
      return this.doc.querySelector(value);
    }
    return value;
  }

  _injectCloseButton(panel) {
    if (!this.doc || !this.doc.createElement || !panel || !panel.appendChild) return null;
    const button = this.doc.createElement('button');
    sequence += 1;
    button.type = 'button';
    button.className = 'ns-menu-close';
    button.dataset.menuClose = '';
    button.setAttribute('aria-label', 'Close menu');
    button.title = 'Close (Esc)';
    button.textContent = '\u00d7';
    panel.appendChild(button);
    return button;
  }
}

/** Page-level singleton factory (kept in one place so every page shares the manager). */
export function createMenuManager(options = {}) {
  return new MenuManager(options);
}
