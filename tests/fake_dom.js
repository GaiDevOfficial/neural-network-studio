/**
 * Minimal DOM doubles for Node-side UI tests.
 *
 * These are deliberately small: only the APIs MenuManager and NodePalette actually use
 * (classList, listeners, containment, appendChild, injected close buttons, bounding rects).
 * They let the lifecycle tests run in `node` with no browser and no jsdom dependency.
 *
 * File: tests/fake_dom.js
 */

export class FakeClassList {
  constructor() {
    this.set = new Set();
  }
  add(...names) {
    for (const name of names) this.set.add(name);
  }
  remove(...names) {
    for (const name of names) this.set.delete(name);
  }
  toggle(name, force) {
    const next = force === undefined ? !this.set.has(name) : Boolean(force);
    if (next) this.set.add(name); else this.set.delete(name);
    return next;
  }
  contains(name) {
    return this.set.has(name);
  }
  get value() {
    return Array.from(this.set).join(' ');
  }
}

export class FakeEvent {
  constructor(type, init = {}) {
    this.type = type;
    this.target = init.target || null;
    this.key = init.key || '';
    this.ctrlKey = Boolean(init.ctrlKey);
    this.metaKey = Boolean(init.metaKey);
    this.defaultPrevented = false;
    this.propagationStopped = false;
    this.dataTransfer = init.dataTransfer || null;
    this.clientX = init.clientX ?? 0;
    this.clientY = init.clientY ?? 0;
    this.pointerId = init.pointerId ?? 1;
    this.button = init.button ?? 0;
  }
  preventDefault() {
    this.defaultPrevented = true;
  }
  stopPropagation() {
    this.propagationStopped = true;
  }
}

export class FakeElement {
  constructor(tag = 'div', doc = null) {
    this.tagName = String(tag).toUpperCase();
    this.ownerDocument = doc;
    this.children = [];
    this.parentNode = null;
    this.hidden = false;
    this._className = '';
    this.classList = new FakeClassList();
    this.dataset = {};
    this.attributes = {};
    this.style = {};
    this._textContent = '';
    this.title = '';
    this.type = '';
    this.id = '';
    this.draggable = false;
    this._handlers = new Map();
    this._rect = { left: 0, top: 0, right: 100, bottom: 100 };
  }

  set className(value) {
    this._className = String(value || '');
    this.classList = new FakeClassList();
    for (const name of this._className.split(/\s+/).filter(Boolean)) this.classList.add(name);
  }

  get className() {
    return this._className;
  }

  set textContent(value) {
    this._textContent = String(value ?? '');
    for (const child of this.children) child.parentNode = null;
    this.children = [];
  }

  get textContent() {
    return this._textContent + this.children.map((child) => child.textContent).join('');
  }

  addEventListener(type, handler) {
    if (!this._handlers.has(type)) this._handlers.set(type, new Set());
    this._handlers.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    if (this._handlers.has(type)) this._handlers.get(type).delete(handler);
  }

  dispatchEvent(event) {
    if (!event.target) event.target = this;
    const handlers = this._handlers.get(event.type);
    if (handlers) for (const handler of Array.from(handlers)) handler(event);
    return !event.defaultPrevented;
  }

  click() {
    return this.dispatchEvent(new FakeEvent('click', { target: this }));
  }

  appendChild(child) {
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    if (!child.ownerDocument) child.ownerDocument = this.ownerDocument;
    this.children.push(child);
    return child;
  }

  removeChild(child) {
    const index = this.children.indexOf(child);
    if (index >= 0) this.children.splice(index, 1);
    child.parentNode = null;
    return child;
  }

  contains(node) {
    if (node === this) return true;
    for (const child of this.children) {
      if (child.contains && child.contains(node)) return true;
    }
    return false;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
  }

  querySelector(selector) {
    if (selector === '[data-menu-close]') {
      for (const child of this.children) {
        if (child.dataset && child.dataset.menuClose !== undefined) return child;
        const nested = child.querySelector ? child.querySelector(selector) : null;
        if (nested) return nested;
      }
      return null;
    }
    const matches = this.querySelectorAll(selector);
    return matches[0] || null;
  }

  querySelectorAll(selector) {
    const className = selector.startsWith('.') ? selector.slice(1) : null;
    const found = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (className && child.classList && child.classList.contains(className)) found.push(child);
        walk(child);
      }
    };
    walk(this);
    return found;
  }

  setBoundingRect(rect) {
    this._rect = rect;
  }

  getBoundingClientRect() {
    return this._rect;
  }

  set innerHTML(value) {
    if (value === '') {
      for (const child of this.children) child.parentNode = null;
      this.children = [];
    }
  }

  get innerHTML() {
    return '';
  }
}

export class FakeDocument extends FakeElement {
  constructor() {
    super('#document', null);
    this._listeners = new Map();
    // JS-created panels mount themselves on document.body; the double must provide one so the
    // suites can assert that mounting happened.
    this.body = new FakeElement('body', this);
    this.body.parentNode = this;
    this.children.push(this.body);
  }

  createElement(tag) {
    return new FakeElement(tag, this);
  }

  addEventListener(type, handler) {
    if (!this._listeners.has(type)) this._listeners.set(type, new Set());
    this._listeners.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    if (this._listeners.has(type)) this._listeners.get(type).delete(handler);
  }

  dispatchEvent(event) {
    if (!event.target) event.target = this;
    const handlers = this._listeners.get(event.type);
    if (handlers) for (const handler of Array.from(handlers)) handler(event);
    return !event.defaultPrevented;
  }

  listenerCount(type) {
    return this._listeners.has(type) ? this._listeners.get(type).size : 0;
  }

  querySelector(selector) {
    const matches = this.querySelectorAll(selector);
    return matches[0] || null;
  }
}
