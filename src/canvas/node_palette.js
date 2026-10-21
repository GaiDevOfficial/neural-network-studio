/**
 * Node palette renderer + drag-to-delete dropzone.
 *
 * Pure presentation: blocks come from `src/graph/node_types.js`; adding a block is delegated
 * through `onAdd(type)`. The palette also acts as the trash target while a canvas block is
 * dragged — `isOver()` is the real geometry test and `setTrashTarget()` drives the red state.
 *
 * DOM-free at import time; safe to import in Node.
 *
 * File: src/canvas/node_palette.js
 */

import { CATEGORY_META, paletteSections } from '../graph/node_types.js';

export class NodePalette {
  constructor(container, options = {}) {
    if (!container) throw new Error('NodePalette: a container element is required.');
    this.container = container;
    this.onAdd = options.onAdd || null;
    this.onDragStart = options.onDragStart || null;
    this.veil = null;
    this.rendered = 0;
    this.trashActive = false;
  }

  render() {
    const doc = this.container.ownerDocument || (typeof document !== 'undefined' ? document : null);
    if (!doc) return;
    this.container.innerHTML = '';
    this.rendered = 0;

    const title = doc.createElement('div');
    title.className = 'ns-palette-title';
    title.textContent = 'Blocks';
    this.container.appendChild(title);

    const { sections, planned } = paletteSections();
    for (const section of sections) {
      const group = doc.createElement('div');
      group.className = 'ns-palette-section';
      group.dataset.category = section.category;

      const head = doc.createElement('div');
      head.className = 'ns-palette-head';
      const dot = doc.createElement('span');
      dot.className = 'ns-palette-dot';
      dot.style.background = section.meta.color;
      head.appendChild(dot);
      const label = doc.createElement('span');
      label.textContent = section.meta.label;
      head.appendChild(label);
      group.appendChild(head);

      for (const block of section.blocks) {
        group.appendChild(this._paletteItem(doc, block));
        this.rendered += 1;
      }
      this.container.appendChild(group);
    }

    if (planned.length > 0) {
      const group = doc.createElement('div');
      group.className = 'ns-palette-section ns-palette-planned';
      const head = doc.createElement('div');
      head.className = 'ns-palette-head';
      head.textContent = 'Planned (engine v2)';
      group.appendChild(head);
      for (const block of planned) {
        const item = doc.createElement('div');
        item.className = 'ns-palette-item is-planned';
        const label = doc.createElement('span');
        label.className = 'ns-palette-label';
        label.textContent = block.label;
        item.appendChild(label);
        const reason = doc.createElement('span');
        reason.className = 'ns-palette-reason';
        reason.textContent = block.reason;
        item.appendChild(reason);
        item.title = block.reason;
        group.appendChild(item);
      }
      this.container.appendChild(group);
    }

    // Trash veil: inert unless a canvas block is being dragged over the palette.
    this.veil = doc.createElement('div');
    this.veil.className = 'ns-palette-trash-veil';
    const veilText = doc.createElement('span');
    veilText.textContent = 'Drop to delete';
    this.veil.appendChild(veilText);
    this.container.appendChild(this.veil);
  }

  _paletteItem(doc, block) {
    const item = doc.createElement('div');
    item.className = 'ns-palette-item';
    item.id = `palette-${block.type}`;
    item.draggable = true;
    item.dataset.type = block.type;

    const dot = doc.createElement('span');
    dot.className = 'ns-palette-dot';
    dot.style.background = (CATEGORY_META[block.category] || {}).color || '#8b95a1';
    item.appendChild(dot);

    const label = doc.createElement('span');
    label.className = 'ns-palette-label';
    label.textContent = block.label;
    item.appendChild(label);
    item.title = block.summary || block.label;

    item.addEventListener('dragstart', (event) => {
      if (event.dataTransfer) {
        event.dataTransfer.setData('application/x-node-type', block.type);
        event.dataTransfer.effectAllowed = 'copy';
      }
      if (this.onDragStart) this.onDragStart(block.type, event);
    });
    item.addEventListener('click', () => {
      if (this.onAdd) this.onAdd(block.type);
    });
    return item;
  }

  /** True when the pointer is inside the palette — the dropzone used for drag-to-delete. */
  isOver(clientX, clientY) {
    if (!this.container || !this.container.getBoundingClientRect) return false;
    const rect = this.container.getBoundingClientRect();
    return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
  }

  setTrashTarget(active) {
    const next = Boolean(active);
    if (this.trashActive === next) return;
    this.trashActive = next;
    this.container.classList.toggle('is-trash-target', next);
    if (this.container.setAttribute) {
      this.container.setAttribute('aria-label', next ? 'Block palette — drop here to delete' : 'Block palette');
    }
  }

  destroy() {
    this.setTrashTarget(false);
    this.container.innerHTML = '';
    this.rendered = 0;
  }
}
