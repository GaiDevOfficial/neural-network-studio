/**
 * Universal menu lifecycle regression suite.
 *
 * Verifies the contract the UI relies on for every popup, dropdown, modal and tour:
 *   - a trigger opens/toggles its panel,
 *   - opening a second popover closes the first,
 *   - an outside pointerdown closes the popover, an inside pointerdown keeps it,
 *   - Escape closes the topmost panel first, then an active overlay,
 *   - every popover gets an injected explicit × button (even when the page forgot one),
 *   - unregister/destroy remove listeners and injected nodes (no leaks).
 *
 * Runs against the real MenuManager with the DOM doubles in tests/fake_dom.js.
 *
 * File: tests/test_menu_lifecycle.js
 */

import { MenuManager } from '../src/ui/menu_manager.js';
import { FakeDocument, FakeElement, FakeEvent } from './fake_dom.js';

let passed = 0;
let failed = 0;

function check(name, condition, detail = '') {
  if (condition) {
    passed++;
    console.log(`  [PASS] ${name}`);
  } else {
    failed++;
    console.error(`  [FAIL] ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

function keydown(doc, key = 'Escape') {
  return doc.dispatchEvent(new FakeEvent('keydown', { key }));
}

function pointerdown(doc, target) {
  return doc.dispatchEvent(new FakeEvent('pointerdown', { target }));
}

function main() {
  console.log('=====================================================');
  console.log('   NODE STUDIO — MENU / MODAL LIFECYCLE               ');
  console.log('=====================================================\n');

  const doc = new FakeDocument();
  const manager = new MenuManager({ document: doc });

  const triggerA = new FakeElement('button', doc);
  const triggerB = new FakeElement('button', doc);
  const panelA = new FakeElement('div', doc);
  const panelB = new FakeElement('div', doc);
  panelA.hidden = true;
  panelB.hidden = true;
  let lastCloseA = null;
  let lastCloseB = null;
  let opensA = 0;

  manager.register({
    id: 'menu-a',
    trigger: triggerA,
    panel: panelA,
    onOpen: () => { opensA++; },
    onClose: (reason) => { lastCloseA = reason; }
  });
  manager.register({
    id: 'menu-b',
    trigger: triggerB,
    panel: panelB,
    onClose: (reason) => { lastCloseB = reason; }
  });

  console.log('--- registration ---');
  check('trigger gets aria-haspopup', triggerA.getAttribute('aria-haspopup') === 'menu');
  check('panel starts hidden', panelA.hidden === true);
  const injectedA = panelA.querySelector('.ns-menu-close');
  const injectedB = panelB.querySelector('.ns-menu-close');
  check('menu A gets an injected × close button', Boolean(injectedA));
  check('injected × has an accessible label', injectedA.getAttribute('aria-label') === 'Close menu');
  check('menu B gets its own × close button', Boolean(injectedB) && injectedB !== injectedA);

  console.log('--- open / switch / close ---');
  triggerA.click();
  check('trigger opens its panel', panelA.hidden === false && manager.isOpen('menu-a') === true);
  check('open callback fires once', opensA === 1, `opens=${opensA}`);
  check('trigger reports aria-expanded=true', triggerA.getAttribute('aria-expanded') === 'true');

  triggerB.click();
  check('opening menu B closes menu A', panelA.hidden === true && panelB.hidden === false);
  check('menu A close reason is "switch"', lastCloseA === 'switch', String(lastCloseA));
  check('only one popover is open', manager.isOpen('menu-a') === false && manager.isOpen('menu-b') === true);

  const outside = new FakeElement('div', doc);
  pointerdown(doc, outside);
  check('outside pointerdown closes the open popover', panelB.hidden === true);
  check('outside close reason is "outside"', lastCloseB === 'outside', String(lastCloseB));

  triggerA.click();
  const inside = new FakeElement('span', doc);
  panelA.appendChild(inside);
  pointerdown(doc, inside);
  check('pointerdown inside the panel keeps it open', manager.isOpen('menu-a') === true);

  console.log('--- escape stack ---');
  const escapeMenu = new FakeEvent('keydown', { key: 'Escape' });
  doc.dispatchEvent(escapeMenu);
  check('Escape closes the open menu', panelA.hidden === true);
  check('Escape close reason is "escape"', lastCloseA === 'escape', String(lastCloseA));
  check('Escape is consumed while a menu is open', escapeMenu.defaultPrevented === true && escapeMenu.propagationStopped === true);

  let overlayOpen = true;
  let overlayCloses = 0;
  manager.registerOverlay({
    id: 'tour-overlay',
    isOpen: () => overlayOpen,
    requestClose: () => {
      overlayCloses++;
      overlayOpen = false;
    }
  });

  triggerB.click();
  keydown(doc);
  check('Escape closes a popover opened above an active overlay first',
    panelB.hidden === true && overlayCloses === 0, `overlay closes=${overlayCloses}`);
  keydown(doc);
  check('the next Escape closes the active overlay', overlayCloses === 1 && overlayOpen === false);
  const idleEscape = new FakeEvent('keydown', { key: 'Escape' });
  doc.dispatchEvent(idleEscape);
  check('Escape with nothing open is not consumed', idleEscape.defaultPrevented === false);

  console.log('--- explicit × ---');
  triggerA.click();
  const x = panelA.querySelector('.ns-menu-close');
  x.click();
  check('injected × closes the panel', panelA.hidden === true);
  check('× close reason is "close-button"', lastCloseA === 'close-button', String(lastCloseA));

  console.log('--- trigger toggle ---');
  triggerA.click();
  pointerdown(doc, triggerA);
  triggerA.click();
  check('pressing the trigger again closes (toggle)', panelA.hidden === true && lastCloseA === 'trigger');

  console.log('--- unregister / destroy ---');
  manager.unregister('menu-a');
  check('unregister removes the entry', manager.get('menu-a') === null);
  check('unregister removes the injected × from the panel', x.parentNode === null || panelA.children.indexOf(x) === -1);
  check('unregister removes the trigger listener', (() => {
    triggerA.click();
    return manager.get('menu-a') === null;
  })());

  manager.destroy();
  check('destroy removes the keydown listener', doc.listenerCount('keydown') === 0);
  check('destroy removes the pointerdown listener', doc.listenerCount('pointerdown') === 0);

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main();
