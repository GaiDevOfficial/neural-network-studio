/**
 * Static DOM cross-check for graph_studio.html <-> canvas_studio.js / tutorial_engine.js.
 * Extracts every getElementById / querySelector id, every tour target and every data-attribute
 * the UI binds, then verifies each one exists in the page (palette items are generated at
 * runtime from the registry, so they are checked against node_types.js instead).
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { paletteSections } from '../src/graph/node_types.js';
import { TOURS } from '../src/ui/tutorial_engine.js';
import { CURRICULUM_TRACKS } from '../src/tutorial/interactive_tour.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(root, 'src', 'graph_studio.html'), 'utf8');
const js = fs.readFileSync(path.join(root, 'src', 'ui', 'canvas_studio.js'), 'utf8');
const playgroundJs = fs.readFileSync(path.join(root, 'src', 'modes', 'absolute_beginner.js'), 'utf8');
const tutorialJs = fs.readFileSync(path.join(root, 'src', 'ui', 'tutorial_engine.js'), 'utf8');
const css = fs.readFileSync(path.join(root, 'src', 'css', 'node_studio.css'), 'utf8');

const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
const runtimeIds = new Set(['probe-result']);
for (const section of paletteSections().sections) {
  for (const block of section.blocks) runtimeIds.add(`palette-${block.type}`);
}

let failed = 0;
const report = (ok, msg) => {
  if (ok) console.log(`  [PASS] ${msg}`);
  else { failed++; console.error(`  [FAIL] ${msg}`); }
};

console.log('--- ids referenced by canvas_studio.js ---');
const refIds = new Set([...js.matchAll(/getElementById\('([^']+)'\)/g)].map((m) => m[1]));
for (const id of refIds) {
  report(htmlIds.has(id) || runtimeIds.has(id), `#${id}`);
}

console.log('--- ids referenced by the playground module ---');
const playgroundIds = new Set([...playgroundJs.matchAll(/\$\('([^']+)'\)/g)].map((m) => m[1]));
for (const id of playgroundIds) {
  report(htmlIds.has(id), `#${id}`);
}

console.log('--- duplicate ids ---');
const idList = [...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
const duplicateIds = [...new Set(idList.filter((id, index) => idList.indexOf(id) !== index))];
report(duplicateIds.length === 0, `no duplicate ids${duplicateIds.length ? ` (${duplicateIds.join(', ')})` : ''}`);

console.log('--- tour targets ---');
for (const [tourId, tour] of Object.entries(TOURS)) {
  for (const step of tour.steps) {
    if (!step.target) continue;
    const id = step.target.replace(/^#/, '').split(' ')[0];
    report(htmlIds.has(id) || runtimeIds.has(id), `${tourId}/${step.id} -> ${step.target}`);
  }
}

console.log('--- curriculum track targets ---');
for (const [trackId, track] of Object.entries(CURRICULUM_TRACKS)) {
  for (const step of track.steps) {
    if (!step.target) continue;
    const id = step.target.replace(/^#/, '').split(' ')[0];
    report(htmlIds.has(id) || runtimeIds.has(id), `${trackId}/${step.id} -> ${step.target}`);
  }
}

console.log('--- universal close contract ---');
report(css.includes('[hidden] { display: none !important; }'), 'hidden always wins over author display rules');
report(css.includes('.ns-menu-close'), 'injected menu × has a style');
report(css.includes('.tour-close'), 'tour × has a style');
report(tutorialJs.includes('data-tour="close"'), 'tour tooltip renders the explicit × button');
report(js.includes('new MenuManager('), 'the page owns a MenuManager instance');
report(js.includes('registerOverlay('), 'active tours register as closable overlays');
report(js.includes("id: 'deep-dive-tour'"), 'deep-dive tutorials register for Escape too');
report(js.includes('menus.register('), 'dropdowns are registered with the lifecycle manager');
report(js.includes('closeActiveMissionForGuide') && js.includes('tutorial.finish(true)'),
  'starting one guide closes the other guide overlay');
report(playgroundJs.includes("data-story"), 'playground story steps are data-driven');
report(/\.mission-card\s*\{[^}]*max-height:\s*calc\(100vh - 24px\)/s.test(css),
  'mission card is bounded to the viewport');
report(/\.mission-card\s*\{[^}]*overflow-y:\s*auto/s.test(css),
  'mission card scrolls instead of overflowing the window');
report(/\.tour-tooltip\s*\{[^}]*max-height:\s*calc\(100vh - 32px\)/s.test(css),
  'tour tooltip is bounded to the viewport');
report(/\.tour-tooltip\s*\{[^}]*overflow-y:\s*auto/s.test(css),
  'tour tooltip scrolls instead of overflowing the window');
report(/\.ns-menu-cursor\s*\{[^}]*z-index:\s*120/s.test(css),
  'cursor menus stay above the mission and tour dim layers');
report(/\.ns-menu\s*\{[^}]*z-index:\s*120/s.test(css),
  'toolbar dropdowns stay above the mission and tour dim layers');
report((js.match(/refreshGuideOverlays\(\);/g) || []).length >= 4,
  'guides are re-anchored after node drags, view changes and inspector rebuilds');

console.log('--- playground markup ---');
for (const id of ['playground', 'playground-brain', 'playground-pad', 'playground-answer',
  'playground-feedback', 'playground-teach', 'playground-stop', 'playground-progress',
  'playground-accuracy', 'playground-exit']) {
  report(htmlIds.has(id), `#${id} exists in the page`);
}
report(/<section id="playground"[^>]*hidden/.test(html), 'playground starts hidden');
report(html.includes('data-tour-track="track_a"') && html.includes('data-tour-track="track_b"') &&
  html.includes('data-tour-track="track_c"'), 'all three curriculum tracks have menu entries');

console.log('--- data attributes ---');
const exportKinds = [...html.matchAll(/data-export="([^"]+)"/g)].map((m) => m[1]);
for (const kind of exportKinds) {
  report(js.includes(`kind === '${kind}'`) || js.includes(`handleExport('${kind}')`) || kind === 'commands', `data-export="${kind}" handled`);
}
const tourButtons = [...html.matchAll(/data-tour-id="([^"]+)"/g)].map((m) => m[1]);
for (const tourId of tourButtons) {
  report(Boolean(TOURS[tourId]), `data-tour-id="${tourId}" exists in TOURS`);
}
for (const tourId of Object.keys(TOURS)) {
  report(tourButtons.includes(tourId), `TOURS.${tourId} has a menu button`);
}
const trackButtons = [...html.matchAll(/data-tour-track="([^"]+)"/g)].map((m) => m[1]);
for (const trackId of trackButtons) {
  report(Boolean(CURRICULUM_TRACKS[trackId]), `data-tour-track="${trackId}" exists in CURRICULUM_TRACKS`);
}
for (const trackId of Object.keys(CURRICULUM_TRACKS)) {
  report(trackButtons.includes(trackId), `CURRICULUM_TRACKS.${trackId} has a menu button`);
}

console.log(failed === 0 ? '\nALL DOM CHECKS PASSED' : `\n${failed} DOM CHECK(S) FAILED`);
process.exit(failed === 0 ? 0 : 1);
