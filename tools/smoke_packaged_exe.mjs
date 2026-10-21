/**
 * Boot-verify the PACKAGED NeuralNetworkStudio.exe through WebView2 CDP.
 *
 * The exe is a PyInstaller windowed app, so stdout cannot be inspected and the
 * random HTTP port is not printed. WebView2 honours
 * WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS, so we launch the real binary with a
 * DevTools port and assert that the Node Studio page itself reaches its ready
 * state. The desktop host opens graph_studio.html directly; there is no classic
 * landing page to navigate through.
 *
 * Run: npm run smoke:packaged
 *      node tools/smoke_packaged_exe.mjs [C:\path\to\NeuralNetworkStudio.exe]
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

if (typeof WebSocket !== 'function') {
  console.error('[packaged] Node 22+ is required (the global WebSocket client drives CDP)');
  process.exit(2);
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const candidates = process.argv[2]
  ? [path.resolve(process.argv[2])]
  : [
      path.join(root, 'dist', 'NeuralNetworkStudio', 'NeuralNetworkStudio.exe'),
      path.join(root, 'dist', 'NeuralNetworkStudio.exe'),
    ];
const exe = candidates.find((candidate) => existsSync(candidate));
if (!exe) {
  console.error('[packaged] missing executable. Checked:');
  for (const candidate of candidates) console.error(`           ${candidate}`);
  console.error('           run "npm run build:exe" first');
  process.exit(2);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let spawnError = null;

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
  });
}

async function jsonList(port) {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/json/list`, { cache: 'no-store' });
    if (!response.ok) return [];
    const list = await response.json();
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}

async function waitForPage(port, match, timeoutMs = 45000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (spawnError) return null;
    const list = await jsonList(port);
    const page = list.find((target) => target.type === 'page'
      && typeof target.url === 'string'
      && target.url.startsWith('http://127.0.0.1:')
      && match(target.url));
    if (page) return page;
    await sleep(400);
  }
  return null;
}

async function connect(wsUrl) {
  const socket = new WebSocket(wsUrl);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('CDP websocket connect timeout')), 10000);
    socket.addEventListener('open', () => { clearTimeout(timer); resolve(); }, { once: true });
    socket.addEventListener('error', (event) => { clearTimeout(timer); reject(new Error(`CDP websocket error: ${event.message || 'unknown'}`)); }, { once: true });
  });
  let nextId = 1;
  const pending = new Map();
  socket.addEventListener('message', (event) => {
    let message = null;
    try { message = JSON.parse(event.data); } catch { return; }
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId++;
    pending.set(id, resolve);
    try {
      socket.send(JSON.stringify({ id, method, params }));
    } catch (error) {
      pending.delete(id);
      reject(error);
    }
  });
  const evaluate = async (expression) => {
    const reply = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (reply.error) throw new Error(JSON.stringify(reply.error));
    const result = reply.result && reply.result.result;
    if (result && result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result ? result.value : undefined;
  };
  return { socket, send, evaluate };
}

const port = await freePort();
const child = spawn(exe, [], {
  env: {
    ...process.env,
    WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS: `--remote-debugging-port=${port} --remote-allow-origins=*`
  },
  stdio: 'ignore',
  windowsHide: false
});
child.on('error', (error) => {
  spawnError = error;
  console.error(`[packaged] failed to launch executable: ${error.message}`);
});

let failed = false;
try {
  const studio = await waitForPage(port, (url) => /\/graph_studio\.html(?:$|[?#])/.test(url));
  if (!studio) throw new Error('the packaged app never exposed its graph_studio.html WebView2 page on CDP');

  const session = await connect(studio.webSocketDebuggerUrl);
  let studioState = null;
  const studioDeadline = Date.now() + 45000;
  while (Date.now() < studioDeadline) {
    try {
      const raw = await session.evaluate(`JSON.stringify({
        href: location.href,
        title: document.title,
        readyState: document.readyState,
        status: document.getElementById('status-text') ? document.getElementById('status-text').textContent : null,
        palette: document.querySelectorAll('.ns-palette-item').length,
        nodes: document.querySelectorAll('.ns-node').length,
        hasTopbar: Boolean(document.querySelector('.ns-topbar')),
        failed: performance.getEntriesByType('resource')
          .filter((e) => e.responseStatus && e.responseStatus >= 400)
          .map((e) => [e.name, e.responseStatus])
      })`);
      studioState = raw ? JSON.parse(raw) : null;
      if (studioState && typeof studioState.status === 'string'
        && studioState.status.startsWith('Ready')
        && studioState.palette >= 20) break;
    } catch {
      // The document may still be installing its module graph; retry.
    }
    await sleep(500);
  }
  console.log('[packaged] graph_studio:', JSON.stringify(studioState));

  const ok = Boolean(studioState)
    && studioState.readyState === 'complete'
    && typeof studioState.status === 'string'
    && studioState.status.startsWith('Ready')
    && studioState.palette >= 20
    && studioState.hasTopbar === true
    && Array.isArray(studioState.failed)
    && studioState.failed.length === 0;
  if (!ok) throw new Error('graph_studio.html did not reach the packaged ready state');
  console.log('\n=====================================================');
  console.log('   PACKAGED EXE BOOT (real WebView2 CDP): PASS');
  console.log('=====================================================');
} catch (error) {
  failed = true;
  console.error('[packaged] FAIL:', error.message);
} finally {
  if (child.pid && process.platform === 'win32') {
    try { spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore' }); } catch { /* ignore */ }
  } else {
    try { child.kill('SIGTERM'); } catch { /* ignore */ }
  }
}
process.exit(failed ? 1 : 0);
