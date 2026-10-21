"""Regression tests for the desktop HTTP host that serves the packaged studio.

The shipped .exe once froze on "Booting..." because desktop_app used a
single-threaded HTTPServer. Chromium opens its ES-module graph over many
parallel connections; while one slow response (the 2 MB MNIST payload) held
the accept loop, the default backlog of 5 overflowed and Windows refused the
extra connections. The module graph never loaded and the page stayed on
"Booting..." forever. These tests pin the threaded host, its deep backlog, and
the no-console error path used by windowed PyInstaller builds.
"""

import concurrent.futures
import os
import sys
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'desktop'))

import desktop_app  # noqa: E402  (path must be set first)


def _module_urls():
    """Every ES module in src, addressed the way the browser requests it."""
    urls = []
    for root, _dirs, files in os.walk(SRC_DIR):
        for name in files:
            if name.endswith('.js'):
                rel = os.path.relpath(os.path.join(root, name), SRC_DIR)
                urls.append('/' + rel.replace(os.sep, '/'))
    return sorted(urls)


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.headers.get('Content-Type', ''), response.read()


class _SlowBoomHandler(desktop_app.StudioHTTPHandler):
    """Real host handler plus a slow endpoint and one that simulates a browser abort."""

    slow_seconds = 1.2

    def do_GET(self):
        if self.path.startswith('/__slow'):
            time.sleep(self.slow_seconds)
            body = b'ok'
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith('/__boom'):
            raise ConnectionResetError('simulated browser abort')
        super().do_GET()


class DesktopHostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = desktop_app.find_free_port()
        cls.httpd = desktop_app.StudioHTTPServer(('127.0.0.1', cls.port), _SlowBoomHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.port}'
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                _get(f'{cls.base}/graph_studio.html', timeout=1)
                break
            except Exception:
                time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_host_is_threaded_with_a_deep_backlog(self):
        self.assertIsInstance(self.httpd, ThreadingHTTPServer)
        self.assertTrue(self.httpd.daemon_threads)
        self.assertGreaterEqual(self.httpd.request_queue_size, 32)

    def test_real_pages_and_modules_are_served_with_correct_mime(self):
        status, content_type, body = _get(f'{self.base}/graph_studio.html')
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith('text/html'), content_type)
        self.assertIn(b'id="status-text"', body)

        status, content_type, body = _get(f'{self.base}/ui/canvas_studio.js')
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith('text/javascript'), content_type)
        self.assertGreater(len(body), 10_000)

        status, content_type, body = _get(f'{self.base}/engine/nn_worker.js')
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith('text/javascript'), content_type)

    def test_root_and_legacy_index_open_node_studio(self):
        """The desktop host must never land on the retired classic studio."""
        for path in ('/', '/index.html'):
            with urllib.request.urlopen(f'{self.base}{path}', timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.geturl().endswith('/graph_studio.html'), response.geturl())
                self.assertIn(b'id="status-text"', response.read())

    def test_parallel_module_burst_survives_one_slow_response(self):
        """The exact packaged-app failure: before the fix 7+ requests were refused."""
        slow_result = {}

        def fetch_slow():
            try:
                slow_result['response'] = _get(f'{self.base}/__slow', timeout=10)
            except Exception as exc:  # pragma: no cover - failure detail
                slow_result['error'] = exc

        slow_thread = threading.Thread(target=fetch_slow, daemon=True)
        slow_thread.start()
        time.sleep(0.15)  # let the slow response occupy a serving thread

        modules = _module_urls()
        self.assertGreaterEqual(len(modules), 30, f'expected a real module graph, found {len(modules)}')
        failures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(modules)) as pool:
            futures = {pool.submit(_get, f'{self.base}{path}'): path for path in modules}
            for future in concurrent.futures.as_completed(futures):
                path = futures[future]
                try:
                    status, content_type, body = future.result()
                except Exception as exc:
                    failures.append(f'{path}: {type(exc).__name__}: {exc}')
                    continue
                if status != 200 or not body:
                    failures.append(f'{path}: HTTP {status}, {len(body)} bytes')

        slow_thread.join(timeout=10)
        self.assertNotIn('error', slow_result, f'slow request failed: {slow_result.get("error")}')
        self.assertEqual(slow_result['response'][0], 200)
        self.assertEqual(failures, [], '\n'.join(failures[:8]))

    def test_windowed_runtime_abort_does_not_take_down_the_server(self):
        """console=False builds have no stdout/stderr; handle_error must stay silent-safe."""
        saved = (sys.stdout, sys.stderr)
        try:
            sys.stdout = None
            sys.stderr = None
            try:
                _get(f'{self.base}/__boom', timeout=5)
            except Exception:
                pass  # the aborted connection is expected
        finally:
            sys.stdout, sys.stderr = saved

        status, content_type, body = _get(f'{self.base}/graph_studio.html')
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith('text/html'), content_type)
        self.assertIn(b'ns-statusbar', body)


if __name__ == '__main__':
    unittest.main(verbosity=2)
