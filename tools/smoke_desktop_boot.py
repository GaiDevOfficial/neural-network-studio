"""Desktop-host boot smoke test (real WebView2).

Serves src/ through desktop_app's own StudioHTTPServer + StudioHTTPHandler --
the exact host used by the packaged .exe -- opens graph_studio.html in a hidden
WebView2 window, and waits for the page to finish booting.

This is the guard against the packaged-app "Booting..." freeze: the host must
deliver the whole ES-module graph even under parallel load, so the status bar
must reach "Ready" and the palette must be rendered.

Run:  python tools/smoke_desktop_boot.py
"""

import json
import os
import sys
import threading
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'desktop'))

import webview  # noqa: E402
import desktop_app  # noqa: E402


def main():
    port = desktop_app.find_free_port()
    httpd = desktop_app.StudioHTTPServer(('127.0.0.1', port), desktop_app.StudioHTTPHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{port}/graph_studio.html'
    print(f'[setup] serving {PROJECT_ROOT}\\src via desktop_app.StudioHTTPServer on port {port}', flush=True)
    print(f'[setup] opening {url} in WebView2\n', flush=True)

    window = webview.create_window('Desktop host boot smoke', url, width=1280, height=860, hidden=True)
    result = {'ok': False, 'detail': {}}

    def ev(js, default=None):
        try:
            return window.evaluate_js(js)
        except Exception:
            return default

    def drive():
        deadline = time.time() + 30
        status = ''
        while time.time() < deadline:
            status = ev("document.getElementById('status-text') ? document.getElementById('status-text').textContent : ''", '') or ''
            if status.startswith('Ready'):
                break
            time.sleep(0.25)

        snapshot = ev(
            "JSON.stringify({"
            "status: document.getElementById('status-text') ? document.getElementById('status-text').textContent : null,"
            "palette: document.querySelectorAll('.ns-palette-item').length,"
            "nodes: document.querySelectorAll('.ns-node').length,"
            "topbar: Boolean(document.querySelector('.ns-topbar')),"
            "readyState: document.readyState,"
            "failedResources: performance.getEntriesByType('resource')"
            "  .filter(function (e) { return e.responseStatus && e.responseStatus >= 400; })"
            "  .map(function (e) { return [e.name, e.responseStatus]; })"
            "})",
            '{}'
        )
        try:
            result['detail'] = json.loads(snapshot or '{}')
        except (TypeError, ValueError):
            result['detail'] = {'raw': snapshot}
        detail = result['detail']
        result['ok'] = (
            isinstance(detail.get('status'), str)
            and detail['status'].startswith('Ready')
            and int(detail.get('palette') or 0) >= 20
            and not detail.get('failedResources')
        )
        print('[boot] status   :', detail.get('status'), flush=True)
        print('[boot] palette  :', detail.get('palette'), 'items', flush=True)
        print('[boot] failed   :', detail.get('failedResources'), flush=True)
        print('[boot] readyState:', detail.get('readyState'), flush=True)
        window.destroy()

    # Hard safety net: never let a suspended renderer hang this script forever.
    threading.Timer(60, window.destroy).start()
    webview.start(drive, debug=False)
    httpd.shutdown()
    httpd.server_close()

    print('\n=====================================================')
    if result['ok']:
        print('   DESKTOP HOST BOOT SMOKE: PASS')
        print('=====================================================')
        return 0
    print('   DESKTOP HOST BOOT SMOKE: FAIL')
    print('   detail:', result['detail'])
    print('=====================================================')
    return 1


if __name__ == '__main__':
    sys.exit(main())
