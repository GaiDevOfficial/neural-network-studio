"""
Node Studio Desktop Host (PyWebView + Local HTTP Server)
Enables 100% Zero-Dependency Standalone Windows Executable (.exe)
Also mounts the Node Studio Data API (SQLite/CSV/JSON) under /api/data/*.
Neural Network Studio Platform
"""

import json
import os
import sys
import tempfile
import threading
import socket
import time
import traceback
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webview

# desktop/ is the script directory in development and inside the PyInstaller bundle.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from studio_data_api import handle_data_request  # noqa: E402

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def get_base_dir():
    if getattr(sys, 'frozen', False):
        # Running inside PyInstaller bundled .exe
        return os.path.join(sys._MEIPASS, 'src')
    else:
        # Running in local development mode
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

class StudioHTTPServer(ThreadingHTTPServer):
    """Threaded HTTP server for the studio's many parallel ES-module requests.

    A browser opens 6+ connections per origin and an ES-module graph can request
    dozens of files at once. With a single-threaded HTTPServer one slow response
    (e.g. the 2 MB MNIST payload) fills the default listen backlog of 5 and the
    overflow is refused by Windows -- Chromium reports ERR_CONNECTION_REFUSED,
    modules never load and the Studio stays on "Booting...". Serving each request
    on its own thread plus a deeper backlog removes that failure mode.
    """
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

class StudioHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=get_base_dir(), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            self.send_response(302)
            self.send_header('Location', '/graph_studio.html')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if parsed.path.startswith('/api/data/'):
            try:
                status, payload = handle_data_request(parsed.path, parse_qs(parsed.query))
            except Exception as exc:  # keep the server alive, answer honestly
                status, payload = 500, {'ok': False, 'error': str(exc)}
            body = json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, format, *args):
        # Suppress noisy console logs in desktop app
        pass

    def handle_error(self, request, client_address):
        """Never let error reporting take down a serving thread.

        Windowed PyInstaller builds have no console streams, and browsers routinely
        abort requests mid-transfer (navigation, prefetch cancellation). Connection
        aborts are normal; anything else goes to a temp log file, never stdout.
        """
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)):
            return
        try:
            log_path = os.path.join(tempfile.gettempdir(), 'neural_network_studio_server.log')
            with open(log_path, 'a', encoding='utf-8') as fh:
                fh.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] client {client_address}: ')
                traceback.print_exc(file=fh)
        except Exception:
            pass

    def end_headers(self):
        # Security & module header policies
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def start_server(port):
    server = StudioHTTPServer(('127.0.0.1', port), StudioHTTPHandler)
    server.serve_forever()

def main():
    port = find_free_port()
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    url = f'http://127.0.0.1:{port}/graph_studio.html'

    icon_path = os.path.join(get_base_dir(), 'assets', 'icon.ico')
    if not os.path.exists(icon_path):
        icon_path = None

    window = webview.create_window(
        title='Node Studio | Neural Network Studio',
        url=url,
        width=1380,
        height=860,
        min_size=(1080, 680),
        background_color='#121417',
        text_select=False
    )

    webview.start(debug=False)

if __name__ == '__main__':
    main()
