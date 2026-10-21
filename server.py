"""
Local Development Server for Node Studio (Neural Network Studio Platform)
Serves ES modules and Web Workers on http://127.0.0.1:8080
Also mounts the Node Studio Data API (SQLite/CSV/JSON) under /api/data/*.
"""

import json
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'desktop'))
from studio_data_api import handle_data_request  # noqa: E402  (path must be set first)

PORT = 8080
DIRECTORY = os.path.join(os.path.dirname(__file__), 'src')

class StudioHTTPServer(ThreadingHTTPServer):
    """Threaded server: a browser requests many ES modules in parallel, and one
    slow response must never fill the listen backlog and refuse the rest."""
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

class DevHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        '.js': 'text/javascript',
        '.mjs': 'text/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.map': 'application/json',
        '.svg': 'image/svg+xml',
        '.wasm': 'application/wasm',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/', '/index.html'):
            self.send_response(302)
            self.send_header('Location', '/graph_studio.html')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if parsed.path.startswith('/api/data/'):
            status, payload = handle_data_request(parsed.path, parse_qs(parsed.query))
            body = json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

def run_server():
    server = StudioHTTPServer(('127.0.0.1', PORT), DevHandler)
    print(f"\n=======================================================")
    print(f"  NODE STUDIO - LOCAL DEVELOPMENT SERVER               ")
    print(f"=======================================================")
    print(f"  URL : http://127.0.0.1:{PORT}/graph_studio.html")
    print(f"  Root: {DIRECTORY}")
    print(f"  Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

if __name__ == '__main__':
    run_server()
