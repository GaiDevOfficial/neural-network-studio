"""
PHASE 3 — TEST 3: Data connector verification (SQLite / CSV / JSON Data API).

Exercises desktop/studio_data_api.py — the stdlib-only read path the Node Studio's data
source blocks consume. Coverage:

  1. Health route.
  2. SQLite: read-only SELECT windows; offset pagination walks every row exactly once.
  3. SQLite guard rails: SELECT-only enforcement, single-statement enforcement,
     missing database, parameter range validation.
  4. CSV: header/columns, windows, pagination, missing file.
  5. JSON: array-of-objects and {columns, rows} forms.
  6. Live HTTP round trip over an ephemeral port — the exact transport the page uses
     (src/graph/data_connectors.js fetches these endpoints).

File: python_core/test_data_connectors.py
"""

import json
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from http.server import HTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "desktop"))

from studio_data_api import handle_data_request, DataApiHTTPHandler  # noqa: E402

ROW_COUNT = 120


class DataConnectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="nn_studio_dataapi_")
        cls.db_path = os.path.join(cls.tmp.name, "records.sqlite")
        connection = sqlite3.connect(cls.db_path)
        connection.execute("CREATE TABLE records (id INTEGER, feature_a REAL, feature_b REAL, verdict TEXT)")
        rows = [
            (i, round(i * 0.5, 2), round((ROW_COUNT - i) * 0.25, 2), "yes" if i % 2 == 0 else "no")
            for i in range(1, ROW_COUNT + 1)
        ]
        connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", rows)
        connection.commit()
        connection.close()

        cls.csv_path = os.path.join(cls.tmp.name, "table.csv")
        with open(cls.csv_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("a,b,label\n")
            for i in range(1, ROW_COUNT + 1):
                handle.write(f"{i},{i * 2},{'yes' if i % 2 == 0 else 'no'}\n")

        cls.json_path = os.path.join(cls.tmp.name, "rows.json")
        with open(cls.json_path, "w", encoding="utf-8") as handle:
            json.dump([
                {"x": 1, "y": 2, "class": "a"},
                {"x": 3, "y": 4, "class": "b", "extra": 5},
            ], handle)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ------------------------------------------------------------------ health

    def test_health(self):
        status, payload = handle_data_request("/api/data/health", {})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertIn("service", payload)

    # ------------------------------------------------------------------ sqlite

    def test_sqlite_pagination_walks_every_row_once(self):
        seen = []
        offset = 0
        pages = 0
        while True:
            status, payload = handle_data_request("/api/data/sqlite", {
                "db": self.db_path,
                "query": "SELECT id, verdict FROM records ORDER BY id",
                "limit": "50",
                "offset": str(offset),
            })
            self.assertEqual(status, 200)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["columns"], ["id", "verdict"])
            seen.extend(row[0] for row in payload["rows"])
            pages += 1
            if not payload["hasMore"]:
                break
            offset += 50
            self.assertLess(pages, 10, "pagination failed to terminate")
        self.assertEqual(seen, list(range(1, ROW_COUNT + 1)), "every row must appear exactly once")
        self.assertEqual(pages, 3)

    def test_sqlite_window_shapes(self):
        status, payload = handle_data_request("/api/data/sqlite", {
            "db": self.db_path,
            "query": "SELECT * FROM records ORDER BY id",
            "limit": "50",
            "offset": "100",
        })
        self.assertEqual(status, 200)
        self.assertEqual(payload["returned"], 20)
        self.assertEqual(payload["offset"], 100)
        self.assertEqual(payload["limit"], 50)
        self.assertFalse(payload["hasMore"])
        self.assertEqual(payload["rows"][0][0], 101)
        self.assertEqual(payload["rows"][-1][0], ROW_COUNT)

    def test_sqlite_readonly_enforced(self):
        for statement in ("DELETE FROM records", "DROP TABLE records", "UPDATE records SET id = 0"):
            status, payload = handle_data_request("/api/data/sqlite", {
                "db": self.db_path, "query": statement, "limit": "10", "offset": "0",
            })
            self.assertEqual(status, 400, statement)
            self.assertFalse(payload["ok"])
            self.assertIn("SELECT", payload["error"])

    def test_sqlite_single_statement_enforced(self):
        status, payload = handle_data_request("/api/data/sqlite", {
            "db": self.db_path, "query": "SELECT 1; SELECT 2", "limit": "10", "offset": "0",
        })
        self.assertEqual(status, 400)
        self.assertIn("Multiple", payload["error"])

    def test_sqlite_missing_db_is_404(self):
        status, payload = handle_data_request("/api/data/sqlite", {
            "db": os.path.join(self.tmp.name, "nope.sqlite"),
            "query": "SELECT 1", "limit": "10", "offset": "0",
        })
        self.assertEqual(status, 404)
        self.assertIn("not found", payload["error"])

    def test_sqlite_invalid_query_is_reported(self):
        status, payload = handle_data_request("/api/data/sqlite", {
            "db": self.db_path, "query": "SELECT * FROM missing_table", "limit": "10", "offset": "0",
        })
        self.assertEqual(status, 400)
        self.assertIn("SQL error", payload["error"])

    def test_limit_range_validated(self):
        status, payload = handle_data_request("/api/data/sqlite", {
            "db": self.db_path, "query": "SELECT 1", "limit": "0", "offset": "0",
        })
        self.assertEqual(status, 400)
        self.assertIn("limit", payload["error"])
        status, payload = handle_data_request("/api/data/sqlite", {
            "db": self.db_path, "query": "SELECT 1", "limit": "999999", "offset": "0",
        })
        self.assertEqual(status, 400)

    # --------------------------------------------------------------------- csv

    def test_csv_windows(self):
        status, payload = handle_data_request("/api/data/csv", {
            "path": self.csv_path, "limit": "50", "offset": "0",
        })
        self.assertEqual(status, 200)
        self.assertEqual(payload["columns"], ["a", "b", "label"])
        self.assertEqual(payload["returned"], 50)
        self.assertTrue(payload["hasMore"])
        self.assertEqual(payload["rows"][0], ["1", "2", "no"])

        status, payload = handle_data_request("/api/data/csv", {
            "path": self.csv_path, "limit": "500", "offset": "0",
        })
        self.assertEqual(payload["returned"], ROW_COUNT)
        self.assertFalse(payload["hasMore"])

    def test_csv_missing_file(self):
        status, payload = handle_data_request("/api/data/csv", {
            "path": os.path.join(self.tmp.name, "nope.csv"), "limit": "10", "offset": "0",
        })
        self.assertEqual(status, 404)

    # -------------------------------------------------------------------- json

    def test_json_array_of_objects(self):
        status, payload = handle_data_request("/api/data/json", {"path": self.json_path})
        self.assertEqual(status, 200)
        self.assertEqual(payload["columns"], ["x", "y", "class", "extra"])
        self.assertEqual(payload["returned"], 2)
        self.assertIsNone(payload["rows"][0][3])  # missing key → null, not dropped

    # -------------------------------------------------------------------- http

    def test_live_http_round_trip(self):
        server = HTTPServer(("127.0.0.1", 0), DataApiHTTPHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/data/health", timeout=10) as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertTrue(health["ok"])

            url = (
                f"http://127.0.0.1:{port}/api/data/sqlite"
                f"?db={self.db_path}&query=SELECT%20id%20FROM%20records%20ORDER%20BY%20id&limit=5&offset=5"
            )
            with urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["returned"], 5)
            self.assertEqual(payload["rows"][0][0], 6)

            try:
                urlopen(f"http://127.0.0.1:{port}/api/data/nothing", timeout=10)
                self.fail("unknown endpoints must return HTTP 404")
            except HTTPError as error:
                self.assertEqual(error.code, 404)
                unknown = json.loads(error.read().decode("utf-8"))
                self.assertFalse(unknown["ok"])  # 404 body still carries the error contract
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
