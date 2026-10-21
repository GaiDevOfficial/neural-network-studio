"""
Local Data API for the Node Studio — SQLite / CSV / JSON row streaming.

Stdlib only (sqlite3 + csv + json). Mounted by:
  - server.py            (development server: /api/data/* routes),
  - desktop/desktop_app.py (packaged .exe), 
  - standalone CLI       (python desktop/studio_data_api.py --port 8099)

Contract (consumed by src/graph/data_connectors.js):
  GET /api/data/health                       -> {"ok": true, "service": ...}
  GET /api/data/sqlite?db=..&query=..&limit=..&offset=..
  GET /api/data/csv?path=..&limit=..&offset=..
  GET /api/data/json?path=..
Every data response: {"ok": true, "columns": [...], "rows": [[...]],
"returned": N, "offset": N, "limit": N, "hasMore": bool}
Errors: HTTP 4xx/5xx + {"ok": false, "error": "..."}.

The database is always opened read-only (SQLite URI mode=ro) and the SQL statement must be
a single SELECT/WITH query — the API is a read path, never a mutation path.

File: desktop/studio_data_api.py
"""

import argparse
import csv
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, quote

DEFAULT_LIMIT = 500
MAX_LIMIT = 5000
MAX_CSV_ROWS = 200_000


class DataApiError(Exception):
    """Raised for any invalid/refused request; carries the HTTP status to report."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


# ------------------------------------------------------------------ helpers

def _param(query, name, default=None):
    if query is None:
        return default
    value = query.get(name, default)
    if isinstance(value, list):
        return value[0] if value else default
    return value


def _int_param(query, name, default, lo, hi):
    raw = _param(query, name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise DataApiError(f'"{name}" must be an integer (got "{raw}").')
    if value < lo or value > hi:
        raise DataApiError(f'"{name}" must be between {lo} and {hi} (got {value}).')
    return value


def _require(query, name):
    value = _param(query, name)
    if value is None or str(value).strip() == "":
        raise DataApiError(f'Missing required parameter "{name}".')
    return str(value)


def _jsonable(value):
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# ------------------------------------------------------------------ queries

def query_sqlite(db_path, sql, limit, offset):
    """Read-only SELECT window over a SQLite database."""
    if not os.path.isfile(db_path):
        raise DataApiError(f"Database file not found: {db_path}", status=404)

    statement = sql.strip().rstrip(";").strip()
    if not statement:
        raise DataApiError("Empty SQL query.")
    if ";" in statement:
        raise DataApiError("Multiple SQL statements are not allowed.")
    first_word = statement.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        raise DataApiError("Only SELECT/WITH queries are allowed on the read-only data API.")

    uri = "file:" + quote(db_path.replace("\\", "/")) + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise DataApiError(f"Cannot open database: {exc}", status=500)

    try:
        cursor = connection.cursor()
        cursor.execute(statement)
        columns = [description[0] for description in (cursor.description or [])]
        fetched = cursor.fetchmany(offset + limit)
    except sqlite3.Error as exc:
        raise DataApiError(f"SQL error: {exc}")
    finally:
        connection.close()

    window = fetched[offset:]
    rows = [[_jsonable(value) for value in row] for row in window]
    return {
        "columns": columns,
        "rows": rows,
        "returned": len(rows),
        "offset": offset,
        "limit": limit,
        "hasMore": len(fetched) == offset + limit,
    }


def query_csv_file(path, limit, offset):
    """Window over a CSV file's data rows (header returned as columns)."""
    if not os.path.isfile(path):
        raise DataApiError(f"CSV file not found: {path}", status=404)
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
    except OSError as exc:
        raise DataApiError(f"Cannot read CSV: {exc}", status=500)

    rows = [row for row in rows if any(str(cell).strip() for cell in row)]
    if not rows:
        raise DataApiError("CSV file has no rows.")
    if len(rows) > MAX_CSV_ROWS + 1:
        raise DataApiError(f"CSV exceeds {MAX_CSV_ROWS} rows — use SQLite for large datasets.")

    columns = [str(cell).strip() for cell in rows[0]]
    data_rows = rows[1:]
    window = data_rows[offset:offset + limit]
    return {
        "columns": columns,
        "rows": [[str(cell) for cell in row] for row in window],
        "returned": len(window),
        "offset": offset,
        "limit": limit,
        "hasMore": offset + limit < len(data_rows),
    }


def query_json_file(path):
    """Read a JSON file: array of objects, or {columns, rows} — returned as-is."""
    if not os.path.isfile(path):
        raise DataApiError(f"JSON file not found: {path}", status=404)
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            body = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataApiError(f"Cannot parse JSON: {exc}", status=500)

    if isinstance(body, list):
        columns = []
        seen = set()
        for row in body:
            if isinstance(row, dict):
                for key in row.keys():
                    if key not in seen:
                        seen.add(key)
                        columns.append(key)
        rows = [[row.get(column) for column in columns] for row in body if isinstance(row, dict)]
        return {"columns": columns, "rows": rows, "returned": len(rows),
                "offset": 0, "limit": len(rows), "hasMore": False}
    if isinstance(body, dict) and isinstance(body.get("rows"), list):
        return {"columns": body.get("columns") or [], "rows": body["rows"],
                "returned": len(body["rows"]), "offset": 0, "limit": len(body["rows"]),
                "hasMore": False}
    raise DataApiError("JSON file must contain an array of objects or {columns, rows}.")


# ------------------------------------------------------------------ routing

def handle_data_request(path, query):
    """Route one /api/data/* request. Returns (status_code, payload_dict)."""
    try:
        if path in ("/api/data/health", "/api/data/health/"):
            return 200, {"ok": True, "service": "Neural Network Studio Data API", "version": 1}

        if path == "/api/data/sqlite":
            limit = _int_param(query, "limit", DEFAULT_LIMIT, 1, MAX_LIMIT)
            offset = _int_param(query, "offset", 0, 0, 10**9)
            result = query_sqlite(_require(query, "db"), _require(query, "query"), limit, offset)
            return 200, {"ok": True, **result}

        if path == "/api/data/csv":
            limit = _int_param(query, "limit", DEFAULT_LIMIT, 1, MAX_LIMIT)
            offset = _int_param(query, "offset", 0, 0, 10**9)
            result = query_csv_file(_require(query, "path"), limit, offset)
            return 200, {"ok": True, **result}

        if path == "/api/data/json":
            result = query_json_file(_require(query, "path"))
            return 200, {"ok": True, **result}

        return 404, {"ok": False, "error": f"Unknown data endpoint: {path}"}
    except DataApiError as exc:
        return exc.status, {"ok": False, "error": str(exc)}
    except Exception as exc:  # pragma: no cover - defensive: report, never crash the host
        return 500, {"ok": False, "error": f"Internal data API error: {exc}"}


# -------------------------------------------------------------- HTTP layer

class DataApiHTTPHandler(BaseHTTPRequestHandler):
    """Minimal JSON server — used by the standalone CLI and integration tests."""

    def log_message(self, format, *args):  # keep test output clean
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        status, payload = handle_data_request(parsed.path, parse_qs(parsed.query))
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def serve(host="127.0.0.1", port=8099):
    server = HTTPServer((host, port), DataApiHTTPHandler)
    print(f"DATA API READY on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Neural Network Studio — Local Data API (SQLite/CSV/JSON)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    args = parser.parse_args()
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
