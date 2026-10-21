/**
 * Node Studio — data connectors.
 *
 * Bridges every supported data source into the platform's verified DatasetManager:
 *   - built-in corpora (real MNIST, generators) via DatasetManager.getBuiltinDatasetAsync,
 *   - uploaded CSVs (bridged through DatasetManager.parseCSV — identical normalization),
 *   - SQLite rows via the local Python Data API (desktop/studio_data_api.py),
 *   - REST endpoints returning JSON rows.
 *
 * Also hosts the batching stream used for incremental/streaming training paths.
 * Pure ES module (uses global fetch) — exercised by tests/test_db_streaming.js.
 *
 * File: src/graph/data_connectors.js
 */

import { DatasetManager } from '../engine/dataset_manager.js';

/* ------------------------------------------------------------ row bridge */

/** Serialize columns + rows into CSV text (the exact input DatasetManager.parseCSV consumes). */
export function rowsToCsvText(columns, rows) {
  if (!Array.isArray(columns) || columns.length === 0) {
    throw new Error('rowsToCsvText: columns must be a non-empty array.');
  }
  const escape = (value) => {
    const text = value === null || value === undefined ? '' : String(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const lines = [columns.map(escape).join(',')];
  for (const row of rows || []) {
    lines.push(columns.map((col, idx) => escape(Array.isArray(row) ? row[idx] : row[col])).join(','));
  }
  return lines.join('\n');
}

/**
 * Convert raw `{columns, rows}` (SQL/CSV/JSON) into a dataset descriptor by running the
 * platform's own CSV pipeline — the last column becomes the target, exactly like an upload.
 */
export function rowsToDataset(columns, rows, options = {}) {
  const text = rowsToCsvText(columns, rows);
  return DatasetManager.parseCSV(text, { name: options.name || 'Connector dataset' });
}

/** Shape metadata used by the visual compiler (context.datasetInfoFor). */
export function datasetInfo(dataset) {
  if (!dataset) return null;
  const classes = dataset.numClasses
    || (dataset.targetNames ? dataset.targetNames.length : null)
    || 2;
  return {
    type: dataset.type || 'classification',
    name: dataset.name || 'dataset',
    numFeatures: dataset.numFeatures,
    numClasses: classes,
    imageDim: dataset.imageDim || null
  };
}

/** Give an un-split X/y dataset the split fields the training run expects. */
export function ensureSplit(dataset, trainRatio = 0.8) {
  if (Array.isArray(dataset.trainX) && Array.isArray(dataset.trainY)) return dataset;
  const split = DatasetManager.splitTrainTest(dataset.X, dataset.y, trainRatio);
  return {
    ...dataset,
    trainX: split.trainX,
    trainY: split.trainY,
    testX: split.testX,
    testY: split.testY
  };
}

/* ------------------------------------------------------- HTTP endpoints */

function joinApi(apiBase, path, params) {
  const base = String(apiBase || '').replace(/\/+$/, '');
  const query = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
    .join('&');
  return `${base}${path}${query ? `?${query}` : ''}`;
}

async function fetchJsonOrThrow(url) {
  let response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new Error(`Data request failed (${url}): ${err.message}`);
  }
  let body;
  try {
    body = await response.json();
  } catch (err) {
    throw new Error(`Data request to ${url} returned non-JSON (HTTP ${response.status}).`);
  }
  if (!response.ok || body.ok === false) {
    throw new Error(body.error || `Data request to ${url} failed (HTTP ${response.status}).`);
  }
  return body;
}

/** One window of rows from the local SQLite Data API. */
export async function fetchSqliteWindow({ apiBase = '', db, query, limit = 500, offset = 0 }) {
  if (!db) throw new Error('SQL source: no database file set.');
  if (!query) throw new Error('SQL source: no query set.');
  const body = await fetchJsonOrThrow(joinApi(apiBase, '/api/data/sqlite', { db, query, limit, offset }));
  return { columns: body.columns || [], rows: body.rows || [], returned: body.returned || 0, hasMore: Boolean(body.hasMore), total: body.total ?? null };
}

/** One window of rows from a CSV file through the local Data API. */
export async function fetchCsvWindow({ apiBase = '', path, limit = 500, offset = 0 }) {
  if (!path) throw new Error('CSV source: no file path set.');
  const body = await fetchJsonOrThrow(joinApi(apiBase, '/api/data/csv', { path, limit, offset }));
  return { columns: body.columns || [], rows: body.rows || [], returned: body.returned || 0, hasMore: Boolean(body.hasMore), total: body.total ?? null };
}

/** Rows from an arbitrary JSON endpoint: an array of objects, or {columns, rows}. */
export async function fetchJsonRows(url) {
  if (!url) throw new Error('REST source: no URL set.');
  const body = await fetchJsonOrThrow(url);
  if (Array.isArray(body)) {
    if (body.length === 0) throw new Error('REST source returned an empty array.');
    const columns = Array.from(body.reduce((set, row) => {
      Object.keys(row || {}).forEach((k) => set.add(k));
      return set;
    }, new Set()));
    return { columns, rows: body };
  }
  if (body && Array.isArray(body.rows)) {
    const columns = body.columns || Object.keys(body.rows[0] || {});
    return { columns, rows: body.rows };
  }
  throw new Error('REST source: expected an array of objects or {columns, rows}.');
}

/* --------------------------------------------------- source resolution */

/**
 * Resolve a source node into a concrete dataset descriptor (async, real fetches).
 * `options.apiBase` targets the local Python Data API; `options.customDatasets` carries
 * uploaded CSVs (id → descriptor); `options.maxRows` caps SQL/CSV paging.
 */
export async function resolveSourceDataset(node, options = {}) {
  if (!node) throw new Error('resolveSourceDataset: missing node.');

  if (node.type === 'source') {
    const datasetId = node.params.datasetId;
    if (datasetId === 'custom_csv') {
      const custom = options.customDatasets && options.customDatasets.custom_csv;
      if (!custom) {
        throw new Error('No CSV uploaded yet — use the Source block\'s “Upload CSV” action first.');
      }
      return { dataset: custom, info: datasetInfo(custom) };
    }
    const dataset = await DatasetManager.getBuiltinDatasetAsync(datasetId, options.datasetOptions || {});
    return { dataset, info: datasetInfo(dataset) };
  }

  if (node.type === 'sql_source') {
    const maxRows = options.maxRows || 2000;
    const pageSize = Math.min(500, maxRows);
    const columns = [];
    const rows = [];
    let offset = 0;
    let guard = 0;
    while (rows.length < maxRows && guard++ < Math.ceil(maxRows / pageSize) + 2) {
      const window = await fetchSqliteWindow({
        apiBase: options.apiBase,
        db: node.params.db,
        query: node.params.query,
        limit: pageSize,
        offset
      });
      if (columns.length === 0) columns.push(...window.columns);
      rows.push(...window.rows);
      if (!window.hasMore) break;
      offset += pageSize;
    }
    if (rows.length === 0) throw new Error('SQL query returned no rows.');
    const dataset = rowsToDataset(columns, rows, { name: `SQL: ${node.params.db}` });
    return { dataset, info: datasetInfo(dataset) };
  }

  if (node.type === 'api_source') {
    const { columns, rows } = await fetchJsonRows(node.params.url);
    if (rows.length === 0) throw new Error('REST source returned no rows.');
    const dataset = rowsToDataset(columns, rows, { name: `REST: ${node.params.url}` });
    return { dataset, info: datasetInfo(dataset) };
  }

  throw new Error(`resolveSourceDataset: unsupported source type "${node.type}".`);
}

/* ---------------------------------------------------------- batching */

/** Deterministic PRNG (mulberry32) for reproducible shuffles. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffledIndexes(count, rand) {
  const index = Array.from({ length: count }, (_, i) => i);
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [index[i], index[j]] = [index[j], index[i]];
  }
  return index;
}

/**
 * Yield real mini-batches `{index, size, total, batchX, batchY}` from a dataset's split.
 * Feeds incremental training paths and the database streaming test.
 */
export function streamBatches(dataset, options = {}) {
  const batchSize = Math.max(1, options.batchSize || 16);
  const split = options.split || 'train';
  const X = split === 'test' ? dataset.testX : dataset.trainX;
  const Y = split === 'test' ? dataset.testY : dataset.trainY;
  if (!Array.isArray(X) || !Array.isArray(Y)) {
    throw new Error('streamBatches: dataset has no trainX/trainY split (call ensureSplit first).');
  }
  const rand = mulberry32(Number.isFinite(options.seed) ? options.seed : 1337);
  const order = options.shuffle === false ? Array.from({ length: X.length }, (_, i) => i) : shuffledIndexes(X.length, rand);

  return {
    get count() {
      return Math.ceil(X.length / batchSize);
    },
    *[Symbol.iterator]() {
      let index = 0;
      for (let start = 0; start < order.length; start += batchSize) {
        const slice = order.slice(start, start + batchSize);
        yield {
          index: index++,
          size: slice.length,
          total: X.length,
          batchX: slice.map((i) => X[i]),
          batchY: slice.map((i) => Y[i])
        };
      }
    }
  };
}
