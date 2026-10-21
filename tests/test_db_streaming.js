/**
 * PHASE 3 — Database streaming verification.
 *
 * Drives the REAL local Data API (desktop/studio_data_api.py, spawned as its own process)
 * through the REAL connector module (src/graph/data_connectors.js) and proves:
 *
 *   1. the SQLite source block's windows arrive intact (columns, page flags),
 *   2. resolveSourceDataset pages through a SQLite table and yields a dataset that went
 *      through the exact DatasetManager.parseCSV normalization as a CSV upload,
 *   3. CSV file windows paginate to every row,
 *   4. REST (JSON) sources parse both array-of-objects and {columns, rows},
 *   5. the error contract surfaces the server's message (not a generic failure),
 *   6. streamBatches feeds real, correctly-shaped mini-batches.
 *
 * File: tests/test_db_streaming.js
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { DatasetManager } from '../src/engine/dataset_manager.js';
import {
  fetchSqliteWindow,
  fetchCsvWindow,
  fetchJsonRows,
  rowsToDataset,
  rowsToCsvText,
  resolveSourceDataset,
  ensureSplit,
  streamBatches
} from '../src/graph/data_connectors.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(HERE, '..');
const PORT = 8137;
const API_BASE = `http://127.0.0.1:${PORT}`;
const ROW_COUNT = 120;

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

function runPython(scriptPath) {
  const result = spawnSync(`python "${scriptPath}"`, { shell: true, encoding: 'utf8' });
  if (result.status !== 0) {
    throw new Error(`python ${scriptPath} failed (${result.status}): ${result.stderr || result.stdout}`);
  }
  return result.stdout;
}

async function waitForHealth(timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${API_BASE}/api/data/health`);
      const body = await response.json();
      if (body.ok) return body;
    } catch {
      // not up yet — retry
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`Data API did not become healthy on ${API_BASE} within ${timeoutMs} ms.`);
}

async function main() {
  console.log('=====================================================');
  console.log('   NODE STUDIO — DATABASE / CONNECTOR STREAMING TEST ');
  console.log('=====================================================\n');

  const workDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nnstudio_db_'));
  const dbPath = path.join(workDir, 'records.sqlite');
  const csvPath = path.join(workDir, 'table.csv');
  const jsonPath = path.join(workDir, 'rows.json');

  // ---- Fixtures (the SQLite file is created by real sqlite3, not hand-written bytes) ----
  const makeDbScript = path.join(workDir, 'make_db.py');
  fs.writeFileSync(makeDbScript, [
    'import sqlite3, sys',
    `connection = sqlite3.connect(r"${dbPath}")`,
    'connection.execute("CREATE TABLE records (feature_a REAL, feature_b REAL, verdict TEXT)")',
    `rows = [(round(i * 0.5, 2), round((${ROW_COUNT} - i) * 0.25, 2), "yes" if i % 2 == 0 else "no") for i in range(1, ${ROW_COUNT} + 1)]`,
    'connection.executemany("INSERT INTO records VALUES (?, ?, ?)", rows)',
    'connection.commit()',
    'connection.close()',
    'print("db ready")'
  ].join('\n'), 'utf8');
  runPython(makeDbScript);

  const csvLines = ['a,b,label'];
  for (let i = 1; i <= ROW_COUNT; i++) {
    csvLines.push(`${i},${i * 2},${i % 2 === 0 ? 'yes' : 'no'}`);
  }
  fs.writeFileSync(csvPath, csvLines.join('\n'), 'utf8');
  fs.writeFileSync(jsonPath, JSON.stringify([
    { x: 1, y: 2, class: 'a' },
    { x: 3, y: 4, class: 'b', extra: 5 }
  ]), 'utf8');

  // ---- Spawn the real Data API process ------------------------------------------------
  console.log(`[setup] Spawning desktop/studio_data_api.py on port ${PORT} ...`);
  const server = spawn('python', [path.join(PROJECT_ROOT, 'desktop', 'studio_data_api.py'), '--port', String(PORT)], {
    cwd: PROJECT_ROOT,
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let serverStderr = '';
  server.stderr.on('data', (chunk) => { serverStderr += String(chunk); });

  try {
    const health = await waitForHealth();
    check('spawned Data API reports healthy', health.ok === true && /Data API/.test(health.service));

    // ---- 1. Direct SQLite window ------------------------------------------------------
    const window1 = await fetchSqliteWindow({
      apiBase: API_BASE, db: dbPath,
      query: 'SELECT feature_a, feature_b, verdict FROM records ORDER BY rowid',
      limit: 50, offset: 0
    });
    check('SQLite window returns the requested rows and columns',
      window1.returned === 50 && JSON.stringify(window1.columns) === JSON.stringify(['feature_a', 'feature_b', 'verdict']),
      `returned=${window1.returned}`);
    check('SQLite window paginates (first page reports hasMore)', window1.rows.length === 50 && window1.hasMore === true && window1.rows[0][2] === 'no');

    // ---- 2. Full source resolution through the connector --------------------------------
    const sqlNode = {
      type: 'sql_source',
      params: { db: dbPath, query: 'SELECT feature_a, feature_b, verdict FROM records ORDER BY rowid' }
    };
    const resolved = await resolveSourceDataset(sqlNode, { apiBase: API_BASE, maxRows: 500 });
    const dataset = resolved.dataset;
    check('SQL source resolves into a classification dataset',
      dataset.type === 'classification' && dataset.numClasses === 2 && dataset.X.length === ROW_COUNT,
      `type=${dataset.type} classes=${dataset.numClasses} rows=${dataset.X.length}`);
    check('resolved targets keep their string labels', JSON.stringify(dataset.targetNames) === JSON.stringify(['no', 'yes']));

    // Connector rows must go through the same normalization as an uploaded CSV.
    const windows = [];
    let offset = 0;
    for (;;) {
      const page = await fetchSqliteWindow({
        apiBase: API_BASE, db: dbPath,
        query: 'SELECT feature_a, feature_b, verdict FROM records ORDER BY rowid',
        limit: 50, offset
      });
      windows.push(...page.rows);
      if (!page.hasMore) break;
      offset += 50;
    }
    const csvEquivalent = DatasetManager.parseCSV(rowsToCsvText(window1.columns, windows), { name: 'equivalent' });
    check('connector rows are numerically identical to a CSV upload of the same rows',
      JSON.stringify(csvEquivalent.X[0]) === JSON.stringify(dataset.X[0]) &&
      JSON.stringify(csvEquivalent.y[0]) === JSON.stringify(dataset.y[0]));
    check('features land in the [-1, 1] range like every studio CSV',
      dataset.X.every((row) => row.every((value) => value >= -1 && value <= 1)));

    // ---- 3. CSV file windows ------------------------------------------------------------
    const csvWindowA = await fetchCsvWindow({ apiBase: API_BASE, path: csvPath, limit: 50, offset: 0 });
    const csvWindowB = await fetchCsvWindow({ apiBase: API_BASE, path: csvPath, limit: 50, offset: 100 });
    check('CSV window returns header columns and the first rows',
      JSON.stringify(csvWindowA.columns) === JSON.stringify(['a', 'b', 'label']) && csvWindowA.returned === 50);
    check('CSV pagination reaches the tail window', csvWindowB.returned === 20 && csvWindowB.hasMore === false);

    // ---- 4. REST / JSON source ----------------------------------------------------------
    const jsonRows = await fetchJsonRows(`${API_BASE}/api/data/json?path=${encodeURIComponent(jsonPath)}`);
    check('REST source parses the JSON endpoint (columns union, null for missing keys)',
      JSON.stringify(jsonRows.columns) === JSON.stringify(['x', 'y', 'class', 'extra']) &&
      jsonRows.rows.length === 2 && jsonRows.rows[0][3] === null,
      JSON.stringify(jsonRows.columns));
    const restDataset = rowsToDataset(jsonRows.columns, jsonRows.rows, { name: 'REST rows' });
    check('REST rows build a real dataset through parseCSV', restDataset.X.length === 2 && restDataset.numFeatures === 3);

    // ---- 5. Error contract ---------------------------------------------------------------
    let errorMessage = '';
    try {
      await fetchSqliteWindow({ apiBase: API_BASE, db: path.join(workDir, 'missing.sqlite'), query: 'SELECT 1', limit: 10, offset: 0 });
    } catch (err) {
      errorMessage = err.message;
    }
    check('a missing database surfaces the server error (not a generic failure)', /not found/i.test(errorMessage), errorMessage);

    // ---- 6. Batching ---------------------------------------------------------------------
    const split = ensureSplit(dataset, 0.8);
    const batches = [...streamBatches(split, { batchSize: 16, seed: 99 })];
    const totalSamples = batches.reduce((sum, batch) => sum + batch.size, 0);
    check('streamBatches covers every training row exactly once',
      totalSamples === split.trainX.length && batches.length === Math.ceil(split.trainX.length / 16),
      `${totalSamples} rows in ${batches.length} batches`);
    check('streamed batches carry 2-feature inputs and 2-value one-hot targets',
      batches.every((batch) => batch.batchX.every((row) => row.length === 2) &&
      batch.batchY.every((row) => row.length === 2)));
  } finally {
    server.kill();
    await new Promise((resolve) => setTimeout(resolve, 300));
    fs.rmSync(workDir, { recursive: true, force: true });
    if (serverStderr.trim()) console.log(`[teardown] server stderr: ${serverStderr.trim().split('\n')[0]}`);
  }

  console.log('\n=====================================================');
  console.log(`   RESULT: ${passed} passed, ${failed} failed`);
  console.log('=====================================================');
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('\n[FATAL]', err);
  process.exit(1);
});
