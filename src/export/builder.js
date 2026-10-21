/**
 * Node Studio — export builder.
 *
 * Assembles every downloadable artifact for a compiled + trained graph:
 *   - model JSON (the platform's `NeuralNetworkStudio-v1` payload — re-loadable by both the
 *     classic studio's "Load Model" and the Node Studio),
 *   - standalone Python source, standalone JavaScript source,
 *   - a real `.npz` weight archive,
 *   - a real ONNX inference graph (hand-written protobuf, no dependency),
 *   - the packaging commands for the .exe / static-web build pipelines.
 *
 * Pure ES module except `downloadBytes`/`downloadText` (browser-only DOM helpers).
 *
 * File: src/export/builder.js
 */

import { generatePythonModel, generateJavaScriptModel, buildModelMeta } from './codegen.js';
import { buildNpzBytes } from './npz_writer.js';
import { buildOnnxBytes } from './onnx_writer.js';

/**
 * @param {object} options
 *   plan         compiled plan from src/graph/node_engine.js compile()
 *   weights      NeuralNetwork.getWeightsSnapshot()
 *   datasetName  display name of the dataset the model trained on
 *   epochs       epoch count reached (from the run controller)
 */
export function buildExportArtifacts({ plan, weights, datasetName = null, epochs = null }) {
  if (!plan || !plan.ok) throw new Error('buildExportArtifacts: compile the graph first (plan.ok !== true).');
  if (!Array.isArray(weights) || weights.length === 0) throw new Error('buildExportArtifacts: empty weights snapshot.');

  const meta = buildModelMeta({ plan, datasetName, epochs });

  const jsonPayload = {
    format: 'NeuralNetworkStudio-v1',
    exportedAt: meta.exportedAt,
    architecture: plan.network.architecture,
    loss: plan.network.loss,
    optimizer: plan.network.optimizer,
    learningRate: plan.network.learningRate,
    weightDecay: plan.network.weightDecay,
    metadata: {
      dataset: meta.dataset,
      datasetId: meta.dataset,
      epochs,
      origin: 'node_studio',
      inputShape: meta.inputShape,
      inputKind: meta.inputKind
    },
    layers: weights
  };

  return {
    meta,
    jsonPayload,
    pythonSource: generatePythonModel({ plan, weights, metadata: meta }),
    jsSource: generateJavaScriptModel({ plan, weights, metadata: meta }),
    npzBytes: buildNpzBytes(weights, meta),
    onnxBytes: buildOnnxBytes({ plan, weights, datasetName: meta.dataset, epochs })
  };
}

/** Packaging commands shipped by the repo (Phase 4 of the build protocol). */
export function buildCommands() {
  return [
    {
      id: 'exe',
      label: 'Standalone Windows executable (PyInstaller)',
      command: 'npm run build:exe',
      detail: 'python desktop/build_exe.py — bundles the UI, engine, Node Studio and the local Data API into dist/NeuralNetworkStudio.exe.'
    },
    {
      id: 'web',
      label: 'Static web build',
      command: 'python desktop/build_exe.py --web-only',
      detail: 'Copies the full src/ tree (Node Studio included) to dist/web — upload to GitHub Pages / Netlify / Vercel.'
    },
    {
      id: 'serve',
      label: 'Development server (with the SQL Data API)',
      command: 'npm start',
      detail: 'python server.py — serves the studio and routes /api/data/* to the local SQLite/CSV Data API.'
    }
  ];
}

/* --------------------------------------------------- browser-only helpers */

export function downloadBytes(bytes, fileName, mimeType = 'application/octet-stream') {
  downloadBlob(new Blob([bytes], { type: mimeType }), fileName);
}

export function downloadText(text, fileName, mimeType = 'text/plain') {
  downloadBlob(new Blob([text], { type: mimeType }), fileName);
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  setTimeout(() => {
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }, 100);
}
