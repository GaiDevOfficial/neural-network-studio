/**
 * Real MNIST Payload Loader
 *
 * Consumes the artifacts produced by `tools/fetch_mnist.py` (raw uint8 IDX payloads
 * re-encoded as flat .bin files + a meta.json manifest) and produces the dataset
 * descriptor used by the rest of the studio.
 *
 * There is no synthetic digit generation anywhere in this module: images decoded here
 * are the published MNIST strokes, split by the ORIGINAL train/test partition.
 *
 * Both the browser path (fetch) and the headless path (Node test harness) funnel into
 * the same `decodeImages` routine, so what the verification suite measures is exactly
 * what the application runs.
 *
 * File: src/engine/mnist_loader.js
 */

export const DEFAULT_BASE_PATH = 'data/mnist_01';

export class MNISTLoadError extends Error {
  constructor(message, cause = null) {
    super(message);
    this.name = 'MNISTLoadError';
    this.cause = cause;
  }
}

export class MNISTLoader {
  /**
   * Fetch + decode the shipped payload from a base URL (browser / served desktop app).
   * @param {string} basePath directory containing meta.json and the four .bin files
   * @returns {Promise<object>} dataset descriptor (see buildDescriptor)
   */
  static async load(basePath = DEFAULT_BASE_PATH, options = {}) {
    if (typeof fetch !== 'function') {
      throw new MNISTLoadError('fetch() is unavailable in this environment');
    }

    const base = basePath.replace(/\/+$/, '');
    const metaUrl = `${base}/meta.json`;

    let meta;
    try {
      const metaResponse = await fetch(metaUrl, { cache: 'no-store' });
      if (!metaResponse.ok) {
        throw new MNISTLoadError(`meta.json request failed with HTTP ${metaResponse.status}`);
      }
      meta = await metaResponse.json();
    } catch (err) {
      if (err instanceof MNISTLoadError) throw err;
      throw new MNISTLoadError(
        `Unable to load ${metaUrl}. Serve the studio over HTTP (python server.py) instead of opening the page directly from disk.`,
        err
      );
    }

    const [trainX, trainY, testX, testY] = await Promise.all([
      MNISTLoader.fetchBinary(`${base}/trainX.bin`),
      MNISTLoader.fetchBinary(`${base}/trainY.bin`),
      MNISTLoader.fetchBinary(`${base}/testX.bin`),
      MNISTLoader.fetchBinary(`${base}/testY.bin`)
    ]);

    MNISTLoader.assertPayloadSizes(meta, { trainX, trainY, testX, testY });
    return MNISTLoader.buildDescriptor(meta, { trainX, trainY, testX, testY }, base, options);
  }

  /** HTTP fetch with strict validation; never returns a partial/empty buffer silently. */
  static async fetchBinary(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) {
      throw new MNISTLoadError(`Request for ${url} failed with HTTP ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    if (!buffer || buffer.byteLength === 0) {
      throw new MNISTLoadError(`Request for ${url} returned an empty payload`);
    }
    return new Uint8Array(buffer);
  }

  /** Structural validation against the manifest (fails loudly instead of training on garbage). */
  static assertPayloadSizes(meta, buffers) {
    const pixels = meta.pixelsPerImage;
    const checks = [
      ['trainX.bin', buffers.trainX, meta.train.count * pixels],
      ['trainY.bin', buffers.trainY, meta.train.count],
      ['testX.bin', buffers.testX, meta.test.count * pixels],
      ['testY.bin', buffers.testY, meta.test.count]
    ];
    for (const [name, buffer, expected] of checks) {
      if (buffer.length !== expected) {
        throw new MNISTLoadError(
          `${name} holds ${buffer.length} bytes but meta.json declares ${expected}. Re-run tools/fetch_mnist.py.`
        );
      }
    }
  }

  /**
   * SHA-256 of a byte buffer. Uses WebCrypto in the browser and node:crypto in Node.
   * Returns null when neither primitive is available (the caller must treat that as "unverified").
   */
  static async sha256Hex(bytes) {
    if (typeof crypto !== 'undefined' && crypto.subtle && typeof crypto.subtle.digest === 'function') {
      const digest = await crypto.subtle.digest('SHA-256', bytes);
      return Array.from(new Uint8Array(digest)).map((b) => b.toString(16).padStart(2, '0')).join('');
    }
    try {
      const { createHash } = await import('node:crypto');
      return createHash('sha256').update(Buffer.from(bytes)).digest('hex');
    } catch (err) {
      return null;
    }
  }

  /**
   * Verify every shipped artifact against the digests recorded in meta.json.
   * @returns {Promise<{ok: boolean, checked: number, mismatches: string[]}>}
   */
  static async verifyIntegrity(meta, buffers) {
    const mismatches = [];
    let checked = 0;

    for (const [name, buffer] of Object.entries(buffers)) {
      const recorded = meta.files && meta.files[name] ? meta.files[name].sha256 : null;
      if (!recorded) continue;
      const actual = await MNISTLoader.sha256Hex(buffer);
      if (actual === null) return { ok: false, checked, mismatches: ['SHA-256 unavailable in this runtime'] };
      checked++;
      if (actual !== recorded) mismatches.push(`${name}: expected ${recorded.slice(0, 16)}..., computed ${actual.slice(0, 16)}...`);
    }

    return { ok: mismatches.length === 0, checked, mismatches };
  }

  /**
   * Decode a flat uint8 image store into normalized Float64Array rows in [0.0, 1.0].
   * @param {Uint8Array} bytes flat row-major grayscale, count * width * height
   * @param {number} count number of images
   * @param {number} width
   * @param {number} height
   * @returns {Float64Array[]}
   */
  static decodeImages(bytes, count, width = 28, height = 28) {
    const pixels = width * height;
    const images = new Array(count);
    for (let i = 0; i < count; i++) {
      const row = new Float64Array(pixels);
      const base = i * pixels;
      for (let p = 0; p < pixels; p++) {
        row[p] = bytes[base + p] / 255.0;
      }
      images[i] = row;
    }
    return images;
  }

  /** Decode a flat uint8 label store into one-hot float vectors. */
  static decodeLabels(bytes, count, numClasses = 2, classIds = [0, 1]) {
    const labels = new Array(count);
    const indexOf = new Map();
    classIds.forEach((id, idx) => indexOf.set(id, idx));

    for (let i = 0; i < count; i++) {
      const raw = bytes[i];
      const oneHot = new Array(numClasses).fill(0.0);
      const mapped = indexOf.has(raw) ? indexOf.get(raw) : -1;
      if (mapped >= 0) {
        oneHot[mapped] = 1.0;
      }
      labels[i] = oneHot;
    }
    return labels;
  }

  /**
   * Turn decoded buffers into the studio's dataset descriptor contract.
   * `trainX/trainY/testX/testY` are provided pre-split (official MNIST partition),
   * which the app honors instead of re-splitting randomly.
   */
  static buildDescriptor(meta, buffers, sourcePath = DEFAULT_BASE_PATH, options = {}) {
    const width = (meta.imageDim && meta.imageDim.width) || 28;
    const height = (meta.imageDim && meta.imageDim.height) || 28;
    const pixels = width * height;
    const classIds = meta.classes || [0, 1];
    const numClasses = classIds.length;

    const keepTrain = Math.min(meta.train.count, options.maxTrain || meta.train.count);
    const keepTest = Math.min(meta.test.count, options.maxTest || meta.test.count);

    const trainX = MNISTLoader.decodeImages(buffers.trainX, keepTrain, width, height);
    const trainY = MNISTLoader.decodeLabels(buffers.trainY, keepTrain, numClasses, classIds);
    const testX = MNISTLoader.decodeImages(buffers.testX, keepTest, width, height);
    const testY = MNISTLoader.decodeLabels(buffers.testY, keepTest, numClasses, classIds);

    return {
      name: `MNIST Handwritten Digits (${classIds.join(' vs ')}, ${width}x${height}, real corpus)`,
      id: 'mnist_01',
      type: 'classification',
      source: 'mnist',
      isImageDataset: true,
      imageDim: { width, height },
      numFeatures: pixels,
      numClasses,
      classIds,
      featureNames: Array.from({ length: pixels }, (_, i) => `px${i}`),
      targetNames: classIds.map((id) => `Digit ${id}`),
      X: trainX,
      y: trainY,
      trainX,
      trainY,
      testX,
      testY,
      provenance: {
        corpus: meta.dataset,
        realData: true,
        sourceMirror: meta.sourceMirror,
        subsetPolicy: meta.subsetPolicy,
        counts: {
          train: keepTrain,
          test: keepTest,
          officialTrain: meta.train.count,
          officialTest: meta.test.count
        },
        checksums: meta.sourceChecksumsMd5,
        sourcePath,
        generatedAt: meta.generatedAt
      }
    };
  }

  /**
   * Headless entry point (Node): build the descriptor straight from buffers that were
   * read off disk, reusing the exact browser decode path.
   */
  static fromBuffers(meta, buffers, options = {}) {
    // Structural validation always compares the shipped files against the FULL counts
    // declared in the manifest; subsetting (maxTrain/maxTest) happens afterwards in
    // buildDescriptor and must never weaken the integrity check.
    MNISTLoader.assertPayloadSizes(meta, buffers);
    return MNISTLoader.buildDescriptor(meta, buffers, options.sourcePath || 'memory', options);
  }

  /**
   * Deterministic thumbnail helper used by the UI and the tests: renders a decoded
   * row into a compact string so a human can eyeball what the tensor actually holds.
   */
  static toAscii(row, width = 28, height = 28) {
    const ramp = ' .:-=+*#%@';
    const lines = [];
    for (let y = 0; y < height; y++) {
      let line = '';
      for (let x = 0; x < width; x++) {
        const value = row[y * width + x];
        const clamped = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
        line += ramp[Math.min(ramp.length - 1, Math.floor(clamped * ramp.length))];
      }
      lines.push(line);
    }
    return lines.join('\n');
  }

  /** Class balance report (used by tests and the dataset panel). */
  static classBalance(labels) {
    const counts = {};
    for (const label of labels) {
      const cls = label.indexOf(Math.max(...label));
      counts[cls] = (counts[cls] || 0) + 1;
    }
    return counts;
  }
}

export default MNISTLoader;
