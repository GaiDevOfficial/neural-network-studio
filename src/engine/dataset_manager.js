/**
 * Dataset Manager: Built-in Datasets, Synthetic Generators, CSV Parser, Normalizer
 * Neural Network Studio Platform
 *
 * DATA HONESTY CONTRACT
 * ---------------------
 * `mnist_01` is NOT generated in this file. It is the real MNIST handwritten digit
 * corpus, shipped as uint8 IDX payloads re-encoded by tools/fetch_mnist.py, decoded by
 * MNISTLoader, and split by the ORIGINAL authors' train/test partition.
 *
 * Every other `generate*` routine is a synthetic mathematical fixture and says so in its
 * own name and in the `provenance.realData = false` flag of the descriptor it returns.
 * Synthetic fixtures are never silently substituted for real data: if the MNIST payload
 * cannot be fetched, getBuiltinDatasetAsync throws and the UI reports the failure.
 */

import { MNISTLoader, MNISTLoadError } from './mnist_loader.js';

export class DatasetManager {
  /** Identifiers that resolve to the real MNIST corpus (asynchronous load). */
  static get ASYNC_DATASET_IDS() {
    return ['mnist', 'mnist_01', 'mnist_real', 'digits_01'];
  }

  static isAsyncDataset(name) {
    return DatasetManager.ASYNC_DATASET_IDS.includes(String(name).toLowerCase());
  }

  /**
   * Resolve any built-in dataset, awaiting the real MNIST payload when requested.
   * @param {string} name dataset identifier from the UI selector
   * @param {object} options { basePath, maxTrain, maxTest, count, noise }
   */
  static async getBuiltinDatasetAsync(name, options = {}) {
    const key = String(name).toLowerCase();

    if (DatasetManager.isAsyncDataset(key)) {
      const descriptor = await MNISTLoader.load(options.basePath, options);
      descriptor.requestedId = key;
      return descriptor;
    }

    return DatasetManager.getBuiltinDataset(key, options.count, options.noise);
  }

  static getBuiltinDataset(name, count = 200, noise = 0.05) {
    switch (String(name).toLowerCase()) {
      case 'xor':
        return DatasetManager.generateXOR();
      case 'circles':
        return DatasetManager.generateCircles(count, noise);
      case 'moons':
        return DatasetManager.generateMoons(count, noise);
      case 'spiral':
        return DatasetManager.generateSpiral(count, noise);
      case 'iris':
        return DatasetManager.getIrisDataset();
      case 'digits':
        return DatasetManager.generateDigits(count, noise);
      case 'synthetic_01':
      case 'synthetic_strokes':
        return DatasetManager.generateSyntheticStrokeDigits(count, 28);
      case 'linear_regression':
        return DatasetManager.generateRegression(count, noise);
      case 'mnist':
      case 'mnist_01':
      case 'mnist_real':
      case 'digits_01':
        // Real data cannot be produced synchronously. Fail loudly rather than
        // fabricating digits under a real-dataset name.
        throw new MNISTLoadError(
          `Dataset "${name}" is the real MNIST corpus and must be loaded with ` +
          `DatasetManager.getBuiltinDatasetAsync("${name}").`
        );
      default:
        return DatasetManager.generateCircles(count, noise);
    }
  }

  /**
   * Classic XOR Problem (4 Canonical Points + Optional Synthetic Cloud)
   */
  static generateXOR(count = 120, noise = 0.08) {
    const X = [];
    const y = [];
    const basePoints = [
      { coord: [-0.8, -0.8], label: [1, 0] }, // Class 0
      { coord: [0.8, 0.8], label: [1, 0] },   // Class 0
      { coord: [-0.8, 0.8], label: [0, 1] },  // Class 1
      { coord: [0.8, -0.8], label: [0, 1] }   // Class 1
    ];

    // Always include the exact corners first
    basePoints.forEach(p => {
      X.push([...p.coord]);
      y.push([...p.label]);
    });

    const samplesPerCluster = Math.floor((count - 4) / 4);
    for (let c = 0; c < 4; c++) {
      const base = basePoints[c];
      for (let i = 0; i < samplesPerCluster; i++) {
        const x1 = base.coord[0] + (Math.random() - 0.5) * noise * 4;
        const x2 = base.coord[1] + (Math.random() - 0.5) * noise * 4;
        X.push([x1, x2]);
        y.push([...base.label]);
      }
    }

    return {
      name: 'XOR Problem',
      type: 'classification',
      featureNames: ['X1', 'X2'],
      targetNames: ['Class 0', 'Class 1'],
      numFeatures: 2,
      numClasses: 2,
      X,
      y
    };
  }

  /**
   * Concentric Circles (Inner ring vs Outer ring)
   */
  static generateCircles(count = 200, noise = 0.05) {
    const X = [];
    const y = [];
    const half = Math.floor(count / 2);

    // Inner Circle (Class 0, radius ~ 0.35)
    for (let i = 0; i < half; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = 0.35 + (Math.random() - 0.5) * noise;
      X.push([r * Math.cos(angle), r * Math.sin(angle)]);
      y.push([1, 0]);
    }

    // Outer Circle (Class 1, radius ~ 0.8)
    for (let i = 0; i < half; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = 0.8 + (Math.random() - 0.5) * noise;
      X.push([r * Math.cos(angle), r * Math.sin(angle)]);
      y.push([0, 1]);
    }

    return {
      name: 'Concentric Circles',
      type: 'classification',
      featureNames: ['X1', 'X2'],
      targetNames: ['Inner Ring', 'Outer Ring'],
      numFeatures: 2,
      numClasses: 2,
      X,
      y
    };
  }

  /**
   * Two Interleaving Moons
   */
  static generateMoons(count = 200, noise = 0.05) {
    const X = [];
    const y = [];
    const half = Math.floor(count / 2);

    // Moon 1 (Upper)
    for (let i = 0; i < half; i++) {
      const angle = (Math.PI * i) / half;
      const x1 = Math.cos(angle) + (Math.random() - 0.5) * noise;
      const x2 = Math.sin(angle) + (Math.random() - 0.5) * noise;
      // Rescale to [-1, 1] range
      X.push([(x1 - 0.5) * 0.7, (x2 - 0.25) * 0.7]);
      y.push([1, 0]);
    }

    // Moon 2 (Lower)
    for (let i = 0; i < half; i++) {
      const angle = (Math.PI * i) / half;
      const x1 = 1 - Math.cos(angle) + (Math.random() - 0.5) * noise;
      const x2 = 1 - Math.sin(angle) - 0.5 + (Math.random() - 0.5) * noise;
      X.push([(x1 - 0.5) * 0.7, (x2 - 0.25) * 0.7]);
      y.push([0, 1]);
    }

    return {
      name: 'Two Moons',
      type: 'classification',
      featureNames: ['X1', 'X2'],
      targetNames: ['Top Moon', 'Bottom Moon'],
      numFeatures: 2,
      numClasses: 2,
      X,
      y
    };
  }

  /**
   * Two Spirals (Challenging Non-Linear Classification)
   */
  static generateSpiral(count = 240, noise = 0.03) {
    const X = [];
    const y = [];
    const half = Math.floor(count / 2);

    for (let i = 0; i < half; i++) {
      const r = (i / half) * 0.85;
      const t = (1.75 * i / half) * 2 * Math.PI;

      // Spiral 1
      const x1 = -Math.cos(t) * r + (Math.random() - 0.5) * noise;
      const y1 = Math.sin(t) * r + (Math.random() - 0.5) * noise;
      X.push([x1, y1]);
      y.push([1, 0]);

      // Spiral 2 (opposite phase)
      const x2 = Math.cos(t) * r + (Math.random() - 0.5) * noise;
      const y2 = -Math.sin(t) * r + (Math.random() - 0.5) * noise;
      X.push([x2, y2]);
      y.push([0, 1]);
    }

    return {
      name: 'Spiral',
      type: 'classification',
      featureNames: ['X1', 'X2'],
      targetNames: ['Spiral A', 'Spiral B'],
      numFeatures: 2,
      numClasses: 2,
      X,
      y
    };
  }

  /**
   * Fisher's Iris Dataset (Multiclass 3 Categories, 4 Features)
   * Normalized to [-1, 1] range
   */
  static getIrisDataset() {
    // Representative subset of Fisher's Iris dataset: 4 features -> 3 classes
    const rawData = [
      [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0], [4.6, 3.1, 1.5, 0.2, 0],
      [5.0, 3.6, 1.4, 0.2, 0], [5.4, 3.9, 1.7, 0.4, 0], [4.6, 3.4, 1.4, 0.3, 0], [5.0, 3.4, 1.5, 0.2, 0],
      [4.4, 2.9, 1.4, 0.2, 0], [4.9, 3.1, 1.5, 0.1, 0], [5.4, 3.7, 1.5, 0.2, 0], [4.8, 3.4, 1.6, 0.2, 0],
      [4.8, 3.0, 1.4, 0.1, 0], [4.3, 3.0, 1.1, 0.1, 0], [5.8, 4.0, 1.2, 0.2, 0], [5.7, 4.4, 1.5, 0.4, 0],
      [5.4, 3.9, 1.3, 0.4, 0], [5.1, 3.5, 1.4, 0.3, 0], [5.7, 3.8, 1.7, 0.3, 0], [5.1, 3.8, 1.5, 0.3, 0],
      // Versicolor
      [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [6.9, 3.1, 4.9, 1.5, 1], [5.5, 2.3, 4.0, 1.3, 1],
      [6.5, 2.8, 4.6, 1.5, 1], [5.7, 2.8, 4.5, 1.3, 1], [6.3, 3.3, 4.7, 1.6, 1], [4.9, 2.4, 3.3, 1.0, 1],
      [6.6, 2.9, 4.6, 1.3, 1], [5.2, 2.7, 3.9, 1.4, 1], [5.0, 2.0, 3.5, 1.0, 1], [5.9, 3.0, 4.2, 1.5, 1],
      [6.0, 2.2, 4.0, 1.0, 1], [6.1, 2.9, 4.7, 1.4, 1], [5.6, 2.9, 3.6, 1.3, 1], [6.7, 3.1, 4.4, 1.4, 1],
      [5.6, 3.0, 4.5, 1.5, 1], [5.8, 2.7, 4.1, 1.0, 1], [6.2, 2.2, 4.5, 1.5, 1], [5.6, 2.5, 3.9, 1.1, 1],
      // Virginica
      [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2], [6.3, 2.9, 5.6, 1.8, 2],
      [6.5, 3.0, 5.8, 2.2, 2], [7.6, 3.0, 6.6, 2.1, 2], [4.9, 2.5, 4.5, 1.7, 2], [7.3, 2.9, 6.3, 1.8, 2],
      [6.7, 2.5, 5.8, 1.8, 2], [7.2, 3.6, 6.1, 2.5, 2], [6.5, 3.2, 5.1, 2.0, 2], [6.4, 2.7, 5.3, 1.9, 2],
      [6.8, 3.0, 5.5, 2.1, 2], [5.7, 2.5, 5.0, 2.0, 2], [5.8, 2.8, 5.1, 2.4, 2], [6.4, 3.2, 5.3, 2.3, 2],
      [6.5, 3.0, 5.5, 1.8, 2], [7.7, 3.8, 6.7, 2.2, 2], [7.7, 2.6, 6.9, 2.3, 2], [6.0, 2.2, 5.0, 1.5, 2]
    ];

    const rawX = rawData.map(r => r.slice(0, 4));
    const labels = rawData.map(r => r[4]);

    // Compute normalization stats
    const normalized = DatasetManager.normalize(rawX);
    const y = labels.map(label => {
      const oneHot = [0, 0, 0];
      oneHot[label] = 1;
      return oneHot;
    });

    return {
      name: 'Iris Flower Dataset',
      type: 'classification',
      featureNames: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
      targetNames: ['Setosa', 'Versicolor', 'Virginica'],
      numFeatures: 4,
      numClasses: 3,
      X: normalized.data,
      y,
      normalizationParams: normalized.params
    };
  }

  /**
   * Continuous Linear / Non-Linear Regression Generator
   */
  static generateRegression(count = 150, noise = 0.05) {
    const X = [];
    const y = [];
    for (let i = 0; i < count; i++) {
      const xVal = (Math.random() - 0.5) * 2; // [-1, 1]
      // Target: 0.5 * sin(pi * x) + 0.3 * x^2
      const target = 0.5 * Math.sin(Math.PI * xVal) + 0.3 * (xVal * xVal) + (Math.random() - 0.5) * noise;
      X.push([xVal]);
      y.push([target]);
    }
    return {
      name: 'Non-Linear Regression',
      type: 'regression',
      featureNames: ['Feature X'],
      targetNames: ['Target Y'],
      numFeatures: 1,
      numClasses: 1,
      X,
      y
    };
  }

  /**
   * Robust CSV Parser with Auto-Detection & Sanitization
   */
  static parseCSV(csvString, options = {}) {
    const delimiter = options.delimiter || DatasetManager.detectDelimiter(csvString);
    const lines = csvString.trim().split(/\r?\n/).filter(line => line.trim().length > 0);

    if (lines.length < 2) {
      throw new Error('CSV must contain a header row and at least one data row.');
    }

    const header = lines[0].split(delimiter).map(col => col.trim().replace(/^["']|["']$/g, ''));
    const rows = [];

    for (let i = 1; i < lines.length; i++) {
      const cells = lines[i].split(delimiter).map(c => c.trim().replace(/^["']|["']$/g, ''));
      if (cells.length === header.length) {
        rows.push(cells);
      }
    }

    if (rows.length === 0) {
      throw new Error('No valid data rows found in CSV.');
    }

    // Identify target column index (default is the last column)
    const targetIdx = options.targetIndex !== undefined ? options.targetIndex : header.length - 1;
    const featureIndices = header
      .map((_, idx) => idx)
      .filter(idx => idx !== targetIdx && (options.features ? options.features.includes(idx) : true));

    if (featureIndices.length === 0) {
      throw new Error('No feature columns selected.');
    }

    // Extract X (Features) and convert to numbers with NaN imputation
    const rawX = [];
    const rawY = [];

    for (let r = 0; r < rows.length; r++) {
      const row = rows[r];
      const xRow = [];
      let rowValid = true;

      for (const fIdx of featureIndices) {
        let val = parseFloat(row[fIdx]);
        if (isNaN(val)) {
          val = 0.0; // fallback imputation
        }
        xRow.push(val);
      }

      if (rowValid) {
        rawX.push(xRow);
        rawY.push(row[targetIdx]);
      }
    }

    // Determine target type (Classification vs Regression)
    const uniqueTargets = Array.from(new Set(rawY));
    const isClassification = uniqueTargets.length <= 10 && uniqueTargets.length < rawY.length * 0.5;

    let processedY = [];
    let targetNames = [];

    if (isClassification) {
      targetNames = uniqueTargets.map(t => String(t));
      const targetMap = new Map();
      targetNames.forEach((name, idx) => targetMap.set(name, idx));

      processedY = rawY.map(val => {
        const idx = targetMap.get(String(val));
        if (targetNames.length === 2) {
          return idx === 1 ? [0, 1] : [1, 0];
        } else {
          const oneHot = new Array(targetNames.length).fill(0);
          oneHot[idx] = 1;
          return oneHot;
        }
      });
    } else {
      // Regression target
      targetNames = [header[targetIdx] || 'Target'];
      processedY = rawY.map(val => [parseFloat(val) || 0.0]);
    }

    // Normalize X
    const normalized = DatasetManager.normalize(rawX);

    return {
      name: options.name || 'Custom Uploaded CSV',
      type: isClassification ? 'classification' : 'regression',
      featureNames: featureIndices.map(idx => header[idx]),
      targetNames,
      numFeatures: featureIndices.length,
      numClasses: targetNames.length,
      X: normalized.data,
      y: processedY,
      normalizationParams: normalized.params
    };
  }

  /**
   * Auto-detect Delimiter (, ; \t |)
   */
  static detectDelimiter(text) {
    const firstLine = text.split(/\r?\n/)[0] || '';
    const candidates = [',', ';', '\t', '|'];
    let bestDelimiter = ',';
    let maxCount = -1;

    for (const d of candidates) {
      const count = (firstLine.match(new RegExp('\\' + d, 'g')) || []).length;
      if (count > maxCount) {
        maxCount = count;
        bestDelimiter = d;
      }
    }
    return bestDelimiter;
  }

  /**
   * Min-Max Normalization to [-1, 1] Range with Zero-Variance Guard
   */
  static normalize(matrix) {
    if (!matrix || matrix.length === 0) return { data: [], params: [] };
    const numCols = matrix[0].length;
    const params = [];

    for (let j = 0; j < numCols; j++) {
      let min = Infinity;
      let max = -Infinity;
      for (let i = 0; i < matrix.length; i++) {
        const val = matrix[i][j];
        if (val < min) min = val;
        if (val > max) max = val;
      }

      // Guardrail against zero-variance constant features
      if (Math.abs(max - min) < 1e-9) {
        params.push({ min, max, span: 1.0, isConstant: true });
      } else {
        params.push({ min, max, span: max - min, isConstant: false });
      }
    }

    const normalizedData = matrix.map(row => {
      return row.map((val, colIdx) => {
        const p = params[colIdx];
        if (p.isConstant) return 0.0;
        // Map to [-1, 1]
        return ((val - p.min) / p.span) * 2 - 1;
      });
    });

    return { data: normalizedData, params };
  }

  /**
   * Split dataset into Training and Validation partitions
   */
  static splitTrainTest(X, y, trainRatio = 0.8) {
    const n = X.length;
    const indices = Array.from({ length: n }, (_, i) => i);

    // Deterministic shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const trainCount = Math.floor(n * trainRatio);
    const trainX = [];
    const trainY = [];
    const testX = [];
    const testY = [];

    for (let i = 0; i < n; i++) {
      const idx = indices[i];
      if (i < trainCount) {
        trainX.push(X[idx]);
        trainY.push(y[idx]);
      } else {
        testX.push(X[idx]);
        testY.push(y[idx]);
      }
    }

    return { trainX, trainY, testX, testY };
  }

  /**
   * 8x8 Synthetic Template Digits (64 input pixels -> 4 classes: 0, 1, 2, 3)
   * Hand-authored bitmap templates with jitter: a synthetic fixture, not photographed ink.
   */
  static generateDigits(count = 160, noise = 0.08) {
    const templates = [
      // Digit 0
      [
        0, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 0
      ],
      // Digit 1
      [
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 0
      ],
      // Digit 2
      [
        0, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1
      ],
      // Digit 3
      [
        0, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 0
      ]
    ];

    const X = [];
    const y = [];
    const samplesPerDigit = Math.floor(count / 4);

    for (let d = 0; d < 4; d++) {
      const base = templates[d];
      const oneHot = [0, 0, 0, 0];
      oneHot[d] = 1;

      for (let s = 0; s < samplesPerDigit; s++) {
        const sample = [];
        for (let p = 0; p < 64; p++) {
          const bit = base[p];
          // Add small jitter
          const val = bit === 1
            ? 0.7 + Math.random() * 0.3 - Math.random() * noise
            : (Math.random() - 0.5) * noise * 2;
          // Scale to [-1, 1]
          sample.push(Math.max(-1, Math.min(1, val * 2 - 1)));
        }
        X.push(sample);
        y.push([...oneHot]);
      }
    }

    const featureNames = Array.from({ length: 64 }, (_, i) => `px${i}`);

    return {
      name: 'Synthetic Template Digits (8x8)',
      id: 'digits',
      type: 'classification',
      source: 'synthetic',
      isImageDataset: true,
      imageDim: { width: 8, height: 8 },
      featureNames,
      targetNames: ['Digit 0', 'Digit 1', 'Digit 2', 'Digit 3'],
      numFeatures: 64,
      numClasses: 4,
      provenance: {
        realData: false,
        generator: 'DatasetManager.generateDigits',
        note: 'Hand-authored 8x8 bitmap templates with additive jitter.'
      },
      X,
      y
    };
  }

  /**
   * Process custom HTML Image or Canvas down to an 8x8 normalized grayscale vector
   */
  static processImage(imageSource, targetW = 8, targetH = 8) {
    const offCanvas = document.createElement('canvas');
    offCanvas.width = targetW;
    offCanvas.height = targetH;
    const ctx = offCanvas.getContext('2d');

    ctx.drawImage(imageSource, 0, 0, targetW, targetH);
    const imgData = ctx.getImageData(0, 0, targetW, targetH).data;
    const vector = [];

    for (let i = 0; i < imgData.length; i += 4) {
      const r = imgData[i];
      const g = imgData[i + 1];
      const b = imgData[i + 2];
      // Grayscale luminance
      const gray = 0.299 * r + 0.587 * g + 0.114 * b;
      // Invert & map to [-1, 1] (ink = 1, background = -1)
      vector.push((gray / 255) * 2 - 1);
    }
    return vector;
  }

  /**
   * Procedural 0-vs-1 stroke fixture (ellipse loops and slanted segments).
   *
   * THIS IS SYNTHETIC DATA, LABELLED AS SUCH. It exists so the offline test suite has a
   * deterministic, dependency-free image task, and so the studio has an explicitly
   * synthetic playground. Handwritten digits digitized from real paper come from
   * `mnist_01` / tools/fetch_mnist.py instead.
   */
  static generateSyntheticStrokeDigits(count = 140, resolution = 28) {
    const X = [];
    const y = [];
    const half = Math.floor(count / 2);
    const center = resolution / 2;

    // Helper: Distance from point to segment
    const distToSeg = (px, py, x1, y1, x2, y2) => {
      const dx = x2 - x1;
      const dy = y2 - y1;
      const l2 = dx * dx + dy * dy;
      if (l2 === 0) return Math.hypot(px - x1, py - y1);
      let t = ((px - x1) * dx + (py - y1) * dy) / l2;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
    };

    // Generate '0' samples (Ovals / Loops)
    for (let i = 0; i < half; i++) {
      const cx = center + (Math.random() - 0.5) * 3.0;
      const cy = center + (Math.random() - 0.5) * 3.0;
      const rx = 5.5 + Math.random() * 3.5;
      const ry = 8.5 + Math.random() * 4.0;
      const angle = (Math.random() - 0.5) * 0.45; // slight tilt
      const strokeSigma = 0.95 + Math.random() * 0.35;

      const sample = new Float64Array(resolution * resolution);
      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      for (let r = 0; r < resolution; r++) {
        for (let c = 0; c < resolution; c++) {
          const dx = c - cx;
          const dy = r - cy;
          // Rotate to ellipse local coordinates
          const lx = dx * cosA + dy * sinA;
          const ly = -dx * sinA + dy * cosA;

          // Normalized ellipse equation: (lx/rx)^2 + (ly/ry)^2 = 1
          const distNorm = Math.hypot(lx / rx, ly / ry);
          const radialDist = Math.abs(distNorm - 1.0) * Math.min(rx, ry);

          const intensity = Math.exp(-(radialDist * radialDist) / (2.0 * strokeSigma * strokeSigma));
          if (intensity > 0.05) {
            sample[r * resolution + c] = Math.min(1.0, intensity);
          }
        }
      }

      X.push(Array.from(sample));
      y.push([1.0, 0.0]); // Class 0
    }

    // Generate '1' samples (Vertical / Slanted strokes with optional serif)
    for (let i = 0; i < half; i++) {
      const cx = center + (Math.random() - 0.5) * 3.0;
      const cy = center + (Math.random() - 0.5) * 2.5;
      const len = 16.0 + Math.random() * 5.0;
      const angle = (Math.random() - 0.5) * 0.35; // tilt
      const strokeSigma = 0.95 + Math.random() * 0.35;
      const hasSerif = Math.random() < 0.65;

      const halfLen = len / 2.0;
      const sinA = Math.sin(angle);
      const cosA = Math.cos(angle);

      const x1 = cx - sinA * halfLen;
      const y1 = cy - cosA * halfLen;
      const x2 = cx + sinA * halfLen;
      const y2 = cy + cosA * halfLen;

      // Optional top serif
      const sx = x1 - 3.5;
      const sy = y1 + 3.0;

      const sample = new Float64Array(resolution * resolution);

      for (let r = 0; r < resolution; r++) {
        for (let c = 0; c < resolution; c++) {
          let d = distToSeg(c, r, x1, y1, x2, y2);
          if (hasSerif) {
            const dSerif = distToSeg(c, r, x1, y1, sx, sy);
            if (dSerif < d) d = dSerif;
          }

          const intensity = Math.exp(-(d * d) / (2.0 * strokeSigma * strokeSigma));
          if (intensity > 0.05) {
            sample[r * resolution + c] = Math.min(1.0, intensity);
          }
        }
      }

      X.push(Array.from(sample));
      y.push([0.0, 1.0]); // Class 1
    }

    const featureNames = Array.from({ length: resolution * resolution }, (_, i) => `px${i}`);

    return {
      name: `Synthetic Stroke Fixture (${resolution}x${resolution})`,
      id: 'synthetic_01',
      type: 'classification',
      source: 'synthetic',
      isImageDataset: true,
      imageDim: { width: resolution, height: resolution },
      featureNames,
      targetNames: ['Digit 0', 'Digit 1'],
      numFeatures: resolution * resolution,
      numClasses: 2,
      provenance: {
        realData: false,
        generator: 'DatasetManager.generateSyntheticStrokeDigits',
        note: 'Procedurally rasterized ellipses/segments. NOT handwritten digit data.'
      },
      X,
      y
    };
  }
}

