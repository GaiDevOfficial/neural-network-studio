/**
 * Production Image Preprocessing Pipeline & Live Canvas Rasterizer
 *
 * Converts a drawn stroke / imported raster / raw luminance buffer into the exact
 * 28x28 normalized tensor that the neural engine consumes, replicating the canonical
 * MNIST preprocessing chain:
 *
 *   luminance -> polarity normalization -> threshold -> stroke bounding box ->
 *   aspect-preserving bilinear downscale into a 20x20 field -> placement into 28x28 ->
 *   center-of-mass recentering -> clamp to [0.0, 1.0]
 *
 * DESIGN NOTE: every numeric stage lives in a pure function that takes plain buffers,
 * so the identical code path runs in the browser (on canvas pixels) and in the Node
 * test harness (on raw luminance buffers). There is no duplicated "test version" of
 * the math anywhere in this project.
 *
 * File: src/pipeline/image_processor.js
 */

export const DEFAULTS = Object.freeze({
  targetWidth: 28,
  targetHeight: 28,
  /** Fraction of the target field the resized stroke may occupy (MNIST uses 20/28). */
  innerFieldRatio: 0.715,
  /** Luminance below this value is treated as background. */
  threshold: 0.08,
  /** Maximum center-of-mass correction in pixels (MNIST keeps the digit near the middle). */
  maxShift: 4,
  /** 'auto' derives polarity from the background ring, true/false force it. */
  invert: 'auto'
});

/** Machine readable degradation codes returned in `warnings`. */
export const WARN = Object.freeze({
  EMPTY_CANVAS: 'EMPTY_CANVAS',
  UNIFORM_INK: 'UNIFORM_INK',
  TINY_STROKE: 'TINY_STROKE',
  NON_FINITE_INPUT: 'NON_FINITE_INPUT',
  LOW_CONTRAST: 'LOW_CONTRAST',
  FULL_FRAME_INK: 'FULL_FRAME_INK'
});

const EPS = 1e-9;

/** Clamp helper (kept local so the module has zero external dependencies). */
export function clamp(value, low, high) {
  if (!Number.isFinite(value)) return low;
  return value < low ? low : (value > high ? high : value);
}

/**
 * Bilinear resample of a float buffer with align_corners = false.
 * srcX = (dstX + 0.5) * (srcW / dstW) - 0.5
 */
export function resizeBilinear(source, srcW, srcH, dstW, dstH) {
  const out = new Float64Array(dstW * dstH);
  if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) return out;

  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    const srcY = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.floor(srcY);
    const wy = srcY - y0;
    const y0c = clamp(y0, 0, srcH - 1);
    const y1c = clamp(y0 + 1, 0, srcH - 1);

    for (let x = 0; x < dstW; x++) {
      const srcX = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.floor(srcX);
      const wx = srcX - x0;
      const x0c = clamp(x0, 0, srcW - 1);
      const x1c = clamp(x0 + 1, 0, srcW - 1);

      const v00 = source[y0c * srcW + x0c];
      const v01 = source[y0c * srcW + x1c];
      const v10 = source[y1c * srcW + x0c];
      const v11 = source[y1c * srcW + x1c];

      const top = v00 * (1 - wx) + v01 * wx;
      const bottom = v10 * (1 - wx) + v11 * wx;
      out[y * dstW + x] = top * (1 - wy) + bottom * wy;
    }
  }
  return out;
}

/**
 * Stroke bounding box over thresholded luminance.
 * Returns null when nothing crosses the threshold.
 */
export function strokeBoundingBox(luminance, width, height, threshold = DEFAULTS.threshold) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let activeCount = 0;
  let inkMass = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = luminance[y * width + x];
      if (value > threshold) {
        activeCount++;
        inkMass += value;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (activeCount === 0 || maxX < minX || maxY < minY) return null;
  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
    activeCount,
    inkMass
  };
}

/**
 * Center of mass over thresholded luminance. Guards the zero-mass division case.
 */
export function centerOfMass(luminance, width, height, threshold = DEFAULTS.threshold) {
  let mass = 0;
  let sumX = 0;
  let sumY = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = luminance[y * width + x];
      if (value > threshold) {
        mass += value;
        sumX += x * value;
        sumY += y * value;
      }
    }
  }

  if (!(mass > EPS)) {
    // Zero-division guard: massless input reports the geometric center instead of NaN.
    return { mass: 0, x: width / 2, y: height / 2 };
  }
  return { mass, x: sumX / mass, y: sumY / mass };
}

/**
 * Background-polarity heuristic: samples a 2px border ring and reports whether the
 * frame is bright (white paper) so dark ink must be flipped to match the model's
 * convention of "ink = high luminance".
 */
export function shouldInvertPolarity(luminance, width, height) {
  if (width < 4 || height < 4) return false;
  let sum = 0;
  let count = 0;
  const ring = 2;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const onRing = x < ring || y < ring || x >= width - ring || y >= height - ring;
      if (!onRing) continue;
      sum += luminance[y * width + x];
      count++;
    }
  }
  if (count === 0) return false;
  return sum / count > 0.5;
}

/**
 * Classify a finished tensor as degenerate (a uniform slab of ink/background) so the
 * studio can refuse to run inference on it instead of inventing a prediction.
 */
export function detectDegenerate(tensor, options = {}) {
  const highWater = options.highWater !== undefined ? options.highWater : 0.97;
  const lowWater = options.lowWater !== undefined ? options.lowWater : 0.02;

  let min = Infinity;
  let max = -Infinity;
  let highCount = 0;
  let lowCount = 0;

  for (let i = 0; i < tensor.length; i++) {
    const value = tensor[i];
    if (value < min) min = value;
    if (value > max) max = value;
    if (value >= highWater) highCount++;
    if (value <= lowWater) lowCount++;
  }

  if (tensor.length === 0) {
    return { degenerate: true, reason: WARN.EMPTY_CANVAS, min: 0, max: 0 };
  }

  const total = tensor.length;
  if (lowCount === total) return { degenerate: true, reason: WARN.EMPTY_CANVAS, min, max };
  if (highCount / total > 0.9) return { degenerate: true, reason: WARN.UNIFORM_INK, min, max };
  // Stroke covering essentially the whole frame carries no digit structure.
  if (max - min < 0.02 && lowCount / total < 0.9) {
    return { degenerate: true, reason: WARN.LOW_CONTRAST, min, max };
  }
  return { degenerate: false, reason: null, min, max };
}

/**
 * THE PIPELINE. Luminance buffer (any size) -> normalized 28x28 tensor.
 *
 * @param {ArrayLike<number>} luminance row-major grayscale, values expected in [0,1]
 * @param {number} width
 * @param {number} height
 * @param {object} options see DEFAULTS
 * @returns {{
 *   tensor: Float64Array, isEmpty: boolean, degenerate: boolean, degenerateReason: string|null,
 *   warnings: string[], scaleX: number, scaleY: number, shiftX: number, shiftY: number,
 *   boundingBox: object|null, activePixelCount: number, inkMass: number, fullFrameInk: boolean,
 *   invertApplied: boolean, polarityConfidence: number
 * }}
 */
export function processLuminance(luminance, width, height, options = {}) {
  const config = { ...DEFAULTS, ...options };
  const targetW = Math.max(1, Math.floor(config.targetWidth));
  const targetH = Math.max(1, Math.floor(config.targetHeight));
  const warnings = [];
  const emptyResult = (extra = {}) => ({
    tensor: new Float64Array(targetW * targetH),
    isEmpty: true,
    degenerate: true,
    degenerateReason: extra.reason || WARN.EMPTY_CANVAS,
    warnings,
    scaleX: 1,
    scaleY: 1,
    shiftX: 0,
    shiftY: 0,
    boundingBox: null,
    activePixelCount: 0,
    inkMass: 0,
    fullFrameInk: false,
    invertApplied: false,
    polarityConfidence: 0,
    ...extra
  });

  if (!luminance || width <= 0 || height <= 0 || luminance.length < width * height) {
    warnings.push(WARN.EMPTY_CANVAS);
    return emptyResult();
  }

  // ---- Stage 1: copy + sanitize (NaN/Inf cannot be allowed to poison the tensor) ----
  const frame = new Float64Array(width * height);
  let nonFinite = 0;
  for (let i = 0; i < frame.length; i++) {
    const raw = luminance[i];
    if (Number.isFinite(raw)) {
      frame[i] = raw < 0 ? 0 : (raw > 1 ? 1 : raw);
    } else {
      frame[i] = 0;
      nonFinite++;
    }
  }
  if (nonFinite > 0) warnings.push(WARN.NON_FINITE_INPUT);

  // ---- Stage 2: polarity normalization ---------------------------------------------
  let invertApplied = false;
  if (config.invert === 'auto') {
    invertApplied = shouldInvertPolarity(frame, width, height);
  } else if (config.invert === true) {
    invertApplied = true;
  }
  if (invertApplied) {
    for (let i = 0; i < frame.length; i++) frame[i] = 1 - frame[i];
  }

  let polarityConfidence = 0;
  {
    const com = centerOfMass(frame, width, height, config.threshold);
    const bg = centerOfMass(
      Float64Array.from(frame, (v) => 1 - v),
      width,
      height,
      config.threshold
    );
    const denom = com.mass + bg.mass;
    polarityConfidence = denom > EPS ? Math.min(com.mass, bg.mass) / denom : 0;
  }

  // ---- Stage 3: bounding box -------------------------------------------------------
  const box = strokeBoundingBox(frame, width, height, config.threshold);
  if (!box) {
    warnings.push(WARN.EMPTY_CANVAS);
    return emptyResult({ invertApplied, polarityConfidence });
  }
  if (box.activeCount < 4) warnings.push(WARN.TINY_STROKE);
  const fullFrameInk = box.width >= width * 0.98 && box.height >= height * 0.98 &&
                       box.inkMass / (width * height) > 0.9;
  if (fullFrameInk) warnings.push(WARN.FULL_FRAME_INK);

  // ---- Stage 4: crop --------------------------------------------------------------
  const crop = new Float64Array(box.width * box.height);
  for (let y = 0; y < box.height; y++) {
    const sourceRow = (box.minY + y) * width;
    for (let x = 0; x < box.width; x++) {
      crop[y * box.width + x] = frame[sourceRow + box.minX + x];
    }
  }

  // ---- Stage 5: aspect preserving downscale into the inner field -------------------
  const innerSize = Math.max(1, Math.floor(Math.min(targetW, targetH) * config.innerFieldRatio));
  let scaleW;
  let scaleH;
  if (box.width >= box.height) {
    scaleW = Math.min(innerSize, box.width);
    scaleH = Math.max(1, Math.round((box.height / box.width) * scaleW));
  } else {
    scaleH = Math.min(innerSize, box.height);
    scaleW = Math.max(1, Math.round((box.width / box.height) * scaleH));
  }

  const resized = resizeBilinear(crop, box.width, box.height, scaleW, scaleH);

  // ---- Stage 6: place into the target field ---------------------------------------
  const canvas = new Float64Array(targetW * targetH);
  let offsetX = Math.floor((targetW - scaleW) / 2);
  let offsetY = Math.floor((targetH - scaleH) / 2);

  for (let y = 0; y < scaleH; y++) {
    const ty = offsetY + y;
    if (ty < 0 || ty >= targetH) continue;
    for (let x = 0; x < scaleW; x++) {
      const tx = offsetX + x;
      if (tx < 0 || tx >= targetW) continue;
      canvas[ty * targetW + tx] = resized[y * scaleW + x];
    }
  }

  // ---- Stage 7: center of mass recentering ----------------------------------------
  const com = centerOfMass(canvas, targetW, targetH, config.threshold);
  let shiftX = 0;
  let shiftY = 0;
  if (com.mass > EPS) {
    const desiredX = (targetW - 1) / 2;
    const desiredY = (targetH - 1) / 2;
    shiftX = clamp(Math.round(desiredX - com.x), -config.maxShift, config.maxShift);
    shiftY = clamp(Math.round(desiredY - com.y), -config.maxShift, config.maxShift);
  }

  let finalTensor = canvas;
  if (shiftX !== 0 || shiftY !== 0) {
    finalTensor = new Float64Array(targetW * targetH);
    for (let y = 0; y < targetH; y++) {
      const sy = y - shiftY;
      if (sy < 0 || sy >= targetH) continue;
      for (let x = 0; x < targetW; x++) {
        const sx = x - shiftX;
        if (sx < 0 || sx >= targetW) continue;
        finalTensor[y * targetW + x] = canvas[sy * targetW + sx];
      }
    }
  }

  // ---- Stage 8: final clamp to [0, 1] ---------------------------------------------
  for (let i = 0; i < finalTensor.length; i++) {
    const value = finalTensor[i];
    finalTensor[i] = Number.isFinite(value) ? (value < 0 ? 0 : (value > 1 ? 1 : value)) : 0;
  }

  // ---- Stage 9: degeneracy classification -------------------------------------------
  // Evaluated on the actual ink content (the resized inner field) rather than the padded
  // 28x28 tensor: the zero padding of a normal drawing must not mask a uniform slab, and
  // a stroke that filled essentially the whole source frame is degenerate by definition.
  const degeneracySource = resized.length > 0 ? resized : finalTensor;
  const degenerateInfo = detectDegenerate(degeneracySource);
  const degenerate = degenerateInfo.degenerate || fullFrameInk;
  const degenerateReason = fullFrameInk ? WARN.FULL_FRAME_INK : degenerateInfo.reason;
  if (degenerate && degenerateReason && !warnings.includes(degenerateReason)) {
    warnings.push(degenerateReason);
  }

  return {
    tensor: finalTensor,
    isEmpty: false,
    degenerate,
    degenerateReason,
    warnings,
    scaleX: scaleW,
    scaleY: scaleH,
    shiftX,
    shiftY,
    boundingBox: box,
    activePixelCount: box.activeCount,
    inkMass: com.mass,
    fullFrameInk,
    invertApplied,
    polarityConfidence
  };
}

/**
 * Extract a luminance buffer from any canvas-like object exposing getImageData().
 * Works with HTMLCanvasElement, OffscreenCanvas, or a Node test double that
 * implements getContext('2d').getImageData().
 */
export function luminanceFromCanvasSurface(surface, options = {}) {
  const width = surface.width;
  const height = surface.height;
  if (!width || !height) return { luminance: new Float64Array(0), width: 0, height: 0, error: 'ZERO_SIZED_SURFACE' };

  const context = surface.getContext('2d');
  if (!context) return { luminance: new Float64Array(0), width, height, error: 'NO_2D_CONTEXT' };

  let imageData;
  try {
    imageData = context.getImageData(0, 0, width, height);
  } catch (err) {
    // Tainted canvas or detached context: report, never fabricate pixels.
    return { luminance: new Float64Array(0), width, height, error: `READ_FAILED: ${err && err.message}` };
  }

  const pixels = imageData.data;
  const luminance = new Float64Array(width * height);
  const useAlpha = options.useAlpha !== false;

  for (let i = 0, p = 0; i < luminance.length; i++, p += 4) {
    const alpha = useAlpha ? pixels[p + 3] / 255 : 1;
    // Rec.601 luma, matching the trained-data convention.
    const luma = (0.299 * pixels[p] + 0.587 * pixels[p + 1] + 0.114 * pixels[p + 2]) / 255;
    luminance[i] = luma * alpha;
  }

  return { luminance, width, height, error: null };
}

export class ImageProcessor {
  /**
   * Browser entry point: canvas pixels -> full preprocessing report.
   */
  static rasterizeCanvas(sourceCanvas, options = {}) {
    const config = { ...DEFAULTS, ...options };
    const targetW = Math.max(1, Math.floor(config.targetWidth));
    const targetH = Math.max(1, Math.floor(config.targetHeight));

    const extracted = luminanceFromCanvasSurface(sourceCanvas, options);
    if (extracted.error) {
      return {
        tensor: new Float64Array(targetW * targetH),
        isEmpty: true,
        degenerate: true,
        degenerateReason: WARN.EMPTY_CANVAS,
        warnings: [WARN.EMPTY_CANVAS],
        error: extracted.error,
        width: targetW,
        height: targetH,
        activePixelCount: 0,
        scaleX: 1,
        scaleY: 1,
        shiftX: 0,
        shiftY: 0,
        boundingBox: null,
        inkMass: 0,
        invertApplied: false
      };
    }

    const result = processLuminance(extracted.luminance, extracted.width, extracted.height, config);
    return { ...result, width: targetW, height: targetH, error: null };
  }

  /**
   * Process an HTML Image / ImageBitmap / VideoFrame element.
   * Auto polarity detection handles the common "black ink on white paper" upload.
   */
  static processImageFile(imageElement, options = {}) {
    if (typeof document === 'undefined') {
      throw new Error('processImageFile requires a DOM environment (use processLuminance in headless tests)');
    }
    const config = { ...DEFAULTS, ...options };
    const scrub = document.createElement('canvas');
    scrub.width = imageElement.naturalWidth || imageElement.width || 280;
    scrub.height = imageElement.naturalHeight || imageElement.height || 280;
    const context = scrub.getContext('2d');
    context.drawImage(imageElement, 0, 0, scrub.width, scrub.height);
    return ImageProcessor.rasterizeCanvas(scrub, config);
  }

  /**
   * Render a normalized tensor back onto a thumbnail canvas (visual confirmation that
   * what the network sees is what the user drew).
   */
  static renderTensorToCanvas(tensor, canvas, width = 28, height = 28) {
    if (!canvas || !tensor) return;
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    const imageData = context.createImageData(width, height);

    for (let i = 0; i < tensor.length && i < width * height; i++) {
      const value = clamp(tensor[i], 0, 1);
      const byteValue = Math.round(value * 255);
      const p = i * 4;
      imageData.data[p] = Math.round(byteValue * 0.1);
      imageData.data[p + 1] = byteValue;
      imageData.data[p + 2] = byteValue;
      imageData.data[p + 3] = 255;
    }

    context.putImageData(imageData, 0, 0);
  }

  /**
   * Convert a raw uint8 grayscale buffer (the encoding used by the shipped MNIST
   * payload) into a normalized Float64Array tensor.
   */
  static tensorFromUint8(bytes, width = 28, height = 28) {
    const expected = width * height;
    const tensor = new Float64Array(expected);
    if (!bytes) return tensor;
    const limit = Math.min(expected, bytes.length);
    for (let i = 0; i < limit; i++) {
      tensor[i] = bytes[i] / 255.0;
    }
    return tensor;
  }

  /**
   * Cosine similarity between two equal length tensors. Used by the verification
   * suite to prove the drawing pipeline reproduces MNIST geometry, and re-usable
   * at runtime for "does this drawing resemble the training distribution" checks.
   */
  static cosineSimilarity(a, b) {
    const length = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < length; i++) {
      const va = Number.isFinite(a[i]) ? a[i] : 0;
      const vb = Number.isFinite(b[i]) ? b[i] : 0;
      dot += va * vb;
      normA += va * va;
      normB += vb * vb;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > EPS ? dot / denom : 0;
  }
}

export default ImageProcessor;
