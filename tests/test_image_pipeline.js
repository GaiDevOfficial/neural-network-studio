/**
 * PHASE 3 - TEST 3: Real image preprocessing pipeline & degenerate-input handling
 *
 * Requirement under test: raw pixels (drawn stroke, uploaded raster, real MNIST digit) are
 * converted into a 28x28 tensor through the SAME pure functions the browser uses, and every
 * hostile input (empty canvas, all-black, all-white, flat gray, NaN/Inf pixels, zero-mass
 * frames) is handled with a defined, honest outcome - never a NaN, never a crash, never a
 * fabricated digit.
 *
 * The final section proves the pipeline geometrically reproduces a REAL MNIST digit after a
 * synthetic 5x upscale and again after a translation, measured with cosine similarity against
 * the original published tensor.
 *
 * File: tests/test_image_pipeline.js
 */

import {
  ImageProcessor,
  processLuminance,
  resizeBilinear,
  strokeBoundingBox,
  centerOfMass,
  shouldInvertPolarity,
  detectDegenerate,
  WARN,
  DEFAULTS
} from '../src/pipeline/image_processor.js';
import { loadMnistForTests, banner } from './mnist_test_utils.js';
import { MNISTLoader } from '../src/engine/mnist_loader.js';

const failures = [];
const checks = { passed: 0, total: 0 };

function check(condition, label) {
  checks.total++;
  if (condition) {
    checks.passed++;
    console.log(`  [PASS] ${label}`);
  } else {
    failures.push(label);
    console.error(`  [FAIL] ${label}`);
  }
}

function fill(width, height, value) {
  return Float64Array.from({ length: width * height }, () => value);
}

function paintRect(frame, width, height, x0, y0, x1, y1, value) {
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      frame[y * width + x] = value;
    }
  }
  return frame;
}

function translate(frame, width, height, dx, dy) {
  const out = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sx = x - dx;
      const sy = y - dy;
      if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
        out[y * width + x] = frame[sy * width + sx];
      }
    }
  }
  return out;
}

function allFinite(tensor) {
  for (let i = 0; i < tensor.length; i++) {
    if (!Number.isFinite(tensor[i])) return false;
  }
  return true;
}

function minMax(tensor) {
  let min = Infinity;
  let max = -Infinity;
  for (const v of tensor) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, max };
}

// ---------------------------------------------------------------------------------------------
// Part 1: degenerate canvas inputs
// ---------------------------------------------------------------------------------------------
function testDegenerateGrids() {
  banner('PART 1 - Degenerate canvas inputs');

  const empty = processLuminance(null, 0, 0);
  check(empty.isEmpty === true && empty.degenerate === true && empty.tensor.length === 784,
    'null input -> empty 28x28 tensor flagged as EMPTY_CANVAS');
  check(empty.warnings.includes(WARN.EMPTY_CANVAS) && allFinite(empty.tensor), 'null input carries the honest warning, tensor all finite');
  check(empty.tensor.every((v) => v === 0), 'no fabricated ink for a null input');

  const short = processLuminance(new Float64Array(50), 10, 10);
  check(short.isEmpty === true && short.tensor.length === 784, 'undersized buffer rejected as empty');

  const allBlack = processLuminance(fill(28, 28, 0), 28, 28);
  check(allBlack.isEmpty === true && allBlack.warnings.includes(WARN.EMPTY_CANVAS) && allFinite(allBlack.tensor),
    'all-black canvas -> EMPTY_CANVAS, no NaN');

  const allWhite = processLuminance(fill(28, 28, 1), 28, 28);
  check(allWhite.invertApplied === true && allWhite.isEmpty === true,
    'all-white canvas is polarity-inverted and then truthfully reported empty');

  const allWhiteForced = processLuminance(fill(28, 28, 1), 28, 28, { invert: false });
  check(allWhiteForced.isEmpty === false && allWhiteForced.degenerate === true &&
        allWhiteForced.degenerateReason === WARN.FULL_FRAME_INK,
    'all-white canvas without inversion -> degenerate FULL_FRAME_INK (refused, not classified)');
  check(allWhiteForced.warnings.includes(WARN.FULL_FRAME_INK), 'FULL_FRAME_INK warning surfaced to the UI');

  const flatGray = processLuminance(fill(28, 28, 0.5), 28, 28, { invert: false });
  check(flatGray.degenerate === true && flatGray.degenerateReason === WARN.LOW_CONTRAST,
    'uniform mid-gray canvas -> degenerate LOW_CONTRAST');
  check(allFinite(flatGray.tensor), 'flat gray tensor is finite');

  const belowThreshold = processLuminance(fill(28, 28, 0.05), 28, 28);
  check(belowThreshold.isEmpty === true, 'every pixel below threshold -> EMPTY_CANVAS (no division by zero)');

  const tiny = fill(28, 28, 0);
  tiny[14 * 28 + 14] = 1;
  const tinyResult = processLuminance(tiny, 28, 28);
  check(tinyResult.isEmpty === false && tinyResult.warnings.includes(WARN.TINY_STROKE),
    'single-pixel stroke -> processed but flagged TINY_STROKE');

  const { min, max } = minMax(tinyResult.tensor);
  check(min >= 0 && max <= 1, 'every produced tensor value lies in [0, 1]');
}

// ---------------------------------------------------------------------------------------------
// Part 2: NaN / infinite pixel handling
// ---------------------------------------------------------------------------------------------
function testNonFinitePixels() {
  banner('PART 2 - NaN / Infinity pixel sanitisation');

  const frame = fill(28, 28, 0);
  paintRect(frame, 28, 28, 8, 4, 19, 23, 1.0);
  frame[3] = NaN;
  frame[50] = Infinity;
  frame[51] = -Infinity;
  frame[52] = 42; // out of range on purpose

  const result = processLuminance(frame, 28, 28);
  check(result.warnings.includes(WARN.NON_FINITE_INPUT), 'NaN/Inf pixels reported through NON_FINITE_INPUT');
  check(allFinite(result.tensor), 'sanitised tensor contains no non-finite values');
  const { max } = minMax(result.tensor);
  check(max <= 1 && result.isEmpty === false, 'out-of-range pixel clamped to [0, 1]; stroke preserved');

  const allNaN = processLuminance(fill(28, 28, NaN), 28, 28);
  check(allNaN.isEmpty === true && allNaN.warnings.includes(WARN.NON_FINITE_INPUT), 'all-NaN frame degrades to EMPTY_CANVAS, never NaN');
}

// ---------------------------------------------------------------------------------------------
// Part 3: pure geometry helpers
// ---------------------------------------------------------------------------------------------
function testGeometryHelpers() {
  banner('PART 3 - Bounding box, center of mass, polarity, bilinear resize');

  const zeros = fill(20, 20, 0);
  check(strokeBoundingBox(zeros, 20, 20) === null, 'empty frame has no bounding box (no zero-mass math)');

  const frame = fill(60, 40, 0);
  paintRect(frame, 60, 40, 20, 14, 39, 23, 0.8);
  const box = strokeBoundingBox(frame, 60, 40);
  check(box && box.minX === 20 && box.maxX === 39 && box.minY === 14 && box.maxY === 23 &&
        box.width === 20 && box.height === 10,
    `bounding box measured exactly (x 20..39, y 14..23)`);

  const com = centerOfMass(zeros, 20, 20);
  check(com.mass === 0 && Number.isFinite(com.x) && Number.isFinite(com.y) && com.x === 10 && com.y === 10,
    'zero-mass center of mass falls back to the geometric center (no division by zero)');

  const comRect = centerOfMass(frame, 60, 40);
  check(Math.abs(comRect.x - 29.5) < 1e-9 && Math.abs(comRect.y - 18.5) < 1e-9,
    'center of mass of the painted rectangle measured exactly');

  check(shouldInvertPolarity(fill(28, 28, 1), 28, 28) === true, 'white-paper ring detected -> invert');
  check(shouldInvertPolarity(fill(28, 28, 0), 28, 28) === false, 'dark-paper ring detected -> no invert');
  check(shouldInvertPolarity(fill(3, 3, 1), 3, 3) === false, 'frames smaller than 4px skip polarity heuristics');

  // Bilinear resize: identity at equal size
  const noise = Float64Array.from({ length: 35 }, (_, i) => Math.sin(i * 1.7) * 0.5 + 0.5);
  const identity = resizeBilinear(noise, 5, 7, 5, 7);
  let maxDiff = 0;
  for (let i = 0; i < noise.length; i++) maxDiff = Math.max(maxDiff, Math.abs(identity[i] - noise[i]));
  check(maxDiff < 1e-15, `bilinear resize at identical size is exact (max diff ${maxDiff.toExponential(1)})`);

  // Known case: 2x2 -> 1x1 is the exact mean
  const quad = Float64Array.from([0.1, 0.3, 0.5, 0.9]);
  const point = resizeBilinear(quad, 2, 2, 1, 1);
  check(Math.abs(point[0] - 0.45) < 1e-12, '2x2 -> 1x1 bilinear equals the arithmetic mean (0.45)');

  // Known case: 1x1 -> any size replicates the value
  const single = resizeBilinear(Float64Array.from([0.7]), 1, 1, 4, 3);
  check(single.length === 12 && single.every((v) => Math.abs(v - 0.7) < 1e-12), '1x1 -> 4x3 replicates the constant');

  // Known case: horizontal ramp downsampled by exactly 2 averages adjacent pairs
  const ramp = Float64Array.from([0, 1, 0, 1, 0, 1, 0, 1]);
  const halved = resizeBilinear(ramp, 8, 1, 4, 1);
  check(Array.from(halved).every((v) => Math.abs(v - 0.5) < 1e-12), '8 -> 4 horizontal halving averages adjacent pairs exactly');

  check(resizeBilinear(noise, 5, 7, 0, 3).length === 0, 'zero-width destination returns an empty buffer, no crash');

  // Aspect ratio is preserved when the stroke is placed in the inner field
  const processed = processLuminance(frame, 60, 40, { invert: false });
  const aspectBefore = box.height / box.width;
  const aspectAfter = processed.scaleY / processed.scaleX;
  check(Math.abs(aspectAfter - aspectBefore) <= 1 / processed.scaleX,
    `aspect ratio preserved into the inner field (${aspectBefore.toFixed(3)} vs ${aspectAfter.toFixed(3)})`);
  check(processed.scaleX <= Math.floor(28 * DEFAULTS.innerFieldRatio) + 1,
    `stroke scaled into the 20-pixel inner field (scaleX=${processed.scaleX})`);

  // detectDegenerate direct behaviour
  check(detectDegenerate(new Float64Array(0)).reason === WARN.EMPTY_CANVAS, 'detectDegenerate: empty tensor -> EMPTY_CANVAS');
  check(detectDegenerate(fill(16, 16, 0)).reason === WARN.EMPTY_CANVAS, 'detectDegenerate: zero tensor -> EMPTY_CANVAS');
  check(detectDegenerate(fill(16, 16, 1)).reason === WARN.UNIFORM_INK, 'detectDegenerate: solid ink -> UNIFORM_INK');
  check(detectDegenerate(fill(16, 16, 0.5)).reason === WARN.LOW_CONTRAST, 'detectDegenerate: flat mid-gray -> LOW_CONTRAST');
  check(detectDegenerate(noise).degenerate === false, 'detectDegenerate: varied content passes');

  // Cosine similarity guards
  check(Math.abs(ImageProcessor.cosineSimilarity(noise, noise) - 1) < 1e-12, 'cosine similarity of a tensor with itself is 1');
  check(ImageProcessor.cosineSimilarity(new Float64Array(9), noise) === 0, 'cosine similarity against zero mass is 0, not NaN');
  const withNaN = Float64Array.from([NaN, 1, Infinity, 0.5]);
  check(Number.isFinite(ImageProcessor.cosineSimilarity(withNaN, withNaN)), 'cosine similarity tolerates non-finite entries');

  // tensorFromUint8
  const bytes = new Uint8Array(784).fill(255);
  const tensor = ImageProcessor.tensorFromUint8(bytes);
  check(tensor.length === 784 && tensor.every((v) => v === 1), 'uint8 255 -> tensor value 1.0');
  const shortBytes = ImageProcessor.tensorFromUint8(new Uint8Array(10));
  check(shortBytes.length === 784 && shortBytes[783] === 0, 'short byte buffer zero-pads instead of throwing');
}

// ---------------------------------------------------------------------------------------------
// Part 4: canvas surface extraction (the exact path a drawn stroke takes)
// ---------------------------------------------------------------------------------------------
function testCanvasExtraction() {
  banner('PART 4 - Canvas extraction (RGBA pixels -> luminance -> tensor)');

  const width = 64;
  const height = 64;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = (y * width + x) * 4;
      const isStroke = x >= 28 && x <= 34;
      const value = isStroke ? 0 : 255;
      data[p] = value;
      data[p + 1] = value;
      data[p + 2] = value;
      data[p + 3] = 255;
    }
  }
  const surface = {
    width,
    height,
    getContext: () => ({ getImageData: () => ({ data, width, height }) })
  };

  const report = ImageProcessor.rasterizeCanvas(surface);
  check(report.error === null && report.isEmpty === false, 'drawn bar rasterised through the production canvas path');
  check(report.invertApplied === true, 'black ink on white canvas polarity-corrected automatically');
  check(report.boundingBox && report.boundingBox.minX === 28 && report.boundingBox.maxX === 34,
    `stroke bounding box recovered from pixels (x 28..34)`);
  check(report.tensor.length === 784 && allFinite(report.tensor), 'canvas path produces a finite 28x28 tensor');

  const transparent = {
    width: 32,
    height: 32,
    getContext: () => ({ getImageData: () => ({ data: new Uint8ClampedArray(32 * 32 * 4), width: 32, height: 32 }) })
  };
  const transparentReport = ImageProcessor.rasterizeCanvas(transparent);
  check(transparentReport.isEmpty === true, 'fully transparent canvas (alpha=0) -> EMPTY_CANVAS, no phantom ink');

  const tainted = {
    width: 32,
    height: 32,
    getContext: () => ({ getImageData: () => { throw new Error('tainted'); } })
  };
  const taintedReport = ImageProcessor.rasterizeCanvas(tainted);
  check(taintedReport.isEmpty === true && /^READ_FAILED/.test(taintedReport.error),
    'unreadable canvas surfaces the real error instead of fabricating pixels');

  const zeroSized = { width: 0, height: 0, getContext: () => null };
  const zeroReport = ImageProcessor.rasterizeCanvas(zeroSized);
  check(zeroReport.isEmpty === true && zeroReport.error === 'ZERO_SIZED_SURFACE', 'zero-sized surface rejected cleanly');
}

// ---------------------------------------------------------------------------------------------
// Part 5: end-to-end with a REAL MNIST digit
// ---------------------------------------------------------------------------------------------
function bestAlignedSimilarity(reference, candidate, maxShift = 3) {
  let best = { similarity: -1, dx: 0, dy: 0 };
  const shifted = new Float64Array(reference.length);
  for (let dy = -maxShift; dy <= maxShift; dy++) {
    for (let dx = -maxShift; dx <= maxShift; dx++) {
      shifted.fill(0);
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const sy = y - dy;
          const sx = x - dx;
          if (sy >= 0 && sy < 28 && sx >= 0 && sx < 28) shifted[y * 28 + x] = candidate[sy * 28 + sx];
        }
      }
      const similarity = ImageProcessor.cosineSimilarity(reference, shifted);
      if (similarity > best.similarity) best = { similarity, dx, dy };
    }
  }
  return best;
}

async function testRealDigitGeometry() {
  banner('PART 5 - Real MNIST digit through the production pipeline');

  const dataset = await loadMnistForTests({ maxTrain: 10, maxTest: 60 });
  let index = 0;
  let bestInk = -1;
  for (let i = 0; i < dataset.testX.length; i++) {
    let ink = 0;
    for (const v of dataset.testX[i]) ink += v;
    if (ink > bestInk) { bestInk = ink; index = i; }
  }
  const original = dataset.testX[index];
  console.log(`  selected test image #${index} (class ${dataset.testY[index].indexOf(1)}, ink mass ${bestInk.toFixed(1)})`);

  // The same digit through two input scales: native 28x28 and a 5x bilinear upscale (140x140).
  const nativeResult = processLuminance(original, 28, 28);
  const upscaled = resizeBilinear(original, 28, 28, 140, 140);
  const upscaledResult = processLuminance(upscaled, 140, 140);
  check(nativeResult.isEmpty === false && nativeResult.degenerate === false, 'native 28x28 real digit processed as valid');
  check(upscaledResult.isEmpty === false && upscaledResult.degenerate === false, 'upscaled 140x140 real digit processed as valid');

  // 1. Scale invariance: the drawing pipeline must produce the same tensor whatever the
  //    canvas resolution, because a real user draws at whatever size their window has.
  const scaleSimilarity = ImageProcessor.cosineSimilarity(nativeResult.tensor, upscaledResult.tensor);
  console.log(`  cos(pipeline(28x28), pipeline(5x upscale)) = ${scaleSimilarity.toFixed(4)}`);
  check(scaleSimilarity >= 0.95, `pipeline is invariant to input resolution (${scaleSimilarity.toFixed(4)} >= 0.95)`);

  // 2. Translation invariance: shifting the digit inside the frame by (5, 7) px must be undone
  //    by the bounding-box crop + center-of-mass recentering, recovering the identical tensor.
  const shifted = translate(upscaled, 140, 140, 5, 7);
  const shiftedResult = processLuminance(shifted, 140, 140);
  const translationSimilarity = ImageProcessor.cosineSimilarity(upscaledResult.tensor, shiftedResult.tensor);
  console.log(`  cos(pipeline(upscaled), pipeline(upscaled + (5,7) px)) = ${translationSimilarity.toFixed(4)} ` +
              `(recentering shift ${shiftedResult.shiftX}, ${shiftedResult.shiftY})`);
  check(translationSimilarity >= 0.99, `translation of the digit is fully recentered (${translationSimilarity.toFixed(4)} >= 0.99)`);

  // 3. Fidelity to the PUBLISHED tensor: shape and scale must match the corpus geometry.
  //    The classic LeCun preprocessing and this pipeline differ by at most one pixel of
  //    centering convention, so the honest claim is: unaligned similarity > 0.85 and
  //    best-shift-aligned similarity > 0.95 with the alignment at most 1 pixel.
  const unaligned = ImageProcessor.cosineSimilarity(original, shiftedResult.tensor);
  const aligned = bestAlignedSimilarity(original, shiftedResult.tensor, 3);
  console.log(`  cos(original, processed) = ${unaligned.toFixed(4)}; best aligned = ${aligned.similarity.toFixed(4)} ` +
              `at (${aligned.dx}, ${aligned.dy}) px`);
  check(unaligned >= 0.85, `processed digit resembles the published tensor (${unaligned.toFixed(4)} >= 0.85)`);
  check(aligned.similarity >= 0.95, `processed digit matches the published geometry within rounding (${aligned.similarity.toFixed(4)} >= 0.95)`);
  check(Math.abs(aligned.dx) + Math.abs(aligned.dy) <= 1,
    `placement convention differs by at most one pixel (best shift ${aligned.dx}, ${aligned.dy})`);

  // Center of mass of the produced tensor must sit near the middle of the field.
  const com = centerOfMass(shiftedResult.tensor, 28, 28);
  const driftX = Math.abs(com.x - 13.5);
  const driftY = Math.abs(com.y - 13.5);
  check(driftX <= DEFAULTS.maxShift + 1 && driftY <= DEFAULTS.maxShift + 1,
    `center of mass recentered (drift ${driftX.toFixed(2)}, ${driftY.toFixed(2)} px)`);

  // ASCII sanity: the processed tensor must still look like the digit, not noise.
  const ascii = MNISTLoader.toAscii(Array.from(shiftedResult.tensor));
  console.log('  processed tensor:');
  for (const line of ascii.split('\n')) console.log(`    |${line}|`);
  check(ascii.includes('#') || ascii.includes('+'), 'processed output still contains digit ink');
}

// ---------------------------------------------------------------------------------------------
async function main() {
  banner('IMAGE PIPELINE VERIFICATION');

  testDegenerateGrids();
  testNonFinitePixels();
  testGeometryHelpers();
  testCanvasExtraction();
  await testRealDigitGeometry();

  banner(failures.length === 0 ? 'IMAGE PIPELINE TEST PASSED' : 'IMAGE PIPELINE TEST FAILED');
  console.log(`  checks passed: ${checks.passed}/${checks.total}`);
  if (failures.length === 0) {
    console.log('  [PASS] Raw pixels -> 28x28 tensor verified for real drawings and hostile inputs\n');
    return 0;
  }
  for (const failure of failures) console.error(`  [FAIL] ${failure}`);
  console.error('');
  return 1;
}

main()
  .then((code) => process.exit(code))
  .catch((err) => {
    console.error('\n[FAIL] Image pipeline test crashed:', err);
    process.exit(1);
  });
