/**
 * Node Studio — NPZ (NumPy archive) writer.
 *
 * Produces real `.npz` files (ZIP container of `.npy` arrays + meta.json) with zero
 * dependencies: the .npy v1.0 binary format and stored (uncompressed) ZIP entries are
 * implemented here directly. The output is verified by loading it with Python's numpy
 * in tests/test_node_exports.js — byte-level compatibility is measured, not assumed.
 *
 * File: src/export/npz_writer.js
 */

/* ------------------------------------------------------------------ npy */

/** Serialize a 1-D or 2-D numeric array as a .npy v1.0 buffer (little-endian float64). */
export function serializeNpy(array) {
  const is2d = Array.isArray(array) && Array.isArray(array[0]);
  const rows = is2d ? array.length : 1;
  const cols = is2d ? (array[0] ? array[0].length : 0) : (array ? array.length : 0);

  for (let r = 0; r < rows; r++) {
    const row = is2d ? array[r] : array;
    for (let c = 0; c < cols; c++) {
      if (!Number.isFinite(row[c])) {
        throw new Error(`serializeNpy: non-finite value at [${r},${c}] — refusing to write a corrupt archive.`);
      }
    }
  }

  const shapeText = is2d ? `(${rows}, ${cols})` : `(${cols},)`;
  let header = `{'descr': '<f8', 'fortran_order': False, 'shape': ${shapeText}, }`;
  // Pad so that magic(6) + version(2) + hlen(2) + header + '\n' is 64-byte aligned.
  const pad = (64 - ((header.length + 10 + 1) % 64)) % 64;
  header += ' '.repeat(pad) + '\n';

  const headerBytes = utf8Bytes(header);
  const dataBytes = new Uint8Array(rows * cols * 8);
  const view = new DataView(dataBytes.buffer);
  let offset = 0;
  for (let r = 0; r < rows; r++) {
    const row = is2d ? array[r] : array;
    for (let c = 0; c < cols; c++) {
      view.setFloat64(offset, Number(row[c]), true);
      offset += 8;
    }
  }

  const out = new Uint8Array(10 + headerBytes.length + dataBytes.length);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0); // \x93NUMPY
  out[6] = 1; // major
  out[7] = 0; // minor
  new DataView(out.buffer).setUint16(8, headerBytes.length, true);
  out.set(headerBytes, 10);
  out.set(dataBytes, 10 + headerBytes.length);
  return out;
}

/* ------------------------------------------------------------------ zip */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function u16(value) {
  const b = new Uint8Array(2);
  new DataView(b.buffer).setUint16(0, value & 0xffff, true);
  return b;
}

function u32(value) {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setUint32(0, value >>> 0, true);
  return b;
}

/**
 * Build a stored-method (no compression) ZIP archive.
 * @param {Array<{name: string, data: Uint8Array}>} entries
 */
export function zipStore(entries) {
  const chunks = [];
  const central = [];
  let offset = 0;

  // Fixed DOS date: 1980-01-01 00:00:00.
  const dosTime = 0;
  const dosDate = (0 << 9) | (1 << 5) | 1;

  for (const entry of entries) {
    const nameBytes = utf8Bytes(entry.name);
    const crc = crc32(entry.data);
    const size = entry.data.length;

    const localHeader = [
      u32(0x04034b50), u16(20), u16(0), u16(0), u16(dosTime), u16(dosDate),
      u32(crc), u32(size), u32(size), u16(nameBytes.length), u16(0)
    ];
    for (const chunk of localHeader) chunks.push(chunk);
    chunks.push(nameBytes);
    chunks.push(entry.data);

    central.push([
      u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(dosTime), u16(dosDate),
      u32(crc), u32(size), u32(size), u16(nameBytes.length), u16(0), u16(0),
      u16(0), u16(0), u32(0), u32(offset)
    ]);
    central[central.length - 1]._name = nameBytes;

    offset += 30 + nameBytes.length + size;
  }

  const centralStart = offset;
  for (const header of central) {
    for (const chunk of header) chunks.push(chunk);
    chunks.push(header._name);
    offset += 46 + header._name.length;
  }
  const centralSize = offset - centralStart;

  const eocd = [
    u32(0x06054b50), u16(0), u16(0), u16(central.length), u16(central.length),
    u32(centralSize), u32(centralStart), u16(0)
  ];
  for (const chunk of eocd) chunks.push(chunk);

  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let cursor = 0;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.length;
  }
  return out;
}

/* ------------------------------------------------------------------ npz */

export function utf8Bytes(text) {
  if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(text);
  // Node fallback without TextEncoder (ancient runtimes only).
  const buffer = Buffer.from(text, 'utf8');
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

/**
 * Build a real `.npz` archive: one pair of .npy arrays per layer + meta.json.
 * @param {Array} weightsSnapshot same shape as NeuralNetwork.getWeightsSnapshot()
 * @param {object} meta JSON metadata (ModelExporter-compatible)
 */
export function buildNpzBytes(weightsSnapshot, meta = {}) {
  if (!Array.isArray(weightsSnapshot) || weightsSnapshot.length === 0) {
    throw new Error('buildNpzBytes: empty weights snapshot.');
  }
  const entries = [];
  weightsSnapshot.forEach((layer, idx) => {
    entries.push({ name: `layer${idx}_weights.npy`, data: serializeNpy(layer.weights) });
    entries.push({ name: `layer${idx}_biases.npy`, data: serializeNpy(layer.biases) });
  });
  entries.push({ name: 'meta.json', data: utf8Bytes(JSON.stringify(meta, null, 2)) });
  return zipStore(entries);
}
