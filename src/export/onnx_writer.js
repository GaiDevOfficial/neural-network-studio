/**
 * Dependency-free ONNX exporter.
 *
 * Emits a real ONNX ModelProto (protobuf wire format written by hand) for the trained dense
 * stack: one `Gemm` per layer (`transB = 1`, float32 initializers for W and b) followed by the
 * layer's activation (`Relu`, `LeakyRelu`, `Sigmoid`, `Tanh` or `Softmax`). Opset 13, IR 8.
 *
 * No `onnx`, `onnxruntime` or npm package is required in the browser. The emitted bytes are
 * verified against the JS engine by loading them in onnxruntime in tests/test_onnx_export.js.
 *
 * File: src/export/onnx_writer.js
 */

const ONNX_IR_VERSION = 8;
const ONNX_OPSET = 13;

/* ------------------------------------------------------------ protobuf core */

function concat(parts) {
  let total = 0;
  for (const part of parts) total += part.length;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function varint(value) {
  let v = Math.trunc(value);
  if (v < 0) {
    // Emit a 64-bit two's-complement varint for negative values.
    let big = BigInt(v);
    big = BigInt.asUintN(64, big);
    const bytes = [];
    while (big > 0x7fn) {
      bytes.push(Number(big & 0x7fn) | 0x80);
      big >>= 7n;
    }
    bytes.push(Number(big));
    return Uint8Array.from(bytes);
  }
  const bytes = [];
  while (v > 0x7f) {
    bytes.push((v & 0x7f) | 0x80);
    v = Math.floor(v / 128);
  }
  bytes.push(v);
  return Uint8Array.from(bytes);
}

function tag(field, wireType) {
  return varint((field << 3) | wireType);
}

function varintField(field, value) {
  return concat([tag(field, 0), varint(value)]);
}

function bytesField(field, bytes) {
  return concat([tag(field, 2), varint(bytes.length), bytes]);
}

function stringField(field, value) {
  return bytesField(field, new TextEncoder().encode(String(value)));
}

function floatField(field, value) {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setFloat32(0, value, true);
  return concat([tag(field, 5), bytes]);
}

function attrFloat(name, value) {
  return concat([stringField(1, name), floatField(2, value), varintField(20, 1)]);
}

function attrInt(name, value) {
  return concat([stringField(1, name), varintField(3, value), varintField(20, 2)]);
}

/* --------------------------------------------------------------- ONNX parts */

function tensorFloat(name, dims, values) {
  const raw = new Uint8Array(values.length * 4);
  const view = new DataView(raw.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat32(i * 4, values[i], true);
  return concat([
    ...dims.map((dim) => varintField(1, dim)),
    varintField(2, 1),                 // TensorProto.DataType.FLOAT
    stringField(8, name),
    bytesField(9, raw)                 // raw_data, little-endian float32
  ]);
}

function valueInfo(name, dims) {
  const dimensions = dims.map((dim) => bytesField(1, varintField(1, dim)));
  const shape = concat(dimensions);
  const tensorType = concat([
    varintField(1, 1),                 // TensorProto.DataType.FLOAT
    bytesField(2, shape)               // TensorShapeProto
  ]);
  return concat([
    stringField(1, name),
    bytesField(2, bytesField(1, tensorType)) // TypeProto.tensor_type
  ]);
}

function nodeProto(inputs, outputs, name, opType, attributes = []) {
  return concat([
    ...inputs.map((input) => stringField(1, input)),
    ...outputs.map((output) => stringField(2, output)),
    stringField(3, name),
    stringField(4, opType),
    ...attributes.map((attribute) => bytesField(5, attribute))
  ]);
}

function activationNode(type, input, output, index) {
  switch (type) {
    case 'relu':
      return nodeProto([input], [output], `act_${index}_relu`, 'Relu');
    case 'leaky_relu':
      // Matches the engine's fixed LEAKY_ALPHA = 0.01.
      return nodeProto([input], [output], `act_${index}_leaky_relu`, 'LeakyRelu', [attrFloat('alpha', 0.01)]);
    case 'sigmoid':
      return nodeProto([input], [output], `act_${index}_sigmoid`, 'Sigmoid');
    case 'tanh':
      return nodeProto([input], [output], `act_${index}_tanh`, 'Tanh');
    case 'softmax':
      return nodeProto([input], [output], `act_${index}_softmax`, 'Softmax', [attrInt('axis', 1)]);
    case 'linear':
    case '':
    case null:
    case undefined:
      return null;
    default:
      throw new Error(`ONNX export does not support activation "${type}".`);
  }
}

function assertFiniteRow(row, label) {
  for (let i = 0; i < row.length; i++) {
    if (!Number.isFinite(row[i])) throw new Error(`ONNX export: non-finite value in ${label}[${i}].`);
  }
}

/* --------------------------------------------------------------- exporter */

/**
 * @param {object} options
 *   plan        compiled graph plan (plan.network.architecture is used for metadata)
 *   weights     NeuralNetwork.getWeightsSnapshot()
 *   datasetName dataset label for the doc string
 *   epochs      measured epochs trained
 * @returns {Uint8Array} ONNX ModelProto bytes
 */
export function buildOnnxBytes({ plan, weights, datasetName = 'dataset', epochs = 0 }) {
  if (!Array.isArray(weights) || weights.length === 0) {
    throw new Error('ONNX export needs a trained weight snapshot.');
  }

  const nodes = [];
  const initializers = [];
  let previous = 'input';

  for (let i = 0; i < weights.length; i++) {
    const layer = weights[i];
    const inputSize = layer.inputSize;
    const outputSize = layer.outputSize;
    if (!Number.isInteger(inputSize) || !Number.isInteger(outputSize) || inputSize <= 0 || outputSize <= 0) {
      throw new Error(`ONNX export: layer ${i} has invalid dimensions.`);
    }
    if (!Array.isArray(layer.weights) || layer.weights.length !== outputSize) {
      throw new Error(`ONNX export: layer ${i} weight matrix does not match outputSize.`);
    }
    if (!Array.isArray(layer.biases) || layer.biases.length !== outputSize) {
      throw new Error(`ONNX export: layer ${i} bias vector does not match outputSize.`);
    }

    const flat = [];
    for (let out = 0; out < outputSize; out++) {
      const row = layer.weights[out];
      if (!Array.isArray(row) || row.length !== inputSize) {
        throw new Error(`ONNX export: layer ${i} row ${out} does not match inputSize.`);
      }
      assertFiniteRow(row, `layer ${i} W[${out}]`);
      for (let j = 0; j < inputSize; j++) flat.push(row[j]);
    }
    assertFiniteRow(layer.biases, `layer ${i} b`);

    const weightName = `W_${i}`;
    const biasName = `b_${i}`;
    initializers.push(tensorFloat(weightName, [outputSize, inputSize], flat));
    initializers.push(tensorFloat(biasName, [outputSize], layer.biases));

    const isLast = i === weights.length - 1;
    const activation = layer.activation || 'linear';
    const gemmOutput = (isLast && (activation === 'linear' || !activation)) ? 'output' : `dense_${i}`;
    nodes.push(nodeProto(
      [previous, weightName, biasName],
      [gemmOutput],
      `gemm_${i}`,
      'Gemm',
      [attrInt('transB', 1)]
    ));

    const activationOut = isLast ? 'output' : `act_${i}`;
    const activationNodeBytes = activationNode(activation, gemmOutput, activationOut, i);
    if (activationNodeBytes) {
      nodes.push(activationNodeBytes);
      previous = activationOut;
    } else {
      previous = gemmOutput;
    }
  }

  const first = weights[0];
  const last = weights[weights.length - 1];
  const graph = concat([
    ...nodes.map((node) => bytesField(1, node)),
    stringField(2, 'neural_network_studio'),
    ...initializers.map((tensor) => bytesField(5, tensor)),
    bytesField(11, valueInfo('input', [1, first.inputSize])),
    bytesField(12, valueInfo('output', [1, last.outputSize]))
  ]);

  const doc = [
    `Neural Network Studio — Node Studio dense export`,
    `dataset=${datasetName}`,
    `epochs=${epochs}`,
    plan && plan.network ? `loss=${plan.network.loss};optimizer=${plan.network.optimizer}` : '',
    `layers=${weights.length}`
  ].filter(Boolean).join(' | ');

  const opset = varintField(2, ONNX_OPSET);
  return concat([
    varintField(1, ONNX_IR_VERSION),
    stringField(2, 'neural-network-studio'),
    stringField(3, '1.0'),
    stringField(4, 'studio.node'),
    stringField(6, doc),
    bytesField(7, graph),
    bytesField(8, opset)
  ]);
}

export const ONNX_INFO = {
  irVersion: ONNX_IR_VERSION,
  opset: ONNX_OPSET
};
