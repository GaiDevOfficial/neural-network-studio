"""
ONNX parity checker used by tests/test_onnx_export.js.

Loads a model emitted by src/export/onnx_writer.js, validates it with the official ONNX
checker, runs every probe through onnxruntime, and prints the measured outputs as JSON.

Usage: python tools/onnx_parity_check.py <payload.json>
payload = {"model": "path/to/model.onnx", "inputs": [[...], ...]}
"""

import json
import sys

import onnx
import onnxruntime as ort


def main() -> int:
    payload_path = sys.argv[1]
    with open(payload_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    model_path = payload["model"]
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = []
    for probe in payload["inputs"]:
        import numpy as np

        array = np.asarray([probe], dtype=np.float32)
        outputs.append(session.run(None, {input_name: array})[0][0].tolist())

    print(json.dumps({
        "ok": True,
        "outputs": outputs,
        "ir_version": model.ir_version,
        "opset": model.opset_import[0].version,
        "producer": model.producer_name,
        "graph_nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "input_dims": [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim],
        "output_dims": [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
