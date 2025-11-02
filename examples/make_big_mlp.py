#!/usr/bin/env python3
import onnx, numpy as np
from onnx import helper, TensorProto
from pathlib import Path

# 4 → 128 → 128 → 64 → 32 → 8 → 3 with ReLU between layers
in_dim = 4
layers = [128, 128, 64, 32, 8, 3]

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, in_dim])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, layers[-1]])

nodes = []
in_name = "x"
in_size = in_dim
inits = []

for i, out_size in enumerate(layers, start=1):
    W = np.random.randn(in_size, out_size).astype("float32") * 0.02
    b = np.zeros((out_size,), dtype="float32")
    Wi = helper.make_tensor(f"W{i}", TensorProto.FLOAT, W.shape, W.ravel())
    bi = helper.make_tensor(f"b{i}", TensorProto.FLOAT, b.shape, b.ravel())
    inits += [Wi, bi]

    mm = helper.make_node("MatMul", [in_name, f"W{i}"], [f"z{i}_mm"])
    add = helper.make_node("Add", [f"z{i}_mm", f"b{i}"], [f"z{i}_add"])
    nodes.append(mm); nodes.append(add)

    if i < len(layers):  # ReLU on hidden layers
        relu = helper.make_node("Relu", [f"z{i}_add"], [f"z{i}_relu"])
        nodes.append(relu)
        in_name = f"z{i}_relu"
    else:
        # Final layer output
        nodes[-1].output[0] = "y"

    in_size = out_size

graph = helper.make_graph(nodes, "big_mlp", [x], [y], initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=11)
onnx.checker.check_model(model)

out_dir = Path("examples/builds/big_mlp"); out_dir.mkdir(parents=True, exist_ok=True)
onnx.save(model, out_dir / "big_mlp.onnx")
np.save(out_dir / "x.npy", np.random.randn(1, in_dim).astype("float32"))
# Optional: dump params
for i, out_size in enumerate(layers, start=1):
    # shapes match construction above
    pass
