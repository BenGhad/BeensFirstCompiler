#!/usr/bin/env python3
import onnx, numpy as np
from onnx import helper, TensorProto
from pathlib import Path

# Compact dual-branch MLP (≈4 MB total)
B = 1
in_dim = 512
out_dim = 512

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [B, in_dim])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [B, out_dim])

# Branch 1 params
W1 = np.random.randn(in_dim, out_dim).astype("float32") * 0.02
b1 = np.zeros((out_dim,), dtype="float32")
W1t = helper.make_tensor("W1", TensorProto.FLOAT, W1.shape, W1.ravel())
b1t = helper.make_tensor("b1", TensorProto.FLOAT, b1.shape, b1.ravel())

# Branch 2 params
W2 = np.random.randn(in_dim, out_dim).astype("float32") * 0.02
b2 = np.zeros((out_dim,), dtype="float32")
W2t = helper.make_tensor("W2", TensorProto.FLOAT, W2.shape, W2.ravel())
b2t = helper.make_tensor("b2", TensorProto.FLOAT, b2.shape, b2.ravel())

n = []
# Branch 1: x→MatMul→Add→ReLU
n += [helper.make_node("MatMul", ["x", "W1"], ["z1_mm"])]
n += [helper.make_node("Add", ["z1_mm", "b1"], ["z1_add"])]
n += [helper.make_node("Relu", ["z1_add"], ["z1"])]
# Branch 2: x→MatMul→Add→ReLU
n += [helper.make_node("MatMul", ["x", "W2"], ["z2_mm"])]
n += [helper.make_node("Add", ["z2_mm", "b2"], ["z2_add"])]
n += [helper.make_node("Relu", ["z2_add"], ["z2"])]
# Merge
n += [helper.make_node("Add", ["z1", "z2"], ["y"])]

g = helper.make_graph(n, "branch_relu", [x], [y], initializer=[W1t, b1t, W2t, b2t])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=11)
onnx.checker.check_model(m)

out_dir = Path("examples/builds/branch_relu"); out_dir.mkdir(parents=True, exist_ok=True)
onnx.save(m, out_dir / "branch_relu.onnx")
np.save(out_dir / "x.npy", np.random.randn(B, in_dim).astype("float32"))
