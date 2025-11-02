#!/usr/bin/env python3
import onnx, numpy as np
from onnx import helper, TensorProto
from pathlib import Path

# x→MatMul(W1)→MatMul(W2)→MatMul(W3)→y
dims = [16, 128, 256, 256, 256, 192, 128, 64, 32, 8]

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, dims[0]])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, dims[-1]])

inits = []
nodes = []
in_name = "x"

for i in range(1, len(dims)):
    W = np.random.randn(dims[i-1], dims[i]).astype("float32") * 0.02
    Wi = helper.make_tensor(f"W{i}", TensorProto.FLOAT, W.shape, W.ravel())
    inits.append(Wi)
    out_name = "y" if i == len(dims)-1 else f"z{i}"
    nodes.append(helper.make_node("MatMul", [in_name, f"W{i}"], [out_name]))
    in_name = out_name

g = helper.make_graph(nodes, "matmul_chain", [x], [y], initializer=inits)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=11)
onnx.checker.check_model(m)

out_dir = Path("examples/builds/matmul_chain"); out_dir.mkdir(parents=True, exist_ok=True)
onnx.save(m, out_dir / "matmul_chain.onnx")
np.save(out_dir / "x.npy", np.random.randn(1, dims[0]).astype("float32"))
