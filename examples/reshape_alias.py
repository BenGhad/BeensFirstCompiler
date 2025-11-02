#!/usr/bin/env python3
import argparse
import onnx, numpy as np
from onnx import helper, TensorProto
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dim", type=int, default=256)
    p.add_argument("--mid_dim", type=int, default=2048)
    p.add_argument("--reshape", type=int, nargs=2, default=[64, 32])  # rows cols
    args = p.parse_args()

    r, c = map(int, args.reshape)
    assert r * c == args.mid_dim, "reshape product must equal mid_dim"

    # I/O
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, int(args.in_dim)])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [int(r), int(c)])

    # Params
    W = np.random.randn(int(args.in_dim), int(args.mid_dim)).astype("float32") * 0.02
    b = np.zeros((int(c),), dtype="float32")  # broadcast on last dim

    Wt = helper.make_tensor("W", TensorProto.FLOAT, W.shape, W.ravel())
    bt = helper.make_tensor("b", TensorProto.FLOAT, b.shape, b.ravel())
    shape_arr = np.array([r, c], dtype=np.int64)
    shape_t = helper.make_tensor("shape", TensorProto.INT64, [2], shape_arr.ravel())

    nodes = [
        helper.make_node("MatMul", ["x", "W"], ["z_mm"]),
        helper.make_node("Reshape", ["z_mm", "shape"], ["z_reshaped"]),
        helper.make_node("Add", ["z_reshaped", "b"], ["z_add"]),
        helper.make_node("Relu", ["z_add"], ["y"]),
    ]

    g = helper.make_graph(nodes, "reshape_alias", [x], [y], initializer=[Wt, bt, shape_t])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=11)
    onnx.checker.check_model(m)

    out_dir = Path("examples/builds/reshape_alias")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(m, out_dir / "reshape_alias.onnx")
    np.save(out_dir / "x.npy", np.random.randn(1, int(args.in_dim)).astype("float32"))

if __name__ == "__main__":
    main()
