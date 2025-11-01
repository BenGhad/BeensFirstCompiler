from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

from ir.ir import Module, Value, Op

# loader + emitter
import onnx, numpy as np, json
from pathlib import Path
from ir.ir import Module, Value, Op

def load(onnx_path: str):
    m = onnx.load(onnx_path)
    g = m.graph

    values = {}
    weight_blobs = {}  # name -> np.array

    def put_val(name, shape, dtype="f32"):
        if name not in values:
            values[name] = Value(name, dtype, list(map(int, shape or [])))

    for vi in list(g.input)+list(g.output)+list(g.value_info):
        shp = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        put_val(vi.name, shp)


    ops = []

    for init in g.initializer:
        arr = onnx.numpy_helper.to_array(init).astype("float32")
        weight_blobs[init.name] = arr
        put_val(init.name, arr.shape)
        ops.append(Op(f"%{init.name}", "const", [], {"name": init.name}))


    for i, n in enumerate(g.node):
        out = n.output[0]
        oid = f"%{out}"
        if n.op_type in ("Identity","Reshape"):
            ops.append(Op(oid, n.op_type.lower(), [n.input[0]]))
        elif n.op_type == "MatMul":
            ops.append(Op(oid, "matmul", [n.input[0], n.input[1]]))
        elif n.op_type == "Add":
            ops.append(Op(oid, "add", [n.input[0], n.input[1]]))
        elif n.op_type == "Relu":
            ops.append(Op(oid, "relu", [n.input[0]]))
        elif n.op_type == "Constant":
            arr = onnx.numpy_helper.to_array(n.attribute[0].t).astype("float32")
            name = f"const_{i}"
            weight_blobs[name] = arr
            put_val(name, arr.shape)
            put_val(out, arr.shape)
            ops.append(Op(oid, "const", [], {"name": name}))
        else:
            raise NotImplementedError(f"Unsupported op: {n.op_type}")

        if n.op_type == "MatMul":
            a, b = values[n.input[0]], values[n.input[1]]
            put_val(out, [a.shape[0], b.shape[1]])
        elif n.op_type in ("Add","Relu","Identity","Reshape"):
            put_val(out, values[n.input[0]].shape)

    entry_inputs  = [g.input[0].name]
    entry_outputs = [g.output[0].name]
    mod = Module(values, ops, entry_inputs, entry_outputs)
    return mod, weight_blobs

def emit_ir_json(onnx_path: str, out_root: str = "builds", project: str | None = None):
    """
    Write:
      builds/<project>/graph.ir.json
      builds/<project>/weights/<name>.npy
    If project is None, uses Path(onnx_path).stem
    """
    onnx_p = Path(onnx_path)
    proj = project or onnx_p.stem
    base = Path(out_root) / proj
    weights_dir = base / "weights"
    base.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    mod, blobs = load(str(onnx_p))
    wmap_rel = {}

    for name, arr in blobs.items():
        p = weights_dir / f"{name}.npy"
        np.save(p, arr)
        wmap_rel[name] = (Path("weights") / f"{name}.npy").as_posix()  # relative to base

    for op in mod.ops:
        if op.op == "const" and "name" in op.attrs:
            op.attrs["file"] = wmap_rel[op.attrs["name"]]

    graph_path = base / "graph.ir.json"
    with open(graph_path, "w") as f:
        json.dump(mod.to_json(), f, indent=2)

    return str(graph_path), {k: str((base / v).resolve()) for k, v in wmap_rel.items()}

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Export IR JSON and .npy weights from an ONNX model."
    )
    parser.add_argument("onnx_path", help="Path to the .onnx model")
    parser.add_argument(
        "-o", "--out-root", default="builds",
        help="Output root directory (default: builds)"
    )
    parser.add_argument(
        "-p", "--project", default=None,
        help="Project name; defaults to ONNX filename stem"
    )
    args = parser.parse_args()

    try:
        graph_path, weights_map = emit_ir_json(
            args.onnx_path, out_root=args.out_root, project=args.project
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)

    print("IR graph:", graph_path)
    if weights_map:
        print("Weights:")
        for name, abs_path in weights_map.items():
            print(f"  {name}: {abs_path}")
