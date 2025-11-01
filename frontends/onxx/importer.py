from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

# loader + emitter
import onnx, numpy as np, json
from pathlib import Path


def load(onnx_path: str):
    m = onnx.load(onnx_path); g = m.graph
    ir = {"meta":{"ir_version":2,"producer":"aicc-0.1","opset":1},
          "values":{}, "consts":{}, "ops":[], "entry":{"inputs":[],"outputs":[]}}

    def put_val(name, shape, dtype="f32"):
        shape = list(map(int, shape or []))
        ir["values"].setdefault(name, {"type":{"dtype":dtype,"shape":shape}})
        ir["values"][name]["type"]["shape"] = shape

    # seed values and entry
    for vi in list(g.input)+list(g.output)+list(g.value_info):
        shp = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        put_val(vi.name, shp)
    ir["entry"]["inputs"]  = [vi.name for vi in g.input]
    ir["entry"]["outputs"] = [vo.name for vo in g.output]

    # initializers → consts
    for init in g.initializer:
        arr = onnx.numpy_helper.to_array(init).astype("float32")
        put_val(init.name, arr.shape, "f32")
        ir["consts"][init.name] = {
            "type":{"dtype":"f32","shape":list(arr.shape)},
            "storage":{"file": str(Path(f"{init.name}.npy"))}  # or {"inline": ...}
        }

    # nodes → ops, no Const ops
    for i, n in enumerate(g.node):
        if n.op_type == "Constant":
            # simplest path for now
            raise NotImplementedError("ONNX Constant not supported in importer v2")
        outs = list(n.output)
        if not outs:  # skip nodes without outputs
            continue
        ir["ops"].append({
            "op_id": f"n{i}",
            "op": n.op_type,
            "inputs": list(n.input),
            "outputs": [outs[0]],
            "attrs": {}
        })

    return ir

def emit_ir_json(onnx_path: str, out_root: str = "builds", project: str | None = None):
    onnx_p = Path(onnx_path)
    proj = project or onnx_p.stem
    base = Path(out_root) / proj
    weights_dir = base / "weights"
    base.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    ir = load(str(onnx_p))

    m = onnx.load(str(onnx_p))
    init_map = {i.name: onnx.numpy_helper.to_array(i).astype("float32") for i in m.graph.initializer}

    weights_map = {}
    for name in ir.get("consts", {}):
        arr = init_map[name]
        p = weights_dir / f"{name}.npy"
        np.save(p, arr)
        weights_map[name] = str(p)
        ir["consts"][name]["storage"]["file"] = (Path("weights") / f"{name}.npy").as_posix()

    graph_path = base / "graph.ir.json"
    with open(graph_path, "w") as f:
        json.dump(ir, f, indent=2)
    return str(graph_path), weights_map


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
