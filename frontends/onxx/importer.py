from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root

import re
import json
import onnx
import numpy as np
from onnx import TensorProto
from onnx import AttributeProto as A

def constant_attr_to_array(n):
    at = {a.name: a for a in n.attribute}
    if "value" in at and at["value"].type == A.TENSOR:
        return onnx.numpy_helper.to_array(at["value"].t)
    if "value_float" in at:
        return np.array(at["value_float"].f, dtype=np.float32)
    if "value_int" in at:
        return np.array(at["value_int"].i, dtype=np.int64)
    if "value_floats" in at:
        return np.array(list(at["value_floats"].floats), dtype=np.float32)
    if "value_ints" in at:
        return np.array(list(at["value_ints"].ints), dtype=np.int64)
    if "value_string" in at or "value_strings" in at:
        raise NotImplementedError("String Constant not supported")
    raise ValueError("Constant node without supported value attribute")

def onnx_dtype_to_ir(elem):
    return {
        TensorProto.FLOAT: "f32", TensorProto.DOUBLE: "f64",
        TensorProto.FLOAT16: "f16", TensorProto.BFLOAT16: "bf16",
        TensorProto.INT8: "i8", TensorProto.UINT8: "u8",
        TensorProto.INT16: "i16", TensorProto.UINT16: "u16",
        TensorProto.INT32: "i32", TensorProto.UINT32: "u32",
        TensorProto.INT64: "i64", TensorProto.UINT64: "u64",
        TensorProto.BOOL: "i1",
    }.get(elem, "unknown")


def mir_dtype(arr):
    bfloat16 = getattr(np, "bfloat16", None)
    mapping = {
        np.float32: "f32", np.float64: "f64", np.float16: "f16",
        np.int8: "i8", np.uint8: "u8", np.int16: "i16", np.uint16: "u16",
        np.int32: "i32", np.uint32: "u32", np.int64: "i64", np.uint64: "u64",
        np.bool_: "i1",
    }
    if bfloat16 is not None:
        mapping[bfloat16] = "bf16"
    return mapping.get(arr.dtype.type, "unknown")


def decode_attr(a):
    from onnx import AttributeProto as A
    if a.type == A.FLOAT: return float(a.f)
    if a.type == A.INT: return int(a.i)
    if a.type == A.STRING: return a.s.decode() if isinstance(a.s, bytes) else str(a.s)
    if a.type == A.FLOATS: return list(a.floats)
    if a.type == A.INTS: return list(a.ints)
    if a.type == A.STRINGS: return [s.decode() if isinstance(s, bytes) else str(s) for s in a.strings]
    if a.type == A.TENSOR:
        arr = onnx.numpy_helper.to_array(a.t)
        dt = onnx_dtype_to_ir(getattr(a.t, "data_type", 0))
        if dt == "unknown":
            dt = mir_dtype(arr)
        return {"tensor": arr.tolist(), "dtype": dt, "shape": list(arr.shape)}
    # Explicitly reject unsupported attribute kinds for now
    from onnx import AttributeProto as AP
    if a.type in (AP.GRAPH, AP.GRAPHS, AP.SPARSE_TENSOR, AP.SPARSE_TENSORS, AP.TENSORS):
        raise NotImplementedError(f"Unsupported attribute type: {a.type}")
    return None


def load(onnx_path: str):
    m = onnx.load(onnx_path)
    g = m.graph
    ir = {
        "meta": {"ir_version": 2, "producer": "aicc-0.1", "opset": 1},
        "values": {}, "consts": {}, "ops": [],
        "entry": {"inputs": [], "outputs": []},
    }

    # opset metadata
    opset_import = {(ei.domain or ""): ei.version for ei in m.opset_import}
    ir["meta"]["opset_import"] = opset_import
    ir["meta"]["opset"] = opset_import.get("", max(opset_import.values(), default=1))

    def put_val(name, shape, dtype=None):
        v = ir["values"].setdefault(name, {"type": {"dtype": "unknown", "shape": []}})
        if shape is not None:
            v["type"]["shape"] = list(shape)
        if dtype:
            v["type"]["dtype"] = dtype

    # shape dim conversion that preserves zeros and symbols
    def dim_to_ir(d):
        tag = d.WhichOneof("value")  # 'dim_value' | 'dim_param' | None
        if tag == "dim_value":
            return int(d.dim_value)   # keep 0 as 0
        if tag == "dim_param" and d.dim_param:
            return d.dim_param        # symbolic
        return -1                     # unknown

    # seed values and entry
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        t = vi.type.tensor_type
        shp = [dim_to_ir(d) for d in t.shape.dim]
        dtype = onnx_dtype_to_ir(t.elem_type)
        put_val(vi.name, shp, dtype)

    ir["entry"]["inputs"] = [vi.name for vi in g.input]
    ir["entry"]["outputs"] = [vo.name for vo in g.output]

    # initializers → consts
    for init in g.initializer:
        arr = onnx.numpy_helper.to_array(init)
        dt = onnx_dtype_to_ir(getattr(init, "data_type", 0))
        if dt == "unknown":
            dt = mir_dtype(arr)
        put_val(init.name, arr.shape, dt)
        ir["consts"][init.name] = {"type": {"dtype": dt, "shape": list(arr.shape)}, "storage": {}}

    # nodes → ops, including Constant into consts table
    for i, n in enumerate(g.node):
        if n.op_type == "Constant":
            outs = [o for o in n.output if o]
            if not outs:
                continue
            arr = constant_attr_to_array(n)
            dt = mir_dtype(arr)
            put_val(outs[0], list(arr.shape), dt)
            ir["consts"][outs[0]] = {"type": {"dtype": dt, "shape": list(arr.shape)}, "storage": {}}
            continue


        outs = [o for o in n.output if o]
        if not outs:
            continue
        for o in outs:
            ir["values"].setdefault(o, {"type": {"dtype": "unknown", "shape": []}})

        ins = [ii for ii in n.input if ii]
        attrs = {a.name: decode_attr(a) for a in n.attribute}
        op_id = n.name or f"n{i}"

        ir["ops"].append({
            "op_id": op_id,
            "id": op_id,                  # compatibility with passes expecting 'id'
            "op": n.op_type,
            "inputs": ins,
            "outputs": outs,
            "attrs": attrs,
            "domain": getattr(n, "domain", ""),
        })

    return ir


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", s)


def emit_ir_json(onnx_path: str, out_root: str = "builds", project: str | None = None):
    onnx_p = Path(onnx_path)
    proj = project or onnx_p.stem
    base = Path(out_root) / proj
    weights_dir = base / "weights"
    base.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    ir = load(str(onnx_p))

    # collect tensor payloads from initializers and Constant nodes
    m = onnx.load(str(onnx_p))
    init_map = {i.name: onnx.numpy_helper.to_array(i) for i in m.graph.initializer}

    const_node_map = {}
    for n in m.graph.node:
        if n.op_type == "Constant":
            outs = [o for o in n.output if o]
            if not outs:
                continue
            const_node_map[outs[0]] = constant_attr_to_array(n)

    weights_map = {}
    for name in ir.get("consts", {}):
        arr = init_map.get(name, const_node_map.get(name))
        if arr is None:
            raise KeyError(f"const '{name}' has no backing data")
        fname = f"{safe_name(name)}.npy"
        p = weights_dir / fname
        np.save(p, arr, allow_pickle=False)
        weights_map[name] = str(p)
        ir["consts"][name]["storage"]["file"] = (Path("weights") / fname).as_posix()

    graph_path = base / "graph.ir.json"
    with open(graph_path, "w") as f:
        json.dump(ir, f, indent=2)
    return str(graph_path), weights_map


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export IR JSON and .npy weights from an ONNX model.")
    parser.add_argument("onnx_path", help="Path to the .onnx model")
    parser.add_argument("-o", "--out-root", default="builds", help="Output root directory (default: builds)")
    parser.add_argument("-p", "--project", default=None, help="Project name; defaults to ONNX filename stem")
    args = parser.parse_args()

    try:
        graph_path, weights_map = emit_ir_json(args.onnx_path, out_root=args.out_root, project=args.project)
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
