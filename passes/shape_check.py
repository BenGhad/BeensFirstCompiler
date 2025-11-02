from pathlib import Path
import numpy as np

def _shape(ir, v):
    return ir["values"][v]["type"]["shape"]

def _as_shape_list(s):
    out = []
    for d in s:
        if isinstance(d, (int, np.integer)):
            out.append(int(d))
        elif isinstance(d, str):
            out.append(d)  # symbolic
        else:
            raise RuntimeError(f"invalid dim {d!r}")
    return out

def _set_shape(ir, v, s):
    s = _as_shape_list(s)
    ir["values"].setdefault(v, {"type": {"dtype": "unknown", "shape": s}})
    ir["values"][v]["type"]["shape"] = s

def _prod_int(dims):
    p = 1
    for d in dims:
        if not isinstance(d, int) or d == -1:
            return None
        p *= d
    return p

def _dims_equal(a, b):
    # equal if same, or if either is unknown/symbolic
    if a == b: return True
    if isinstance(a, int) and a == -1: return True
    if isinstance(b, int) and b == -1: return True
    if isinstance(a, str) or isinstance(b, str): return True
    return False

def _load_const_vector(ir, name, base_dir: Path):
    c = ir["consts"].get(name)
    if not c: return None
    rel = c.get("storage", {}).get("file")
    if not rel: return None
    path = base_dir / rel
    arr = np.load(path, allow_pickle=False)
    return arr.tolist()

def shape_check(ir, base_dir: Path):
    for s in ir["entry"]["inputs"] + ir["entry"]["outputs"]:
        if s not in ir["values"]:
            raise RuntimeError(f"unknown value: {s}")

    for op in ir["ops"]:
        k = op["op"].lower()
        ins = op["inputs"]
        out = op["outputs"][0]

        if k == "identity":
            _set_shape(ir, out, _shape(ir, ins[0]))

        elif k == "reshape":
            # ONNX Reshape: inputs = [data, shape]
            if len(ins) < 2:
                raise RuntimeError("reshape requires 2 inputs")
            in_shape = _shape(ir, ins[0])
            shape_vec = _load_const_vector(ir, ins[1], base_dir)
            if shape_vec is None:
                raise RuntimeError("reshape shape must be constant for now")
            # apply ONNX semantics: 0 copies dim, -1 infers one dim
            target = []
            neg1_pos = -1
            for i, d in enumerate(shape_vec):
                d = int(d)
                if d == 0:
                    # copy i-th input dim if available
                    if i < len(in_shape):
                        target.append(in_shape[i])
                    else:
                        raise RuntimeError("reshape 0 out of range for input rank")
                elif d == -1:
                    if neg1_pos != -1:
                        raise RuntimeError("reshape allows only one -1")
                    neg1_pos = len(target)
                    target.append(-1)
                else:
                    target.append(d)
            # infer -1 if all known products available
            if neg1_pos != -1:
                in_prod = _prod_int(in_shape)
                out_known = _prod_int([x for x in target if isinstance(x, int) and x != -1])
                if in_prod is not None and out_known not in (None, 0) and in_prod % out_known == 0:
                    target[neg1_pos] = in_prod // out_known
                # else leave as -1
            _set_shape(ir, out, target)

        elif k == "matmul":
            A = _shape(ir, ins[0]); B = _shape(ir, ins[1])
            if len(A) != 2 or len(B) != 2:
                raise RuntimeError(f"MatMul rank-2 expected, got A{A}, B{B}")
            M, K = A; Kb, N = B
            if not _dims_equal(K, Kb):
                raise RuntimeError(f"MatMul K mismatch: {K} vs {Kb}")
            _set_shape(ir, out, [M, N])

        elif k == "add":
            X = _shape(ir, ins[0]); Y = _shape(ir, ins[1])
            if len(X) == 2 and X == Y:
                _set_shape(ir, out, X)
            elif len(X) == 2 and len(Y) == 1 and _dims_equal(X[1], Y[0]):
                _set_shape(ir, out, X)
            elif len(Y) == 0:  # scalar add
                _set_shape(ir, out, X)
            else:
                raise RuntimeError(f"unsupported add shapes: {X} + {Y}")

        elif k == "relu":
            _set_shape(ir, out, _shape(ir, ins[0]))

        else:
            raise RuntimeError(f"unhandled op in shape_check: {k}")

    for o in ir["entry"]["outputs"]:
        _shape(ir, o)
    return ir
