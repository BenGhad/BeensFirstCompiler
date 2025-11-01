def _shape(ir, v): return ir["values"][v]["type"]["shape"]

def _set_shape(ir, v, s):
    ir["values"].setdefault(v, {"type":{"dtype":"f32","shape":list(map(int, s))}})
    ir["values"][v]["type"]["shape"] = list(map(int, s))

def shape_check(ir):
    for s in ir["entry"]["inputs"] + ir["entry"]["outputs"]:
        if s not in ir["values"]: raise RuntimeError(f"unknown value: {s}")

    for op in ir["ops"]:
        k = op["op"].lower(); ins = op["inputs"]; out = op["outputs"][0]
        if k == "identity":
            _set_shape(ir, out, _shape(ir, ins[0]))
        elif k == "reshape":
            tgt = op.get("attrs", {}).get("shape", {}).get("shape")
            if not isinstance(tgt, list): raise RuntimeError("reshape requires attrs.shape")
            _set_shape(ir, out, tgt)
        elif k == "matmul":
            A = _shape(ir, ins[0]); B = _shape(ir, ins[1])
            if len(A)!=2 or len(B)!=2: raise RuntimeError(f"MatMul rank-2 expected, got A{A}, B{B}")
            M,K = A; Kb,N = B
            if K!=Kb: raise RuntimeError(f"MatMul K mismatch: {K} vs {Kb}")
            _set_shape(ir, out, [M,N])
        elif k == "add":
            X = _shape(ir, ins[0]); Y = _shape(ir, ins[1])
            if len(X)==2 and X==Y: _set_shape(ir, out, X)
            elif len(X)==2 and len(Y)==1 and X[1]==Y[0]: _set_shape(ir, out, X)
            else: raise RuntimeError(f"unsupported add shapes: {X} + {Y}")
        elif k == "relu":
            _set_shape(ir, out, _shape(ir, ins[0]))
        else:
            raise RuntimeError(f"unhandled op in shape_check: {k}")

    for o in ir["entry"]["outputs"]:
        _shape(ir, o)
    return ir
