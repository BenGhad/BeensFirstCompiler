def _dtype(ir, v): return ir["values"][v]["type"]["dtype"]

def dtype_check(ir):
    vals = ir["values"]; consts = ir.get("consts", {})

    # consumers map: value -> [(op, idx)]
    consumers = {}
    for op in ir["ops"]:
        k = op["op"].lower()
        for idx, v in enumerate(op.get("inputs", [])):
            consumers.setdefault(v, []).append((k, idx))

    # consts: permit non-f32 only if used as reshape's shape input (idx==1)
    for name, c in consts.items():
        dt = c["type"]["dtype"]
        if dt == "f32":
            continue
        uses = consumers.get(name, [])
        if all(op == "reshape" and idx == 1 for op, idx in uses):
            continue
        raise RuntimeError(f"unsupported const dtype for {name}: {dt}")

    def dt(v): return vals[v]["type"]["dtype"]
    def set_f32(v):
        if dt(v) in ("unknown", "f32"): vals[v]["type"]["dtype"] = "f32"
        elif dt(v) != "f32": raise RuntimeError(f"dtype mismatch on {v}: {dt(v)}")

    # Phase-3: inputs must be f32
    for x in ir["entry"]["inputs"]:
        set_f32(x)

    for op in ir["ops"]:
        k = op["op"].lower(); ins = op["inputs"]; outs = op["outputs"]

        if k == "matmul":
            # LHS activation f32, RHS const f32
            set_f32(ins[0])
            if ins[1] not in consts: raise RuntimeError("matmul RHS must be const")
            if consts[ins[1]]["type"]["dtype"] != "f32": raise RuntimeError("matmul RHS const not f32")
            for o in outs: set_f32(o)

        elif k == "add":
            # both sides f32; one may be const f32
            for i in ins:
                if i in consts:
                    if consts[i]["type"]["dtype"] != "f32": raise RuntimeError("add const not f32")
                else:
                    set_f32(i)
            for o in outs: set_f32(o)

        elif k in ("relu",):
            set_f32(ins[0]);  [set_f32(o) for o in outs]

        elif k == "reshape":
            # data path f32; shape input may be int
            set_f32(ins[0]);  [set_f32(o) for o in outs]

        else:
            raise RuntimeError(f"unhandled op in dtype_check: {k}")

    # Final guard: every non-shape value must be f32
    offenders = [v for v, info in vals.items()
                 if info["type"]["dtype"] != "f32"
                 and not all(op == "reshape" and idx == 1 for op, idx in consumers.get(v, []))]
    if offenders:
        raise RuntimeError("non-f32 tensors remain: " + ", ".join(offenders))
    return ir
