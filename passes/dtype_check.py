def _dtype(ir, v): return ir["values"][v]["type"]["dtype"]

def dtype_check(ir):
    for name, c in ir.get("consts", {}).items():
        if c["type"]["dtype"] != "f32":
            raise RuntimeError(f"unsupported const dtype for {name}")
    for op in ir["ops"]:
        k = op["op"].lower(); ins = op["inputs"]; out = op["outputs"][0]
        if k == "matmul":
            if not (_dtype(ir, ins[0]) == _dtype(ir, ins[1]) == "f32"): raise RuntimeError("matmul requires f32")
            ir["values"][out]["type"]["dtype"] = "f32"
        elif k == "add":
            if len(op["inputs"]) == 2 and _dtype(ir, ins[0]) == _dtype(ir, ins[1]):
                ir["values"][out]["type"]["dtype"] = _dtype(ir, ins[0])
            elif len(op["inputs"]) == 2 and _dtype(ir, ins[1]) == _dtype(ir, ins[0]):
                ir["values"][out]["type"]["dtype"] = _dtype(ir, ins[0])
            else:
                raise RuntimeError("add dtype mismatch")

        elif k in ("relu","identity","reshape"):
            ir["values"][out]["type"]["dtype"] = _dtype(ir, ins[0])
        else:
            raise RuntimeError(f"unhandled op in dtype_check: {k}")
    return ir
