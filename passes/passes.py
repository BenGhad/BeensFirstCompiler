import json, argparse
from .shape_check import shape_check
from .dtype_check import dtype_check


def load_ir(p):
    with open(p) as f:
        return json.load(f)

def dce(ir):
    live = set(ir["entry"]["outputs"])
    kept = []
    for op in reversed(ir["ops"]):
        outs = op.get("outputs", [])
        if any(o in live for o in outs):
            kept.append(op)
            for x in op["inputs"]:
                live.add(x)
    ir["ops"] = list(reversed(kept))
    return ir

def canonicalize(ir):
    for op in ir["ops"]:
        op["op"] = op["op"].lower()

    rewrites = {}
    for op in ir["ops"]:
        if op["op"] == "identity":
            rewrites[op["outputs"][0]] = op["inputs"][0]

    if rewrites:
        def root(x):
            while x in rewrites:
                x = rewrites[x]
            return x

        for op in ir["ops"]:
            op["inputs"] = [root(i) for i in op["inputs"]]
        ir["entry"]["outputs"] = [root(o) for o in ir["entry"]["outputs"]]
        ir["entry"]["inputs"]  = [root(i) for i in ir["entry"]["inputs"]]
        ir["ops"] = [op for op in ir["ops"] if op["op"] != "identity"]
    return ir


def run_all(path):
    ir = load_ir(path)

    # optimize
    ir = canonicalize(ir)
    ir = dce(ir)

    # validate
    ir = shape_check(ir)
    ir = dtype_check(ir)

    with open(path, "w") as f:
        json.dump(ir, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to IR JSON file (will be overwritten)")
    args = p.parse_args()
    run_all(args.input)
