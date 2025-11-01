import json, argparse

def _bare(s: str) -> str:
    return s[1:] if isinstance(s, str) and s.startswith("%") else s

def load_ir(p):
    with open(p) as f:
        return json.load(f)

def dce(ir):
    live = set(ir["entry"]["outputs"])
    for op in reversed(ir["ops"]):
        if _bare(op["id"]) in live:
            for x in op["inputs"]:
                live.add(x)
    ir["ops"] = [op for op in ir["ops"] if _bare(op["id"]) in live]
    return ir

def canonicalize(ir):
    # normalize op kinds
    for op in ir["ops"]:
        op["op"] = op["op"].lower()

    # collapse identity chains
    parent = {}
    for op in ir["ops"]:
        if op["op"] == "identity":
            parent[_bare(op["id"])] = op["inputs"][0]

    def root(x):
        x = _bare(x)
        while x in parent:
            x = parent[x]
        return x

    if parent:
        for op in ir["ops"]:
            op["inputs"] = [root(i) for i in op["inputs"]]
        ir["entry"]["outputs"] = [root(o) for o in ir["entry"]["outputs"]]
        ir["entry"]["inputs"]  = [root(i) for i in ir["entry"]["inputs"]]
        ir["ops"] = [op for op in ir["ops"] if op["op"] != "identity"]
    return ir

def run_all(path):
    ir = load_ir(path)
    ir = canonicalize(ir)
    ir = dce(ir)
    with open(path, "w") as f:
        json.dump(ir, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to IR JSON file (will be overwritten)")
    args = p.parse_args()
    run_all(args.input)
