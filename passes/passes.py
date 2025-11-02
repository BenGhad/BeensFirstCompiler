# passes.py
import json, argparse
from pathlib import Path
from collections import defaultdict, deque
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
            for x in op.get("inputs", []):
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
            op["inputs"]  = [root(i) for i in op["inputs"]]
            op["outputs"] = [root(o) if o in rewrites else o for o in op["outputs"]]
        ir["entry"]["outputs"] = [root(o) for o in ir["entry"]["outputs"]]
        ir["entry"]["inputs"]  = [root(i) for i in ir["entry"]["inputs"]]
        ir["ops"] = [op for op in ir["ops"] if op["op"] != "identity"]
    return ir

def topo_sort(ir):
    # op producer per value
    producer = {}
    for op in ir["ops"]:
        for o in op.get("outputs", []):
            producer[o] = op["id"]
    indeg = {op["id"]: 0 for op in ir["ops"]}
    succ  = defaultdict(list)
    for op in ir["ops"]:
        oid = op["id"]
        for i in op.get("inputs", []):
            p = producer.get(i)
            if p:
                indeg[oid] += 1
                succ[p].append(oid)
    q = deque([oid for oid, d in indeg.items() if d == 0])
    id2op = {op["id"]: op for op in ir["ops"]}
    order = []
    while q:
        oid = q.popleft()
        order.append(id2op[oid])
        for v in succ[oid]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(ir["ops"]):
        raise RuntimeError("cycle detected: graph must be a DAG")
    ir["ops"] = order
    return ir

def plan_memory(ir):
    # number of elements from concrete shapes
    def numel(shape):
        return int(__import__("numpy").prod(shape)) if isinstance(shape, list) and all(isinstance(d, int) and d > 0 for d in shape) else None

    sizes = {v: numel(info["type"]["shape"]) for v, info in ir["values"].items()}
    consts = set(ir.get("consts", {}).keys())
    inputs = set(ir["entry"]["inputs"])

    # def index
    def_idx = {v: -1 for v in inputs}
    for idx, op in enumerate(ir["ops"]):
        for o in op.get("outputs", []):
            def_idx[o] = idx

    # last use index
    last_use = {}
    for idx, op in enumerate(ir["ops"]):
        for i in op.get("inputs", []):
            last_use[i] = idx
    # ensure entry outputs are considered used at the end
    end_idx = len(ir["ops"])
    for o in ir["entry"]["outputs"]:
        last_use[o] = max(last_use.get(o, -1), end_idx)

    # aliasing ops (no storage change)
    alias_ops = {"reshape"}  # can extend later
    # force aliasing: out takes input’s storage
    aliases = {}
    for op in ir["ops"]:
        if op["op"] in alias_ops:
            src = op["inputs"][0]
            dst = op["outputs"][0]
            aliases[dst] = src

    # collapse alias chains
    def root(v):
        while v in aliases:
            v = aliases[v]
        return v

    # intervals for activations that need storage
    need = [v for v in sizes if v not in consts]
    intervals = []
    for v in need:
        sz = sizes[v]
        if sz is None:
            continue  # planner needs concrete shapes only
        intervals.append((def_idx.get(v, -1), last_use.get(v, -1), sz, v))
    intervals.sort(key=lambda x: x[0])

    # first-fit on a 1D arena
    slots = []  # list of {end, size, offset}
    alloc = {}
    cur_off = 0
    for start, end, sz, v in intervals:
        # respect aliasing
        r = root(v)
        if r != v and r in alloc:
            alloc[v] = alloc[r]
            continue
        placed = False
        for s in slots:
            if s["end"] <= start and s["size"] >= sz:
                alloc[v] = s["offset"]
                s["end"] = end
                placed = True
                break
        if not placed:
            alloc[v] = cur_off
            slots.append({"end": end, "size": sz, "offset": cur_off})
            cur_off += sz

    ir["meta"]["arena_elems"] = max([0] + [a + sizes[v] for v, a in alloc.items() if sizes.get(v)])
    ir["meta"]["offsets"] = alloc
    return ir

def run_all(path):
    ir = load_ir(path)

    ir = canonicalize(ir)
    ir = dce(ir)

    # validate before planning
    base_dir = Path(path).parent
    ir = shape_check(ir, base_dir=base_dir)
    ir = dtype_check(ir)

    # prepare schedule + memory
    ir = topo_sort(ir)
    ir = plan_memory(ir)

    # prune dead consts
    used = set()
    for op in ir["ops"]:
        for x in op.get("inputs", []):
            if x in ir.get("consts", {}): used.add(x)
    ir["consts"] = {k: v for k, v in ir.get("consts", {}).items() if k in used}

    with open(path, "w") as f:
        json.dump(ir, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    args = p.parse_args()
    run_all(args.input)
