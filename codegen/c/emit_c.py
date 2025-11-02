# emit_c.py
import json, numpy as np, sys
from pathlib import Path

MATMUL = r"""
static void kernel_matmul(const float* A, const float* B, float* C, int M, int K, int N){
  for(int i=0;i<M;++i)
    for(int k=0;k<K;++k){
      float a = A[i*K + k];
      for(int j=0;j<N;++j)
        C[i*N + j] += a * B[k*N + j];
    }
}
"""

ADD_INPLACE = r"""
static void kernel_add(float* X, const float* B, int N){
  for(int i=0;i<N;++i) X[i] += B[i];
}
"""

RELU_INPLACE = r"""
static void kernel_relu(float* X, int N){
  for(int i=0;i<N;++i) X[i] = X[i] > 0.f ? X[i] : 0.f;
}
"""

def emit(ir_json, out_c):
    ir_path = Path(ir_json)
    base_dir = ir_path.parent
    ir = json.loads(ir_path.read_text())

    if len(ir["entry"]["outputs"]) < 1:
        raise RuntimeError("no graph outputs")

    # Phase-3 emits f32 only
    if any(v["type"]["dtype"] not in ("f32",) for v in ir["values"].values()):
        raise RuntimeError("emitter supports only f32 tensors")

    # C prologue
    c = ["#include <stddef.h>", '#include "aicc_runtime.h"']
    c += [MATMUL, ADD_INPLACE, RELU_INPLACE]

    # embed weights
    wid = 0
    for name in sorted(ir.get("consts", {}).keys()):
        file_field = ir["consts"][name]["storage"]["file"]
        wpath = (base_dir / file_field).resolve() if not Path(file_field).is_absolute() else Path(file_field)
        if not wpath.exists():
            raise FileNotFoundError(f"weight not found: {wpath}")
        arr = np.load(wpath, allow_pickle=False)
        flat = ",".join(f"{float(x):.9g}" for x in arr.flatten())
        sym = f"W{wid}"
        c.append(f"static const float {sym}[] = {{{flat}}};")
        ir["consts"][name]["symbol"] = sym
        wid += 1

    # helpers
    def dims_of(val): return ir["values"][val]["type"]["shape"]
    def numel(shape):
        return int(np.prod(shape)) if isinstance(shape, list) and all(isinstance(d, int) and d > 0 for d in shape) else None

    arena_elems = ir["meta"].get("arena_elems")
    if arena_elems is None or arena_elems <= 0:
        raise RuntimeError("arena_elems missing or zero; run passes.run_all first")

    offsets = ir["meta"]["offsets"]
    name2idx = {name: i for i, name in enumerate(ir["entry"]["inputs"])}

    def ptr_expr(val):
        if val in name2idx:
            return f"inputs[{name2idx[val]}].data"
        off = offsets.get(val)
        if off is None:
            raise RuntimeError(f"no storage for value '{val}'")
        return f"(arena + {off})"

    # buffer cap and entry
    c.append("int aicc_run(const aicc_tensor* inputs, aicc_tensor* outputs){")
    c.append(f"  enum {{ ARENA_ELEMS = {arena_elems} }};")
    c.append("  static float arena[ARENA_ELEMS];")

    # emit ops in topo order
    for op in ir["ops"]:
        opk = op["op"].lower()
        oid = op.get("op_id", op.get("id", "?"))

        if opk == "matmul":
            A, B = op["inputs"]
            O = op["outputs"][0]
            Ashp, Bshp = dims_of(A), dims_of(B)
            if not (len(Ashp) == 2 and len(Bshp) == 2):
                raise RuntimeError(f"[{oid}] MatMul expects rank-2")
            M, K = Ashp; Kb, N = Bshp
            if not (isinstance(K,int) and isinstance(Kb,int) and K == Kb):
                raise RuntimeError(f"[{oid}] MatMul K mismatch")
            srcA = ptr_expr(A)
            if B not in ir["consts"]:
                raise RuntimeError(f"[{oid}] RHS must be const in Phase-3 minimal")
            srcB = ir["consts"][B]["symbol"]
            dst  = ptr_expr(O)
            c.append(f"  /* MatMul {A}({M}x{K}) x {B}({Kb}x{N}) -> {O} */")
            c.append(f"  for(int i=0;i<{M}*{N};++i) {dst}[i]=0.f;")
            c.append(f"  kernel_matmul({srcA}, {srcB}, {dst}, {M},{K},{N});")

        elif opk == "add":
            a, b = op["inputs"]; O = op["outputs"][0]
            # choose activation vs weight
            if a in ir["consts"] and b in ir["consts"]:
                raise RuntimeError(f"[{oid}] Add has two consts; unsupported")
            act = b if a in ir["consts"] else a
            w   = a if a in ir["consts"] else b

            dst = ptr_expr(O)
            src = ptr_expr(act)
            shp = dims_of(act)
            if numel(shp) is None: raise RuntimeError(f"[{oid}] activation shape must be concrete")
            total = numel(shp)
            sym = ir["consts"][w]["symbol"]
            wshape = ir["values"][w]["type"]["shape"]

            c.append(f"  /* Add {act} + {w} -> {O} */")
            c.append(f"  if ({dst} != {src}) for(int i=0;i<{total}; ++i) {dst}[i] = {src}[i];")
            if len(wshape) == 0:
                c.append(f"  for(int i=0;i<{total}; ++i) {dst}[i] += {sym}[0];")
            elif len(wshape) == 1 and isinstance(wshape[0], int) and wshape[0] == shp[1]:
                c.append(f"  for(int i=0;i<{shp[0]}; ++i) kernel_add({dst} + i*{shp[1]}, {sym}, {shp[1]});")
            elif len(wshape) == 2 and wshape == shp:
                c.append(f"  kernel_add({dst}, {sym}, {total});")
            else:
                raise RuntimeError(f"[{oid}] unsupported broadcast for Add: act{shp}, w{wshape}")

        elif opk == "relu":
            x = op["inputs"][0]; O = op["outputs"][0]
            dst = ptr_expr(O); src = ptr_expr(x)
            shp = dims_of(x); total = numel(shp)
            if total is None: raise RuntimeError(f"[{oid}] relu shape must be concrete")
            c.append(f"  /* ReLU {x} -> {O} */")
            c.append(f"  if ({dst} != {src}) for(int i=0;i<{total}; ++i) {dst}[i] = {src}[i];")
            c.append(f"  kernel_relu({dst}, {total});")

        elif opk == "reshape":
            # alias handled by planner; nothing to do
            c.append(f"  /* Reshape alias */")

        else:
            raise RuntimeError(f"[{oid}] unhandled op {opk}")

    # write all outputs
    c.append("  for (int oi=0; oi< (int)1e9; ++oi) { /* dummy to avoid unused warning */ break; }")
    for i, oname in enumerate(ir["entry"]["outputs"]):
        src = ptr_expr(oname)
        c.append(f"  {{ size_t n=1; for(size_t d=0; d<outputs[{i}].rank; ++d) n*=outputs[{i}].shape[d];")
        c.append(f"    for(size_t k=0;k<n;++k) outputs[{i}].data[k] = {src}[k]; }}")

    c.append("  return 0; }")
    Path(out_c).write_text('\n'.join(c))

if __name__ == "__main__":
    emit(sys.argv[1], sys.argv[2])
