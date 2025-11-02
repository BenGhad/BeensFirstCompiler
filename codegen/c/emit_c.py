# emit_c.py
import json, numpy as np, sys
from pathlib import Path

MATMUL = r"""
#include <cblas.h>
static void kernel_matmul(const float* A,const float* B,float* C,int M,int K,int N){
  long work = (long)M*(long)K*(long)N;          // ~ MACs
  if (work < 2000000L) {                        // tune threshold
    for(int i=0;i<M;++i){
      for(int j=0;j<N;++j) C[i*N+j] = 0.f;
      for(int k=0;k<K;++k){
        float a = A[i*K + k];
        for(int j=0;j<N;++j) C[i*N + j] += a * B[k*N + j];
      }
    }
    return;
  }
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,
              M,N,K,1.0f,A,K,B,N,0.0f,C,N);
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
    must_f32 = set(ir["entry"]["inputs"]) | set(ir["entry"]["outputs"])
    for op in ir["ops"]:
        k = op["op"].lower(); ins = op["inputs"]; outs = op["outputs"]
        if k == "matmul":
            must_f32.add(ins[0]);  must_f32.update(outs)
        elif k == "add":
            must_f32.update(ins);  must_f32.update(outs)
        elif k == "relu":
            must_f32.update(ins);  must_f32.update(outs)
        elif k == "reshape":
            must_f32.add(ins[0]);  must_f32.update(outs)

    off = {v: ir["values"][v]["type"]["dtype"] for v in must_f32 if ir["values"][v]["type"]["dtype"] != "f32"}
    if off:
        listed = ", ".join(f"{k}({dt})" for k, dt in off.items())
        raise RuntimeError(f"emitter supports only f32 tensors on data paths; found: {listed}")

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
    out2idx  = {name: i for i, name in enumerate(ir["entry"]["outputs"])}

    def ptr_expr(val):
        if val in name2idx:
            return f"inputs[{name2idx[val]}].data"
        if val in out2idx:
            return f"outputs[{out2idx[val]}].data"   # compute directly into outputs
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
            c.append(f"  kernel_matmul({srcA}, {srcB}, {dst}, {M},{K},{N});")

        elif opk == "add":
            a, b = op["inputs"]; O = op["outputs"][0]
            a_const = a in ir["consts"]
            b_const = b in ir["consts"]

            Ashp = dims_of(a)
            Bshp = dims_of(b)
            if numel(Ashp) is None or numel(Bshp) is None:
                raise RuntimeError(f"[{oid}] Add requires concrete shapes")
            totalA = numel(Ashp)
            totalB = numel(Bshp)

            dst = ptr_expr(O)

            # -------- case 1: both activations (z1 + z2) --------
            if not a_const and not b_const:
                srcA = ptr_expr(a)
                srcB = ptr_expr(b)
                if Ashp != Bshp:
                    raise RuntimeError(f"[{oid}] Add act+act requires same shape, got {Ashp} vs {Bshp}")
                c.append(f"  /* Add act+act {a} + {b} -> {O} */")
                c.append(f"  if ({dst} != {srcA}) for(int i=0;i<{totalA}; ++i) {dst}[i] = {srcA}[i];")
                c.append(f"  kernel_add({dst}, {srcB}, {totalA});")
                continue

            # Decide activation vs weight-const
            act = b if a_const else a
            w   = a if a_const else b
            src = ptr_expr(act)
            shp = dims_of(act)
            total = numel(shp)
            wshape = ir["values"][w]["type"]["shape"]

            c.append(f"  /* Add act+const {act} + {w} -> {O} */")
            # materialize activation into dst if needed
            c.append(f"  if ({dst} != {src}) for(int i=0;i<{total}; ++i) {dst}[i] = {src}[i];")

            # const payload
            if w not in ir["consts"]:
                raise RuntimeError(f"[{oid}] expected const on one Add input")
            sym = ir["consts"][w]["symbol"]

            # broadcasts
            if len(wshape) == 0:
                c.append(f"  for(int i=0;i<{total}; ++i) {dst}[i] += {sym}[0];")
            elif len(wshape) == 1 and isinstance(wshape[0], int) and len(shp) == 2 and wshape[0] == shp[1]:
                # row-wise bias
                c.append(f"  for(int i=0;i<{shp[0]}; ++i) kernel_add({dst} + i*{shp[1]}, {sym}, {shp[1]});")
            elif wshape == shp and total == totalB:
                # full tensor add
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

    c.append("  return 0; }")
    Path(out_c).write_text('\n'.join(c))

if __name__ == "__main__":
    emit(sys.argv[1], sys.argv[2])
