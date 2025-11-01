import json, numpy as np, pathlib, sys
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

ADD = r"""
static void kernel_add(float* X, const float* B, int N){
  for(int i=0;i<N;++i) X[i] += B[i];
}
"""

RELU = r"""
static void kernel_relu(float* X, int N){
  for(int i=0;i<N;++i) X[i] = X[i] > 0.f ? X[i] : 0.f;
}
"""

def find_weight(ir, name):
    for o in ir["ops"]:
        if o["op"].lower() == "const" and (
            o["id"][1:] == name or o["attrs"].get("name") == name
        ):
            return o["attrs"]["sym"]
    raise KeyError(name)

def is_const_value(ir, name):
    try:
        find_weight(ir, name)
        return True
    except KeyError:
        return False



def emit(ir_json, out_c):
    ir_path = Path(ir_json)
    base_dir = ir_path.parent

    ir = json.loads(ir_path.read_text())
    c = ["#include <stddef.h>", '#include "aicc_runtime.h"']
    c += [MATMUL, ADD, RELU]

    # embed weights
    wid = 0
    for op in ir["ops"]:
        if op["op"].lower() == "const":
            file_field = op["attrs"]["file"]
            wpath = Path(file_field)
            if not wpath.is_absolute():
                wpath = (base_dir / wpath).resolve()
            if not wpath.exists():
                raise FileNotFoundError(
                    f"weight not found: {wpath} (from attrs.file={file_field}, base={base_dir})"
                )
            arr = np.load(wpath)
            flat = ",".join(f"{float(x):.9g}" for x in arr.flatten())
            c.append(f"static const float W{wid}[] = {{{flat}}};")
            op["attrs"]["sym"] = f"W{wid}"
            op["attrs"]["size"] = int(arr.size)
            op["attrs"]["shape"] = ir["values"][op["id"][1:]]["shape"]
            wid += 1

    c.append("int aicc_run(const aicc_tensor* inputs, aicc_tensor* outputs){")
    c.append("  const float* x = inputs[0].data;")
    c.append("  enum { BUF_CAP = 128*128 };")
    c.append("  static float buf0[BUF_CAP];")
    c.append("  static float buf1[BUF_CAP];")
    c.append("  float* cur = buf0;")
    c.append("  float* nxt = buf1;")
    c.append("  size_t cur_elems = 0;")

    cur_M = None
    cur_N = None
    for op in ir["ops"]:
        oid, opk = op["id"][1:], op["op"].lower()
        if opk == "matmul":
            A, B = op["inputs"]
            Ashp = ir["values"][A]["shape"]
            Bshp = ir["values"][B]["shape"]
            if not (len(Ashp) == 2 and len(Bshp) == 2):
                raise RuntimeError(f"MatMul expects rank-2, got A{Ashp}, B{Bshp}")
            M, K = Ashp
            Kb, N = Bshp
            if K != Kb:
                raise RuntimeError(f"MatMul K mismatch: {K} vs {Kb}")
            srcA = "x" if A == ir["entry"]["inputs"][0] else "cur"
            srcB = find_weight(ir, B)
            c.append(f"  /* MatMul {A}({M}x{K}) x {B}({Kb}x{N}) */")
            c.append(f"  if ((size_t)({M}*{N}) > BUF_CAP) return -2;")
            c.append(f"  for(int i=0;i<{M}*{N};++i) nxt[i]=0.f;")
            c.append(f"  kernel_matmul({srcA}, {srcB}, nxt, {M},{K},{N});")
            c.append(f"  cur_elems = (size_t)({M}*{N});")
            c.append("  { float* t = cur; cur = nxt; nxt = t; }")
            cur_M, cur_N = M, N

        elif opk == "add":
            a, b = op["inputs"]

            # pick the constant side regardless of position
            if is_const_value(ir, b):
                w = b
            elif is_const_value(ir, a):
                w = a
            else:
                raise RuntimeError("Phase-1 Add requires one const operand")

            bshape = ir["values"][w]["shape"]
            sym = find_weight(ir, w)

            # support (M,N)+(N) or (M,N)+(M,N)
            if len(bshape) == 1 and bshape[0] == cur_N:
                c.append(f"  for(int i=0;i<{cur_M}; ++i) kernel_add(cur + i*{cur_N}, {sym}, {cur_N});")
            elif len(bshape) == 2 and bshape[0] == cur_M and bshape[1] == cur_N:
                c.append(f"  kernel_add(cur, {sym}, {cur_M*cur_N});")
            else:
                raise RuntimeError(f"unsupported add shape: {bshape}, expected [{cur_N}] or [{cur_M},{cur_N}]")

        elif opk == "relu":
            c.append(f"  kernel_relu(cur, {cur_M*cur_N});")
        elif opk in ("identity", "reshape", "const"):
            pass
        else:
            raise RuntimeError(f"unhandled op {opk}")

    c.append("  size_t out_elems = 1;")
    c.append("  for (size_t d=0; d<outputs[0].rank; ++d) out_elems *= outputs[0].shape[d];")
    c.append("  if (out_elems != cur_elems) return -3;")
    c.append("  for (size_t i=0;i<out_elems;++i) outputs[0].data[i]=cur[i];")
    c.append("  return 0; }")

    Path(out_c).write_text("\n".join(c))

if __name__ == "__main__":
    emit(sys.argv[1], sys.argv[2])
