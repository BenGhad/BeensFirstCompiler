# AICC: ONNX → C Compiler

**Elevator pitch:** Compile ONNX models to portable C with planned memory and BLAS speed.

---

## About the project

### What it does
Converts ONNX graphs into optimized C. plans a single tensor arena, and emits C that calls `cblas_sgemm` for large matmuls and a tight kernel for small ones. Writes into outputs in place when safe. A Python tester validates against ONNX Runtime and reports accuracy and speed.

### Inspiration
Jake Errington(For making me do this) and Christophe Dubach(For posting the COMP520 outline in advance)


### What it be doing
1. **Importer**
   - Reads ONNX. Normalizes float compute paths to `f32`.
   - Lower GEMM computations into Matmul and Adds when possible.
2. **IR**
   - Initial IR was planned out with a dataclass.
   - JSON with `values` (dtype, shape), `ops`, `consts`, and explicit entry I/O.
3. **Passes**
   - Canonicalize, DCE, shape and dtype checks, topo sort(only support DAGs for now).
   - **Memory planner:** first-fit packs live intervals into a single arena;
4. **Codegen**
   - Emits `kernel_matmul`, `kernel_add`, `kernel_relu`, plus a `cblas_sgemm` fast path.
   - Embeds weights as `static const float[]`. Computes directly into output buffers when possible.

```c
// Small-matmul kernel; large cases call cblas_sgemm.
static void kernel_matmul(const float* A,const float* B,float* C,int M,int K,int N){
  for(int i=0;i<M;++i){
    for(int j=0;j<N;++j) C[i*N+j]=0.f;
    for(int k=0;k<K;++k){
      float a=A[i*K+k];
      for(int j=0;j<N;++j) C[i*N+j]+=a*B[k*N+j];
    }
  }
}
```

# How to use
Run demo.py