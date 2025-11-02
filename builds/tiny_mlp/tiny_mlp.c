#include <stddef.h>
#include "aicc_runtime.h"

static void kernel_matmul(const float* A, const float* B, float* C, int M, int K, int N){
  for(int i=0;i<M;++i)
    for(int k=0;k<K;++k){
      float a = A[i*K + k];
      for(int j=0;j<N;++j)
        C[i*N + j] += a * B[k*N + j];
    }
}


static void kernel_add(float* X, const float* B, int N){
  for(int i=0;i<N;++i) X[i] += B[i];
}


static void kernel_relu(float* X, int N){
  for(int i=0;i<N;++i) X[i] = X[i] > 0.f ? X[i] : 0.f;
}

static const float W0[] = {0.844303787,-0.736737967,0.898725033,-0.349847168,-1.33939695,-0.654554009,0.858656228,0.930095136,0.311900854,-0.11983905,-0.890809357,0.929849625};
static const float W1[] = {-0.841090202,-2.27875137,-0.453444958};
int aicc_run(const aicc_tensor* inputs, aicc_tensor* outputs){
  enum { ARENA_ELEMS = 4 };
  static float arena[ARENA_ELEMS];
  /* MatMul x(1x4) x W(4x3) -> z1 */
  for(int i=0;i<1*3;++i) (arena + 0)[i]=0.f;
  kernel_matmul(inputs[0].data, W0, (arena + 0), 1,4,3);
  /* Add z1 + b -> z2 */
  if ((arena + 0) != (arena + 0)) for(int i=0;i<3; ++i) (arena + 0)[i] = (arena + 0)[i];
  for(int i=0;i<1; ++i) kernel_add((arena + 0) + i*3, W1, 3);
  /* ReLU z2 -> y */
  if ((arena + 0) != (arena + 0)) for(int i=0;i<3; ++i) (arena + 0)[i] = (arena + 0)[i];
  kernel_relu((arena + 0), 3);
  for (int oi=0; oi< (int)1e9; ++oi) { /* dummy to avoid unused warning */ break; }
  { size_t n=1; for(size_t d=0; d<outputs[0].rank; ++d) n*=outputs[0].shape[d];
    for(size_t k=0;k<n;++k) outputs[0].data[k] = (arena + 0)[k]; }
  return 0; }