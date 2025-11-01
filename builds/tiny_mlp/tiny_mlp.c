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
  const float* x = inputs[0].data;
  enum { BUF_CAP = 12 };
  static float buf0[BUF_CAP];
  static float buf1[BUF_CAP];
  float* cur = buf0;
  float* nxt = buf1;
  size_t cur_elems = 0;
  /* MatMul x(1x4) x W(4x3) */
  if ((size_t)(1*3) > BUF_CAP) return -2;
  for(int i=0;i<1*3;++i) nxt[i]=0.f;
  kernel_matmul(x, W0, nxt, 1,4,3);
  cur_elems = (size_t)(1*3);
  { float* t = cur; cur = nxt; nxt = t; }
  for(int i=0;i<1; ++i) kernel_add(cur + i*3, W1, 3);
  kernel_relu(cur, 3);
  size_t out_elems = 1;
  for (size_t d=0; d<outputs[0].rank; ++d) out_elems *= outputs[0].shape[d];
  if (out_elems != cur_elems) return -3;
  for (size_t i=0;i<out_elems;++i) outputs[0].data[i]=cur[i];
  return 0; }