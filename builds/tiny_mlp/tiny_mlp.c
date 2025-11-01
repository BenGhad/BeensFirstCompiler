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
  float* y = outputs[0].data;
  (void)y;
  static float buf0[1*128*128];
  static float buf1[1*128*128];
  float* tmp0 = buf0; float* tmp1 = buf1;
  for(int i=0;i<1*3;++i) tmp0[i]=0.f;
  kernel_matmul(x, W0, tmp0, 1,4,3);
  kernel_add(tmp0, W1, 3);
  kernel_relu(tmp0, 3);
  for(size_t i=0;i<outputs[0].shape[0]*outputs[0].shape[1];++i) outputs[0].data[i]=tmp0[i];
  return 0; }