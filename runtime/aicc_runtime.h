#pragma once
#include <stddef.h>

typedef struct {
  float* data;
  size_t rank;
  const size_t* shape;
} aicc_tensor;

__attribute__((visibility("default")))
int aicc_run(const aicc_tensor* inputs, aicc_tensor* outputs);
