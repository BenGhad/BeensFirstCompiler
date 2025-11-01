#pragma once
#include <stddef.h>

// Only do floats 4 now :broken_heart:
typedef struct {
  float* data;
  const size_t* shape;  // len = rank
  size_t rank;
} aicc_tensor;


int aicc_run(const aicc_tensor* inputs, aicc_tensor* outputs);
