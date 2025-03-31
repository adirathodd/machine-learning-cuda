#ifndef LR_KERNELS_H
#define LR_KERNELS_H

#include <vector>
#include <cuda_runtime.h>

__global__ void updateWeights(const float *weights);

#endif