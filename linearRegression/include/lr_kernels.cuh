#ifndef LR_KERNELS_H
#define LR_KERNELS_H

#include <vector>
#include <cuda_runtime.h>

__global__ void computeG0(int numFeatures, int t, int k, float bias, float *X_train, float *y_train, float *weights, float *g_0);
__global__ void reduceSum(float* input, float* output, int N);
__global__ void computeGi(int numCols, int i, int t, int k, float bias, float *X_train, float *y_train, float *weights, float *g_i);

#endif