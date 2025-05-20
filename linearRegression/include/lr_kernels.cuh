#ifndef LR_KERNELS_H
#define LR_KERNELS_H

#include <vector>
#include <cuda_runtime.h>

__global__
void compute_residuals(
    const float * __restrict__ X,
    const float * __restrict__ y,
    const float * __restrict__ w,
    float * __restrict__ res, // [N]
    float b, int N, int d
);

__global__ void compute_g_kernel(
    const float* __restrict__ X,    // [N×d]
    const float* __restrict__ r,    // [N]   residuals
    float*       grad,              // [d]   output gradients
    int             N,
    int             d
);

__global__ void predict_kernel(
    const float * __restrict__ X, // [Nxd]
    const float * __restrict__ y, // [N]
    const float * __restrict__ w, // [d]
    float *preds, float *mse_r, // [N]
    int N, int d, float bias
);

#endif