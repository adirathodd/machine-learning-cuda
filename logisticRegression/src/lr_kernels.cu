#include <lr_kernels.cuh>
#include <stdio.h>

__global__
void compute_residuals(
    const float * __restrict__ X,
    const float * __restrict__ y,
    const float * __restrict__ w,
    float * __restrict__ res, // [N]
    float b, int N, int d
){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if(j >= N) return;

    // dot product X[j] * w
    const float *xj = X + j * d;
    float pred = b;

    for(int i = 0; i < d; ++i){
        pred += w[i] * xj[i];
    }

    res[j] = pred - y[j];

}

__global__ void compute_g_kernel(
    const float* __restrict__ X,    // [N×d]
    const float* __restrict__ r,    // [N]   residuals
          float*       grad,        // [d]   output gradients
    int             N,
    int             d
) {
    extern __shared__ float sdata[];
    int feat = blockIdx.x; // i
    int thread = threadIdx.x;
    float sum = 0.0f;

    for(int j = thread; j < N; j+=blockDim.x)
        sum += r[j] * X[j * d + feat];

    sdata[thread] = sum;
    __syncthreads();

    // sum reduction for this thread (feature)
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(thread < stride)
            sdata[thread] += sdata[thread + stride];
        
            __syncthreads();
    }

    if (thread == 0) grad[feat] = sdata[0]; // update grad for this feature
}

__global__ void predict_kernel(
    const float * __restrict__ X, // [Nxd]
    const float * __restrict__ y, // [N]
    const float * __restrict__ w, // [d]
    float *preds, float *mse_r, // [N]
    int N, int d, float bias
){
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // j (row)

    if (tid >= N) return;
    
    const float *Xj = X + tid * d;

    preds[tid] = bias;

    for(int i = 0; i < d; i++) preds[tid] += Xj[i] * w[i];

    mse_r[tid] = powf(y[tid] - preds[tid], 2);
}