#include <lr_kernels.cuh>
#include <stdio.h>

__global__ void computeG0(int numFeatures, int t, int k, float bias, float *X_train, float *y_train, float *weights, float *g_0) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int end = t * (k + 1), start = t * k;
    
    if(j < start || j >= end) return;

    float dot = 0.0f;
    for (int i = 0; i < numFeatures; i++) {
        dot += X_train[j * numFeatures + i] * weights[i];
    }

    float curr_g0 = bias + dot - y_train[j];

    g_0[j - start] = curr_g0;
}

__global__ void computeGi(int numCols, int numRows, int i, int t, int k, float bias, float *X_train, float *y_train, float *weights, float *g_i) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int end = t * (k+1), start = t * k;
    if (j < start || j >= end) {
        return;
    }

    float dot = 0.0f;
    for (int f = 0; f < numCols; f++) {
        dot += X_train[j * numCols + f] * weights[f];
    }

    float curr_gi = (bias + dot - y_train[j]) * X_train[j * numCols + i];

    g_i[j - start] = curr_gi;
}

__global__ void reduceSum(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = 0.0f;
    if (i < N)
        sum = input[i];
    if (i + blockDim.x < N)
        sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}
