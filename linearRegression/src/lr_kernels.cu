#include <lr_kernels.cuh>

__global__ void computeG0(int numFeatures, int t, int k, float bias, float *X_train, float *y_train, float *weights, float *g_0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int end = t * k, start = (t-1) * (k + 1);
    if (idx < start || idx >= end) {
        return;
    }

    float dot = 0.0f;
    for (int i = 0; i < numFeatures; i++) {
        dot += X_train[idx * numFeatures + i] * weights[i];
    }
    dot += bias;
    float curr_g0 = dot - y_train[idx];

    g_0[idx - start] = curr_g0;
    
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
