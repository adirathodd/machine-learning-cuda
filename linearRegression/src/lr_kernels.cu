#include <lr_kernels.cuh>
#include <stdio.h>

__global__ void computeG0(int numFeatures, int t, int k, float bias, 
     float * X_train,  float * y_train, 
     float * weights, float *g_0) {
    // Compute the global index
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute start and end indices for the current mini-batch
    int start = t * k; 
    int end = t * (k + 1);  // If each batch has exactly k elements, consider setting end = start + k
    
    if(j < start || j >= end) return;
    
    float dot = 0.0f;
    // Compute dot product for training example j
    for (int i = 0; i < numFeatures; i++) {
        dot += X_train[j * numFeatures + i] * weights[i];
    }
    
    float curr_g0 = bias + dot - y_train[j];
    // Store the gradient output, offset by the start index
    g_0[j - start] = curr_g0;
}

__global__ void computeGi(int numCols, int i, int t, int k, float bias, 
     float * X_train, float * y_train, 
     float * weights, float *g_i) {
    // Compute the global index for the current batch element
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int start = t * k;
    int end = t * (k + 1);  // If each batch has exactly k elements, consider setting end = start + k
    
    if (j < start || j >= end) {
        return;
    }
    
    float dot = 0.0f;
    // Compute dot product over all features for training example j
    for (int f = 0; f < numCols; f++) {
        dot += X_train[j * numCols + f] * weights[f];
    }
    
    // Multiply the error by the i-th feature
    float curr_gi = (bias + dot - y_train[j]) * X_train[j * numCols + i];
    g_i[j - start] = curr_gi;
}

__global__ void reduceSum( float* input, float* output, int N) {
    // Shared memory for partial sums
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    // Each block processes 2 * blockDim.x elements
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float sum = 0.0f;
    if (i < N)
        sum = input[i];
    if (i + blockDim.x < N)
        sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Binary tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}
