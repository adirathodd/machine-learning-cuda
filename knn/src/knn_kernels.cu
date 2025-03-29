#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <knn_kernels.h>

__global__ void computeDistances(const float* X_train, const float* X_i, float* distances, int numTrain, int numFeatures) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numTrain) {
        float sum = 0.0;
        for (int j = 0; j < numFeatures; j++) {
            float diff = X_train[tid * numFeatures + j] - X_i[j];
            sum += diff * diff;
        }
        
        distances[tid] = sqrtf(sum);
    }
}

