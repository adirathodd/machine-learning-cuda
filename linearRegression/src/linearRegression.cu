#include <linearRegression.cuh>
#include <random>
#include <cuda_runtime.h>
#include <lr_kernels.cuh>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(err);                                             \
    }                                                          \
} while(0)

void linearRegression::fit(vector<float> X_train, vector<float> y_train, int numRows, int numCols, float p, int epochs) {
    this->X_train = X_train, this->y_train = y_train, this->p = p;
    float *d_X_train, *d_y_train, *d_weights;

    // initialize weights and bias
    std::mt19937 rng(42);
    this->bias = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    this->weights = new float[numCols];
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    int k = 100;
    int numBatches = numRows / k;
    if (numRows % k != 0) numBatches++;

    CUDA_CHECK(cudaMalloc((void**)&d_X_train, numRows * numCols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y_train, numRows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_weights, numCols * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X_train, X_train.data(), numRows * numCols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train.data(), numRows * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, weights, numCols * sizeof(float), cudaMemcpyHostToDevice));

    float g[numCols];

    for(int e = 0; e < epochs; e++) {

        for (int i = 0; i < numCols; i++) {
            printf("Weight %d: %f\n", i+1, weights[i]);
        }
        printf("-------------------------------------\n\n");
        
        for(int t = 0; t < numBatches; t++) {
            // compute g_0
            float g_0 = compute_g0(numCols, t, k, d_X_train, d_y_train, d_weights);

            //compute loss for each feature
            for(int i = 0; i < numCols; i++){
                g[i] = compute_gi(numCols, i, t, k, d_X_train, d_y_train, d_weights);
            }

            //update bias
            this->bias = this->bias - (this->p * g_0);

            //update weights
            for(int i = 0; i < numCols; i++) this->weights[i] = this->weights[i] - (this->p * g[i]);
            CUDA_CHECK(cudaMemcpy(d_weights, weights, numCols * sizeof(float), cudaMemcpyHostToDevice));

        }
        
    }

    printf("Final Bias: %f\n", bias);
    for (int i = 0; i < numCols; i++) {
        printf("Final Weight %d: %f\n", i+1, weights[i]);
    }

    return;
}

float linearRegression::predict(vector<float> row) {
    float prediction = 0.0f;
    for (int i = 0; i < row.size(); i++) {
        prediction += row[i] * weights[i];
    }
    return prediction;
}

float linearRegression::compute_g0(int numCols, int t, int k, float *d_X_train, float *d_y_train, float *d_weights) {
    float *d_g0;
    CUDA_CHECK(cudaMalloc((void**)&d_g0, k * sizeof(float)));

    int threadsPerBlock = 32;
    int blocksPerGrid = (k + threadsPerBlock - 1) / threadsPerBlock;
    computeG0<<<blocksPerGrid, threadsPerBlock>>>(numCols, t, k, bias,
                                                d_X_train, d_y_train, d_weights, d_g0);

    CUDA_CHECK(cudaDeviceSynchronize());

    int reduceThreads = 32;
    int reduceBlocks = (k + reduceThreads * 2 - 1) / (reduceThreads * 2);
    size_t sharedMemSize = reduceThreads * sizeof(float);
    float *d_partialSums;
    CUDA_CHECK(cudaMalloc((void**)&d_partialSums, reduceBlocks * sizeof(float)));

    reduceSum<<<reduceBlocks, reduceThreads, sharedMemSize>>>(d_g0, d_partialSums, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    int s = reduceBlocks;
    while (s > 1) {
        int threads = (s < reduceThreads * 2 ? s / 2 : reduceThreads);
        int blocks = (s + threads * 2 - 1) / (threads * 2);
        reduceSum<<<blocks, threads, threads * sizeof(float)>>>(d_partialSums, d_partialSums, s);
        CUDA_CHECK(cudaDeviceSynchronize());
        s = blocks;
    }

    float h_g0;
    CUDA_CHECK(cudaMemcpy(&h_g0, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_g0));
    CUDA_CHECK(cudaFree(d_partialSums));

    return h_g0;
}

float linearRegression::compute_gi(int numCols, int i, int t, int k, float *d_X_train, float *d_y_train, float *d_weights) {
    float *d_g_i;
    CUDA_CHECK(cudaMalloc((void**)&d_g_i, k * sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (k + threadsPerBlock - 1) / threadsPerBlock;

    // d_g_i[j] = (bias + dot(X[j], w) - y[j]) * X[j * numCols + i]
    computeGi<<<blocksPerGrid, threadsPerBlock>>>(numCols, i, t, k, bias,
                    d_X_train, d_y_train, d_weights, d_g_i);
    CUDA_CHECK(cudaDeviceSynchronize());

    int reduceThreads = 256;
    int reduceBlocks = (k + reduceThreads * 2 - 1) / (reduceThreads * 2);
    size_t sharedMemSize = reduceThreads * sizeof(float);
    float *d_partialSums;
    CUDA_CHECK(cudaMalloc((void**)&d_partialSums, reduceBlocks * sizeof(float)));

    reduceSum<<<reduceBlocks, reduceThreads, sharedMemSize>>>(d_g_i, d_partialSums, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    int s = reduceBlocks;
    while (s > 1) {
        int threads = (s < reduceThreads * 2 ? s / 2 : reduceThreads);
        int blocks = (s + threads * 2 - 1) / (threads * 2);
        reduceSum<<<blocks, threads, threads * sizeof(float)>>>(d_partialSums, d_partialSums, s);
        CUDA_CHECK(cudaDeviceSynchronize());
        s = blocks;
    }

    float h_g_i;
    CUDA_CHECK(cudaMemcpy(&h_g_i, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_g_i));
    CUDA_CHECK(cudaFree(d_partialSums));

    return h_g_i;
}
