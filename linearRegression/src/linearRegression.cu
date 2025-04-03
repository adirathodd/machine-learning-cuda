#include <linearRegression.cuh>
#include <random>
#include <cuda_runtime.h>
#include <lr_kernels.cuh>

float compute_g0(int numCols, int t, int batchSize, float bias, float *d_X_train, float *d_y_train, float *d_weights);


void linearRegression::fit(vector<float> X_train, vector<float> y_train, int numRows, int numCols, float p, int epochs) {
    this->X_train = X_train;
    float *d_X_train, *d_y_train, *d_weights;
    this->y_train = y_train;
    this->p = p;

    std::mt19937 rng(42);
    this->bias = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    this->weights = new float[numCols];
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < numCols; i++) {
        weights[i] = dist(rng);
    }
    
    int batchSize = 32;
    int numBatches = numRows / batchSize;
    if (numRows % batchSize != 0) numBatches++;

    cudaMalloc((void**)&d_X_train, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&d_y_train, numRows * sizeof(float));
    cudaMalloc((void**)&d_weights, numCols * sizeof(float));
    cudaMemcpy(d_X_train, X_train.data(), numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_train, y_train.data(), numRows * sizeof(float), cudaMemcpyHostToDevice);

    for(int i = 0; i < epochs; i++) {
        for(int t = 1; t <= numBatches; t++) {
            // update weights
            cudaMemcpy(d_weights, weights, numCols * sizeof(float), cudaMemcpyHostToDevice);
            float g_0 = compute_g0(numCols, t, batchSize, this->bias, d_X_train, d_y_train, d_weights);
            printf("Epoch %d, Batch %d, g_0: %f\n", i, t, g_0);
        }
    }
}

float linearRegression::predict(vector<float> row) {
    float prediction = 0.0f;
    for (int i = 0; i < row.size(); i++) {
        prediction += row[i] * weights[i];
    }
    return prediction;
}

float compute_g0(int numCols, int t, int batchSize, float bias, float *d_X_train, float *d_y_train, float *d_weights) {
    float *d_g0;
    cudaMalloc((void**)&d_g0, batchSize * sizeof(float));

    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    computeG0<<<blocksPerGrid, threadsPerBlock>>>(numCols, t, batchSize, bias,
                                                d_X_train, d_y_train, d_weights, d_g0);

    cudaDeviceSynchronize();

    int reduceThreads = 256;
    int reduceBlocks = (batchSize + reduceThreads * 2 - 1) / (reduceThreads * 2);
    size_t sharedMemSize = reduceThreads * sizeof(float);
    float *d_partialSums;
    cudaMalloc((void**)&d_partialSums, reduceBlocks * sizeof(float));

    reduceSum<<<reduceBlocks, reduceThreads, sharedMemSize>>>(d_g0, d_partialSums, batchSize);
    cudaDeviceSynchronize();

    int s = reduceBlocks;
    while (s > 1) {
        int threads = (s < reduceThreads * 2 ? s / 2 : reduceThreads);
        int blocks = (s + threads * 2 - 1) / (threads * 2);
        reduceSum<<<blocks, threads, threads * sizeof(float)>>>(d_partialSums, d_partialSums, s);
        cudaDeviceSynchronize();
        s = blocks;
    }

    float h_g0;
    cudaMemcpy(&h_g0, d_partialSums, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_g0);
    cudaFree(d_partialSums);

    return h_g0;
}