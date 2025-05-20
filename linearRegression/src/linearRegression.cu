#include <linearRegression.cuh>
#include <random>
#include <lr_kernels.cuh>
#include <cub/cub.cuh>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(err));  \
        exit(err);                                             \
    }                                                          \
} while(0)

void linearRegression::fit(float *X_train, float *y_train, int N, int d, float p, int epochs) {
    this->X_train = X_train, this->y_train = y_train, this->p = p;
    this->N = N, this->d = d;
    float *g = (float *)calloc(sizeof(float), d); // gradients

    this->weights = (float *)malloc(sizeof(float) * d);
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f/std::sqrt(d));
    bias = dist(rng);
    for(int i = 0; i < d; ++i)
        weights[i] = dist(rng);

    cudaMalloc(&d_X, sizeof(float) * N * d);
    cudaMalloc(&d_y, sizeof(float) * N);
    cudaMalloc(&d_w, sizeof(float) * d);
    cudaMemcpy(d_X, X_train, sizeof(float) * N * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_train, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weights, sizeof(float) * d, cudaMemcpyHostToDevice);

    int e, i;

    for(e = 1; e <= epochs; e++){
        float g0 = compute_g(g);
        this->bias -= this->p * g0;
        
        #pragma unroll 5
        for(i = 0; i < d; i++) weights[i] -= p * g[i];
        cudaMemcpy(d_w, weights, sizeof(float) * d, cudaMemcpyHostToDevice);

        // if (e % 100 == 0) {
        //     printf("Epoch - %d\n", e);

        //     for(i = 0; i < d; i++) printf("weight[%d]=%f\n", i, weights[i]);
        //     printf("bias=%f\n\n", bias);
        // }
    }

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);

    printf("\nFinal Weights and Bias:\n");
    printf("Bias - %f\n", bias);
    for(i = 0; i < d; i++) printf("Weight[%d]=%f\n", i + 1, weights[i]);
    printf("\n");
}

float *linearRegression::predict(float *X_test, float *y_test, int N, int d, float *MSE){
    float *preds = (float *)malloc(sizeof(float) * N);

    float *d_mse_r, *d_preds;
    cudaMalloc(&d_mse_r, sizeof(float) * N);
    cudaMalloc(&d_preds, sizeof(float) * N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaMalloc(&d_X, sizeof(float) * N * d);
    cudaMalloc(&d_y, sizeof(float) * N);
    cudaMalloc(&d_w, sizeof(float) * d);
    cudaMemcpy(d_X, X_test, sizeof(float) * N * d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_test, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, weights, sizeof(float) * d, cudaMemcpyHostToDevice);

    predict_kernel<<<blocks, threads>>>(d_X, d_y, d_w, d_preds, d_mse_r, N, d, bias);

    cudaDeviceSynchronize();

    // compute mse
    float *d_mse = nullptr;
    cudaMalloc(&d_mse, sizeof(float));

    void * tmp = nullptr;
    size_t tmp_bytes = 0;

    cub::DeviceReduce::Sum(tmp,tmp_bytes, d_mse_r, d_mse, N);

    cudaMalloc(&tmp, tmp_bytes);
    cub::DeviceReduce::Sum(tmp, tmp_bytes, d_mse_r, d_mse, N);

    cudaFree(tmp);
    cudaMemcpy(MSE, d_mse, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(preds, d_preds, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_mse_r);
    cudaFree(d_preds);
    cudaFree(d_mse);

    *MSE /= N;

    return preds;
}

float linearRegression::compute_g(float *g_array){
    float *d_residuals = nullptr;
    cudaMalloc(&d_residuals, N * sizeof(float));

    // compute g0
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_residuals<<<blocks, threads>>>(d_X, d_y, d_w, d_residuals, bias, N, d);

    cudaDeviceSynchronize();

    // sum reduction
    float *d_g0 = nullptr;
    cudaMalloc(&d_g0, sizeof(float));

    void * tmp = nullptr;
    size_t tmp_bytes = 0;

    cub::DeviceReduce::Sum(tmp,tmp_bytes, d_residuals, d_g0, N);

    cudaMalloc(&tmp, tmp_bytes);
    cub::DeviceReduce::Sum(tmp, tmp_bytes, d_residuals, d_g0, N);

    cudaFree(tmp);
    
    float g0 = 0.0f;
    cudaMemcpy(&g0, d_g0, sizeof(float), cudaMemcpyDeviceToHost);
    // compute rest of gradients
    blocks = d;
    size_t shmem = sizeof(float) * threads;
    float *d_g;
    cudaMalloc(&d_g, sizeof(float) * N);

    compute_g_kernel<<<blocks, threads, shmem>>>(d_X, d_residuals, d_g, N, d);
    cudaDeviceSynchronize();
    cudaMemcpy(g_array, d_g, sizeof(float) * d, cudaMemcpyDeviceToHost);

    cudaFree(d_residuals);
    cudaFree(d_g0);

    return g0;
}