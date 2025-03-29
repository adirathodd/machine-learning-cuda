#include <knn.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <knn_kernels.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <map>
#include <cfloat>

knn::knn(int rows, int cols, int k) : X_train(nullptr), y_train(nullptr), rows(rows), cols(cols), k(k) {}

knn::~knn() {}

void knn::fit(float *X_train, float *y_train){
    this->X_train = X_train;
    this->y_train = y_train;
}

float knn::findMajority(int *indices, float *distances){
	std::map<int, int>votes;
	float minDistance = FLT_MAX;
	int minLabel = 0;
	
	for(int i = 0; i < this->k; i++){
		int idx = indices[i];
		int label = static_cast<int>(this->y_train[idx]);
		float dist = distances[idx];
		votes[label] += 1;
		
		if(dist < minDistance){
			minDistance = dist, minLabel = label;
		}
	}
	
	int resLabel = 0, resVotes = 0;
	
	for(auto it = votes.begin(); it != votes.end(); ++it){
		int label = it->first, count = it->second;

		if(count > resVotes){
			resLabel = label, resVotes = count;
		} else if (count == resVotes && minLabel == label){
			resLabel = label;
		}
	}

	return resLabel;
}

float knn::predict(float *X_i){
	float *d_X_train, *d_X_i, *d_distances;

	cudaMalloc(&d_X_train, sizeof(float) * this->rows * this->cols);
	cudaMalloc(&d_X_i, sizeof(float) * this->cols);
 	cudaMalloc(&d_distances, sizeof(float) * this->rows);

	cudaMemcpy(d_X_train, this->X_train, sizeof(float) * this->rows * this->cols, cudaMemcpyHostToDevice);
	cudaMemcpy(d_X_i, X_i, this->cols * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
    	int blocksPerGrid = (this->rows + threadsPerBlock - 1) / threadsPerBlock;
    	computeDistances<<<blocksPerGrid, threadsPerBlock>>>(d_X_train, d_X_i, d_distances, this->rows, this->cols);
	
	cudaDeviceSynchronize();

	thrust::device_ptr<float> device_ptr = thrust::device_pointer_cast(d_distances);
	thrust::device_vector<int> d_indices(this->rows);
	thrust::sequence(d_indices.begin(), d_indices.end());
	
	thrust::sort_by_key(device_ptr, device_ptr + this->rows, d_indices.begin());

	int *h_sortedIndices = (int*)malloc(this->k * sizeof(int));
cudaMemcpy(h_sortedIndices, thrust::raw_pointer_cast(d_indices.data()), this-> k * sizeof(int), cudaMemcpyDeviceToHost);
	
	float *h_distances = (float *)malloc(this->rows * sizeof(float));
	cudaMemcpy(h_distances, thrust::raw_pointer_cast(device_ptr), this->rows * sizeof(float), cudaMemcpyDeviceToHost);
	
	float res = findMajority(h_sortedIndices, h_distances);
		
	cudaFree(d_X_train);
	cudaFree(d_X_i);
	cudaFree(d_distances);

	return res;
}
