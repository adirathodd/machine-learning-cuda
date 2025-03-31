#ifndef KNN_KERNELS_H
#define KNN_KERNELS_H

__global__ void computeDistances(const float *X_train, const float *X_test, float *distances, int numTrain, int numFeatures);

#endif