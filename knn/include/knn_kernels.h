#ifndef KNN_KERNELS_H
#define KNN_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

__global__ void computeDistances(const float *X_train, const float *X_test, float *distances, int numTrain, int numFeatures);

#ifdef __cplusplus
}
#endif

#endif

