#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

class linearRegression {
	private:
		float p, bias;
		float *weights, *d_w; // Vector (d-dimensional)
		float *X_train, *d_X; // 2D Matrix (N x d)
		float *y_train, *d_y; // 2D Matrix (N x 1)
		int N, d;
	
	protected:
		float compute_g(float *g_array);
	
	public:
		void fit(float *X_train, float *y_train, int N, int d, float p, int epochs);
		float *predict(float *X_test, float *y_test, int N, int d, float *mse);

};

#endif