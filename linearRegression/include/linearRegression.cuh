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
		float *weights; // Vector (d-dimensional)
		float *X_train; // 2D Matrix (N x d)
		float *y_train; // 2D Matrix (N x 1)
	
	protected:
		float compute_g0();
		float *compute_g();
		void update_weights();
	
	public:
		void fit(float *X_train, float *y_train, int N, int d, float p, int epochs);
		float *predict(float *X_test, float *y_test, int N, int d);

};

#endif