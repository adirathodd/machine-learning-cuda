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
        float *weights;
        vector<float> X_train;
        vector<float> y_train;
    
    protected:
        float compute_g0(int numCols, int t, int k, float *d_X_train, float *d_y_train, float *d_weights);
        float compute_gi(int numCols, int i, int t, int k, float *d_X_train, float *d_y_train, float *d_weights);

    public:
        linearRegression(){}
        ~linearRegression() {};

        void fit(vector<float> X_train, vector<float> y_train, int numRows, int numCols, float p, int epochs);
        float predict(vector<float> row);

};

#endif