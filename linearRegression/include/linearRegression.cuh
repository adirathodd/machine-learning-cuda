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

    public:
        linearRegression(){}
        ~linearRegression() {};

        void fit(vector<float> X_train, vector<float> y_train, int numRows, int numCols, float p, int epochs);
        float predict(vector<float> row);        
};

#endif