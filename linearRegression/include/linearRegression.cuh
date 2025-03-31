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
        float p;
        float *weights;

    public:
        linearRegression(float p = 0.1){
            this->p = p;
        }

        ~linearRegression() {};

        void fit(vector<vector<float>> X_train, vector<vector<float>> y_train);
        float predict(vector<float> row);        
};

#endif