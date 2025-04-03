#include <csv.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <lr_kernels.cuh>
#include <algorithm>
#include <random>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <linearRegression.cuh>

int main(){
    int numRows, numCols;
    float *data = loadCSV("data/student_performance.csv", &numRows, &numCols);

    // Create and shuffle row indices
    std::vector<int> indices(numRows);
    for (int i = 0; i < numRows; i++){
        indices[i] = i;
    }
    std::mt19937 g(5);
    std::shuffle(indices.begin(), indices.end(), g);

    int trainRows = static_cast<int>(numRows * 0.8);
    int testRows = numRows - trainRows;

    std::vector<float> X_train(trainRows * (numCols - 1));
    std::vector<float> y_train(trainRows);
    std::vector<float> X_test(testRows * (numCols - 1));
    std::vector<float> y_test(testRows);

    for (int i = 0; i < trainRows; i++){
        int idx = indices[i];
        std::memcpy(&X_train[i * (numCols - 1)],
                    &data[idx * numCols],
                    (numCols - 1) * sizeof(float));
        y_train[i] = data[idx * numCols + (numCols - 1)];
    }


    for (int i = trainRows; i < numRows; i++){
        int idx = indices[i];
        int testIndex = i - trainRows;
        std::memcpy(&X_test[testIndex * (numCols - 1)],
                    &data[idx * numCols],
                    (numCols - 1) * sizeof(float));
        y_test[testIndex] = data[idx * numCols + (numCols - 1)];
    }

    delete[] data;

    linearRegression lr;
    lr.fit(X_train, y_train, trainRows, numCols - 1, 0.01f, 100);
}
