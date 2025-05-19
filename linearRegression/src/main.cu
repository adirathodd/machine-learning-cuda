#include <csv.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <iostream>
#include <linearRegression.cuh>
#include <lr_kernels.cuh>

int main(){
    int numRows, numCols;
    float *data = loadCSV("data/student_performance.csv", &numRows, &numCols);
    int numFeatures = numCols - 1;

    //scale the features
    std::vector<float> attribute_means[numFeatures] = {0};
    std::vector<float> attribute_deviations[numFeatures] = {0};

    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numFeatures; j++)
            attribute_means[j] += data[i * numCols + j];

    for (int j = 0; j < numFeatures; j++)
        attribute_means[j] /= numRows;

    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numFeatures; j++)
            attribute_deviations[j] += powf(data[i * numCols + j] - attribute_means[j], 2);

    for (int j = 0; j < numFeatures; j++) {
        attribute_deviations[j] = sqrt(attribute_deviations[j] / numRows);
        if (attribute_deviations[j] == 0.0f)
            attribute_deviations[j] = 1.0f;
    }

    float *scaled_features = (float *)malloc(sizeof(float) * numRows * numFeatures);
    for (int i = 0; i < numRows; i++)
        for (int j = 0; j < numFeatures; j++)
            scaled_features[i * numFeatures + j] =
                (data[i * numCols + j] - attribute_means[j]) / attribute_deviations[j];
            
    // Create and shuffle row indices
    std::vector<int> indices(numRows);
    for (int i = 0; i < numRows; i++){
        indices[i] = i;
    }
    std::mt19937 g(5);
    std::shuffle(indices.begin(), indices.end(), g);

    int trainRows = static_cast<int>(numRows * 0.8);
    int testRows = numRows - trainRows;

    std::vector<float> X_train(trainRows * numFeatures);
    std::vector<float> y_train(trainRows);
    std::vector<float> X_test(testRows * numFeatures);
    std::vector<float> y_test(testRows);

    for (int i = 0; i < trainRows; i++){
        int idx = indices[i];
        std::memcpy(&X_train[i * numFeatures],
                    &scaled_features[idx * numFeatures],
                    (numFeatures) * sizeof(float));
        y_train[i] = data[idx * numCols + numFeatures];
    }


    for (int i = trainRows; i < numRows; i++){
        int idx = indices[i];
        int testIndex = i - trainRows;
        std::memcpy(&X_test[testIndex * numFeatures],
                    &scaled_features[idx * numFeatures],
                    (numFeatures) * sizeof(float));
        y_test[testIndex] = data[idx * numCols + numFeatures];
    }

    delete[] data;

    // for(int i = 0; i < 5; i++){
    //     printf("Row %d: ", i);
    //     for(int j = 0; j < numCols; j++){
    //         printf("%f ", scaled_features[i * (numFeatures) + j]);
    //     }
    //     printf("\n");
    // }

    linearRegression lr;
    lr.fit(X_train, y_train, trainRows, numCols - 1, 0.001, 10);
}
