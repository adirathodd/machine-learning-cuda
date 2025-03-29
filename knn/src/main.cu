#include <stdio.h>
#include <stdlib.h>
#include <readCSV.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include <knn.h>

void shuffle(int *array, int n) {
    srand(1);
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

int main(int argc, char *argv[]){	
	if(argc != 1){
		printf("Usage - %s\n", argv[0]);
		return -1;
	}

	float *data = loadCSV("/home/adirathodd/Documents/intro-to-ml/knn/data/iris.data");	   
	int trainRows = numRows * 0.8, testRows = numRows - trainRows;
	
	int indices[numRows];

	for(int i = 0; i < numRows; i++) indices[i] = i;
	
	shuffle(indices, numRows);

	float *X_train = (float *)malloc(trainRows * (numCols-1) * sizeof(float));
	float *y_train = (float *)malloc(trainRows  * sizeof(float));
	float *X_test = (float *)malloc(testRows * (numCols-1) * sizeof(float));
	float *y_test = (float *)malloc(testRows * sizeof(float));
	
	for(int i = 0; i < trainRows; i++){
		int rowIndex = indices[i];
	        memcpy(&X_train[i * (numCols-1)], &data[rowIndex * numCols], (numCols - 1) * sizeof(float));
		memcpy(&y_train[i], &data[rowIndex * numCols + 4], sizeof(float));
	}

	for(int i = trainRows; i < numRows; i++){
                int rowIndex = indices[i];
                memcpy(&X_test[(i - trainRows) *(numCols-1)], &data[rowIndex * numCols], (numCols - 1) * sizeof(float));
                memcpy(&y_test[i-trainRows], &data[rowIndex * numCols + 4], sizeof(float));
        }
	
	//printData(X_train, X_test, y_train, y_test, numCols - 1, trainRows, testRows);
	
	for(int k = 1; k < 10; k++){
		knn model(trainRows, numCols - 1, k);
		model.fit(X_train, y_train);
		float correct = 0.0;

		for(int i = 0; i < testRows; i++) {
			float res = model.predict(&X_test[i * (numCols - 1)]);
			if(res == y_test[i]){
				correct++;
			}
		}

		printf("K-%d-NN Model Accuracy = %.2f%%\n", k, (correct / testRows) * 100);
	}

	free(data);
	free(X_train);
	free(y_train);
	free(X_test);
	free(y_test);

	return 0;
}
