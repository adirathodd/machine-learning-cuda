#include <stdio.h>
#include <stdlib.h>
#include <readCSV.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

void shuffle(int *array, int n) {
    srand(time(NULL));
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void printData(float *X_train, float *X_test, float *y_train, float *y_test, int numFeatures, int trainRows, int testRows){
	// Print X_train
	printf("X_train:\n");
	for (int i = 0; i < trainRows; i++) {
	    printf("Row %d: ", i);
	    for (int j = 0; j < numFeatures; j++) {
		printf("%f ", X_train[i * numFeatures + j]);
	    }
	    printf("\n");
	}

	// Print y_train
	printf("\ny_train:\n");
	for (int i = 0; i < trainRows; i++) {
	    printf("%f ", y_train[i]);
	}
	printf("\n");

	// Print X_test
	printf("\nX_test:\n");
	for (int i = 0; i < testRows; i++) {
	    printf("Row %d: ", i);
	    for (int j = 0; j < numFeatures; j++) {
		printf("%f ", X_test[i * numFeatures + j]);
	    }
	    printf("\n");
	}

	// Print y_test
	printf("\ny_test:\n");
	for (int i = 0; i < testRows; i++) {
	    printf("%f ", y_test[i]);
	}
	printf("\n");

}

int main(int argc, char *argv[]){	
	if(argc != 2){
		printf("Usage - %s <filepath>\n", argv[0]);
		return -1;
	}

	float *data = loadCSV(argv[1]);	
	
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
	
	printData(X_train, X_test, y_train, y_test, numCols - 1, trainRows, testRows);
	return 0;
}
