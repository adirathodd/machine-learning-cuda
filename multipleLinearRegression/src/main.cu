#include <csv.h>
#include <stdio.h>
#include <stdlib.h>
#include <lr_kernels.cuh>
#include <linearRegression.cuh>

#define SEED 42

void split_data(const float* data,
                int numRows,
                int numCols,
                float **X_train,
                float **y_train,
                float **X_test,
                float **y_test,
				float split_rate)
{
    int numFeatures = numCols - 1;
    int train_N     = (int)(numRows * split_rate);
    int test_N      = numRows - train_N;

    /* allocate output arrays */
    *X_train = (float*)malloc(sizeof(float) * train_N * numFeatures);
    *y_train = (float*)malloc(sizeof(float) * train_N);
    *X_test  = (float*)malloc(sizeof(float) * test_N  * numFeatures);
    *y_test  = (float*)malloc(sizeof(float) * test_N);
    if (!*X_train || !*y_train || !*X_test || !*y_test) {
        fprintf(stderr, "split_data: allocation failed\n");
        exit(EXIT_FAILURE);
    }

    /* build and shuffle index array */
    int *indices = (int*)malloc(sizeof(int) * numRows);
    if (!indices) {
        fprintf(stderr, "split_data: indices malloc failed\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < numRows; ++i) indices[i] = i;
    srand(SEED);
    for (int i = numRows - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    /* fill training set */
    for (int i = 0; i < train_N; ++i) {
        const float *row = data + indices[i] * numCols;
        for (int j = 0; j < numFeatures; ++j) {
            (*X_train)[i * numFeatures + j] = row[j];
        }
        (*y_train)[i] = row[numFeatures];
    }

    /* fill test set */
    for (int i = 0; i < test_N; ++i) {
        const float *row = data + indices[train_N + i] * numCols;
        for (int j = 0; j < numFeatures; ++j) {
            (*X_test)[i * numFeatures + j] = row[j];
        }
        (*y_test)[i] = row[numFeatures];
    }

    free(indices);
}


int main(int argc, char *argv[]){

    if(argc != 2){
        printf("Usage - %s <nepochs>\n", argv[0]);
        return -1;
    }

	int numRows, numCols;
	float *data = loadCSV("data/student_performance.csv", &numRows, &numCols);

    // standardize features

    const int cols_to_std[] = {0, 1, 3, 4};
    const int n_std_cols = 4;

    for(int c=0; c < n_std_cols; ++c){
        int col = cols_to_std[c];

        double sum = 0.0;
        for(int i = 0; i < numRows; ++i) sum += data[i * numCols + col];
        double mean = sum / numRows;

        double sq_sum = 0.0, v;
        for(int i = 0; i < numRows; i++){
            v = data[i * numCols + col] - mean;
            sq_sum += v * v;
        }
        double stddev = std::sqrt(sq_sum / numRows);
        if(stddev == 0.0) stddev = 1.0;

        for(int i = 0; i < numRows; ++i){
            data[i * numCols + col] = static_cast<float>((data[i * numCols + col] - mean) / stddev);
        }
    }

	int numFeatures = numCols - 1, epochs = strtol(argv[1], NULL, 10);
	float train = 0.8,  p = 0.00001;

	int train_N = numRows * train;
    int test_N = numRows - train_N;
	
	float *X_train, *X_test, *y_train, *y_test;
	
	split_data(data, numRows, numCols, &X_train, &y_train, &X_test, &y_test, 0.80f);

	linearRegression lr;
	lr.fit(X_train, y_train, train_N, numFeatures, p, epochs);

    float mse = 0.0f;
    float *preds = lr.predict(X_test, y_test, test_N, numFeatures, &mse);

    for(int i = 0; i < 10; i++){
        printf("Real - %f, Pred - %f\n", y_test[i], preds[i]);
    }

    printf("MSE - %f\n", mse);
}