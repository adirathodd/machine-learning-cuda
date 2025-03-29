#ifndef KNN_H
#define KNN_H

class knn {
public:
    knn(int rows, int cols, int k);

    ~knn();

    void fit(float* X_train, float* y_train);

    float predict(float* row);

private:
    float* X_train;
    float* y_train;
    int rows, cols, k;

    float findMajority(int *indices, float *distances);
};

#endif // KNN_H

