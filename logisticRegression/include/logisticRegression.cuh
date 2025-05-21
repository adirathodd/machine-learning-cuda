#ifndef LOGISTIC_REGRESSION_CUH
#define LOGISTIC_REGRESSION_CUH

class LogisticRegression {
public:
	LogisticRegression(int num_features, int num_classes);
	~LogisticRegression();

	void fit(float* X, float* y, int num_samples, int num_epochs, float learning_rate);
	void predict(float* X, float* y_pred, int num_samples);
};

#endif