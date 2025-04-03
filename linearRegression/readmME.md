# CUDA Accelerated Linear Regression with Gradient Descent

## Overview
This project implements linear regression using gradient descent and leverages CUDA to accelerate the training process. The focus is on parallelizing the computationally intensive gradient computations by distributing the workload across CUDA threads. This approach enables efficient processing of large datasets and faster model convergence.

## Dataset
For this project, we use the **Student Performance** dataset from Kaggle. The dataset is designed for multiple linear regression tasks and includes various features that influence student performance. You can download the dataset from the following link:

[Student Performance Dataset on Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression?resource=download)

## How It Works

1. **Data Preparation:**  
   - Load the dataset (CSV) containing student performance data.
   - Perform any necessary preprocessing such as handling missing values, normalization, and splitting the data into training and testing sets.

2. **CUDA Acceleration:**  
   - Transfer data to device memory to leverage GPU parallelism.
   - Compute gradients in parallel using CUDA kernels.
   - Use efficient parallel reduction techniques to aggregate gradients across threads.

3. **Gradient Descent Optimization:**  
   - Update model weights using the computed gradients.
   - Repeat the gradient computation and weight update iteratively until convergence.

4. **Model Evaluation:**  
   - Assess the performance of the regression model using evaluation metrics like Mean Squared Error (MSE) and R-squared.
   - Analyze the model coefficients to understand the influence of each feature on student performance.

## Summary
By utilizing CUDA for linear regression, this project demonstrates how parallel computing can significantly speed up the training process of machine learning models. The Student Performance dataset serves as an excellent test case for exploring the benefits of CUDA-accelerated gradient descent in a multiple linear regression setting.

