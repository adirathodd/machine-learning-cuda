# Linear Regression with Gradient Descent

## Overview
This project implements linear regression using gradient descent and leverages CUDA to accelerate the training process. The focus is on parallelizing the computationally intensive gradient computations by distributing the workload across CUDA threads.

## How It Works

1. **Data Preparation:**  
   - We use the Boston Housing dataset for training.  

2. **Gradient Computation Kernel:**  
   - CUDA kernels compute the error and partial gradients for each training example in parallel.
   - A parallel reduction is used to sum the partial gradients across threads to obtain the overall gradient.

3. **Iterative Optimization:**  
   - The gradient computation and weight update steps are repeated for a predetermined number of iterations or until convergence.
   - CUDA's parallel processing significantly speeds up these iterative computations, especially on large datasets.

## Dataset
The project uses the Boston Housing dataset, a classic dataset for regression tasks. For more details and a hands-on example, check out the [Kaggle Notebook](https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset/notebook).

## Summary
By leveraging CUDA for linear regression, this project achieves fast, scalable gradient computation and weight updates. The use of parallel processing in gradient descent allows for efficient handling of large datasets and faster convergence, making it a powerful approach for accelerating machine learning workflows.