# Logistic Regression

This module implements a logistic regression model from scratch in C++ and accelerates matrix operations using NVIDIA CUDA. It processes CSV datasets, computes the maximum likelihood estimates for weights via batch gradient descent, and performs binary classification.

## Features

- CSV data loading and preprocessing (feature scaling and splitting data)  
- GPU-accelerated matrix and vector operations using CUDA kernels  
- Binary classification with sigmoid activation  
- Train/test split with reproducible fixed random seed  
- Evaluation metrics: accuracy, precision, recall, and F1 score 

## Dataset

This module uses the Pima Indians Diabetes Dataset from Kaggle:  
https://www.kaggle.com/datasets/kandij/diabetes-dataset

The CSV file (`diabetes.csv`) contains medical diagnostic measurements (e.g., glucose, blood pressure, BMI) and a binary label indicating diabetes onset.

## Results

- **Accuracy:** N/A 
- **Precision:** N/A
- **Recall:** N/A 
- **F1 Score:** N/A