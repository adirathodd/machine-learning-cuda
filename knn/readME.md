# K-Nearest Neighbors (KNN) Classifier

This module implements a K-Nearest Neighbors classifier from scratch in C++ and accelerates distance computations using NVIDIA CUDA. It processes CSV datasets, splits them into training and test sets, and performs predictions with majority voting and distance-based tie-breaking.

## Features

- CSV data loading and preprocessing
- Reproducible train/test split with fixed random seed
- GPU-accelerated distance calculations via CUDA kernels
- Efficient neighbor sorting using Thrust

## Dataset

This module includes the Iris dataset for classification experiments. The dataset contains 150 samples with four features (sepal length, sepal width, petal length, petal width) across three classes (setosa, versicolor, virginica). Use the provided `iris.data` file in the `data/` directory.

## Results

![KNN Results](knn.png)