"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Assignment 3
B. Chan, Z. Zhang, D. Fleet
"""

import matplotlib.pyplot as plt
import numpy as np

from utils import load_pickle_dataset

def visualize_2d_data(X, y):
    """ This function generates a 2D scatter plot given the inputs and their corresponding labels.
    Inputs with different classes are represented with different colours.

    Args:
    - X (ndarray (shape: (N, D))): A NxD matrix consisting N D-dimensional inputs.
    - y (ndarray (shape: (N, 1))): A N-column vector consisting N scalar outputs (labels).
    """
    assert len(X.shape) == len(y.shape) == 2, f"Input/output pairs must be 2D-arrays. X: {X.shape}, y: {y.shape}"
    (N, D) = X.shape
    assert N == y.shape[0], f"Number of samples must match for input/output pairs. X: {N}, y: {y.shape[0]}"
    assert D == 2, f"Expected 2 features. Got: {D}"
    assert y.shape[1] == 1, f"Y must be a column vector. Got: {y.shape}"

    # ====================================================
    # TODO: Implement your solution within the box
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=len(X[:]), edgecolor='k')
    plt.show()
    # ====================================================


if __name__ == "__main__":
    # Support generic_1, generic_2, generic_3
    dataset = "generic_3"

    assert dataset in ("generic_1", "generic_2", "generic_3", "wine"), f"Invalid dataset: {dataset}"

    dataset_path = f"./datasets/{dataset}.pkl"
    data = load_pickle_dataset(dataset_path)
    visualize_2d_data(data['train_X'], data['train_y'])
