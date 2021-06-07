"""
CSCC11 - Introduction to Machine Learning, Winter 2021, Assignment 3
B. Chan, Z. Zhang, D. Fleet
"""

import _pickle as pickle
import numpy as np

def softmax(logits):
    """ This function applies softmax function to the logits.

    Args:
    - logits (ndarray (shape: (N, K))): A NxK matrix consisting N K-dimensional logits.

    Output:
    - (ndarray (shape: (N, K))): A NxK matrix consisting N K-categorical distribution.
    """
    e_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e_logits / np.sum(e_logits, axis=1, keepdims=True)

def load_pickle_dataset(file_path):
    """ This function loads a pickle file given a file path.

    Args:
    - file_path (str): The path of the pickle file

    Output:
    - (dict): A dictionary consisting the dataset content.
    """
    return pickle.load(open(file_path, "rb"))
