import numpy as np
import pandas as pd

def calc_standard_matrices(condition_matrix, g_val):
    """

    :param condition_matrix:
    :param g_val:
    :return:
    """

    # Computes Time x Time correlation matrix
    for i in max(condition_matrix.shape):
        t_matrix = np.corrcoef(condition_matrix[i].T)

def calc_standard_matrices_abl(condition_matrix, g_val):

    # Computes Time x Time correlation matrix
    for i in max(condition_matrix.shape):
        t_matrix = np.corrcoef(condition_matrix[i].T)