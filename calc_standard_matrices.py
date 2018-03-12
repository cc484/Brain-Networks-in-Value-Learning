import numpy as np
import pandas as pd
from multislice_static_signed import multislice_stat_si


def calc_standard_matrices(condition_matrix, g_val):
    """

    :param condition_matrix:
    :param g_val:
    :return:
    """

    # partition assignments structured as 2d array of dicts
    max_cms = max(condition_matrix.shape)
    s = np.asarray([[dict() for i in range(100)] for j in range(max_cms)])
    q = np.zeros((max_cms, 100))

    # Computes Time x Time correlation matrix
    for i in max_cms:
        t_matrix = np.corrcoef(condition_matrix[i].T)
        s[i], q[i] = multislice_stat_si(t_matrix, gval)

        # number of communities
        n_coms = np.asarray([[dict() for i in range(100)] for j in range(max_cms)])
        n_coms[i] = max(s[i])

        # state flexibility
        f = np.asarray([[dict() for i in range(100)] for j in range(max_cms)])
        f[id] = np.size(np.where(abs(np.diff(s[i]))>0)) / (max(t_matrix.shape) - 1)

        # computing promiscuity
        prm = np.empty(n_coms[i].shape)
        prm_temp = np.empty(n_coms[i].shape)
        for j in range(n_coms[i]):
            prm_temp[j] = np.size(np.where(s[i] == j))
        prm[i] = np.var(prm_temp)

        # null model for time x time matrix
        t_matrix_u = np.triu(t_matrix, 1)

        # threshold on weights of matrix


def calc_standard_matrices_abl(condition_matrix, g_val):

    # Computes Time x Time correlation matrix
    for i in max(condition_matrix.shape):
        t_matrix = np.corrcoef(condition_matrix[i].T)