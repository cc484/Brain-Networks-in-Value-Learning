import numpy as np
import pandas as pd
import networkx as nx
from louvain import mod_max
from consensus_iterative import cons_iter


def brain_st_extr(condition_matrix, g_val):
    """
    Extract the brain state at a certain time points as a
    time x time correlation matrix (tt_matrix)

    :param condition_matrix:
    :param g_val:

    :return:
    b_final:
    qpc_total:
    bn_final:
    nqpc_total:
    """

    cm_size = condition_matrix.shape  # size of condition matrix
    max_cms = np.max(cm_size)  # max dimension of the condition matrix size

    s = pd.DataFrame(max_cms, 100)
    sn = pd.DataFrame(max_cms, 100)

    q = np.zeros((cm_size, 100), float)
    qn = np.zeros((cm_size, 100), float)

    for i in range(cm_size):
        # Compute time x time correlation matrix
        tt_matrix = np.corrcoef(condition_matrix[i].T)
        tt_graph = nx.from_numpy_array(tt_matrix)

        # Compute null model for tt_matrix
        tt_matrix_u = np.triu(tt_matrix, 1)
        x_vector = tt_matrix_u[abs(tt_matrix_u).ravel.nonzero() > 0]
        p = np.random.permutation(np.size(x_vector))
        n_matrix = np.zeros(tt_matrix.size)
        ind = abs(tt_matrix_u).ravel.nonzero > 0
        n_matrix[ind] = tt_matrix_u[ind[p]]

        for j in np.max(tt_matrix.size):
            n_matrix[i][j] = 1

        # Apply community detection using mod_mox
        for runI in range(100):
            s[i][runI], q[i][runI] = mod_max(tt_graph, g_val)
            sn[i][runI], qn[i][runI] = mod_max(tt_graph, g_val)

    s_total = pd.DataFrame(max_cms, 1)
    for itr in max_cms:
        s_total[itr] = s[itr][1]
        for runItr in range(2, 100):
            s_total[itr] = np.hstack((s_total[itr], s[itr][runItr]))

    qpc_total = 0
    s_total2 = []
    for sessI in max_cms:
        s2, q2, x_new3, qpc = cons_iter(s_total[sessI].T)
        s_total2[sessI] = s2.T
        qpc_total = qpc_total + qpc

    b = []
    for sessI in max_cms:
        for runI in range(100):
            for j in np.max(np.max(s_total2[sessI])):
                br = b[runI]
                br[:][j] = np.mean(condition_matrix[sessI].ravel.nonzero(s_total2[sessI]))

    """
    dim = np.ndim(b[1])
    M = np.concatenate(dim + 1, B[:])

    b_final[sessI] = np.nanmean(M, 3)

    bn_final[sessI] = np.nanmean(M, 3)
    nqpc_total = 0

    return (b_final, qpc_total, bn_final, nqpc_total)
    """




