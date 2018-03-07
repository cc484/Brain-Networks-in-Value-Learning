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

    cm_size = list(condition_matrix.shape)  # size of condition matrix
    max_cms = max(cm_size)  # max dimension of the condition matrix size

    s = pd.DataFrame()
    sn = pd.DataFrame()

    q = np.zeros((max_cms, 100))
    qn = np.zeros((max_cms, 100))

    for i in range(max_cms):
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
            n_matrix[i, j] = 1

        # Apply community detection using mod_mox
        for runI in range(100):
            s[i, runI], q[i, runI] = mod_max(tt_graph, g_val)
            sn[i, runI], qn[i, runI] = mod_max(tt_graph, g_val)

    s_total = pd.DataFrame(max_cms, 1)
    for itr in max_cms:
        s_total[itr] = s[itr, 1]
        for runItr in range(2, 100):
            s_total[itr] = np.hstack((s_total[itr], s[itr][runItr]))

    qpc_total = 0
    s_total2 = pd.DataFrame()
    for sessI in max_cms:
        s2, q2, x_new3, qpc = cons_iter(s_total[sessI].T)
        s_total2[sessI] = s2.T
        qpc_total = qpc_total + qpc

    b = pd.DataFrame()
    b_final = pd.DataFrame()
    for sessI in max_cms:
        for runI in range(100):
            for j in np.max(np.max(s_total2[sessI])):
                br = b[runI]
                st2_ind = s_total2[sessI]
                br[:, j] = np.mean(condition_matrix[sessI][np.where(st2_ind[:, runI] == j), 1], 1).T

        dim = np.ndim(b[1])
        m = np.concatenate(dim + 1, b[:])
        b_final[sessI] = np.nanmean(m, 3)

    sn_total = pd.DataFrame(max_cms, 1)
    for iter in max_cms:
        sn_total[iter] = sn[iter, 1]
        for runIter in range(2, 100):
            sn_total[iter] = np.hstack((sn_total[iter], s[iter, runIter]))

    nqpc_total = 0
    sn_total2 = pd.DataFrame()
    for sessI in range(max_cms):
        sn2, qn2, xn_new3, nqpc = cons_iter(sn_total[sessI].T)
        sn_total2[sessI] = sn2.T
        nqpc_total = nqpc_total + nqpc

    bn_final = pd.DataFrame()
    for sessI in range(max_cms):
        for runI in range(100):
            for j in range(max(max(sn_total2[sessI]))):
                b[runI][:, j] = np.mean(condition_matrix[sessI][np.where(st2_ind[:, runI] == j), 1], 1).T
        dim = np.ndim(b[1])
        m = np.concatenate(b[:], axis=dim+1)
        bn_final[sessI] = np.nanmean(m, 3)

    return b_final, qpc_total, bn_final, nqpc_total





