import numpy as np
import pandas as pd
from multislice_static_signed import multislice_stat_si
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

    s = np.asarray([[dict() for i in range(100)] for j in range(max_cms)])
    sn = np.asarray([[dict() for i in range(100)] for j in range(max_cms)])

    q = np.zeros((max_cms, 100))
    qn = np.zeros((max_cms, 100))

    for i in range(max_cms):
        # Compute time x time correlation matrix
        tt_arr = np.corrcoef(condition_matrix[:, i].T)
        tt_matrix = np.asmatrix(tt_arr)
        print(tt_matrix)
        print(tt_arr)

        # Compute null model for tt_matrix
        tt_matrix_u = np.triu(tt_matrix, 1)
        x_vector = tt_matrix_u[np.where(np.absolute(tt_matrix_u) > 0)]

        n_matrix = np.zeros(tt_matrix.shape)
        p = np.random.permutation(x_vector.size)
        ind = np.where(np.absolute(tt_matrix_u) > 0)
        for x in ind:
            for y in p:
                ip = ind[y]
                n_matrix[x] = tt_matrix_u[ip]

        n_matrix = n_matrix + n_matrix.T
        for j in range(max(tt_matrix.shape)):
            n_matrix[i, j] = 1

        # Apply community detection using mod_mox
        for runI in range(100):
            s[i, runI], q[i, runI] = multislice_stat_si(tt_matrix, g_val)
            sn[i, runI], qn[i, runI] = multislice_stat_si(n_matrix, g_val)

    s_total = np.asarray([dict() for i in range(max_cms)])
    print(s_total)
    for itr in range(max_cms):
        s_total[itr] = s[itr, 0]
        for runItr in range(1, 100):
            s_total[itr] = np.hstack((s_total[itr], s[itr, runItr]))

    qpc_total = 0
    s_total2 = pd.DataFrame()
    for sessI in range(max_cms):
        s2, q2, x_new3, qpc = cons_iter(s_total[sessI].T)
        s_total2[sessI] = s2.T
        qpc_total = qpc_total + qpc

    b = pd.DataFrame()
    b_final = pd.DataFrame()
    for sessI in range(max_cms):
        for runI in range(100):
            for j in np.max(np.max(s_total2[sessI])):
                br = b[runI]
                st2_ind = s_total2[sessI]
                br[:, j] = np.mean(condition_matrix[sessI][np.where(st2_ind[:, runI] == j), 1], 1).T

        dim = np.ndim(b[1])
        m = np.concatenate(dim + 1, b[:])
        b_final[sessI] = np.nanmean(m, 3)

    sn_total = np.asarray([dict() for i in range(max_cms)])
    for itr in range(max_cms):
        sn_total[itr] = sn[itr, 0]
        for runIter in range(1, 100):
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
                st2_ind = sn_total2[sessI]
                b[runI][:, j] = np.mean(condition_matrix[sessI][np.where(st2_ind[:, runI] == j), 1], 1).T
        dim = np.ndim(b[1])
        m = np.concatenate(b[:], axis=dim+1)
        bn_final[sessI] = np.nanmean(m, 3)

    ret_tup = (b_final, qpc_total, bn_final, nqpc_total)
    return ret_tup





