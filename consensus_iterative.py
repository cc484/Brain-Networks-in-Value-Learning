import numpy as np
from louvain import mod_max


def cons_iter(c):
    """
    construct a consensus (representative) partition using the iterative thresholding
    procedure
    :param C:
        pxn matrix of community assignments where p is the
        number of optimizations and n the number of node
    :return:
    s2: pxn matrix of new community assignments
    q2: associated modularity value
    x_new3: thresholded nodal association matrix
    qpc: quality of the consensus (lower == better)
    """

    n_part = np.size(c[:, 1]) # number of partitions
    m = np.size(C[1, :]) # size of the network

    c_rand3 = np.zeros(np.size(c))
    x = np.zeros(m, m)
    x_rand3 = x

    for i in range(n_part):
        pr = np.random.permutation(m)
        c_rand3[i, :] = c[i, pr]

    for i in range(n_part):
        for k in range(m):
            for p in range(m):
                if np.equal(c[i][k], c[i][p]):
                    x[k][p] = x[k][p] + 1
                else:
                    x_rand3[k][p] = x_rand3[k][p] + 0

    x_new3 = np.zeros(m, m)
    x_new3[x > max(max(np.triu(x_rand3, 1)))] = x[x > max(max(np.triu(x_rand3, 1)))]

    s2 = []
    q2 = []
    for i in range(n_part):
        s2[i][:], q2[i] = mod_max(x_new3, 1)

    qpc = np.sum(np.sum(np.abs(np.diff(s2))))

    return s2, q2, x_new3, qpc

