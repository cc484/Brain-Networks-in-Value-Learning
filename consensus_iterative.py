import numpy as np
from multislice_static_signed import multislice_stat_si


def cons_iter(c):
    """
    construct a consensus (representative) partition using the iterative thresholding
    procedure
    :param c:
        p x n matrix of community assignments where p is the
        number of optimizations and n the number of node
    :return:
    s2: pxn matrix of new community assignments
    q2: associated modularity value
    x_new3: thresholded nodal association matrix
    qpc: quality of the consensus (lower == better)
    """
    print(" ")
    print(c)
    print(c.shape)
    n_part = c.shape[0]  # number of partitions
    m = c.shape[1]  # size of the network

    c_rand3 = np.zeros(c.shape)  # permuted version of c
    x = np.zeros((m, m))  # nodal association matrix for c
    x_rand3 = x  # random nodal association matrix for c_rand3

    # NODAL ASSOCIATION MATRIX

    # random permutation matrix
    for i in range(n_part):
        pr = np.random.permutation(m)
        c_rand3[i, :] = c[i, pr]

    for k in range(m):
        for p in range(m):
            # element [i, j] indicate the number of times that node i and node j have been
            # assigned to the same community
            if np.equal(c[k], c[p]):
                x[k, p] = x[k, p] + 1
            else:
                x_rand3[k, p] = x_rand3[k, p] + 0

            # element [i, j] indicate the number of times node i and node j are expected
            # to be assigned to the same community by chance
            if np.equal(c_rand3[i, k], c_rand3[i, p]):
                x_rand3[k, p] = x_rand3[k, p] + 1
            else:
                x_rand3[k, p] = x_rand3[k, p] + 0

    x_new3 = np.zeros(m, m)
    x_new3[x > max(max(np.triu(x_rand3, 1)))] = x[x > max(max(np.triu(x_rand3, 1)))]

    s2 = []
    q2 = []
    for i in range(n_part):
        s2[i, :], q2[i] = multislice_stat_si(x_new3, 1)

    qpc = np.sum(np.sum(np.abs(np.diff(s2))))

    return s2, q2, x_new3, qpc

