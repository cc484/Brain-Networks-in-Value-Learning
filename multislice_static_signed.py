import numpy as np
import networkx as nx
from louvain import mod_max


def multislice_stat_si(a, g_plus):
    a_plus = a
    a_plus[a < 0] = 0
    k_plus = np.sum(a_plus)
    p = np.divide(k_plus * k_plus.T, np.sum(k_plus))
    b = np.subtract(a, np.multiply(p, g_plus))
    # print(b)
    b = np.asarray(b)
    print(b)
    b_graph = nx.from_numpy_array(b)
    s, q = mod_max(b_graph, resolution=g_plus, randomize=True)

    return s, q

