import numpy as np


def put_back(values_S, support):
    """
    TODO: document

    Parameters
    ----------
    values_S: array-like, shape (S, )

    support: array-like of bools, shape (d, )

    Output
    ------
    values: array-like, shape (d, )
    """
    assert len(values_S) <= len(support)

    d = len(support)
    values = np.zeros(d)

    idx_S = 0
    for idx in range(d):
        if support[idx]:
            values[idx] = values_S[idx_S]
            idx_S += 1

    return values
