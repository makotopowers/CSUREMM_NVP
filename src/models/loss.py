import numpy as np
from icecream import ic


def cost(q, d, c_o, c_u):
    """Compute the cost of producing order quantities q when the true demand was d for
    each of n days. The cost on the ith day is:
        cost = c_o * (q[i] - d[i] if =+ c_u * d[i]

    Parameters
    ----------
    d : array-like of shape (n, 1)
        Demand for each day.
    q : array-like of shape (n, 1)
        Ordering quantity for each day.
    c_o : float
        Overage cost.
    c_u : float
        Underage cost.

    Returns
    -------
    array-like of shape (n,)
        The cost incurred on each day.
    """

    return c_u * np.squeeze((d-q)*[d > q]) + c_o * np.squeeze((q-d) * [q > d])
