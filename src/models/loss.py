import numpy as np


def inventory_loss(y_true, y_pred, overage, underage):
    """Loss function for computing the cost of having overage and underages in inventory.

    Parameters
    ----------
    y_true : _type_
        _description_
    y_pred : _type_
        _description_
    overage : _type_
        _description_
    underage : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return np.sum(y_true > y_pred) * underage + np.sum(y_true < y_pred) * overage
