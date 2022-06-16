from re import A
import numpy as np
import cvxpy as cp

from icecream import ic


def NV_ERM2(d, X, c_o_, c_u_, lambd_=1, p=1, verbose=1):
    """ The Nadaraya–Watson Vapnik Empirical Risk Minimization 2 algorithm (NV_ERM2) predicts
    the optimal ordering for the newsvendor problem. 

    NV_ERM2 assumes the demand distribution is unknown and the descion make only has access to 
    n historical demand points (d) and p exogenous variables for each demand point (X). NV_ERM2 
    estimates the demand function with a linear model. NV_ERM2 is a linear program with 4n constraints
    and a (p + 1 + 2n)-dimensional decision vector.

    Questions
    ---------
    1. How do we recast arrays in cvxpy so we don't need d to be (n, 1) and instead can have d be (n,)?
    2. How can we garuantee that we only produce an integer quantity? 
    3. What exactly are we modeling with a linear equation? Is it the true demand, the order quantity, 
    etc.? What is q(x) really modeling?

    References
    ----------
    1. [The Big Data Newsvendor: Practical Insights from Machine Learning](https://pubsonline.informs.org/doi/abs/10.1287/opre.2018.1757)
    NV-ERM2 algorithm.

    Parameters
    ----------
    d : array-like of shape (n, 1)
        n demand observations, generally from the past n time periods, ie. days, weeks, etc.
    X : array-like of shape (n, p)
        n exogenous variables each with p features that are associated with the n demand observations. These
        observations may depend on such as seasonality, weather, location, etc.
    c_o_ : float
        Overage cost.
    c_u_ : float
        Underage cost.
    lambd_ : int, optional
        _description_, by default 1
    p : int, optional
        If non-zero, add the p-norm squared as a regularization term to the objective function. If zero, no
        regularization term is added. Acceptable p values are {0, 1, 2} so the optimization problem will be 
        linear, linear, and quadratic respectivally. By default, 1.
    verbose : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """

    # define variables
    n, p = X.shape
    q = cp.Variable((p + 1, 1))  # +1 to lift the data
    u = cp.Variable((n, 1))
    o = cp.Variable((n, 1))

    # define parameters
    c_u = cp.Parameter(nonneg=True, value=c_u_)
    c_o = cp.Parameter(nonneg=True, value=c_o_)
    lambd = cp.Parameter(nonneg=True, value=lambd_)

    # define mean function
    def mean(x):
        return cp.sum(x) / x.size

    # define the optimization problem
    if p:
        reg = lambd_ * np.linalg.norm(q, ord=p) ** 2
        objective = cp.Minimize(mean(c_u * u + c_o * o) + reg)
    else:
        objective = cp.Minimize(mean(c_u * u + c_o * o))
    constraints = [
        u >= 0,
        o >= 0,
        u >= d - q[0] - X @ q[1:],
        o >= q[0] + X @ q[1:] - d
    ]
    prob = cp.Problem(objective, constraints)

    # solve the problem
    prob.solve()

    # print statements
    if verbose:
        print('Status: {}'.format(prob.status))
        print('Optimal Empirical Risk: {}'.format(prob.value))
        print('Optimal Order Quantities: {}'.format(q.value.T))


n, p = 3, 5

np.random.seed(1)
d = np.random.randn(n, 1)
X = np.random.randn(n, p)

c_u, c_o = 1, 1
NV_ERM2(d, X, c_o, c_u, p=2)
