from re import A
import numpy as np
import cvxpy as cp

from icecream import ic
from loss import cost


class NV_ERM2:

    def __init__(self, lambd=1, norm=1):
        """Initialize the Nadaraya–Watson Vapnik Empirical Risk Minimization 2 algorithm (NV_ERM2). 
        NV_ERM2 predicts the ordering quantity for the newsvendor problem by solving the problem:

        NV_ERM2 assumes the demand distribution is unknown and stationary. It also assumes the demand
        is linear in the covariates. NV_ERM2 is a linear program with 4n constraints and a 
        (p + 1 + 2n)-dimensional decision vector.

        References
        ----------
        1. [The Big Data Newsvendor: Practical Insights from Machine Learning](https://pubsonline.informs.org/doi/abs/10.1287/opre.2018.1757)

        Parameters
        ----------
        lambd : float, optional
            The regularization parameter, by default 1. Note: 'lambd' is shorthand for lambda.
        norm : int, optional
            The norm of the regularization. p=2 is the L2 norm, p=1 is the L1 norm, 
            and p=0 disables the regularization. By default, p=1.
        """

        self.lambd = lambd
        self.norm = norm

        self.d = None
        self.X = None
        self.c_o = None
        self.c_u = None

        self.w = None
        self.o = None
        self.u = None

    def fit(self, d, X, c_o, c_u, verbose=0):

        # record method parameters
        self.d = d
        self.X = X
        self.c_o = c_o
        self.c_u = c_u

        # define variables for cvxpy
        n, p = X.shape
        w = cp.Variable((p + 1, 1))  # +1 to lift the data
        u = cp.Variable((n, 1))
        o = cp.Variable((n, 1))

        # define parameters for cvxpy
        c_u = cp.Parameter(nonneg=True, value=c_u)
        c_o = cp.Parameter(nonneg=True, value=c_o)
        lambd = cp.Parameter(nonneg=True, value=self.lambd)

        # define mean function for cvxpy
        def mean(x):
            return cp.sum(x) / x.size

        # define the optimization problem
        if self.norm:
            regularization = lambd * cp.norm(w, self.norm) ** 2
            objective = cp.Minimize(mean(c_u * u + c_o * o) + regularization)
        else:
            objective = cp.Minimize(mean(c_u * u + c_o * o))
        constraints = [
            u >= 0,
            o >= 0,
            u >= d - w[0] - X @ w[1:],
            o >= w[0] + X @ w[1:] - d
        ]
        prob = cp.Problem(objective, constraints)

        # solve the problem
        prob.solve()

        # record optimal values
        self.w = w.value
        self.u = u.value
        self.u = o.value

        # print statements
        if verbose:
            print('Status: {}'.format(prob.status))
            print('Optimal Empirical Risk: {}'.format(prob.value))
            print('Optimal Order quantities: {}'.format(w.value.T))

        return prob

    def predict(self, X):
        return self.w[0] + X @ self.w[1:]
