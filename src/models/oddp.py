import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binom
from scipy.optimize import minimize_scalar

import warnings
from icecream import ic


class ODDP:
    """
    Implement the Optimal Data-Driven Policy (ODDP) from the paper 
    [How Big Should Your Data Really Be? Data-Driven Newsvendor: Learning One Sample at a Time](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3878155)
    """

    def __init__(self, beta=None, c_u=None, c_o=None):

        # record orginal parameters
        self.beta = c_u / (c_u + c_o) if beta is None else beta

        self.n = None
        self.d_unsorted = None
        self.d_sorted = None

        self.k = None
        self.gamma = None
        self.degenerate = None
        self.mu = None
        self.upper_bound = None

    def _bernstein(self, y, i, n):
        """Computer a Bernstein polynomial defined as
            B_{i, n}(y) = \sum_{j=i}^n binom(n, j) y^j (1-y)^{n-j}
        where
            binom(n, j) = n! / (j! (n-j)!)

        Parameters
        ----------
        y : float
            Probability of a single success (between 0 and 1, inclusive).
        i : int
            The trial we begin with.
        n : int
            The number of trials.

        Returns
        -------
        float
            The value of the Bernstein polynomial.
        """
        j_values = np.arange(i, n + 1)
        return np.sum(binom.pmf(j_values, n, y))

    def regret(self, mu, lambd):
        """ Compute the expected relative regret (from now on refered to as regret) of a mixture of order statistics policy
        on the newsvendor problem. The regret is defined as
            R(\pi^\lambda, \mu))
            &=
            \sum_{i=1}^n \lambda_i
                \frac{(1 - B_{i, n}(1 - \mu)) * (1 - \mu - \beta) + \beta * \mu}
                     {\min((1 - \beta) * (1 - \mu), \beta * \mu)}
            - 1
        Note that this is not a bound on the relative regret but rather the exact value of the relative regret.

        References
        ----------
        1. Theorem 1 in [How Big Should Your Data Really Be? Data-Driven Newsvendor: Learning One Sample at a Time](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3878155)
           Note: In theorem 1, they use q and we use \beta instead. The beta in the function argument refers to the
           bernoulli distribution characterized by \mu. So we simply dennote this by \mu.

        Parameters
        ----------
        mu : float
            The probability of a single success (between 0 and 1, inclusive).
        lambd : array_like of shape (n,)
            The vector that characterizes the mixture of order statistics policy. lambd is shorthand for lambda.
            The ith element of lambd represents the weight that the ith order statistic is given. Note: all entries
            of lambd must be non-negative and sum to 1.

        Returns
        -------
        float
            The exact relative regret of the mixture of order statistics policy.
        """

        # error check
        if np.sum(lambd) != 1:
            warnings.warn(
                "lambd does not sum to 1. This may cause problems with the relative regret computation."
            )

        # initial values
        n = len(lambd)
        bernstein_vector = np.array(
            [self._bernstein(1 - mu, i, n) for i in range(1, n+1)]
        )

        numerator = (1 - bernstein_vector) * \
            (1 - mu - self.beta) + self.beta * mu
        denominator = min((1 - self.beta) * (1 - mu), self.beta * mu)

        return np.sum(lambd * numerator / denominator) - 1

    def _manual_max_regret(self, policy, lower, upper, n_samples=100):
        """Manually compute the maximum of the expected relative regret over mu in [lower, upper].
        This method will only be called if the scipy.optimize.minimize_scalar method fails.

        Notes
        -----
        1. The equation for regret is undefined as mu approaches 0 or 1. Thus I include a buffer of 0.01 to avoid
        the undefined behavior.

        Parameters
        ----------
        policy : array_like of shape (n,)
            The vector that characterizes the mixture of order statistics policy.
        lower : float
            The lower bound on mu, must be no lower than 0.
        upper : float
            The upper bound on mu, must be no greater than 1.
        n_samples : int, optional
            The number of samples for mu, by default 1000

        Returns
        -------
        self.max_regret_val : float
            The supremum of the expected relative regret over mu.

        """

        # compute mu values to sample from
        mus = np.linspace(lower, upper, n_samples)

        # compute the mu that maximizes the regret
        regrets = np.array([self.regret(mu, policy) for mu in mus])
        self.max_regret_val = np.max(regrets)
        self.mu = mus[np.argmax(regrets)]

        return self.max_regret_val

    def max_regret(self, policy, lower, upper):
        """Compute the maximum regret over all mu values in [lower, upper] for a given policy.

        Parameters
        ----------
        policy : array_like of shape (n,)
            The vector that characterizes the mixture of order statistics policy.
        lower : float
            The lower bound on mu, must be no lower than 0.
        upper : float
            The upper bound on mu, must be no greater than 1.

        Returns
        -------
        self.max_regret_val : float
            The supremum of the expected relative regret over mu.
        """

        # add buffer to avoid undefined behavior
        buffer = 0.001
        lower = lower + buffer if lower == 0 else lower
        upper = upper - buffer if upper == 1 else upper

        # define objective function; -1 because we want to maximize the regret
        def obj(mu, policy):
            return -1 * self.regret(mu, policy)

        # compute the maximum regret
        sol = minimize_scalar(
            obj,
            bounds=(lower, upper),
            args=(policy,),
            method='bounded'
        )

        max_regret_val = -1 * sol.fun
        if sol.success:
            max_regret_val = -1 * sol.fun
        else:
            max_regret_val = self._manual_max_regret(policy, lower, upper)
        return max_regret_val

    def _get_os_policy(self, k, n):
        """Returns the kth order statistic policy for n observations.

        Parameters
        ----------
        k : int
            The order statistic.
        n : int
            The number of observations.

        Returns
        -------
        array-like of shape (n,)
            The kth order statistic policy.
        """
        os_policy = np.zeros(n)
        os_policy[k] = 1
        return os_policy

    def _min_policy_optimal(self, d):
        """Check if the minimum policy is optimal. The minimum policy always returns the smallest demand
        from the demand history.

        References
        ----------
        1. Proposition 2 in [How Big Should Your Data Really Be? Data-Driven Newsvendor: Learning One Sample at a Time](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3878155)

        Parameters
        ----------
        d : array_like of shape (n,)
            The demand history.

        Returns
        -------
        bool
            True if the minimum policy is optimal, False otherwise.
        """

        min_policy = self._get_os_policy(0, self.n)
        return self.max_regret(min_policy, 0, 1 - self.beta) > self.max_regret(min_policy, 1 - self.beta, 1)

    def _max_policy_optimal(self, d):
        """Check if the maximum policy is optimal. The minimum policy always returns the greatest demand
        from the demand history.

        References
        ----------
        1. Proposition 2 in [How Big Should Your Data Really Be? Data-Driven Newsvendor: Learning One Sample at a Time](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3878155)

        Parameters
        ----------
        d : array_like of shape (n,)
            The demand history.

        Returns
        -------
        bool
            True if the maximum policy is optimal, False otherwise.
        """

        max_policy = self._get_os_policy(-1, self.n)
        return self.max_regret(max_policy, 0, 1 - self.beta) < self.max_regret(max_policy, 1 - self.beta, 1)

    def _get_k(self, d, verbose):
        """Compute the kth order statistic for a given demand history using binary search.

        Parameters
        ----------
        d : array_like of shape (n,)
            The demand history.
        verbose : bool
            If True, print the progress of the binary search.

        Returns
        -------
        k : int
            The kth order statistic.
        """

        # initial values
        lo, hi = 0, len(d) - 1

        while lo < hi:

            # compute the midpoint
            mid = lo + (hi - lo) // 2

            if verbose:
                print('lo = {}, hi = {}, mid = {}'.format(lo, hi, mid))

            # compute the mid-th order statistics policy
            policy = self._get_os_policy(mid, self.n)

            # update the pointers i, j
            if self.max_regret(policy, 1 - self.beta, 1) >= self.max_regret(policy, 0, 1 - self.beta):
                lo = mid + 1
            else:
                hi = mid
        k = hi
        return k

    def _dual_regret(self, mu, gamma, policy_1, policy_2):
        """Compute the dual regret defined as:
            gamma * regret(mu, policy_1) - (1 - gamma) * regret(mu, policy_2)

        Parameters
        ----------
        mu : float
            The value of mu.
        gamma : float
            The weight given between the first policy and the second policy.
        policy_1 : array_like of shape (n,)
            The first policy.
        policy_2 : array_like of shape (n,)
            The second policy.

        Returns
        -------
        float
            The dual regret.
        """
        return gamma * self.regret(mu, policy_1) + (1 - gamma) * self.regret(mu, policy_2)

    def _get_gamma(self, k):
        """Compute gamma, the weight given between the first policy and the second policy.

        Parameters
        ----------
        k : int
            The kth order statistic.

        Returns
        -------
        float
            The weight given between the first policy and the second policy.
        """

        # define the kth and k-1th order statistic policies
        os_policy_k = self._get_os_policy(k, self.n)
        os_policy_k_1 = self._get_os_policy(k - 1, self.n)

        # define the objective function; multiply by -1 because we want to maximize, not minimize
        def obj(mu, gamma, policy_1, policy_2):
            return -1 * self._dual_regret(mu, gamma, policy_1, policy_2)

        # define bounds on mu
        buffer = 0.001
        lower_1, upper_1 = 0 + buffer, 1 - self.beta
        lower_2, upper_2 = 1 - self.beta, 1 - buffer

        # objective function for computing the gamma that solves the root
        def obj_root(gamma, policy_1, policy_2):

            # compute the maximum regret of two policies over mu on interval [0, 1 - beta]
            sol_1 = minimize_scalar(
                obj,
                bounds=(lower_1, upper_1),
                args=(gamma, policy_1, policy_2),
                method='bounded'
            )

            # compute the maximum regret of two policies over mu on interval [1 - beta, 1]
            sol_2 = minimize_scalar(
                obj,
                bounds=(lower_2, upper_2),
                args=(gamma, policy_1, policy_2),
                method='bounded'
            )

            max_regret_1 = -1 * sol_1.fun
            max_regret_2 = -1 * sol_2.fun
            abs_diff = np.sqrt(np.square(max_regret_1 - max_regret_2))
            return abs_diff

        gamma_0 = 0.5
        sol = minimize_scalar(obj_root,
                              gamma_0,
                              args=(os_policy_k, os_policy_k_1),
                              bounds=(0, 1),
                              method='bounded'
                              )
        gamma = sol.x
        return gamma

    def closest_idx(self, array, value):
        """Returns the index where the values of array and value are closest

        Parameters
        ----------
        array : array-like of shape (n,)
            _description_
        value : scalar or array-like of shape (n,)
            _description_

        Returns
        -------
        int
            The index of the closest value in array to value.
        """
        return np.nanargmin(np.abs(array - value))

    # def _get_gamma_2(self, k, n_mu_samples=100, n_gamma_samples=100):
    #     """Compute gamma, the weight given between the first policy and the second policy.
    #     This is a slower, less accurate brute force method to compute gamma recommended by Omar.
    #     No need to use this.

    #     Parameters
    #     ----------
    #     k : int
    #         The kth order statistic.
    #     n_mu_samples : int, optional
    #         The number of samples to use for computing mu, by default 100
    #     n_gamma_samples : int, optional
    #         The number of samples to use for computing gamma, by default 100

    #     Returns
    #     -------
    #     float
    #         The weight given between the first policy and the second policy.
    #     """

    #     # define the kth and k-1th order statistic policies
    #     os_policy_k = self._get_os_policy(k, self.n)
    #     os_policy_k_1 = self._get_os_policy(k - 1, self.n)

    #     # define samples
    #     buffer = 0.01
    #     mus = np.linspace(0 + buffer, 1 - buffer, n_mu_samples)
    #     gammas = np.linspace(0, 1, n_gamma_samples)

    #     # find the index of the mu value that is closest to 1 - beta
    #     split_idx = self.closest_idx(mus, 1 - self.beta)

    #     # compute the dual regret for every combination of mu and gamma
    #     M = np.zeros((n_mu_samples, n_gamma_samples))
    #     for ix, iy in np.ndindex((len(mus), len(gammas))):
    #         M[ix, iy] = self._dual_regret(
    #             mus[ix], gammas[iy], os_policy_k, os_policy_k_1)

    #     # split M into two matricies, one with mu <= 1 - beta and one with mu >= 1 - beta
    #     # compute the mu that maximizes the dual regret in both matrices across all gamma values
    #     # find the gamma value that makes the maximum the same in both matrices
    #     A, B = M[:split_idx+1, :], M[split_idx:, :]
    #     A_max, B_max = np.max(A, axis=0), np.max(B, axis=0)
    #     gamma_idx = self.closest_idx(A_max, B_max)
    #     gamma = gammas[gamma_idx]

    #     return gamma

    def upper_bound_regret(self, k, gamma):
        """Computes the upper bound on the expected relative regret

        Parameters
        ----------
        k : int
            The kth order statistic.
        gamma : float
            The weight given between the first policy and the second policy.

        Returns
        -------
        float
            The upper bound on the expected relative regret.
        """

        # add buffer to avoid undefined behavior
        buffer = 0.001
        lower = 1 - self.beta
        upper = 1 - buffer

        # define the kth and k-1th order statistic policies
        os_policy_k = self._get_os_policy(k, self.n)
        os_policy_k_1 = self._get_os_policy(k - 1, self.n)

        # define objective function; -1 because we want to maximize, not minimize the regret
        def obj(mu):
            return -1 * (gamma * self.regret(mu, os_policy_k) + (1 - gamma) * self.regret(mu, os_policy_k_1))

        # compute the maximum regret
        sol = minimize_scalar(
            obj,
            bounds=(lower, upper),
            method='bounded'
        )

        mu = sol.x
        upper_bound_regret = -1 * sol.fun
        return mu, upper_bound_regret

    def fit(self, d, verbose=0):
        """Compute the kth order statistic (k) and the weight given between the first policy and the second policy (gamma).

        Parameters
        ----------
        d : array-like of shape (n,)
            The demand history over the past n units of time.
        verbose : int, optional
            If non-zero, print values, by default 0.

        Returns
        -------
        float
            Gamma, the weight given between the first policy and the second policy.
        """

        # sort the demand for order statistics usage
        self.n = len(d)
        self.d_unsorted = d
        self.d_sorted = np.sort(d)
        d = self.d_unsorted

        # compute the kth order statistic and gamma
        self.degenerate = True
        if self._min_policy_optimal(d):  # degenerate case
            self.k = 1
            self.gamma = 1
        elif self._max_policy_optimal(d):  # degenerate case
            self.k = len(d)
            self.gamma = 1
        else:  # non-degenerate case
            self.k = self._get_k(d, verbose)
            self.gamma = ic(self._get_gamma(self.k))
            self.degenerate = False
            # self.gamma = self._get_gamma_2(self.k)

        # compute the upper bound on the expected relative regret
        self.mu, self.upper_bound = self.upper_bound_regret(self.k, self.gamma)

        if verbose:
            print('k = {}, gamma = {}'.format(self.k, self.gamma))
            print('The upper bound on the expected relative regret is: {}'.format(
                self.upper_bound))

        return self.k

    def predict(self):
        """Predict the demand for the next unit of time.

        Returns
        -------
        float
            The predicted demand.
        """

        # define the kth and k-1th order statistic policies
        os_policy_k = self._get_os_policy(self.k, self.n)
        os_policy_k_1 = self._get_os_policy(self.k - 1, self.n)

        return self.gamma * np.sum(os_policy_k * self.d_sorted) + (1 - self.gamma) * np.sum(os_policy_k_1 * self.d_sorted)

    def fit_and_predict(self, d, verbose=0):
        """Compute the kth order statistic (k) and the weight given between the first policy and the 
        second policy (gamma). Then predict the demand for the next unit of time.

        Parameters
        ----------
        d : array-like of shape (n,)
            The demand history over the past n units of time.
        verbose : int, optional
            If non-zero, print values, by default 0.

        Returns
        -------
        float
            The predicted demand.
        """
        self.fit(d, verbose=verbose)
        return self.predict()


def recreate_figure_3(beta=0.9, n=20, outdir='reports/figures/'):
    """Plot regret versues mu for different policies to recreate Figure 3 from 
    [How Big Should Your Data Really Be? Data-Driven Newsvendor: Learning One Sample at a Time](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3878155)

    Notes
    -----
    1. In the paper, the kth order statistic means we want the kth smallest item. But 
    because lists are 0-indexed, the kth order statistic should return the item at index k-1.
    So we have this off by one factor.


    References
    ----------
    1. The numbers beta=0.9, n=20 are taken from Figure 3 in the paper's appendix.

    Parameters
    ----------
    beta : float, optional
        _description_, by default 0.9
    n : int, optional
        _description_, by default 20
    outdir : str, optional
        _description_, by default 'reports/figures/'
    """

    # initialize the policies
    oddp = ODDP(beta=beta)
    k = int(np.ceil(n * beta))
    SAA_policy = oddp._get_os_policy(k - 1, n)
    k_policy = oddp._get_os_policy(k, n)

    # compute the regret for all mu values
    buffer = 0.001
    mus = np.linspace(0 + buffer, 1 - buffer, 100)
    SAA_regret = [oddp.regret(mu, SAA_policy) for mu in mus]
    k_regret = [oddp.regret(mu, k_policy) for mu in mus]

    # plot the regret
    fig, ax = plt.subplots()
    ax.scatter(mus, SAA_regret, s=2)
    ax.plot(mus, SAA_regret, linewidth=1,
            label=r'$\pi^{OS_{\lceil \beta n \rceil}}$' + ' (SAA)')
    ax.scatter(mus, k_regret, s=2)
    ax.plot(mus, k_regret, linewidth=1,
            label=r'$\pi^{OS_{\lceil \beta n \rceil + 1}}$')
    ax.set(xlabel='Mu',
           ylabel='Regret',
           title='Regret for beta = {}'.format(beta)
           )
    ax.legend(prop={'size': 15})

    # save the figure
    fig.savefig('{}regret_beta_{}.png'.format(outdir, beta))



