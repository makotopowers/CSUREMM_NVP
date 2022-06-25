import numpy as np
from scipy.stats import norm
from statsmodels.tsa.api import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.ar_model import AutoReg

class methods:
        
    def mean(self,array):
        return np.mean(array)

    def median(self, array):
        return np.median(array)

    def naive(self, array):
        return array[-1]

    def holt(self, array):
        fit1 = Holt(array, initialization_method="estimated").fit(
        smoothing_level=0.8, smoothing_trend=0.2, optimized=False)

        fcast1 = fit1.forecast(1)
        return fcast1

    def autoreg(self, array):
        return AutoReg(array)

    def SAA(self, array, underage, overage):
        q = underage / (overage + underage) 
        return sorted(array)[int(np.ceil(q*len(array)))-1]

    def normsinv(self, array, underage, overage):
        q =  underage / (overage + underage)

        mean = np.mean(array)
        std = np.std(array)
        return mean + std * norm.ppf(q)




