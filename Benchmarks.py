import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm 
from icecream import ic

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.tsa.arima.model as ARIMA

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

    def arima(self, array, interval = 5):
        mod = ARIMA.ARIMA(array, order=(0,0,2)).fit()
        pred = mod.predict(start=len(array), end=len(array))
        return pred

    def autoreg(self, array):
        return AutoReg(array)

    def SAA(self, array, overage=0.5, underage=0.5):
        q = underage / (overage + underage) 
        return sorted(array)[int(np.ceil(q*len(array)))]

    def normsinv(self, array, overage=0.5, underage=0.5):
        q =  underage / (overage + underage)

        mean = np.mean(array)
        std = np.std(array)
        return mean + std * norm.ppf(q)




