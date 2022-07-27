



'''

This file is used to write algorithms that will be used to play decisions in the multi-item,
multi-feature newsvendor problem.

Each algorithm is defined by its own function. The function get_algos()
returns a dictionary of the algorithms, with the key being the name of the algorithm and the value being the algorithm it 

Each function has the same signature, even if the parameter passed in is not used. This is for ease of use in other files. 


'''

## Imports

import numpy as np
from icecream import ic
from scipy.stats import norm
from statsmodels.tsa.api import Holt

from src.models.oddp import ODDP

## NOTE: Each function below should have the signature:
## 
##         def function_name(array, underage, overage):
##              function body
##              return int value 
##
## This may change in the future when algorithms that require more parameters are added.


def mean(array, underage, overage):
    return np.mean(array)


def median(array, underage, overage):
    return np.median(array)


def naive(array, underage, overage):
    return array[-1]


def holt(array, underage, overage):
    if len(array)<10:
        return np.nan

    fit1 = Holt(array, initialization_method="estimated").fit(
    smoothing_level=0.8, smoothing_trend=0.2, optimized=False)

    fcast1 = fit1.forecast(1)
    return fcast1[0]


def normsinv(array, underage, overage):
    q =  underage / (overage + underage)

    mean = np.mean(array)
    std = np.std(array)
    return max(mean + std * norm.ppf(q),0)


def oddp(array, underage, overage):
    x = ODDP(c_u = underage, c_o = overage)
    return x.fit_and_predict(array)


def SAA(intervals, trends, seasons, noises, underage, overage):
    q = underage / (overage + underage) 
    return sorted(intervals)[int(np.ceil(q*len(intervals)))-1]

def SAA_on_trend(intervals, trends, seasons, noises, underage, overage):
    pass

def SAA_on_trend_and_seasonal(intervals, trends, seasons, noises, underage, overage):
    pass

def SAA_on_noise(intervals, trends, seasons, noises, underage, overage):
    pass

def SAA_on_noise_and_seasonal(intervals, trends, seasons, noises, underage, overage):
    pass




## NOTE: Make sure to add the algorithm to the dictionary below when you add a new algorithm.

def get_SAA():

    algos = {
        "SAA": SAA
    }

    return algos


def get_all_algos():

    algos = {
        "Naive": naive,
        "Holt": holt,
        "SAA": SAA,
        "Normsinv": normsinv,
        "Oddp": oddp
    }

    return algos




