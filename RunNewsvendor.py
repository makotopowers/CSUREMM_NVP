



'''

This code is for the ranking and comparison of different algorithms for the multi-item, multi-feature
newsvendor problem. It uses vaex to read in and process data, and then basic numpy methods to generate a comparison.

The data sets are the JD data set, and the RRS data set, which are processed in the DataReader file. 


'''

## Imports 

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import ruptures as rpt
from statsmodels.tsa.seasonal import seasonal_decompose


import Methods
import Data
import warnings
warnings.filterwarnings("ignore")


def rolling_window(ts, window):
    shape = (ts.size - window + 1, window)
    strides = ts.strides * 2
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

def normal_ts(ts):
    return [ts[:i] for i in range(1,len(ts)+1)]


def predict_normal(methods, sequence, underage, overage): 
    predictions = dict()
    for key in methods:
        predictions[key] = methods[key](sequence, underage, overage)
    return predictions

def predict_zero(methods, sequence, underage, overage): 
    predictions = dict()

    
    predictions = {key : 0 for key in methods}

    # |D_t < D_{t-1} + epsilon| => No spikes considered. 

    epsilon = 100
    if sequence[-1] > 3 and sequence[-2] > 3 and sequence[-1] < sequence[-2] + epsilon: 
        sequence = sequence[sequence > 3]
        for key in methods:
            predictions[key] = methods[key](sequence, underage, overage)
    return predictions


def predict_cpd(methods, sequence, underage, overage): 
    predictions = dict()

    model="rbf"
    a = np.nan_to_num(sequence)
    
    algo = rpt.Pelt(model=model).fit(a)
    result = algo.predict(pen=4)
    for key in methods:
        if len(result) > 1:
            predictions[key] = methods[key](sequence[result[-2]:], underage, overage)
        else:
            predictions[key] = methods[key](sequence, underage, overage)
    return predictions


def get_cost(predictions, actual, c_u, c_o):
    costs = dict()

    for key in predictions:
        costs[key] =  abs(actual-predictions[key])*c_o if actual < predictions[key] else abs(actual-predictions[key])*c_u
    return costs


def all_predictions(ts, methods, underage, overage, window=1,seasons=None, actual=None, roll_window=False, compare=None):
    
    if roll_window:
        intervals = rolling_window(ts, window)
        window = window
        start = 5 if len(ts) < 50 else 40
    else:
        intervals = normal_ts(ts)
        window, start = 1, 15
    
    predictions = {key: np.array([np.nan for i in range(start + window )]) for key in methods}
    costs = {key: np.array([np.nan for i in range(start + window)]) for key in methods}

   
    for i in range(start, len(intervals)):
       
        if compare == None:
            prediction = predict_normal(methods, intervals[i], underage, overage)
        elif compare == 'sparse': 
            prediction = predict_zero(methods, intervals[i], underage, overage)
        elif compare == 'cpd':
            prediction = predict_cpd(methods, intervals[i], underage, overage)
        elif compare == 'zero':
            prediction = {key : 0 for key in methods}
        if seasons is not None:
            if i + 1 < len(seasons)-1:
                prediction = {key : prediction[key] + seasons[i-1] for key in prediction}

        if window + i < len(ts):
            if actual is None:
                cost = get_cost(prediction, ts[window + i], underage, overage)
            else:
                cost = get_cost(prediction, actual[window + i], underage, overage)

            for key in cost:
                costs[key] = np.append(costs[key], cost[key])

        for key in prediction:
            predictions[key] = np.append(predictions[key], prediction[key])

    return predictions, costs



## Given a time series ts, return the average cost as a function of window size. 
## For a window of size i, the average cost is averaged across all days when a benchmark method 
## uses a sliding window. The window sizes range from wstart to wend. 
## parameters:
## ts : array-like of shape (n, ). Time series data.
## underage : float. The cost of having less stock than demand.
## overage : float. The cost of having more stock than demand.
## wstart : int. The starting window size.
## wend : int. The ending window size.
## Returns: dict of float

def cost_vs_window_size(ts, underage, overage, wstart=10, wend=50):

    average_costs = {key: np.array([np.nan for i in range(wstart)]) for key in Methods.methods().algos}

    for i in range(wstart,wend): # 50 if RRS, len(ts) if JD
        costs = all_predictions(ts, i, underage, overage, all=False)[1]

        for key in costs:
            average_costs[key] = np.concatenate((average_costs[key], np.nanmean(costs[key])))

    return average_costs

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Generate rankings for each method.
## parameters: None. The function will fetch methods from Benchmarks.methods().
## Returns: dict of int. The ranking of each method.

## NOTE: WARNING: This function is slow. 

def rank():

    data = Data.prepare_data()
    methods = Methods.get_algos()
    rankings = {key: np.array([]) for key in methods}
    quantiles = [[10,1], [2,1], [1,1], [1,2], [1,10]]

    for datakey in data:
        ts = data[datakey][1] # For shortening [:min(len(data[datakey][1]), 60)]
        for q in quantiles:
        
            predictions, costs = all_predictions(ts, methods, window=1, underage=q[0], overage=q[1], roll_window=False)
            
            for key in costs:
                rankings[key] = np.append(rankings[key], np.nanmean(costs[key])) 

                ic(datakey, q)

        

        # for i in range(10, 20):
            
        #     if i >= len(ts)-10:
        #         break
        #     predictions, costs = all_predictions(ts, methods, window=i, underage=q[0], overage=q[1], roll_window=True)
        #     for key in costs:
        #         rankings[key] = np.append(rankings[key], np.nanmean(costs[key]))

    ordering = [(key, np.nanmean(rankings[key])) for key in rankings]
    std = {key: np.nanstd(rankings[key]) for key in rankings}
    ordering = sorted(ordering, key=lambda x: x[1])
    ranks = [(key[0], i + 1) for i, key in enumerate(ordering)]
    
    return ranks, ordering, std
        

if __name__=='__main__':
    data = Data.prepare_data()
