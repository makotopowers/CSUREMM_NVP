



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


def normal_ts(ts):
    return [ts[:i] for i in range(1,len(ts)+1)]


def predict(methods, intervals, trends, seasons, noises, underage, overage): 
    predictions = dict()
    for key in methods:
        predictions[key] = methods[key](intervals, trends, seasons, noises, underage, overage)
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


# def get_cost(predictions, actual, c_u, c_o):
#     costs = dict()

#     for key in predictions:
#         costs[key] =  abs(actual-predictions[key])*c_o if actual < predictions[key] else abs(actual-predictions[key])*c_u
#     return costs


def newsvendor_plays(ts, trend, seasons, noise, methods, underage, overage, compare=None):

    intervals = normal_ts(ts)
    trends = normal_ts(trend)
    seasons = normal_ts(seasons)
    noises = normal_ts(noise)

    window, start = 1, 15
    
    predictions = {key: np.array([np.nan for i in range(start + window )]) for key in methods}

    
    for i in range(start, len(intervals)):
        prediction = predict(methods, intervals[i], trends[i], seasons[i], noises[i], underage, overage)

        predictions = {key: np.append(predictions[key], prediction[key]) for key in predictions}

    return predictions ## all the predictions from a data stream. 

def newsvendor_costs(ts, predictions, underage, overage):
    costs = dict()

    for key in predictions:
        costs[key] = np.where(ts < predictions, abs(ts-predictions)*overage, abs(ts-predictions)*underage)

    return costs
            # abs(ts-predictions)*overage if ts < predictions else abs(ts-predictions)*underage
    




    # costs = {key: np.array([np.nan for i in range(start + window)]) for key in methods}
    #     if seasons is not None:
    #         if i + 1 < len(seasons)-1:
    #             prediction = {key : prediction[key] + seasons[i-1] for key in prediction}

    #     if window + i < len(ts):
    #         if actual is None:
    #             cost = get_cost(prediction, ts[window + i], underage, overage)
    #         else:
    #             cost = get_cost(prediction, actual[window + i], underage, overage)

    #         for key in cost:
    #             costs[key] = np.append(costs[key], cost[key])

    #     for key in prediction:
    #         predictions[key] = np.append(predictions[key], prediction[key])

    # return predictions, costs

