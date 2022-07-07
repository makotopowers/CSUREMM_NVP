



'''

This code is for the ranking and comparison of different algorithms for the multi-item, multi-feature
newsvendor problem. It uses vaex to read in and process data, and then basic numpy methods to generate a comparison.

The data sets are the JD data set, and the RRS data set, which are processed in the DataReader file. 


'''

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Imports 

import matplotlib.pyplot as plt
import numpy as np
import vaex
from icecream import ic

import Benchmarks
import DataReader

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Set plt settings so that it shows up on dark or light mode. 

plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Extract all consequtive subsequences of the time series ts of length window.
## parameters: 
## ts : array-like of shape (n, ). Time series data. This won't work for an array of shape (n, 1).
## window : int The window size. The time series data is split into subarrays each of length window. 
## Returns: array-like of shape (n - window + 1, window)

def rolling_window(ts, window):
    shape = (ts.size - window + 1, window)
    strides = ts.strides * 2
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Extract subsequences of the time series ts of length 1, 2, ... , len(ts).
## parameters: 
## ts : array-like of shape (n, ). Time series data.  
## Returns: list of array-like

def normal_ts(ts):
    return [ts[:i] for i in range(1,len(ts)+1)]

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
    
## With a given historical demand sequence and underage and overage costs, predict the future demand
## for each of the benchmark methods.
## parameters: 
## sequence : array-like of shape (n, ). Historical demand sequence.
## underage : float. The cost of having less stock than demand.
## overage : float. The cost of having more stock than demand.
## Returns: dict of int

def compare(methods, sequence, underage, overage): 
    predictions = dict()

    for key in methods:
        predictions[key] = methods[key](sequence, underage, overage)
    return predictions

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Given predictions, return the cost incurred by playing those predictions. 
## parameters: 
## predictions : dict of int. The predictions made by the benchmark methods.
## actual : int. The actual demand.
## c_u : float. The cost of having less stock than demand.
## c_o : float. The cost of having more stock than demand.

def get_cost(predictions, actual, c_u, c_o):
    costs = dict()

    for key in predictions:
        costs[key] =  abs(actual-predictions)*c_o if actual < predictions else abs(actual-predictions)*c_u
    return costs

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Given a time series ts, return predictions as well as corresponding costs for each of the benchmark methods.
## May use a rolling window, or not, determined by the rolling_window parameter.
## parameters:
## ts : array-like of shape (n, ). Time series data.
## window : int. The window size. The time series data is split into subarrays each of length window.
## underage : float. The cost of having less stock than demand.
## overage : float. The cost of having more stock than demand.
## rolling_window : bool. Whether to use a rolling window or not.
## Returns: dict of int, dict of float

def all_predictions(ts, methods, window, underage, overage, rolling_window=True):

    if rolling_window:
        intervals = rolling_window(ts, window)
        window, start = window, 0
    else:
        intervals = normal_ts(ts)
        window, start = 1, 0

    predictions = {key: np.array([np.nan for i in range(start + window )]) for key in Benchmarks.methods().algos}
    costs = {key: np.array([np.nan for i in range(start + window)]) for key in Benchmarks.methods().algos}

    for i in range(start, len(intervals)):
        prediction = compare(methods, intervals[i], underage, overage)

        if window + i < len(ts):
            cost = get_cost(prediction, ts[window + i], underage, overage)

            for key in cost:
                costs[key] = np.concatenate((costs[key], cost[key]))

        for key in prediction:
            predictions[key] = np.concatenate((predictions[key], prediction[key]))

    return predictions, costs

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

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
    average_costs = {key: np.array([np.nan for i in range(wstart)]) for key in Benchmarks.methods().algos}

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

def rank():
    data = DataReader.prepare_data()
    methods = Benchmarks.get_algos()
    rankings = {key: np.array([]) for key in methods}
    quantiles = [[10,1], [2,1], [1,1], [1,2], [1,10]]

    for key in data:
        for ts in data[key]:
            for q in quantiles:

                predictions, costs = all_predictions(ts, methods, window=1, underage=q[0], overage=q[1], rolling_window=False)
                for key in costs:
                    rankings[key] = np.concatenate((rankings[key], np.nanmean(costs[key]))) 

                for i in range(10, 100):
                    if i >= len(ts)-10:
                        break
                    predictions, costs = all_predictions(ts, methods, window=i, underage=q[0], overage=q[1], rolling_window=True)
                    for key in costs:
                        rankings[key] = np.concatenate((rankings[key], np.nanmean(costs[key])))
                
    return {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1])}
        
    



#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Generate a bunch of figures. 
def figures(path, features, aggs):
    ts = vaex.open(path)
    ts = ts.groupby(by='new_dates').agg({feature: agg for feature, agg in zip(features, aggs)})

    ic('Data loaded. ')
        

    combos = [[20,1], [10,1], [2,1], [1,1], [1,2], [1,10], [1,20]]

    for combo in combos:
        predictions,costs = all_predictions(data, 0, combo[0], combo[1], all=True)
        for key in predictions:
            plt.plot(predictions[key], label=key)
        
        plt.plot(data, label="Actual")
        plt.legend()
        plt.title("Predictions for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
        plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+".png")
        plt.close()
        for key in costs:
            plt.plot(costs[key], label=key+"_cost")
            
        plt.legend()
        plt.title("Costs for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
        plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+"_cost.png")
        plt.close()
        ic("Checkpoint 1")

        costs = cost_vs_window_size(data, combo[0], combo[1])
        for key in costs:
            plt.plot(costs[key], label=key)
        plt.legend()
        plt.title("Average Costs vs Window Size for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
        plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+"_cost_vs_window.png")
        plt.close()

        ic("Checkpoint 2")

        for i in range(1,min(len(data),50)):
            ic("check 2", i)
            predictions = all_predictions(data, i, combo[0], combo[1], all=False)[0]
            for key in predictions:
                plt.plot(predictions[key], label=key)
            plt.plot(data, label="Actual")
            plt.legend()
            plt.title("Predictions with Window Size (" + (str(i)) + ") for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
            plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+"_W"+str(i)+".png")
            plt.close()

        ic(f'Combo: {combo}' + ' done.')

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

if __name__=='__main__':
    path = "/Users/makotopowers/Desktop/CSUREMM/reports/figures/RR_FIGS/different_quantiles"
    #figures(JD=False)
    data = np.load("/Users/makotopowers/Desktop/CSUREMM/data/raw/h24_all_data.npy").transpose(1,0)[0]
    m = DataReader.Data("/Users/makotopowers/Desktop/CSUREMM/data/raw/JD_order_data.csv", RR=True)
    
   
    ic(data)

    preds, costs = all_predictions(data, 30, 1, 1, all=False)

    ic(preds, costs)

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
