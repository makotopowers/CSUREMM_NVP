import matplotlib.pyplot as plt
import numpy as np
from icecream import ic

import Benchmarks
import DataReader


def rolling_window(ts, window):
    """ Extract all consequtive subsequences of the time series ts of length window.
    
    Notes
    --------
    1.  If you update any values in the returned array is changes the values in the 
    original array, and vice-versa.
    
    References
    ----------
    https://stackoverflow.com/questions/27852343/split-python-sequence-time-series-array-into-subsequences-with-overlap

    Parameters
    ----------
    ts : array-like of shape (n, )
        Time series data. This won't work for an array of shape (n, 1).
    window : int
        The window size. The time series data is split into subarrays each of length window. 

    Returns
    -------
    array-like of shape (n - window + 1, window)
        _description_
    """
    
    shape = (ts.size - window + 1, window)
    strides = ts.strides * 2
    return np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

def normal_ts(ts):
    
    return [ts[:i] for i in range(1,len(ts)+1)]
    

def compare(array, underage, overage):
    methods = Benchmarks.methods()
    
    mean = methods.mean(array)
    median = methods.median(array)
    holt = methods.holt(array)[0] if len(array) >= 10 else np.nan
    naive = methods.naive(array)
    saa = methods.SAA(array, underage, overage)
    normsinv = methods.normsinv(array, underage, overage)

    
    
    return np.array([mean, median, holt, naive, saa, normsinv]) 

def get_cost(predictions, actual, c_u, c_o):
    
    return np.where(actual<predictions , abs(actual-predictions)*c_o, abs(actual-predictions)*c_u)
    
def get_loss(predictions, actual):
    
    return np.subtract(predictions, actual)


def all_predictions(ts, window, underage, overage, all=True):
    no_data_chunk = [np.nan for i in range(6)]
    if all:
        intervals = normal_ts(ts)
        window, start = 1,0
        predictions = np.array([no_data_chunk for i in range(start + 1)])
        losses = np.array([no_data_chunk for i in range(start + 1)])
        costs = np.array([no_data_chunk for i in range(start + 1)])
        
    else: 
        intervals = rolling_window(ts, window)
        start=0
        predictions = np.array([no_data_chunk for j in range(window+start)])
        losses = np.array([no_data_chunk for j in range(window +start )])
        costs = np.array([no_data_chunk for j in range(window +start )])

    for i in range(start, len(intervals)):
        prediction = compare(intervals[i], underage, overage)


        

        if window + i < len(ts):

            


            loss = get_loss(prediction, ts[window + i])
            loss = np.reshape(loss, (1, len(loss)))
            losses = np.concatenate((losses, loss), axis=0)

            cost = get_cost(prediction, ts[window + i], underage, overage)
            cost = np.reshape(cost, (1, len(cost)))
            costs = np.concatenate((costs, cost))

        prediction = np.reshape(prediction, (1, len(prediction)))
        predictions = np.concatenate((predictions, prediction))

    return predictions, losses, costs

def cost_vs_window_size(ts, underage, overage):
    to_plot = np.array([[np.nan for i in range(6)] for i in range(2)])
    for i in range(2, len(ts)):
        
        preds, losses, costs = all_predictions(ts, i, underage, overage, all=False)
        
        
        
        average_cost = np.nansum(costs, axis=0)/(365-i)      # 365 if RRS, 31 if JD
        average_cost = np.reshape(average_cost, (1, len(average_cost)))

        to_plot = np.concatenate((to_plot, average_cost), axis=0)
    return to_plot

def loss_vs_window_size(ts):
    to_plot = np.array([[np.nan for i in range(6)] for i in range(2)])
    
    for i in range(2, len(ts)):
        losses = all_predictions(ts, i, 1, 1, all=False)[1]
        average_loss = np.nansum(abs(losses), axis=0)/(31-i)     # 365 if RRS, 31 if JD
        to_plot = np.vstack((to_plot, average_loss))
    return to_plot

def figures(JD=True):
    if JD:
        dataframe = DataReader.Data("/Users/makotopowers/Desktop/CSUREMM/data/raw/JD_order_data.csv")
        data = dataframe.extract_feature(feature=None, interval=24)[0]

    else:
        dataframe = DataReader.Data("/Users/makotopowers/Desktop/CSUREMM/data/raw/JD_order_data.csv")
        jd = ic(dataframe.extract_feature(feature=None, interval=24)[0])
        data = np.load("/Users/makotopowers/Desktop/CSUREMM/data/raw/h24_all_data.npy").transpose(1,0)[0]
        ic(data)
    ic('Data loaded. ')
        
    
    titles = ["Mean", "Median", "Holt", "Naive", "SAA", "NormsInv"]
    labels = ["Prediction", "Loss", "Cost"]
    combos = [[20,1], [10,1], [2,1], [1,1], [1,2], [1,10], [1,20]]

    for combo in combos:
        predictions, losses, costs = all_predictions(data, 0, combo[0], combo[1], all=True)
        predictions, losses, costs = predictions.transpose(1,0), losses.transpose(1,0), costs.transpose(1,0)
        ic(predictions[1])
        for u, v in zip(titles, predictions):
            plt.plot(v, label=u)
        
        plt.plot(data, label="Actual")
        plt.legend()
        plt.title("Predictions for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
        plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+".png")
        plt.close()
        for u, v in zip(titles, costs):
            plt.plot(v, label=u+" Cost")
        plt.legend()
        plt.title("Costs for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
        plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+"_cost.png")
        plt.close()
        ic("Checkpoint 1")

        costs = cost_vs_window_size(data, combo[0], combo[1])
        costs = costs.transpose(1,0)
        for u, v in zip(titles, costs):
            plt.plot(v, label=u+" Cost")
        plt.legend()
        plt.title("Average Costs vs Window Size for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
        plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+"_cost_vs_window.png")
        plt.close()

        ic("Checkpoint 2")

        for i in range(1,min(len(data),50)):
            predictions = all_predictions(data, i, combo[0], combo[1], all=False)[0]
            predictions = predictions.transpose(1,0)
            for u, v in zip(titles, predictions):
                plt.plot(v, label=u)
            plt.plot(data, label="Actual")
            plt.legend()
            plt.title("Predictions with Window Size (" + (str(i)) + ") for Underage: ("+str(combo[0])+") Overage: ("+str(combo[1]) + ")")
            plt.savefig(path+"/"+str(combo[0])+"_"+str(combo[1])+"_W"+str(i)+".png")
            plt.close()

        ic(f'Combo: {combo}' + ' done.')


if __name__=='__main__':
    path = "/Users/makotopowers/Desktop/CSUREMM/reports/figures/RR_FIGS/different_quantiles"
    figures(JD=False)


