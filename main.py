from crypt import methods
import DataReader
import Benchmarks
from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm 

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
    chunks=[]
    for i in range(1,len(ts)+1):
        chunks.append(ts[:i])
    return chunks


def compare(slice, show=False):
    methods = Benchmarks.methods()
    

    mean = methods.mean(slice)
    median = methods.median(slice)
    holt = methods.holt(slice)[0] if len(slice) >= 10 else None
    
    naive = methods.naive(slice)
    saa = methods.SAA(slice)
    normsinv = methods.normsinv(slice)

    return [mean, median, holt, naive, saa, normsinv]


def all_preds(ts, window, all=True):
    ic(ts)
    if all:
        intervals = normal_ts(ts)
        window = 0
        start =1
        preds = np.array([[None for i in range(6)],
                          [None for i in range(6)]])
        losses = np.array([[0 for i in range(6)],
                           [0 for i in range(6)]])
        
    else: 
        intervals = rolling_window(ts, window)
        start=0
        p_starter = [[None for i in range(6)] for j in range(window)]
        l_starter = [[0 for i in range(6)] for j in range(window)]
        preds = np.array(p_starter)
        losses = np.array(l_starter)

    
    for i in range(start, len(intervals)):
        pred = compare(intervals[i], show=False)
        if window + i < len(ts)-1:
            loss = get_losses(ts, pred, window + i +1)
            losses = np.vstack((losses, loss))
            
    
        preds = np.vstack((preds, pred))
    
    return ic(preds), ic(losses)


def get_losses(all_data, predictions, day):
    
    losses = []
    predics = predictions.copy()
    for i in range(len(predics)):
        ic(predics[i])
        if predics[i] == None:
            losses.append(None)
        else:
            losses.append(predics[i]- all_data[day])
    return losses

if __name__=='__main__':
    data = DataReader.Data("/Users/makotopowers/Desktop/CSUREMM/data/raw/JD_order_data.csv")
    ts = data.extract_feature(bucket_length=24, all=True)[0]
    preds, losses = all_preds(ts, 10, all=True)
    ic(preds.transpose(1,0))
    title = ["mean", "median", "holt", "naive", "saa", "normsinv"]
    
    #preds, losses = all_preds(ts, window=10)
    #ic(preds), ic(preds.shape), ic(losses), ic(losses.shape)
    for i in range(len(losses.transpose(1,0))):
        
        plt.plot(losses.transpose(1,0)[i], label=f"{title[i]}")
    #plt.plot(ts, label="Actual", linewidth=3, color='black')
    plt.legend()
    plt.title(f"Comparison of Losses using All Data, Full Window" )
    plt.savefig("/Users/makotopowers/Desktop/CSUREMM/reports/figures/full_ts/all_data/all_data_losses.png")
    #plt.show()

#loss as a function of window length


    
    
    
    

    