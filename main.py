from crypt import methods

from sqlalchemy import over
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

def compare(slice):
    methods = Benchmarks.methods()
    
    mean = methods.mean(slice)
    median = methods.median(slice)
    holt = methods.holt(slice)[0] if len(slice) >= 10 else None
    naive = methods.naive(slice)
    saa = methods.SAA(slice, overage=10, underage =1)
    normsinv = methods.normsinv(slice, overage=10, underage =1)
    #ic(saa)
    return [mean, median, holt, naive, saa, normsinv]


def all_preds(ts, window, all=True):
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
        p_starter = [[None for i in range(6)] for j in range(window+start)]
        l_starter = [[0 for i in range(6)] for j in range(window +start )]
        preds = np.array(p_starter)
        losses = np.array(l_starter)

    
    for i in range(start, len(intervals)):
        #ic(intervals[i])

        pred = compare(intervals[i])
        #ic (pred)
        if window + i < len(ts):
            #ic("get_losses called")
            loss = get_losses(ts, pred, window + i)
            losses = np.vstack((losses, loss))
            
    
        preds = np.vstack((preds, pred))
        #ic("in all_preds", preds), ic("in all_preds",losses)
    #ic(f"{window}",losses)
    return preds, losses


def get_losses(all_data, predictions, day):
    
    losses = []
    predics = predictions.copy()
    for i in range(len(predics)):
       
        if predics[i] == None:
            losses.append(None)
        else:
            losses.append(abs(predics[i]- all_data[day]))

    #ic("get_losses Loss", losses)
    return losses

def all_window_sizes(ts, all=False):
    to_plot = np.array([[None, None, None, None, None, None],[None, None, None, None, None, None]])
    for i in range(2, len(ts)):
        #ic(i)
        preds, losses = all_preds(ts, i, all=False)
        l = losses.copy()
        l[l==None] = 0
        #ic(l)
        total_losses = np.sum(l, axis=0)
        total_losses = total_losses/(31-i)
        #ic("l: ",l, l.shape, "i: ", i)
        #ic("total_losses: ", total_losses, total_losses.shape)
        
        to_plot = np.vstack((to_plot, total_losses))
    return to_plot


if __name__=='__main__':
    data = DataReader.Data("/Users/makotopowers/Desktop/CSUREMM/data/raw/JD_order_data.csv")
    feat = data.extract_feature(feature="sku_ID", bucket_length=24,)
    sku_ts, indices = data.get_most(feat)
    #ic(feat[indices[1]])

    title = ["mean", "median", "holt", "naive", "saa", "normsinv"]
    for i in range(20):
        
        a = feat[indices[i]]
        p = all_window_sizes(a, all=False)
    #p, l = all_preds(a, 10, all=False)
    #ic(p)
        for j in range(len(p.transpose(1,0))):
            plt.plot(p.transpose(1,0)[j], label = title[j])
        plt.title(f"Loss as function of window size for sku_ID: {indices[i]}")
        plt.legend()
        plt.savefig(f"/Users/makotopowers/Desktop/CSUREMM/reports/figures/as_function_of_window_size/cu1_co10/sku_ID_{indices[i]}.png")
        plt.close()
    #plt.plot(a, label="Actual", linewidth=3, color='black')
    #plt.show()
    #preds, losses = all_preds(ts, 10, all=False)
    #ic(losses)
    #losses[losses==None] = 0
    #losses = ic(np.sum(losses, axis=0))
    #losses = losses/(31-3)
    #ic(losses)
    #ic(losses.shape)
    #title = ["mean", "median", "holt", "naive", "saa", "normsinv"]
    #plt.plot(preds)
    #plt.show()
    #preds, losses = all_preds(ts, window=10)
    #ic(preds), ic(preds.shape), ic(losses), ic(losses.shape)
   # for i in range(len(losses.transpose(1,0))):
        
        #plt.plot(losses.transpose(1,0)[i], label=f"{title[i]}")
    #plt.plot(ts, label="Actual", linewidth=3, color='black')
    #plt.legend()
    #plt.title(f"Comparison of Losses using All Data, Full Window" )
    
    #plt.show()

#loss as a function of window length


    
    
    
    
#|________|
#|c_u vs c_o        |
#|________|
#Each SKU_ID
#Average loss of all SKUs, compare to average loss of "all_data"$
#


#look at other data set 
#loss as a function of different window sizes