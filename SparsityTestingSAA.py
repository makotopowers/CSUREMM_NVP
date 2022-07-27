import pandas as pd
from icecream import ic
import numpy as np
#from mle import * 
import Methods
import vaex
#import torch 
import matplotlib.pyplot as plt
import Data
# import hana_ml
# import hana_ml.dataframe as df

from statsmodels.tsa.seasonal import seasonal_decompose

import ruptures as rpt 

import RunNewsvendor
import warnings
warnings.filterwarnings("ignore")


def sparsity(x):
    zs = 0
    for i in x:
        if i in [0,1,2]:
            zs += 1
    return zs/len(x)
            

if __name__=='__main__':
    data = Data.prepare_data(0, 100)
    sparsity_scores = dict()
    for key in data:
        sparsity_scores[key] = sparsity(data[key][1])


    for key in data:
            
        x = seasonal_decompose(data[key][1], model='additive', two_sided=False, period=7)
        data[key] = np.vstack((data[key], x.trend))
        data[key] = np.vstack((data[key], x.seasonal))
        data[key] = np.vstack((data[key], x.resid))

    all_costs_3 = []
    all_costs_cpd = []
    all_costs_norm = []
    all_costs_z = []

    improve3 = 0
    improvecpd = 0
    improvenorm = 0
    improvez = 0

    average_relative_improvement3 = 0
    average_relative_improvement_cpd = 0
    average_relative_improvement_norm = 0
    average_relative_improvement_z = 0

    sparse3 = []
    sparsecpd = []
    sparsenorm = []
    sparsez = []

    average_relative_cost3 = 0 
    average_relative_cost_cpd = 0
    average_relative_cost_norm = 0
    average_relative_cost_z = 0

    cost_sparse3 = []
    cost_sparsecpd = []
    cost_sparsenorm = []
    cost_sparsez = []

    count = 0
    for key in data:
        count = ic(count)+1
        ic("doen")
        ic("key")

        preds, costs = RunNewsvendor.all_predictions(data[key][1] , Methods.get_algos(), 1,1,1, seasons=None,roll_window=False, compare=None)

        predscpd, costscpd = RunNewsvendor.all_predictions(data[key][2], Methods.get_algos(), 1,1,1, actual=data[key][1],roll_window=False, compare='cpd')
        preds3, costs3 = RunNewsvendor.all_predictions(data[key][2], Methods.get_algos(), 1,1,1, actual=data[key][1],roll_window=False, compare='sparse')
        prednorm, costnorm = RunNewsvendor.all_predictions(data[key][2], Methods.get_algos(), 1,1,1, actual=data[key][1],roll_window=False, compare=None)
        predz, costz = RunNewsvendor.all_predictions(data[key][2], Methods.get_algos(), 1,1,1, actual=data[key][1],roll_window=False, compare='zero')

        algo='SAA'
        if (np.nanmean(costs3[algo]) - np.nanmean(costs[algo]))/np.nanmean(costs[algo]) > 0.75:
            continue
        all_costs_3.append( [sparsity_scores[key], (np.nanmean(costs3[algo]) - np.nanmean(costs[algo]))/np.nanmean(costs[algo])])
        if (np.nanmean(costscpd[algo]) - np.nanmean(costs[algo]))/np.nanmean(costs[algo]) > 0.75:
            continue
        all_costs_cpd.append([sparsity_scores[key],(np.nanmean(costscpd[algo]) - np.nanmean(costs[algo]))/np.nanmean(costs[algo])])
        if (np.nanmean(costnorm[algo])-np.nanmean(costs[algo]))/np.nanmean(costs[algo]) > 0.75:
            continue
        all_costs_norm.append([sparsity_scores[key],(np.nanmean(costnorm[algo])-np.nanmean(costs[algo]))/np.nanmean(costs[algo])])
        if (np.nanmean(costz[algo])-np.nanmean(costs[algo]))/np.nanmean(costs[algo]) > 0.75:
            continue
        all_costs_z.append([sparsity_scores[key],(np.nanmean(costz[algo])-np.nanmean(costs[algo]))/np.nanmean(costs[algo])])


    x31, y31 = zip(*all_costs_3)
    xcpd1, ycpd1 = zip(*all_costs_cpd)
    xnorm1, ynorm1 = zip(*all_costs_norm)
    xz1, yz1 = zip(*all_costs_z)

    x3, y3 = np.array(x31), np.array(y31)
    xcpd, ycpd = np.array(xcpd1), np.array(ycpd1)
    xnorm, ynorm = np.array(xnorm1), np.array(ynorm1)
    xz, yz = np.array(xz1), np.array(yz1)

    arrayx= np.array(all_costs_3)
    arraycpd= np.array(all_costs_cpd)
    arraynorm= np.array(all_costs_norm)
    arrayz= np.array(all_costs_z)
    
    plt.figure(figsize=(12,8)) 
    plt.title(r'Using RRS datasets: Relative Cost of $X$ against SAA vs Sparsity')
    plt.xlabel('Sparsity')
    plt.ylabel(r'Relative Cost, $\frac{C_x-C_saa}{C_saa}$')
    plt.xlim(-0.1,1.1)

    ic("making lines")
    interval = 0.1
    bounds = 1
    while interval <= 1:
        av3 = []
        avcpd = []
        avnorm = []
        avz = []
        for i in range(len(x31)):
            if x31[i] <= interval and x31[i] >= interval - 0.1:
                av3.append(y31[i])
        for i in range(len(xcpd)):
            if xcpd1[i] <= interval and xcpd1[i] >= interval - 0.1:
                avcpd.append(ycpd1[i])
        for i in range(len(xnorm)):
            if xnorm1[i] <= interval and xnorm1[i] >= interval - 0.1:
                avnorm.append(ynorm1[i])
        for i in range(len(xz)):
            if xz1[i] <= interval and xz1[i] >= interval - 0.1:
                avz.append(yz1[i])

        plt.axhline(y=np.nanmean(av3), xmin=(bounds * 1/12), xmax=((bounds+1)*1/12), linewidth=2, color='r')
        plt.axhline(y=np.nanmean(avcpd), xmin=(bounds * 1/12), xmax=((bounds+1)*1/12), linewidth=2, color='b')
        
        plt.axhline(y=np.nanmean(avz), xmin=(bounds * 1/12), xmax=((bounds+1)*1/12), linewidth=2, color='k')
        interval += 0.1
        bounds += 1

    plt.scatter(x31,y31, color='red', label='Sparse')
    plt.scatter(xcpd1, ycpd1, color='blue', label='CPD')
    plt.scatter(xz1, yz1, color='black', label='Zero')
    plt.grid(True)
    plt.legend()
    plt.savefig('FINALSPARSITYVSCOST.png', dpi=300)

    plt.show()
