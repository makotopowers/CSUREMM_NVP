

## Imports 

import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import ruptures as rpt
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns

import Methods
import Data
import RunNewsvendor
import warnings
warnings.filterwarnings("ignore")

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



def sparsity(x):
    zs = 0
    for i in x:
        if i in [0,1,2]:
            zs += 1
    return zs/len(x)

def modify_SAA():
    samples = 100
    data, stream_types, titles = Data.synthetic_data(100, samples)
    # ic(data.keys())

    sparsity_scores = dict()
    for key in data:
        sparsity_scores[key] = sparsity(data[key])

    for key in data:
            
        x = seasonal_decompose(data[key], model='additive', two_sided=False, period=2)
        data[key] = np.vstack((data[key], x.trend))
        data[key] = np.vstack((data[key], x.seasonal))
        data[key] = np.vstack((data[key], x.resid))


    

    combos = [[10,1], [2,1], [1,1], [1,2], [1,10]]
    for q in combos:
        all_costs = {t: {} for t in stream_types }
        count = 0
        printProgressBar(0, len(stream_types)*samples, prefix = f'Progress on combo {q}:', suffix = 'Complete', length = 100)
        for key in data:
            
            modifiers = {
                'cpd': [data[key][1], data[key][0], data[key][2], 'cpd'], 
                'norm': [data[key][1], data[key][0], data[key][2], None], 
                'cpd-noise': [data[key][3], data[key][0], data[key][1] + data[key][2], 'cpd']
            }
            preds, costs = RunNewsvendor.all_predictions(data[key][0], Methods.get_SAA(), q[0],q[1], seasons=None,roll_window=False, compare=None)

            mod_preds = {}
            mod_costs = {}
            for modifier in modifiers:
                mod_preds[f"pred_{modifier}"], mod_costs[f"costs_{modifier}"] = \
                RunNewsvendor.all_predictions(modifiers[modifier][0], Methods.get_SAA(), q[0], q[1], actual=modifiers[modifier][1],seasons=modifiers[modifier][2], compare=modifiers[modifier][3])

            algo='SAA'

            for modifier in mod_costs:
                d_stream_type = key[:2]
                try:
                    all_costs[d_stream_type][modifier].append((np.nanmean(mod_costs[modifier][algo]) - np.nanmean(costs[algo]))/np.nanmean(costs[algo]))
                except KeyError:
                    all_costs[d_stream_type][modifier] = [(np.nanmean(mod_costs[modifier][algo]) - np.nanmean(costs[algo]))/np.nanmean(costs[algo])]
            
            printProgressBar(count, len(stream_types)*samples, prefix = f'Progress on combo {q}:', suffix = 'Complete', length = 100)
            count+=1

            # fig, ax = plt.subplots(4,1, figsize=(10,5))
            # fig.suptitle(f'{key}')
            # ax[0].plot(data[key][0])
            # ax[0].grid(True)
            # ax[0].set_ylabel('Value')
            # ax[0].set_title('Original')

            # ax[1].plot(data[key][1])
            # ax[1].set_ylabel('Value')
            # ax[1].grid(True)
            # ax[1].set_title('Trend')

            # ax[2].plot(data[key][2])
            # ax[2].set_ylabel('Value')
            # ax[2].grid(True)
            # ax[2].set_title('Seasonal')

            # ax[3].plot(data[key][3])
            # ax[3].set_ylabel('Value')
            # ax[3].set_xlabel('Time')
            # ax[3].grid(True)
            # ax[3].set_title('Noise')

            # plt.tight_layout()
            # plt.savefig(f'reports/figures/synth_data_samples/{key}.png')
            # plt.close()



                        

        
    

        

        
        count = 0
        
        for d_stream in all_costs:

            sns.set()
            plt.tight_layout()
            plt.title (f"{titles[count]} with cu = {q[0]}, co = {q[1]}")


            plt.grid(True)
            plt.xlabel(r'Average after $n$ iterations')
            plt.ylabel('Relative cost')

            
            for modifier in all_costs[d_stream]:
                average = np.array([np.nanmean(all_costs[d_stream][modifier][:i]) for i in range(len(all_costs[d_stream][modifier]))])
                plt.plot(average, label=modifier)
            plt.legend()

            
            plt.savefig(f"reports/figures/synth_data_results/{d_stream}_{q}.png")
            plt.close()

            sns.set()
            plt.tight_layout()
            plt.grid(True)
            for modifier in all_costs[d_stream]:
                plt.plot(all_costs[d_stream][modifier], label=modifier)
            plt.xlabel(r'Iteration')
            plt.ylabel('Relative cost')
            plt.title(f"{titles[count]} with cu = {q[0]}, co = {q[1]}")
            plt.legend()
            plt.savefig(f"reports/figures/synth_data_results/{d_stream}_{q}_iterations.png")
            plt.close()
            count += 1

        


modify_SAA()
