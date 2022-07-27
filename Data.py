



'''

This file is used to prepare the data for the analysis.
It reads in the data from hdf5 files (already converted from csv using vaex)
and prepares it for the analysis using various conversions to make computations possible.

Running prepare_data() will return a dictionary which contains array-like objects representing time series 
for each of the following:
    - JD_all: the demand sequence of JD orders
    - JD_sku: the demand sequence of JD orders for the 20 SKUs with the most data
    - RRS_all: the demand sequence of RRS orders
    - RRS_sku: the demand sequence of RRS orders for the 20 SKUs with the most data

The JD dataset is over 30 days, and the RRS dataset is over 365 days. 


'''

import datetime

import numpy as np
import vaex
from icecream import ic
import warnings
warnings.filterwarnings("ignore")


def convert(date_string : str) -> datetime.datetime:
    '''Converts a string of the form "YYYY-MM-DD HH:MM:SS" to a datetime object

    Parameters
    ----------
    date_string : str
        A string representing a date in the format YYYY-MM-DD HH:MM:SS

    Returns
    -------
    datetime.datetime
        A datetime object representing the date
    '''
    return datetime.datetime.strptime(str(date_string), "%Y-%m-%d %H:%M:%S")


def comp(date : datetime.datetime, start_date_string : str) -> int:
    '''Computes the number of days between two dates

    Parameters
    ----------
    date : datetime.datetime
        A datetime object representing the date

    start_date_string : str
        A string representing the start date

    Returns
    -------
    int 
        The number of days between the two dates
    '''
    x = datetime.datetime.strptime(start_date_string, "%Y-%m-%d %H:%M:%S")
    return (date - x).days


def all_data(pathto : str = None) -> np.array:
    '''Reads in all the data from the hdf5 files and returns it as a numpy array

    Parameters
    ----------
    pathto : str, optional
        The path to the hdf5 file to read in. If not specified, the default path is used.

    Returns
    -------
    np.array
        A numpy array containing all the data
    '''
    if pathto is None:
        pathto = "data/raw/JD_order_data.csv.hdf5"

    all_data : vaex.dataframe = vaex.open(pathto)
 
    
    dates : np.array = all_data.new_dates.values.reshape(1,-1) 
    values : np.array = all_data.quantity.values.reshape(1,-1)


    return values
    

def prepare_data(JD_number : int = 100, RRS_number : int = 100) -> dict[np.array, np.array]:
    '''Prepares the data for the analysis

    Parameters
    ----------
    JD_number : int
        Number of JD items to include in the dataset
    
    RRS_number : int
        Number of RRS items to include in the dataset

    Returns
    -------
    dict(np.ndarray)
        A dictionary containing the following arrays:
            - JD_all: the demand sequence of JD orders
            - JD_sku: the demand sequence of JD orders for the JD_number SKUs with the most data
            - RRS_all: the demand sequence of RRS orders
            - RRS_sku: the demand sequence of RRS orders for the RRS_number SKUs with the most data
        Each array has a the day number as the first row and the SKU demand sequence as the second row.
        Each np.ndarray is of shape (2, days)
   
    '''

    data : dict = dict()

    
    data['JD_all'] = all_data('data/processed/JD_all.hdf5')
    
    JD_sku = vaex.open('data/processed/JD_sku.hdf5')
    JD_sku['sku_ID'] = JD_sku.sku_ID.apply(lambda x: str(x))

    JD_by_sku : vaex.dataframe = vaex.open('data/processed/JD_by_sku.hdf5')

    
    JD_by_sku = [str(JD_by_sku.sort(by=['quantity', 'sku_ID'], ascending=[False, True]).sku_ID.values[i]) for i in range(JD_number)]


    JD_skus : dict = dict()

    for sku in JD_by_sku:
        JD_skus[sku] = np.column_stack((JD_sku[JD_sku.sku_ID == sku].new_dates.values, JD_sku[JD_sku.sku_ID == sku].quantity.values))


    for key in JD_skus:
        JD_skus[key] = JD_skus[key][JD_skus[key][:,0].argsort()]
        for i in range(31):
            try: 
                if (JD_skus[key][i][0] != i):
                    JD_skus[key] = np.insert(JD_skus[key], i, [i, 0], axis=0)

            except(IndexError):
                JD_skus[key] = np.vstack((JD_skus[key],[i, 0]))

    for key in JD_skus:
        data['JD_'+key] = JD_skus[key].transpose(1,0)[1]

    
    RRS_all = vaex.open('data/processed/RRS_all.hdf5')
    

    
    dates: int = RRS_all.new_dates.values.reshape(1,-1) 
    values = RRS_all.ORDER_AMT.values.reshape(1,-1)



    to_delete = np.where(dates[0] < 0)


    dates, values = np.delete(dates[0], to_delete, axis=0).reshape(1,-1), np.delete(values[0], to_delete, axis=0).reshape(1,-1)

    to_add = np.vstack((dates, values))

    to_add = to_add[:, to_add[0,:].argsort()].transpose(1,0)

    
    to_del_RRS = []
    
    count = len(to_add)

    i=0
    while i < count:
        try: 
            if (to_add[i][0] != i):
                if (to_add[i][0] < 0):
                    try:
                        to_del_RRS.append(i)
                    except:
                        to_del_RRS = [i]
                else:
                    to_add = np.insert(to_add, i, [i, 0], axis=0)
                    if count < 365:
                        count += 1
        except(IndexError):
            to_add = np.vstack((to_add,[i, 0]))
        i += 1
    
            

    x = np.delete(to_add, to_del_RRS, axis=0)
    del to_add
    to_add = x

    data['RRS_all'] = to_add.transpose(1,0)[1]
   

    RRS_skus = dict()
    RRS_sku = vaex.open('data/processed/RRS_sku.hdf5')

    RRS_by_sku = vaex.open('data/processed/RRS_by_sku.hdf5')
    RRS_by_sku = [str(RRS_by_sku.sort(by=['ORDER_AMT', 'RRS_MATE_CODE'], ascending=[False, True]).RRS_MATE_CODE.values[i]) for i in range(RRS_number)]

    for sku in RRS_by_sku:

        RRS_skus[sku] = np.column_stack((RRS_sku[RRS_sku.RRS_MATE_CODE == sku].new_dates.values, RRS_sku[RRS_sku.RRS_MATE_CODE == sku].ORDER_AMT.values))
        
    to_del = {}
    for key in RRS_skus:
        RRS_skus[key] = RRS_skus[key][RRS_skus[key][:,0].argsort()]
        
        count = len(RRS_skus[key])
        i=0
        while i < count:
            try: 
                if (RRS_skus[key][i][0] != i):
                    if (RRS_skus[key][i][0] < 0):
                        try:
                            to_del[key].append(i)
                        except:
                            to_del[key] = [i]
                    else:    
                        RRS_skus[key] = np.insert(RRS_skus[key], i, [i, 0], axis=0)
                        if count < 365:
                            count += 1
                

            except(IndexError):
                RRS_skus[key] = np.vstack((RRS_skus[key],[i, 0]))
            i += 1
    
            
    for key in to_del:
        x = np.delete(RRS_skus[key], to_del[key], axis=0)
        del RRS_skus[key]
        RRS_skus[key] = x

    for key in RRS_skus:
        data['RRS'+key] = RRS_skus[key].transpose(1,0)[1]

    to_del = []
    to_add = []

    return data


def synthetic_data(length : int, samples : int) -> dict[np.ndarray]:
    '''_summary_

    Parameters
    ----------
    length : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    data = dict()
    for sample in range(samples):
        t1 = np.array([100 + s + np.random.normal(0, np.sqrt(1+100+s)) for s in range(length)])
        t2 = np.array([100 + s**2 + np.random.normal(0, np.sqrt(1+100+s**2)) for s in range(length)])
        t3 = np.array([max(100*(s%2), 50) + np.random.normal(0, np.sqrt(1+max(100*(s%2), 50))) for s in range(length)])
        t4 = np.array([max(100*(s%2), 50) + s + np.random.normal(0, np.sqrt(1+max(100*(s%2), 50)+s)) for s in range(length)])

        # t6 = np.array([100*np.random.binomial(1, 0.8)  for s in range(length)])
        # t7 = np.array([max(100*(s%4), 140) for s in range(length)])

        data[f't1_{sample}'] = t1
        data[f't2_{sample}'] = t2
        data[f't3_{sample}'] = t3
        data[f't4_{sample}'] = t4

     
        # t5 = []
        # for s in range(length):
        #     x = np.random.choice([100,80,50,130, 170, 200])
        #     t5.append(x + s + np.random.normal(0, np.sqrt(1+x+s)))
        # data[f't5_{sample}'] = np.array(t5)

    labels = ['t1', 't2', 't3', 't4']
    titles = [
        r't1: $\mu=100+s$', 
        r't2: $\mu=100+s^2$', 
        r't3: $\mu = 100$ if $s$ odd, 50 if $s$ even', 
        r't4: $\mu = 100 + s$ if $s$ odd, $50 + s$ if $s$ even', 
        
    ]
    return data, labels, titles

        
    

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    data = synthetic_data(100, 1)
    ic(data)
    