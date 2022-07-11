



'''

This file is used to prepare the data for the analysis.
It reads in the data from hdf5 files (already converted from csv using vaex)
and prepares it for the analysis using various conversions to make computations possible.

Running prepare_data() will return a dictionary which contains array-like objects representing time series 
for each of the following:
    - JD_all: the total number of JD orders
    - JD_sku: the total number of JD orders for the 20 SKUs with the most data
    - RRS_all: the total number of RRS orders
    - RRS_sku: the total number of RRS orders for the 20 SKUs with the most data

The JD dataset is over 30 days, and the RRS dataset is over 365 days. 


'''

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Imports

import datetime

import numpy as np
import vaex
from icecream import ic

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Convert string to datetime object.
## parameters:
## date_string : string representing a date
## Returns:
## datetime object

def convert(date_string):
    return datetime.datetime.strptime(str(date_string), "%Y-%m-%d %H:%M:%S")

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

## Compute the number of days between two dates, the first date being the day of the first order.
## parameters:
## date_string : datetime object
## Returns:
## int value representing the number of days between the two dates

def comp(date_string, start_date_string):
    x = datetime.datetime.strptime(start_date_string, "%Y-%m-%d %H:%M:%S")
    return (date_string - x).days

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#


def correlate():
    JD_correlations = dict()
    JD_all = vaex.open("data/raw/JD_order_data.csv.hdf5")
    JD_all['promise'] = JD_all.promise.apply(lambda x : int(x) if x != '-' else np.nan)
    JD_col_names = ['promise', 'original_unit_price', 'final_unit_price', 'direct_discount_per_unit', 'quantity_discount_per_unit', 'bundle_discount_per_unit', 'coupon_discount_per_unit']
    for column in JD_col_names:
        JD_correlations[str(column)] = JD_all.correlation(x = 'quantity', y=f'{column}')

    RRS_correlations = dict()
    RRS_all = vaex.open("data/raw/1+2_orders_SKU_details.hdf5")
    ic(RRS_all.column_names)
    RRS_col_names = []
    for column in RRS_col_names:
        RRS_correlations[str(column)] = RRS_all.correlation(x = 'quantity', y=f'{column}')

    return JD_correlations, RRS_correlations

## Prepare the data for the analysis.
## parameters:
## None
## Returns:
## data : dictionary containing array-like objects representing time series

def prepare_data():

    data = dict()
    #----------------------------------------------------------------------------------------------------------------------#
    ## JD dataset processing ##

    JD_all = vaex.open('data/processed/JD_all.hdf5')
    
    dates, values = JD_all.new_dates.values.reshape(1,-1), JD_all.quantity.values.reshape(1,-1)
    data['JD_all'] = np.vstack((dates, values)).data
    
    JD_sku = vaex.open('data/processed/JD_sku.hdf5')
    JD_sku['sku_ID'] = JD_sku.sku_ID.apply(lambda x: str(x))

    JD_skus = dict()
    JD_by_sku = vaex.open('data/processed/JD_by_sku.hdf5')

    JD_by_sku = [str(JD_by_sku.sort(by=['quantity', 'sku_ID'], ascending=[False, True]).sku_ID.values[i]) for i in range(20)]

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
        data['JD'+key] = JD_skus[key].transpose(1,0)

    ic("Finished Processing JD Data. (1/2)")

    #----------------------------------------------------------------------------------------------------------------------#
    ## RRS dataset processing ##

    RRS_all = vaex.open('data/processed/RRS_all.hdf5')

    dates, values = RRS_all.new_dates.values.reshape(1,-1), RRS_all.ORDER_AMT.values.reshape(1,-1)
    to_delete = np.where(dates[0] < 0)
    dates, values = np.delete(dates[0], to_delete, axis=0).reshape(1,-1), np.delete(values[0], to_delete, axis=0).reshape(1,-1)

    data['RRS_all'] = ic(np.vstack((dates, values)))

    RRS_skus = dict()
    RRS_sku = vaex.open('data/processed/RRS_sku.hdf5')

    RRS_by_sku = vaex.open('data/processed/RRS_by_sku.hdf5')
    RRS_by_sku = [str(RRS_by_sku.sort(by=['ORDER_AMT', 'RRS_MATE_CODE'], ascending=[False, True]).RRS_MATE_CODE.values[i]) for i in range(3)]

    for sku in RRS_by_sku:
        RRS_skus[sku] = np.column_stack((RRS_sku[RRS_sku.RRS_MATE_CODE == sku].new_dates.values, RRS_sku[RRS_sku.RRS_MATE_CODE == sku].ORDER_AMT.values))

    for key in RRS_skus:
        RRS_skus[key] = RRS_skus[key][RRS_skus[key][:,0].argsort()]
        for i in range(365):
            try: 
                if (RRS_skus[key][i][0] != i):
                    RRS_skus[key] = np.insert(RRS_skus[key], i, [i, 0], axis=0)

            except(IndexError):
                RRS_skus[key] = np.vstack((RRS_skus[key],[i, 0]))

    for key in RRS_skus:
        data['RRS'+key] = RRS_skus[key].transpose(1,0)

    ic("Finished Processing RRS Data. (2/2)")

    #----------------------------------------------------------------------------------------------------------------------#

    return data

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
    x = prepare_data()
    ic(x)
