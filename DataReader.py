



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

## Prepare the data for the analysis.
## parameters:
## None
## Returns:
## data : dictionary containing array-like objects representing time series

def prepare_data():

    #----------------------------------------------------------------------------------------------------------------------#
    ## JD dataset processing ##

    data = dict()
    JD = vaex.open('data/raw/JD_order_data.csv.hdf5')
    JD['order_time'] = JD.order_time.apply(lambda x: x.split('.')[0])
    JD['dates'] = JD.order_time.apply(convert)
    JD['new_dates'] = JD.dates.apply(lambda x: comp(x, '2018-03-01 00:00:00'))
    
    ic("Finished Reading JD Data. (1/6)")

    JD_all = JD.groupby(by='new_dates').agg({'quantity': 'sum'}).quantity.values.reshape(1,-1)
    dates, values = JD_all.quantity.values.reshape(1,-1), JD_all.new_dates.values.reshape(1,-1)
    data['RRS_all'] = np.column_stack((dates, values))


    ic(JD_all)
    data['JD_all'] = JD_all

    ic("Finished Processing 'ALL' JD Data. (2/6)")

    JD_sku = JD.groupby(by=['new_dates', 'sku_ID']).agg({'quantity': 'sum'})
    JD_sku['sku_ID'] = JD_sku.sku_ID.apply(lambda x: str(x))

    JD_skus = dict()
    JD_by_sku = JD_sku.groupby(by='sku_ID').agg({'quantity': 'sum'})
    JD_by_sku = [str(JD_by_sku.sort(by=['quantity', 'sku_ID'], ascending=[False, True]).sku_ID.values[i]) for i in range(20)]

    for sku in JD_by_sku:
        JD_skus[sku] = np.column_stack((JD_sku[JD_sku.sku_ID == sku].new_dates.values.to_numpy(), JD_sku[JD_sku.sku_ID == sku].quantity.values))

    for key in JD_skus:
        JD_skus[key] = JD_skus[key][JD_skus[key][:,0].argsort()]

    data['JD_sku'] = JD_sku

    ic("Finished Processing 'SKU' JD Data. (3/6)")

    #----------------------------------------------------------------------------------------------------------------------#
    ## RRS dataset processing ##

    RRS = vaex.open('data/raw/1+2_orders_SKU_details.hdf5')

    ic("Finished Reading RRS Data. (4/6)")

    RRS['dates'] = RRS.order_date.apply(convert)
    RRS['new_dates'] = RRS.dates.apply(lambda x: comp(x, '2018-09-01 00:00:00'))

    RRS_all = RRS.groupby(by='new_dates').agg({'ORDER_AMT': 'sum'})
    dates, values = RRS_all.ORDER_AMT.values.reshape(1,-1), RRS_all.new_dates.values.reshape(1,-1)
    data['RRS_all'] = np.column_stack((dates, values))

    ic("Finished Processing 'ALL' RRS Data. (5/6)")

    RRS_sku = RRS.groupby(by=['new_dates', 'RRS_MATE_CODE']).agg({'ORDER_AMT': 'sum'})
   
    RRS_sku['RRS_MATE_CODE'] = RRS_sku.RRS_MATE_CODE.apply(lambda x: str(x))
    RRS_skus = dict()
    RRS_by_sku = RRS_sku.groupby(by='RRS_MATE_CODE').agg({'ORDER_AMT': 'sum'})
    RRS_by_sku = [str(RRS_by_sku.sort(by=['ORDER_AMT', 'RRS_MATE_CODE'], ascending=[False, True]).RRS_MATE_CODE.values[i]) for i in range(20)]

    for sku in RRS_by_sku:
        RRS_skus[sku] = np.column_stack((RRS_sku[RRS_sku.RRS_MATE_CODE == sku].new_dates.values.to_numpy(), RRS_sku[RRS_sku.RRS_MATE_CODE == sku].ORDER_AMT.values))

    for key in RRS_skus:
        RRS_skus[key] = RRS_skus[key][RRS_skus[key][:,0].argsort()]
    data['RRS_sku'] = RRS_sku

    ic("Finished Processing 'SKU' RRS Data. (6/6)")

    #----------------------------------------------------------------------------------------------------------------------#

    return data

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#



if __name__ == '__main__':
    RRS = vaex.open('data/raw/1+2_orders_SKU_details.hdf5')
    data = dict()
    RRS['dates'] = RRS.order_date.apply(convert)
    RRS['new_dates'] = RRS.dates.apply(lambda x: comp(x, '2018-10-01 00:00:00'))

    ic("Finished Processing 'ALL' RRS Data. (5/6)")

    RRS_sku = RRS.groupby(by=['new_dates', 'RRS_MATE_CODE']).agg({'ORDER_AMT': 'sum'})
   
    #RRS_sku['RRS_MATE_CODE'] = RRS_sku.RRS_MATE_CODE.apply(lambda x: str(x))
    
    ic(RRS_sku)
    RRS_skus = dict()
    RRS_by_sku = RRS_sku.groupby(by='RRS_MATE_CODE').agg({'ORDER_AMT': 'sum'})


    ic(RRS_by_sku)
    RRS_by_sku = [str(RRS_by_sku.sort(by=['ORDER_AMT', 'RRS_MATE_CODE'], ascending=[False, True]).RRS_MATE_CODE.values[i]) for i in range(20)]
    ic(RRS_by_sku)

    ic(RRS_by_sku[0])
    ic(RRS_sku[RRS_sku.RRS_MATE_CODE == RRS_by_sku[0]].new_dates.values.to_numpy(), RRS_sku[RRS_sku.RRS_MATE_CODE == RRS_by_sku[0]].ORDER_AMT.values)
    ic(RRS_sku[RRS_sku.RRS_MATE_CODE == RRS_by_sku[0]].new_dates.values.to_numpy().shape, RRS_sku[RRS_sku.RRS_MATE_CODE == RRS_by_sku[0]].ORDER_AMT.values.shape)

    for sku in RRS_by_sku:
        RRS_skus[sku] = np.column_stack((RRS_sku[RRS_sku.RRS_MATE_CODE == sku].new_dates.values.to_numpy(), RRS_sku[RRS_sku.RRS_MATE_CODE == sku].ORDER_AMT.values))

    for key in RRS_skus:
        RRS_skus[key] = RRS_skus[key][RRS_skus[key][:,0].argsort()]
    data['RRS_sku'] = RRS_skus

    ic("Finished Processing 'SKU' RRS Data. (6/6)")
    

#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#

