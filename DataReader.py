import numpy as np
import pandas as pd
import dask.dataframe as dd
from numba import jit

class Data:
    def __init__(self, file_name, RR=False):
        self.RR = RR
        if RR:
            pass
        else:
            self.df = pd.read_csv(file_name)


    def read_RR(self, feature=None, interval=24):
        df1 = pd.read_csv("/Users/makotopowers/Desktop/CSUREMM/1.orders.csv", dtype={'distc_dest_org': 'object'})
        df2 = pd.read_csv("/Users/makotopowers/Desktop/CSUREMM/2.SKU_details.csv")
        
        df1 = df1[['order_date', 'order_no']]
        df2 = df2[['ORDER_NO', 'RRS_MATE_CODE', 'ORDER_AMT']]
        
        df = df2.merge(df1, how='inner', left_on='ORDER_NO', right_on='order_no')

        df['hour'] = pd.to_datetime(df['order_date']).dt.hour//interval
        df['day'] = pd.to_datetime(df['order_date']).dt.dayofyear-1
        df['datetimes'] = (df['day'])*(24//interval) + df['hour']
        df = df.drop(columns=['hour', 'day', 'order_date', 'ORDER_NO', 'order_no'])

        
        u = list(df['datetimes'].unique())
        for i in range(8760):
            if i in u:
                continue

            else:
                new_row = {'RRS_MATE_CODE':np.nan, 'ORDER_AMT':0, 'datetimes':i}
                df = df.append(new_row, ignore_index=True)
        
        df.sort_values(by=['datetimes'], inplace=True)

        if feature == None:
            quantity = df[['datetimes', 'ORDER_AMT']]
            quantity = quantity.groupby(['datetimes']).sum() 
            q = q.to_numpy().fillna(0)
            return q

        array = pd.pivot_table(df, index=feature, columns = 'datetimes', values='ORDER_AMT', aggfunc='sum',dropna=False).fillna(0)
        array = array.to_numpy()
        return array

    def extract_feature(self, interval=1, feature=None):
        if self.RR:
            if feature:
                return np.load("/Users/makotopowers/Desktop/CSUREMM/data/raw/h24_by_sku.npy")

            else:
                return np.load("/Users/makotopowers/Desktop/CSUREMM/data/raw/h24_all_data.npy")
            


        else:
            self.df['hour'] = pd.to_datetime(self.df['order_time']).dt.hour//interval
            self.df['day'] = pd.to_datetime(self.df['order_time']).dt.day-1
            self.df['datetimes'] = (self.df['day'])*(24//interval)  + self.df['hour']

            if feature == None:
                quantity = self.df[['datetimes', 'quantity']]
                quantity = quantity.groupby(['datetimes']).sum()
                return quantity.fillna(0).to_numpy().transpose(1,0)
            else:
                quantity_feature = self.df[['datetimes', feature, 'quantity']]
                array = quantity_feature.pivot_table(index=feature, columns = 'datetimes', values='quantity', aggfunc='sum').fillna(0).to_numpy()
                return array




    def remove_bad_data(self, array):
        indices = []
        for i in range(array.shape[0]):
            if np.sum(array[i]) < 100:
                indices.append(i)
        return np.delete(array, indices, axis=0)
        



    def most_data(self, array):
        most = []
        indices = []
        most_data = []
        for i in range(array.shape[0]):
            most.append(tuple([i, np.sum(array[i])]))
        most = sorted(most, key=lambda x: x[1], reverse=True)
        for u, v in most:
            indices.append(u)
            most_data.append(array[u])
        
        return most_data[:20], indices[:20]




    def get_most(self, array):
        self.remove_bad_data(array)
        most_data, indices = self.most_data(array)
        return most_data, indices
        




