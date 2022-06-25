import numpy as np
import pandas as pd
from numba import jit

class Data:
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)

    def extract_feature(self, interval=1, feature=None):
        
        self.df['hour'] = pd.to_datetime(self.df['order_time']).dt.hour//interval
        self.df['day'] = pd.to_datetime(self.df['order_time']).dt.day-1
        self.df['datetimes'] = (self.df['day'])*(24//interval)  + self.df['hour']

        if feature == None:
            quantity = self.df[['datetimes', 'quantity']]
            quantity = quantity.groupby(['datetimes']).sum()
            return quantity.fillna(0).to_numpy().transpose(1,0)
    
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
        




