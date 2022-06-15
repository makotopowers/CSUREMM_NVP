import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as ARIMA
from icecream import ic


#use numba on loops and numpy functions with @jit(nopython=True)
from numba import jit
from sympy import true 

        
#turn pandas dataframe into numpy array and run baseline predictions
class BaselinePredictor:


    #constructor
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)


    def extract_feature(self, bucket_length=1, feature=None, all=False):
        '''
        bucket_length: length of time step. feature is the feature to be bucketed.
        all: if true, function returns total order data.
        
        '''
        
        self.df['hour'] = pd.to_datetime(self.df['order_time']).dt.hour
        self.df['day'] = pd.to_datetime(self.df['order_time']).dt.day
        self.df['hour'] = self.df['hour']//bucket_length
        self.df['datetimes'] = (self.df['day']-1)*(24//bucket_length)  + self.df['hour']
        

        if all == True:
            seasons = self.df[['datetimes', 'quantity']]
            seasons = seasons.groupby(['datetimes']).sum()
            return seasons.fillna(0).to_numpy().transpose(1,0)
    
        seasons = self.df[['datetimes', feature, 'quantity']]
        array = seasons.pivot_table(index=feature, columns = 'datetimes', values='quantity', aggfunc='sum').fillna(0).to_numpy()
        return array


    def remove_bad_skus(self, season_array):
        indices = []
        for i in range(season_array.shape[0]):
            if np.sum(season_array[i]) < 100:
                indices.append(i)
        return np.delete(season_array, indices, axis=0)



    def generate_figures(self, season_array, indices, directory, output_dir='/Users/makotopowers/Desktop/CSUREMM/'):
        for i in indices:
            plt.plot(season_array[i], label=i)
            plt.savefig(output_dir+directory+'/'+str(i)+'.png')
            plt.close()
        

    def most_data(self, array):
        most = []
        for i in range(array.shape[0]):
            most.append(tuple([i, np.sum(array[i])]))
        indices = []
        most = sorted(most, key=lambda x: x[1], reverse=True)
        for u, v in most:
            indices.append(u)
        
        return most[:20], indices[:20]

    def group_sku(self, digits):
        new = self.df 
        new['sku_ID'] = new['sku_ID'][:digits]
        return new 


    def seasonal_median(self, season_array, time, interval):
        return np.median(season_array[:time])

    
        
    def SAA(self, array, overage, underage, sample):
        q = underage / (overage + underage) 
        
        return sorted(array[:30])[int(np.ceil(q*sample))]

        

if __name__ == "__main__":
    #Read in data)

    make = BaselinePredictor("/Users/makotopowers/Desktop/CSUREMM/data/raw/JD_order_data.csv")
    season_arrays = make.extract_feature(bucket_length=24, all=True)
    #print(season_arrays.shape)
    #print(np.sum(season_arrays))

    #new = make.remove_bad_skus(season_arrays)
    #print(np.sum(new))
    #ic(new.shape)
    #print(new)
    ic(type(season_arrays))
    ic(season_arrays.shape)
    most, indices = make.most_data(season_arrays)
    ic(season_arrays)
    
    #make.generate_figures(new, indices, 'bucket_24')
    #print(make.most_data(new))
    print(make.SAA(season_arrays[0], 0.5, 0.5, 30))
    



