import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as ARIMA


#use numba on loops and numpy functions with @jit(nopython=True)
from numba import jit 


#read in csv file to dataframe
class DataReader:

    
    #constructor
    def __init__(self, file_name):
        self.df = pd.read_csv(file_name)
        #self.df = self.df.reset_index()


    #rename columns
    def rename_columns(self, new_columns):
        self.df = self.df.rename(columns={self.df.columns[0]:new_columns[0],self.df.columns[1]:new_columns[1]})
        print(self.df.columns)
        
    #display data: head, columns, info 
    def display_data(self):
        print(self.df.head(30))
        print(self.df.columns)
        print(self.df.info())
        print(self.df.describe())

    #plot data
    def plot_data(self):
        self.df.plot.scatter(x=self.df.columns[0], y=self.df.columns[2])
        plt.show()

    #drop a column
    def drop_column(self, column):
        self.df = self.df.drop(column, axis=1)

    #trim rows 
    def trim_rows(self,start, end = None):
        self.df = self.df.iloc[start:end]

    #split column into two columns
    def split_column(self, column):
        self.df = self.df.request_time.str.split(expand=True)
        self.df = self.df.drop(1, axis=1)
        self.df.rename(columns={0: column}, inplace=True)
        self.df = self.df.groupby([column])[column].count()

    #save data to csv
    def save_data(self, path):
        self.df.to_csv(path, index=False)

    #merge dataframes on column
    def merge_data(self, column):
        self.df = pd.merge(self.df, self.df, on=column)
        self.df = self.df.drop(1, axis=1)
        self.df = self.df.drop(2, axis=1)
        self.df = self.df.groupby([column])[column].count()
        
    #choose features
    def choose_features(self, features):
        pass


#take two pandas dataframes and join them on a feature 
class FeatureChooser:

    #constructor
    def __init__(self, table_1, table_2):
        self.table_data_1 = table_1.df
        self.table_data_2 = table_2.df
        
    #merge two tables on a column
    def join_tables(self, feature):
        self.joined_table = self.table_data_1.merge(self.table_data_2, on=feature)
        return self.joined_table
    
    #merge two tables on two columns 
    def join_tables(self, left, right):
        joined_df = pd.merge(self.table_data_1,self.table_data_2, how='left',left_on=left,right_on=right)
        self.joined_table = joined_df


#turn pandas dataframe into numpy array and run baseline predictions
class BaselinePredictor:

    #constructor
    def __init__(self, df):
        self.df = df.df
        
    #takes in a feature and returns time series for each value of that feature
    def extract_feature(self, feature):
        seasons = self.df[['order_date', feature]]
        seasons['count'] = [1 for i in range(len(seasons))]
        return seasons.pivot_table(index='order_date', columns = feature, values='count', aggfunc='sum').fillna(0)
    
    def group_sku(self, digits):
        new = self.df 
        new['sku_ID'] = new['sku_ID'][:digits]
        return new 

    def mean(self):
        #implement
        pass

    def median(self):
        #implement
        pass

    def seasonal_median(self):
        #implement
        pass

    def moving_average(self):
        #implement
        pass

    def s_moving_average(self):
        #implement
        pass

    def s_naive(self):
        #implement
        pass
    
    def ets(self):
        #implement
        pass

    def s_arima(self):
        #implement
        pass
        
    def sample_ave_approx(self):
        #implement
        pass



if __name__ == "__main__":
    pass



