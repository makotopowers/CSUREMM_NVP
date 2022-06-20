import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.arima.model as ARIMA
from scipy.stats import norm 
from icecream import ic
import time


class methods:

    def __init__(self):
        string = "Initializing Class. \nHello there. How are you?\n"
        for char in string:
            print(char, end="")
            time.sleep(0.03)
        
    def moving_average(self):
        string = "Applying benchmark [MOVING AVERAGE]."
        for char in string:
            print(char, end="")
            time.sleep(0.03)
            #implement
            pass

    def s_moving_average(self):
        #implement
        pass

    def s_naive(self, season_array, time, interval):
        return season_array[time-1]

    def ets(self):
        #implement
        pass

    def s_arima(self, season_array, time, interval = 5):
        #print(season_array)
        #print(season_array[:time])
        mod = ARIMA.ARIMA(season_array[:time], order=(0,1,0)).fit()
        pred = mod.predict(start=time, end=time)
        print(pred)
        return pred


    def mean(self):
        #implement
        pass

    def median(self):
        #implement
        pass


if __name__=="__main__":
    m = methods()
    m.moving_average()