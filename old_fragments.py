def moving_average(self):
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

def mse(self, losses):
    total = 0
    for loss in losses:
        total += loss**2

    return total / len(losses)

def estimates (self, season_array, function, interval=1):
        estimates = []
        period = 1

        while period < len(season_array)-1:
            estimate = function(season_array, period, interval)
            period+=1
            estimates.append(estimate)
            
        return estimates

def loss_sequence(self, season_array, estimates, function = None, interval=1):
    #estimate = None
    loss = [0]
    #estimates = []
    period = 1
    
    
    while period < len(season_array)-1:
        loss.append(estimates[0][period-1] - season_array[period])
        period+=1 
        '''
        estimate = function(season_array, period, interval)
        period+=1
        estimates.append(estimate)
        loss.append(estimate-season_array[period])
        '''

    return loss



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
