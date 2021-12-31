import numpy as np
import matplotlib.pyplot as plt

class TimeSeries:
    
    """
    This class initializes our dataset and also allows us to plot the main variables of our dataset.
    The methods of this class calculates the moving average along with the regression coefficients. 
    Using some methods described below, we will be able to create a model that will allow us to predict
    future observations of our variable of interest
    
    Attributes
    --------------
    feature: float numpy array
                    This array will have data on montly usage of WiFi from 2011-2014
    
    feature_name: str
                name of the feature; in this case- monthly usage of WiFi
    """
    
    def __init__(self, filename,column,feature_name):
        array_library=np.loadtxt(filename,dtype=str,delimiter=",", skiprows=1,usecols=column)# we don't necessarily need month and year

        self.feature=array_library.astype(np.float)#converting str numpy array into float numpy array
        self.feature_name=feature_name
    
    
    def plot_monthly(self):
        """Plot monthly usage of WiFi Cummulative usage of WiFi
        
        In this method, we plot the monthly usage of WiFi and the cummulative usage of WiFi
        """
        fig,ax=plt.subplots()
        ax.plot(self.feature)
        ax.set(title="Monthly WiFi usage in the library",
        xlabel="Time",
        ylabel=self.feature_name)
        
    def moving_average(self,M):
        """Calculating Moving Average
        
        In this method, we calculate the moving average array and also plot it along with our original
        datapoints
        
        Parameters
        ------------
        M: int
            time steps for which we want to consider while calculating the moving average
            
        Returns
        ------------
        moving_average: float numpy array
                        this contains the moving average corresponding to each time step
        
        time_step: int numpy array
                    this contains the time steps for which we can calculate the moving average given a 
                    specific M
        
        main_trimmed: float numpy array
                    this contains the monthly WiFi usage for all the time units that have not been 
                    trimmed. So if M=3, MAIN_TRIMMED will have data of WiFi usage from second month 
                    till (n-1)th month
                    
        self.feature: float numpy array
                        this contains complete data of monthly usage of WiFi
        """
        
        length=len(self.feature)#length of our main dataset
        m=0
        master=np.zeros(length-M+1)#intializing the dataset where we want to store our moving average
        master=np.reshape(master,(-1,1))
        x=0
        y=M-1
        
        while(m<M):
            to_add=self.feature[x:length-y]
            to_add=np.reshape(to_add,(-1,1))
            master=master+to_add
            x+=1
            y-=1
            m+=1
        
        #MASTER contains the sum of previous and the next observations. So if M=3, The first observation \n
        # in MASTER will be the sum of 1st, 2nd, and 3rd observation in the original dataset
        moving_average=master/M
        k=int((M-1)/2)
        time_step=np.arange(k+1,length-k+1,1)
        time_step=np.reshape(time_step, (-1,1))
        
        #MAIN_TRIMMED is an array of the original dataset but with data only for the valid time steps
        main_trimmed=self.feature[k:length-k]
        main_trimmed=np.reshape(main_trimmed, (-1,1))
        
        fig, ax=plt.subplots()
        ax.plot(time_step,moving_average,color='orange',linestyle='--', label="Moving Average")
        ax.plot(time_step,main_trimmed,color='purple',linestyle=':', label=self.feature_name)
        ax.set_xticks(np.arange(min(time_step), max(time_step)+1, 2))

        ax.legend()
        ax.set(title='Monthly usage and Moving Average where M=%i' %M,
                xlabel="Time",
                ylabel="Usage")
        
        return(moving_average, time_step, main_trimmed, self.feature)
    
    
        
    def without_intercept(self, moving_average, time_step):
        """Calculating the slope
        
        In this method, we regress time_step on moving_Average and calculate the slope coefficient 
        along with Residual.
        
        Parameters
        --------------
        moving_average: float numpy array
                        contains moving average for all the valid timesteps
        
        time_step: int numpy array
                    contains all the valid time steps
                    
        Returns
        --------------
        BETA: float numpy array
                by construction, this will have only one value which is the regression coefficient
                
        R: float numpy array
            this contains the squared residual between predited annd actual values
        """
        BETA,R, _, _ = np.linalg.lstsq(time_step, moving_average, rcond=None)
        
        return BETA, R
    
        
    def with_intercept(self,moving_average,time_step):
        """Calculating intercept and slope
        
        In this method we regress time step on moving average and calculate the slope and intercept
        along with the squared residual
        
        Parameters
        --------------
        moving_average: float numpy array
                        contains moving average for all the valid timesteps
        
        time_step: int numpy array
                    contains all the valid time steps
                    
        Returns
        --------------
        BETA: float numpy array
                by construction, this will have only two value which is the regression coefficient
                
        R: float numpy array
            this contains the squared residual between predited annd actual values
        """
        length=len(moving_average)
        a=np.ones(length)
        time_step=np.insert(time_step,0,a,axis=1)#to calculate the intercept, we include a column of 1's in our X matrix
        
        BETA, R, _, _ = np.linalg.lstsq(time_step,moving_average, rcond=None)
        return BETA, R
        
        
    def residual_only_slope(self,beta,moving_average,time_step):
        """Customized function for calculating residuals using Beta
        
        In this method, we do a few numpy operations to estimate  predicted y and further calculate 
        squared residual
        
        Parameters
        -------------
        beta: float
            the slope that we get on regressing time_step on moving_average without the intercept
        
        moving_average: float numpy array
                        contains moving average for all the valid timesteps
        
        time_step: int numpy array
                    contains all the valid time steps
                    
        Returns
        -------------
        summed: float
                sum of squared residual
        """
        y_cap=time_step*beta
        res_unsquared=y_cap-moving_average
        res_squared=np.square(res_unsquared)
        summed=np.sum(res_squared)
        return summed
        
    
    def residual_with_intercept(self,beta,moving_average,time_step):
        """Customized function for calculating residuals using Beta
        
        In this method, we do a few numpy operations to estimate  predicted y and further calculate 
        squared residual
        
        Parameters
        -------------
        beta: float
            the slope that we get on regressing time_step on moving_average with the intercept
        
        moving_average: float numpy array
                        contains moving average for all the valid timesteps
        
        time_step: int numpy array
                    contains all the valid time steps
                    
        Returns
        -------------
        summed: float
                sum of squared residual
        """
        intercept=beta[0]
        slope=beta[1]
        y_cap=intercept+time_step*slope
        res_unsquared=y_cap-moving_average
        res_squared=np.square(res_unsquared)
        summed=np.sum(res_squared)
        return summed
    


    
    