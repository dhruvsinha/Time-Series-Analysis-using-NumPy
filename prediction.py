import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class Prediction:
    """
    This class initializes the dataset and also allows us to standardize our main features and target
    variables. The methods in this class will also calculate the regression coefficients and squared
    residuals
    
    Attributes:
    --------------
    array: float numpy array
            contains all the feature variables and the target variable
    
    BETA: LIST
             we will store the (intercept, slope) combination for each feature in this list
    
    RESIDUAL: LIST
            we will store the squared residual for each variable in this list
            
    feature_names: LIST
            name of each feature
            
    target_name: str
            name of the target variable
            
    feature_array: float numpy array
            data for each feature is in this array
            
    target_array: float numpy array
            data for target variable is in this array
    """
    
    def __init__(self, filename,columns,feature_names, target_name):
        
        array=np.loadtxt(filename,dtype=str,delimiter=",", skiprows=1,usecols=columns)
        rows, cols = array.shape
        self.array=np.empty((rows,cols), float)
        
        #cleaning the column and converting them into float
        for col in range(cols):
            column=array[:,col]#extract the column
            column[column=='']='0'#fill empty spaces with 0
            column=column.astype(np.float)#converting string columns to float
            column=np.reshape(column, (-1,1))
            self.array=np.append(self.array,column,axis=1)#adding the new column to another array
            
        self.BETA=[]
        self.RESIDUAL=[]
        
        self.feature_names=feature_names
        self.target_name=target_name
        self.array=np.delete(self.array,np.s_[0:len(columns)], axis=1)#the first 7 columns are empty; so delete
        self.feature_array=np.delete(self.array,np.s_[len(columns)-1], axis=1)#the last column is our target variable
        self.target_array=self.array[:,len(columns)-1]
        

    def plot_relation(self):
        """Plot relationship between feature variable and the target variable
        
        This methods plots relationship between each of our features and our target variable- median house
        value
        """
        _,num_cols=self.feature_array.shape
        for col in range(num_cols):
            x=self.feature_array[:,col]
            y=self.target_array
            fig,ax=plt.subplots()
            ax.scatter(x,y,s=5)
            ax.set(title="Relationship between {} and {}".format(self.feature_names[col],self.target_name),
               xlabel=self.feature_names[col],
               ylabel=self.target_name)
            
    def standardize(self):
        """Standardize each variable
        
        In this method, we standardize (or normalize) feature and target variables
        """
        standardized_features=np.empty((20640,6), float)
        _,num_cols=self.feature_array.shape
        
        for col in range(num_cols):
            x=self.feature_array[:,col]
            x=np.reshape(x,(-1,1))
            x_mean=np.mean(x)
            x_std=np.std(x)
            x=(x-x_mean)/x_std
            standardized_features=np.append(standardized_features,x,axis=1)
        
        self.feature_array=np.delete(standardized_features,np.s_[0:6], axis=1)
        target_mean=np.mean(self.target_array)
        target_std=np.std(self.target_array)
        self.target_array=(self.target_array-target_mean)/target_std
        
        
    def best_fit(self):
        """
        In this method we regress our target variable on each feature and store their regression coefficients
        We use these regression coefficients to calculate the residual. The variable with the minimum 
        residual is our variable of interest
        
        Returns
        -----------
        minimum_res: str
                    The name of the variable that has minimum residual
                    
        BETA: LIST
            this list contains (intercept, slope) combination for each variable
            
        feature_names: LIST
                        this list contains the name of all the features 
        """
        length=len(self.target_array)
        ones=np.ones(length)
        _,num_cols=self.feature_array.shape
            
        for col in range(num_cols):
            x=self.feature_array[:,col]
            x=np.reshape(x,(-1,1))
            x=np.insert(x,0,ones,axis=1)# adding a column of one's as we are interested in intercepts too
            beta,_,_,_=np.linalg.lstsq(x,self.target_array,rcond=None)
            self.BETA.append(beta)
            
        # calculating Residual using the regression coefficients
        for col in range(num_cols):
            beta=self.BETA[col]
            x=self.feature_array[:,col]
            betanot=beta[0]
            betaone=beta[1]
                
            y_cap=betanot+betaone*x
            res_unsquared=y_cap-self.target_array
            res_squared=np.square(res_unsquared)
            summed=np.sum(res_squared)
            self.RESIDUAL.append(summed)
                
        minimum_res=self.feature_names[self.RESIDUAL.index(min(self.RESIDUAL))]
        return (minimum_res, self.BETA, self.feature_names)
        
            
            
    def multivariate(self):
        """
        In this method, we calculate the result of multi-feature regression and also do some hypothesis
        testing. We also square median housing age to capture if housing age has quadratic relationship
        with our target variable
        
        Returns
        -----------
        beta_round: LIST
                    list of all regression coefficients for each feature in multi-feature regression
                    
        regressor_OLS.summary(): function with some information
                                this is a table that has information on p-value
        """
        length=len(self.target_array)
        ones=np.ones(length)
        multivariate=np.insert(self.feature_array,0,ones,axis=1)# we also want the intercept so we add a column of 1's in our dataset
        age=self.feature_array[:,0]
        age_squared=np.square(age)
        multivariate=np.insert(multivariate,7,age_squared,axis=1)#squaring the age variable and including it in our list of features
        beta,a,b,c=np.linalg.lstsq(multivariate,self.target_array,rcond=None)
        beta_round=[round(num,4) for num in beta]
        
        y=self.target_array
        regressor_OLS = sm.OLS(endog = y, exog = multivariate).fit()
        return (beta_round, regressor_OLS.summary())
        

