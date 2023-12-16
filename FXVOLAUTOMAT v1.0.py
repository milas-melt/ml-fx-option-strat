import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as si


""" ******************************************************************************************************* """

def euro_vanilla(S, K, T, rd,rf, sigma, option = 'call'):
    """
     Black Scholes Model to evaluate FX option Pricing
     input:     
         #S: spot price
         #K: strike price
         #T: time to maturity
         #r: interest rate
         #sigma: volatility of underlying asset
         # option - flag variable "call" or "put"
     output: Option Price in %
    """

    d1      =   (np.log(S / K) + (rd - rf + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2      =   d1 - (sigma * np.sqrt(T))
   
    if option   == 'call':
        phi     = 1
        
    elif option   == 'put':
        phi       = -1
        
    result      =   phi*(S *np.exp(-rf * T)* si.norm.cdf(phi*d1, 0.0, 1.0) - K * np.exp(-rd * T) * si.norm.cdf(phi*d2, 0.0, 1.0))
    
    return result


""" ******************************************************************************************************* """




def getData():
    """
     Reads all the data from the Data file and constructs df_main which holds all the predictors
     input: nothing - Make sure you have the .xlsx file in the same folder as this .py file
     output: 
         df_main- contains all the predictors
         df_spot- Spot price for all the currencies
         df_eurfwd- 1M EUR FWD rates
         df_Implied_vols- ATM Implied Volatilities for all the FX pairs
         df_deporates- 3M deposit rates for the given currencies
         df_realized_vol - 2 month realized vols for all the currencies in df_spot
    """
    
    sheetname       = ["FX SPOT", "ATM VOLS","3M 25D RR", "3M DEPOSIT RATES","10Y YIELD","EQUITY INDICES","COMDTY","CREDIT SPREADS","IMM POSITIONING"]
    levels          = ["ATM VOLS", "3M 25D RR"]
    filename        = "DataTables.xlsx"
    
    # Dataframe of Spot prices
    df_spot         = pd.read_excel(filename,"FX SPOT",parse_dates=True, index_col='Dates')
    df_spot         = df_spot[df_spot.index.dayofweek < days_in_week]
    
    # Dataframe of EURUSD 1M forward rates prices
    df_eurfwd       = pd.read_excel(filename,"1M EUR FWD",parse_dates=True, index_col='Dates')
    df_eurfwd       = df_eurfwd[df_eurfwd.index.dayofweek < days_in_week]
    
    # Dataframe of 1M Implied Volatilities
    df_Implied_vols = pd.read_excel(filename,"ATM VOLS",parse_dates=True, index_col='Dates')
    df_Implied_vols = df_Implied_vols[df_Implied_vols.index.dayofweek < days_in_week]
    
    # Dataframe of Deposit rates
    df_deporates    = pd.read_excel(filename,"3M DEPOSIT RATES",parse_dates=True, index_col='Dates')
    df_deporates    = df_deporates[df_deporates.index.dayofweek < days_in_week]
    
    #df_main holds all the data - predictors all 373 of them
    df_main         = pd.DataFrame(index = df_spot.index)
    
    # Calculating spot returns to be further used in calculating 2M realized volatilities
    df_returns              = df_spot.pct_change()
    df_realized_vol         = pd.DataFrame(df_returns.rolling(window = days_in_month*2).std()*np.sqrt(days_in_year), index = df_spot.index, columns = df_spot.columns).shift(1)
    df_realized_vol.columns = [str(col) + 'Vol2M' for col in df_realized_vol.columns]
    
    
    # Calculating 1W change in realized Volatilities
    df_1W_vol_per_change            = (df_realized_vol.astype(float) / df_realized_vol.astype(float).shift(days_in_week) - 1) 
    df_1W_vol_per_change.columns    = [str(col) + '1W' for col in df_1W_vol_per_change.columns]
    
    
    # Calculating 1month change in realized Volatilities
    df_1M_vol_per_change            = (df_realized_vol.astype(float) / df_realized_vol.astype(float).shift(days_in_month) - 1) 
    df_1M_vol_per_change.columns    = [str(col) + '1M' for col in df_1M_vol_per_change.columns]
    
    
    # Adding the Volatilite, 1W change in vols and 1M change in realized vols to the master dataframe
    df_main     =   df_main.join(df_realized_vol)
    df_main     =   df_main.join(df_1W_vol_per_change)
    df_main     =   df_main.join(df_1M_vol_per_change)
    
    
    #Looping through all the sheets and individual predictors to calculate 1week and 1month change 
    #and joining them in the Master dataframe - df_main
    for sheet in sheetname:
        df      =       pd.DataFrame()
        df      =       pd.read_excel(filename,sheet,parse_dates=True, index_col='Dates')
        df      =       df[df.index.dayofweek < days_in_week] # removing all the weekend dates from the dataset
    
        if sheet in levels:
            df_main     =       df_main.join(df.shift(1))
        
        print("Reading sheet", sheet)
        df_1W_per_change            = (df.astype(float) / df.astype(float).shift(days_in_week) - 1) 
        df_1W_per_change.columns    = [str(col) + '1W' for col in df_1W_per_change.columns]
        df_1M_per_change            = (df.astype(float) / df.astype(float).shift(days_in_month) - 1) 
        df_1M_per_change.columns    = [str(col) + '1M' for col in df_1M_per_change.columns]
        
        df_main         =   df_main.join(df_1W_per_change.shift(1))
        df_main         =   df_main.join(df_1M_per_change.shift(1))
        
    print("Reading sheet JPM EASI")
    df_easi         =   pd.read_excel(filename,"JPM EASI",parse_dates=True, index_col='Dates')
    df_easi         =   df_easi[df_easi.index.dayofweek < days_in_week]
    df_easi.fillna(0, inplace = True)
    
    # JPM EASI is an index value between -100 to +100, so we have divided by total range (200) to find out change in 1W and 1M
    df_easi_1W          = (df_easi.astype(float) - df_easi.astype(float).shift(days_in_week))/ 200
    df_easi_1W.columns  = [str(col) + '1W' for col in df_easi_1W.columns]
    df_easi_1M          = (df_easi.astype(float) - df_easi.astype(float).shift(days_in_month))/200 
    df_easi_1M.columns  = [str(col) + '1M' for col in df_easi_1M.columns]
    df_main             = df_main.join(df_easi_1W.shift(1))
    df_main             = df_main.join(df_easi_1M.shift(1))
    
    return df_main, df_spot, df_eurfwd, df_Implied_vols, df_deporates, df_realized_vol, df_returns



""" ******************************************************************************************************* """


def getDependentVariable(method = "BS"):
    """
     Creates the dependent variable "Y" from the given data set
     input: uses global variables returned from the getData() function
            method - "BS" - Black Scholes Model is used to form Y
                        Calculate the price of ATM straddle and subtract this from the option payoff 
                   - "RV-IV" - compute the 1 month ahead realized volatility and subtract the Implied Volatility from this
     output: 
             Y: 1/-1 boolean variable depending upon whether the strategy returned a positive payoff or not
    """
    
    if method   ==  "BS":
        #df_Y holds all the variables to be used to calculate option's price
        df_Y    =   pd.DataFrame(index = df_spot.index)
        df_Y    =   df_Y.join(df_spot["EURUSD CURNCY"])
        df_Y    =   df_Y.join(df_eurfwd["EUR1M BGN Curncy"])
        df_Y    =   df_Y.join(df_deporates["EUDRC CURNCY"])
        df_Y    =   df_Y.join(df_deporates["USDRC CURNCY"])
        df_Y    =   df_Y.join(df_Implied_vols["EURUSDV1M Curncy"])
        df_Y.dropna(inplace = True)
        
        # Calculating price of the call and puts and then adding value to calculate value of the ATM straddle
        option                  =   pd.DataFrame()
        option["Put"]           =   euro_vanilla(df_Y["EURUSD CURNCY"], df_Y["EUR1M BGN Curncy"], days_in_month/days_in_year, df_Y["USDRC CURNCY"]/100,df_Y["EUDRC CURNCY"]/100, df_Y["EURUSDV1M Curncy"]/100, option = 'put')/df_Y["EUR1M BGN Curncy"]
        option["Call"]          =   euro_vanilla(df_Y["EURUSD CURNCY"], df_Y["EUR1M BGN Curncy"], days_in_month/days_in_year, df_Y["USDRC CURNCY"]/100,df_Y["EUDRC CURNCY"]/100, df_Y["EURUSDV1M Curncy"]/100, option = 'call')/df_Y["EUR1M BGN Curncy"]
        option["ATM Straddle"]  =   option["Call"] + option["Put"]
        
        # Calculating % gain in the price from the 1M ahead spot to current Strike price to calculate in the monneyness
        option["Option ITM"]    =   (df_Y["EURUSD CURNCY"].shift(-days_in_month)-df_Y["EUR1M BGN Curncy"])/df_Y["EUR1M BGN Curncy"]
        option["Option ITM"]    =   option["Option ITM"].abs()
        option["Payoff"]        =   option["Option ITM"] - option["ATM Straddle"]

        Y   =   pd.DataFrame(data = 0, index=option.index , columns=["Y"])
        Y[option["Payoff"]>-(.3/100)] = 1
        Y[option["Payoff"]<-(.3/100)] = -1
        
 #      Y[option["Payoff"]>0] = 1
 #      Y[option["Payoff"]<0] = 0
    
    elif method     ==  "RV-IV":
        # Calculating 1M ahead realized volatility for EURUSD and annualising 
        df                          =   pd.DataFrame(df_returns["EURUSD CURNCY"].shift(-(days_in_month-1)))
        df_1MRV                     =   pd.DataFrame(df.rolling(window = days_in_month).std()*np.sqrt(days_in_year), index = df.index, columns = df.columns)*100
        RV_IV                       =   pd.DataFrame(data = (df_1MRV['EURUSD CURNCY'] - df_Implied_vols["EURUSDV1M Curncy"]),columns=["RV-IV"])
        
        Y   =   pd.DataFrame(data = 0,index = RV_IV.index,columns = ["Y"])
        Y[RV_IV>-1] = 1
        Y[RV_IV<-1] = -1
        option = []
        
    return Y, option



""" ******************************************************************************************************* """


def traintestsplit(X,Y,split):
    """
     Splits the data set into training & Quarantine/test set from the given dataframes X & Y
     input: X & Y Dataframes as formed in the main, 
            split - the % split between training & test/quarantine dataset
     output: trainX,testX, trainY, testY

    """
    trainX      =   X.iloc[0:round(split*X.shape[0]),:]
    testX       =   X.iloc[round(split*X.shape[0]):,:]
    trainY      =   Y.iloc[0:round(split*Y.shape[0]),:]
    testY       =   Y.iloc[round(split*Y.shape[0]):,:]
    
    return trainX,testX, trainY, testY


""" ******************************************************************************************************* """

def standardize(trainX,testX):
    """
     Standardizing the data set
     input: trainX & testX - dataframes containing the predictors to be standardized
     output: train_X_stdz,test_X_stdz - standardized predictors
    """
    scaler          =   StandardScaler()

    # Fit on training set only.
    scaler.fit(trainX)

    # Apply transform to both the training set and the test set.
    train_X_stdz    =   pd.DataFrame(scaler.transform(trainX))
    test_X_stdz     =   pd.DataFrame(scaler.transform(testX))
    
    return train_X_stdz,test_X_stdz


""" ******************************************************************************************************* """

def plot_pca_components(x,y):
    plt.plot(range(x),y)
    plt.title(" Explained Variance vs No of PCA components")
    plt.xlabel('No of PCA components')
    plt.ylabel('Variance explained')
    plt.show()

""" ******************************************************************************************************* """


def model_pca(variance):
    
    # Using PCA for dimension reductionality of the 373 features
    # Finding principal components which explain "variance" of the variability of the features as input to Machine Learning models
    pca             = PCA(variance)
    pca.fit(train_X_stdz) 
    
    print('Number of reduced components which explain '+ str(variance*100)+'% of the feature set:'+str(pca.n_components_))
    #print('Variance explained by each component of the Principal Components: ',pca.explained_variance_ratio_)
    plot_pca_components(pca.n_components_,pca.explained_variance_ratio_)
    pca_train_X     = pd.DataFrame(pca.transform(train_X_stdz))
    pca_test_X      = pd.DataFrame(pca.transform(test_X_stdz))
    
    return pca_train_X, pca_test_X




""" ******************************************************************************************************* """


"""""""""""""""""""""""""""""""""""""""START OF MAIN"""""""""""""""""""""""""""""""""""""""
split           = 0.8 # denominate the train test split ratio
days_in_month   = 22
days_in_week    = 5
days_in_year    = 252
predictions     = {}
pca_list        = [0.861, 0.919, 0.986]
pca_train_X     = {}
pca_test_X      = {}

# Reading data from variable file
df_main, df_spot, df_eurfwd, df_Implied_vols, df_deporates, df_realized_vol, df_returns     =       getData()

# Creating the dependent variable
Y, option       =   getDependentVariable(method = "BS")  # method BS - Black Scholes option Payoff OR method RV-IV : Realized vol - Implied Vold


# joining the Y dataframe to master dataframe so that all the indices of the X and Y data match
data            =   Y.join(df_main)
data_monthly    =   data.resample("BM").last()

# Selecting the time period to run the data set on
data            =   data["2007":"2016"]

# Segragating X and Y form the dataframe "data" for further calculations
#Y = pd.DataFrame(data_monthly.iloc[:,0])
Y               =   pd.DataFrame(data.iloc[:,0])

#X = data_monthly.iloc[:,1:]
X               =   data.iloc[:,1:]
X.replace([np.inf, -np.inf], np.nan, inplace = True)
X.fillna(0, inplace = True)

# Creating the train test split
trainX,testX, trainY, testY     =   traintestsplit(X,Y,split)

# Standardizing predictors
train_X_stdz,test_X_stdz        =   standardize(trainX,testX)

# creating the principal components
for item in pca_list:
    pca_train_X[item], pca_test_X[item] = model_pca(item)


"""""""""""""""""""""""""""""""""""""""END OF MAIN"""""""""""""""""""""""""""""""""""""""