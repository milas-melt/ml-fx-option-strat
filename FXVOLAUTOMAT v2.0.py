
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
import scipy.stats as si
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import data_util as du


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """



def optimizeHyperParameters(train_X,train_Y,test_X,test_Y, parameter, algo, cv_folds):
    """
     Optimize the models Hyper Parameter.
     For simplicity we have used only one Hyperparameter for the model to regularize
     input: training and test data set in dataframe format, the algorithm to be applied and the number of folds for cross validation
     output: index of the tuned HyperParameter,training accuracy & test accuracy
    """
    
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    temp = train_X.copy()

    # Initializing variables to store statistics
    rmse_val        = np.empty(len(parameter)) #to store rmse values for different k
    train_accuracy  = np.empty(len(parameter))
    test_accuracy   = np.empty(len(parameter))

    for i, k in enumerate(parameter):
        #print('Running Loop for k:',k)
        acctrain    = []
        acctest     = []
        error       = []
        
        # Setup up the model parameters as per the chosen Algorithm to tune
        if algo     ==  "KNN":
            model                   = KNeighborsClassifier(n_neighbors=k)
            
        elif algo   ==  "Logistic": 
            polynomial_features     = PolynomialFeatures(degree=k)
            train_X                 = polynomial_features.fit_transform(temp)
            model                   = LogisticRegression()
            
        elif algo   ==  "Ridge":
            model                   = RidgeClassifier(k, normalize=True)
            
        elif algo   ==  "SVM":
            model                   = LinearSVC(C=k, multi_class='ovr')
            #model = LinearSVC(C=k, dual=False, fit_intercept=False,intercept_scaling=1, loss='squared_hinge',multi_class='ovr', penalty='l1', random_state=None,verbose=0)
            
        elif algo   ==  "CART":
            model                   = DecisionTreeClassifier(max_depth=k)
            
        elif algo   ==  "Random Forest":
            model                   = RandomForestClassifier(bootstrap=True, criterion='gini',max_depth=k, max_features='auto', \
                                      min_samples_leaf=1, min_samples_split=2, n_estimators=1000, verbose=0,warm_start=False)
            
        else:
            print("Please enter the correct Algorithm to apply")
         
        # Iterate over all the 10 folds for each parameter     
        for train, test in tscv.split(train_X,train_Y):

            model.fit(train_X[train], train_Y[train,0])
            pred    =   model.predict(train_X[test])
            
            # Append the scores to the respective training and test scores list
            acctrain.append(model.score(train_X[train], train_Y[train,0]))
            acctest.append(model.score(train_X[test], train_Y[test,0]))
            
            #calculate rmse
            error.append(sqrt(mean_squared_error(train_Y[test,0],pred))) 


        #Compute accuracy on the training set
        train_accuracy[i]   = np.mean(acctrain)
        
        #Compute accuracy on the Cross Validation set
        test_accuracy[i]    = np.mean(acctest)
        
        #store rmse values
        rmse_val[i]         =   np.mean(error)
        
    # Finding index of the Hyperparameter for which the cross validation set accuracy was the highest
    idx     =   np.argmax(test_accuracy)

    return idx, train_accuracy,test_accuracy      


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """



def plot_parameters_vs_accuracies(parameters, test_accuracy, train_accuracy,x_label,title):
    """
     Plots the Graph of Varying Hyper parameters vs the Training and the Cross Validation Accuracies
     input: Hyperparameter array, training & test accuracies, Label for the X axis and the chart title
     output: prints the graph
    """
    plt.title(title)
    plt.plot(parameters, test_accuracy, label = 'Cross Validation Accuracy')
    plt.plot(parameters, train_accuracy, label = 'Training Accuracy')
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.show()
    
    
""" ******************************************************************************************************* """
""" ******************************************************************************************************* """
    
    

def evaluate_performance(model,pred,train_X,train_Y,test_X,test_Y):
    """
     Evaluates the performance of the ML algorithm
     input: model,pred,train_X,train_Y,test_X,test_Y
     output: RMSE error, train_accuracy,test_accuracy
             prints the Confusion Matrices & the Classification Report
    """    
    #Compute accuracy on the training set
    train_accuracy  = model.score(train_X, train_Y)

    #Compute accuracy on the testing set OOS accuracy of the model
    test_accuracy   = model.score(test_X, test_Y)

    print('Training Accuracy Score for the Optimized model is:', train_accuracy)
    print('OOS Accuracy Score for the Optimized model is:', test_accuracy)

    # Computing the Root Mean squared error for the Test set
    error           = sqrt(mean_squared_error(test_Y,pred)) #calculate rmse
    
    print("Printing Confusion Matrix")
    print(confusion_matrix(test_Y, pred))
    
    print("Printing Classification Report")
    print(classification_report(test_Y, pred))
    
    return error, train_accuracy,test_accuracy


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """


#Fitting kNN model on the training set to identify and optimize Hyperparameter k - no of neighbours 
#using 10 fold cross validation
def knnmodel(train_X,train_Y,test_X,test_Y):
    """
     K - Nearest Neighbour model
     input: training and test data set in dataframe format
     output: no of neighbours, training accuracy & test accuracy
    """

    train_X     = np.array(train_X)
    train_Y     = np.array(train_Y)
    test_X      = np.array(test_X)
    test_Y      = np.array(test_Y)

    neighbors   = np.arange(1, 10)
    algo        = "KNN"
    
    # Optimizing Hyperparameter - k - number of neighbors
    idx, train_accuracy,test_accuracy       =       optimizeHyperParameters(train_X,train_Y,test_X,test_Y, neighbors, algo, cv_folds = 10)

    # Generate plot of Training and Cross Validation Accuracies vs varying Hyperparameter
    plot_parameters_vs_accuracies(neighbors, test_accuracy, train_accuracy,'Number of Neighbors','k-NN: Varying Number of Neighbors')
    
    #Fitting the model on the training set with the optimized K value
    print("Optimized number of Neighbours to fit the final model: ",neighbors[idx])
    knn         =       KNeighborsClassifier(n_neighbors=neighbors[idx])

    # Fit the classifier to the entire training data
    knn.fit(train_X, train_Y)
    
    #Predicting the output from the Test Set
    pred=knn.predict(test_X)

    # Evaluate the performance of the kNN Model
    error, train_accuracy,test_accuracy = evaluate_performance(knn,pred,train_X,train_Y,test_X,test_Y)
    print('RMSE value for k= ' , neighbors[idx] , 'is:', error)
    
    return idx, train_accuracy,test_accuracy , pred


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """

# Implementing Logistic Regression Model
    
def plotROC(test_Y,y_pred_prob):
    """
     Plots the Receiver Operating Characteristics of the probability classifier model
     input: test set Y variable and model prediction probabilities
     output: NIL - ROC Curve
    """
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(test_Y, y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    print("AUC: {}".format(roc_auc_score(test_Y, y_pred_prob)))
    
    
""" ******************************************************************************************************* """
""" ******************************************************************************************************* """



def logisticregression(train_X,train_Y,test_X,test_Y):
    """
     Logistic Regression Model
     input: training and test data set in dataframe format
     output: training accuracy & test accuracy
    """

    train_X     = np.array(train_X)
    train_Y     = np.array(train_Y)
    test_X      = np.array(test_X)
    test_Y      = np.array(test_Y)

    # Setup arrays to store train and test accuracies and defining the range of polynomial function to be used
    poly        = np.arange(1, 3)

    algo        = "Logistic"
    
    # Optimizing Hyperparameter - degree of polynomial fit
    idx, train_accuracy,test_accuracy       =       optimizeHyperParameters(train_X,train_Y,test_X,test_Y, poly, algo, cv_folds = 5)

    # Generate plot of Training and Cross Validation Accuracies vs varying Hyperparameter - degree of polynomial features
    plot_parameters_vs_accuracies(poly, test_accuracy, train_accuracy,'Degree of Polynomial','Logistic Regression: Varying Number of polynomial features')

    #Fitting the model on the training set with the optimized degree value
    print("Optimized degree of polynomial to fit the final model: ",poly[idx])
   
    #Transforming the feature set to the optimized degree 
    polynomial_features     =   PolynomialFeatures(degree=poly[idx])
    trainX_poly             =   polynomial_features.fit_transform(train_X)
    testX_poly              =   polynomial_features.fit_transform(test_X)
    
    # Applying Logistic regression to the transformed predictors 
    reg_all     =   LogisticRegression()
    reg_all.fit(trainX_poly, trainY)
    
    # Predicting in Sample training predictions
    train_predictions   =   reg_all.predict(trainX_poly)
    
    # Predicting Out of SAmple Test Predictions
    test_predictions    =   reg_all.predict(testX_poly)
    y_pred_prob         =   reg_all.predict_proba(testX_poly)[:,1]
        
    print("Plotting the ROC Curve")
    plotROC(test_Y,y_pred_prob)
  
    # Evaluating performance of the Logistic Regression Model
    error, train_accuracy,test_accuracy     =   evaluate_performance(reg_all,test_predictions,trainX_poly,train_Y,testX_poly,test_Y)
    print('RMSE value for polynomial with degree= ' , poly[idx] , 'is:', error)
    
    return train_accuracy,test_accuracy


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """


# Implementing Ridge Regularization 

def ridge_Classification(train_X,train_Y,test_X,test_Y):
    """
     Ridge Regularization Classifier
     input: training and test data set in dataframe format
     output: training accuracy & test accuracy
    """
    
    train_Xi    = np.array(train_X)
    train_Y     = np.array(train_Y)
    test_X      = np.array(test_X)
    test_Y      = np.array(test_Y)

    alpha       = np.arange(0.0000004, 10,0.01)
    
    algo        = "Ridge"
    # Optimizing Hyperparameter - degree of polynomial fit
    idx, train_accuracy,test_accuracy       =       optimizeHyperParameters(train_Xi,train_Y,test_X,test_Y, alpha, algo, cv_folds = 10)

    # Generate plot of Training and Cross Validation Accuracies vs varying Hyperparameter - degree of polynomial features
    plot_parameters_vs_accuracies(alpha, test_accuracy, train_accuracy,'Strength of Hyperparameter L2','Ridge: Varying L2 parameter')

    #Fitting the model on the training set with the optimized L2 value
    ridge       =   RidgeClassifier(alpha[idx], normalize=True)

    # Fit the classifier to the training data
    ridge.fit(train_Xi, train_Y)
    
    # Predicting the dependent variable for the Test set/Quarantine Set
    pred        =   ridge.predict(test_X)
    ridge_coef  =   ridge.coef_    

    # Plot the coefficients
    plt.plot(range(len(train_X.columns)), ridge_coef.T)
    plt.xticks(range(len(train_X.columns)), train_X.columns.values, rotation=60)
    plt.margins(0.02)
    plt.show()
     
    # Evaluating the performance of the Ridge Regularization Classifier
    error, train_accuracy,test_accuracy     =       evaluate_performance(ridge,pred,train_Xi,train_Y,test_X,test_Y)
    print('RMSE value for L1 = ' , alpha[idx] , 'is:', error)
    
    return train_accuracy,test_accuracy


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """


# Implementing Linear Support Vector Machines 

def Linear_SVM(train_X,train_Y,test_X,test_Y):
    """
    Support Vector Machine Classifier
     input: training and test data set in dataframe format
     output: training accuracy & test accuracy
    """
    
    train_X     = np.array(train_X)
    train_Y     = np.array(train_Y)
    test_X      = np.array(test_X)
    test_Y      = np.array(test_Y)
    
    param       = np.arange(0.000004, 0.0005,0.00001)
  
    algo        = "SVM"
    
    # Optimizing Hyperparameter - degree of polynomial fit
    idx, train_accuracy,test_accuracy       =   optimizeHyperParameters(train_X,train_Y,test_X,test_Y, param, algo, cv_folds = 10)

    # Generate plot of Training and Cross Validation Accuracies vs varying Hyperparameter - degree of polynomial features
    plot_parameters_vs_accuracies(param, test_accuracy, train_accuracy,'Strength of Hyperparameter C','Support Vector Machine: Varying Regularization Parameter')

    #Fitting the model on the training set with the optimized C value
#    svm = LinearSVC(C=param[idx], dual=False, fit_intercept=False,intercept_scaling=1, loss='squared_hinge',multi_class='ovr', penalty='l1', random_state=None,verbose=0)
    svm     =   LinearSVC(C=param[idx],  multi_class='ovr')

    # Fit the classifier to the training data
    svm.fit(train_X, train_Y)
    pred=svm.predict(test_X)   

    # Evaluating the performance of the Support Vector Machine Classifier
    error, train_accuracy,test_accuracy     =   evaluate_performance(svm,pred,train_X,train_Y,test_X,test_Y)
    print('RMSE value for L1 = ' , param[idx], 'is:', error)
    
    return train_accuracy,test_accuracy



""" ******************************************************************************************************* """
""" ******************************************************************************************************* """

# Implementin Decision Tree as a Classification problem to predict up or down moves in Y

def CART(train_X,train_Y,test_X,test_Y):
    """
     Decision Tree Classifier
     input: training and test data set in dataframe format
     output: training accuracy & test accuracy
    """

    train_X     = np.array(train_X)
    train_Y     = np.array(train_Y)
    test_X      = np.array(test_X)
    test_Y      = np.array(test_Y)
    
    depth       = np.arange(1,10,1)
   
    algo        = "CART"
    
    # Optimizing Hyperparameter - degree of polynomial fit
    idx, train_accuracy,test_accuracy   =   optimizeHyperParameters(train_X,train_Y,test_X,test_Y, depth, algo, cv_folds = 10)

    # Generate plot of Training and Cross Validation Accuracies vs varying Hyperparameter - degree of polynomial features
    plot_parameters_vs_accuracies(depth, test_accuracy, train_accuracy,'Tree Depth','Regression tree: Varying Depth ')

    #Fitting the model on the training set with the optimized depth
    decision_tree                       = DecisionTreeClassifier(max_depth=depth[idx])

    # Fit the classifier to the training data
    decision_tree.fit(train_X, train_Y)
    # Predicting the dependent variable for the Test set/Quarantine Set
    pred=decision_tree.predict(test_X)

    # Evaluating the performance of the Decision Tree Classifier
    error, train_accuracy,test_accuracy     =   evaluate_performance(decision_tree,pred,train_X,train_Y,test_X,test_Y)
    print('RMSE value for optimized depth of tree: ' , depth[idx] , 'is:', error)

    return train_accuracy,test_accuracy


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """


# Implementin Decision Tree as a Classification problem to predict up or down moves in Y

def RandomForest(train_X,train_Y,test_X,test_Y):
    """
     Random Forest Classifier
     input: training and test data set in dataframe format
     output: training accuracy & test accuracy
    """

    train_X     = np.array(train_X)
    train_Y     = np.array(train_Y)
    test_X      = np.array(test_X)
    test_Y      = np.array(test_Y)
    
    depth       = np.arange(1,10,1)

    algo        = "Random Forest"
    
    # Optimizing Hyperparameter - degree of polynomial fit
    idx, train_accuracy,test_accuracy       =   optimizeHyperParameters(train_X,train_Y,test_X,test_Y, depth, algo, cv_folds = 10)

    # Generate plot of Training and Cross Validation Accuracies vs varying Hyperparameter - tree depth
    plot_parameters_vs_accuracies(depth, test_accuracy, train_accuracy,'Tree Depth','Random Forests: Varying Depth ')

    #Fitting the model on the training set with the optimized depth
    rforest         =       RandomForestClassifier(bootstrap=True, criterion='gini',max_depth=depth[idx], max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=1000, verbose=0,warm_start=False)

    # Fit the classifier to the training data
    rforest.fit(train_X, train_Y)
    pred            =       rforest.predict(test_X)

    # Evaluating the performance of the Random Forest Classifier
    error, train_accuracy,test_accuracy     =       evaluate_performance(rforest,pred,train_X,train_Y,test_X,test_Y)
    print('RMSE value for optimized depth of tree: ' , depth[idx] , 'is:', error)

    return train_accuracy,test_accuracy


""" ******************************************************************************************************* """
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
        

    
    elif method     ==  "RV-IV":
        # Calculating 1M ahead realized volatility for EURUSD and annualising 
        df                          =   pd.DataFrame(df_returns["EURUSD CURNCY"].shift(-(days_in_month-1)))
        df_1MRV                     =   pd.DataFrame(df.rolling(window = days_in_month).std()*np.sqrt(days_in_year), index = df.index, columns = df.columns)*100
        RV_IV                       =   pd.DataFrame(data = (df_1MRV['EURUSD CURNCY'] - df_Implied_vols["EURUSDV1M Curncy"]),columns=["RV-IV"])
        
        Y   =   pd.DataFrame(data = 0,index = RV_IV.index,columns = ["Y"])
        Y[RV_IV>-1] = 1
        Y[RV_IV<-1] = -1
        
#        Y[RV_IV>0] = 1
#        Y[RV_IV<0] = 0
        option = []
        
    return Y, option


""" ******************************************************************************************************* """
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
""" ******************************************************************************************************* """


def plot_pca_components(x,y):
    plt.plot(range(x),y)
    plt.title(" Explained Variance vs No of PCA components")
    plt.xlabel('No of PCA components')
    plt.ylabel('Variance explained')
    plt.show()


""" ******************************************************************************************************* """
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
method = "BS"
Y, option       =   getDependentVariable(method = method)  # menthod BS - Black Scholes option Payoff OR method RV-IV : Realized vol - Implied Vold


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


# creating dataframes to store the training and test performance
training_results        =   pd.DataFrame(data = 0 , index= ["KNN","Logistic","Ridge","CART","SVM","Random Forest"],columns= ["Raw Data","Normalised","PCA 98.6","PCA 91.9","PCA 86.1"])
test_results            =   pd.DataFrame(data = 0 , index= ["KNN","Logistic","Ridge","CART","SVM","Random Forest"],columns= ["Raw Data","Normalised","PCA 98.6","PCA 91.9","PCA 86.1"])

# Running the kNN model on the dataset
#idx861, training_results.loc["KNN","PCA 86.1"],test_results.loc["KNN","PCA 86.1"], predictions["PCA861"]       = knnmodel(pca_train_X[pca_list[0]],trainY,pca_test_X[pca_list[0]],testY)
#idx919, training_results.loc["KNN","PCA 91.9"],test_results.loc["KNN","PCA 91.9"], predictions["PCA919"]       = knnmodel(pca_train_X[pca_list[1]],trainY,pca_test_X[pca_list[1]],testY)
#idx986, training_results.loc["KNN","PCA 98.6"],test_results.loc["KNN","PCA 98.6"], predictions["PCA986"]       = knnmodel(pca_train_X[pca_list[2]],trainY,pca_test_X[pca_list[2]],testY)
#idxX, training_results.loc["KNN","Raw Data"] ,test_results.loc["KNN","Raw Data"], predictions["RawData"]       = knnmodel(trainX,trainY,testX,testY)
idxStdX, training_results.loc["KNN","Normalised"],test_results.loc["KNN","Normalised"], predictions["Norm"]    = knnmodel(train_X_stdz,trainY,test_X_stdz,testY)

# #Running Logistic Regression Model on the dataset
#training_results.loc["Logistic","PCA 86.1"],test_results.loc["Logistic","PCA 86.1"]     = logisticregression(pca_train_X[pca_list[0]],trainY,pca_test_X[pca_list[0]],testY)
#training_results.loc["Logistic","PCA 91.9"],test_results.loc["Logistic","PCA 91.9"]     = logisticregression(pca_train_X[pca_list[1]],trainY,pca_test_X[pca_list[1]],testY)
#training_results.loc["Logistic","PCA 98.6"],test_results.loc["Logistic","PCA 98.6"]     = logisticregression(pca_train_X[pca_list[2]],trainY,pca_test_X[pca_list[2]],testY)
#training_results.loc["Logistic","Raw Data"] ,test_results.loc["Logistic","Raw Data"]    = logisticregression(trainX,trainY,testX,testY)
#training_results.loc["Logistic","Normalised"],test_results.loc["Logistic","Normalised"] = logisticregression(train_X_stdz,trainY,test_X_stdz,testY)
#
# #Running Risge Classifier Model on the dataset
#training_results.loc["Ridge","PCA 86.1"],test_results.loc["Ridge","PCA 86.1"]           = ridge_Classification(pca_train_X[pca_list[0]],trainY,pca_test_X[pca_list[0]],testY)
#training_results.loc["Ridge","PCA 91.9"],test_results.loc["Ridge","PCA 91.9"]           = ridge_Classification(pca_train_X[pca_list[1]],trainY,pca_test_X[pca_list[1]],testY)
#training_results.loc["Ridge","PCA 98.6"],test_results.loc["Ridge","PCA 98.6"]           = ridge_Classification(pca_train_X[pca_list[2]],trainY,pca_test_X[pca_list[2]],testY)
#training_results.loc["Ridge","Raw Data"] ,test_results.loc["Ridge","Raw Data"]          = ridge_Classification(trainX,trainY,testX,testY)
#training_results.loc["Ridge","Normalised"],test_results.loc["Ridge","Normalised"]       = ridge_Classification(train_X_stdz,trainY,test_X_stdz,testY)
#
##Running Support Vector Classifier Model on the dataset
#training_results.loc["SVM","PCA 86.1"],test_results.loc["SVM","PCA 86.1"]               = Linear_SVM(pca_train_X[pca_list[0]],trainY,pca_test_X[pca_list[0]],testY)
#training_results.loc["SVM","PCA 91.9"],test_results.loc["SVM","PCA 91.9"]               = Linear_SVM(pca_train_X[pca_list[1]],trainY,pca_test_X[pca_list[1]],testY)
#training_results.loc["SVM","PCA 98.6"],test_results.loc["SVM","PCA 98.6"]               = Linear_SVM(pca_train_X[pca_list[2]],trainY,pca_test_X[pca_list[2]],testY)
#training_results.loc["SVM","Raw Data"] ,test_results.loc["SVM","Raw Data"]              = Linear_SVM(trainX,trainY,testX,testY)
#training_results.loc["SVM","Normalised"],test_results.loc["SVM","Normalised"]           = Linear_SVM(train_X_stdz,trainY,test_X_stdz,testY)
#
# #Running Decision Tree Model on the dataset
#training_results.loc["CART","PCA 86.1"],test_results.loc["CART","PCA 86.1"]             = CART(pca_train_X[pca_list[0]],trainY,pca_test_X[pca_list[0]],testY)
#training_results.loc["CART","PCA 91.9"],test_results.loc["CART","PCA 91.9"]             = CART(pca_train_X[pca_list[1]],trainY,pca_test_X[pca_list[1]],testY)
#training_results.loc["CART","PCA 98.6"],test_results.loc["CART","PCA 98.6"]             = CART(pca_train_X[pca_list[2]],trainY,pca_test_X[pca_list[2]],testY)
#training_results.loc["CART","Raw Data"] ,test_results.loc["CART","Raw Data"]            = CART(trainX,trainY,testX,testY)
#training_results.loc["CART","Normalised"],test_results.loc["CART","Normalised"]         = CART(train_X_stdz,trainY,test_X_stdz,testY)
#
# #Running Random Forest Classifier Model on the dataset
#training_results.loc["Random Forest","PCA 86.1"],test_results.loc["Random Forest","PCA 86.1"]       = RandomForest(pca_train_X[pca_list[0]],trainY,pca_test_X[pca_list[0]],testY)
#training_results.loc["Random Forest","PCA 91.9"],test_results.loc["Random Forest","PCA 91.9"]       = RandomForest(pca_train_X[pca_list[1]],trainY,pca_test_X[pca_list[1]],testY)
#training_results.loc["Random Forest","PCA 98.6"],test_results.loc["Random Forest","PCA 98.6"]       = RandomForest(pca_train_X[pca_list[2]],trainY,pca_test_X[pca_list[2]],testY)
#training_results.loc["Random Forest","Raw Data"] ,test_results.loc["Random Forest","Raw Data"]      = RandomForest(trainX,trainY,testX,testY)
#training_results.loc["Random Forest","Normalised"],test_results.loc["Random Forest","Normalised"]   = RandomForest(train_X_stdz,trainY,test_X_stdz,testY)


# Storing the predictions for the Out of Sample Test predictions from the kNN Normalised model
df_Signal               =       pd.DataFrame(data = predictions["Norm"] ,index = testY.index, columns = ["Signal kNN Norm"])
#df_Signal[df_Signal<1]  =       -1

# Resampling the signal to Business month
df_Signal_Monthly        =       df_Signal.resample("BM").last()

if method == "BS":
    # Reindexing the Option Payoffs i.e. Option Payoff - ATM Straddle cost = New payoff
    df_Return               =       pd.DataFrame(data = option["Payoff"].reindex(df_Signal_Monthly.index))
    weightdRet              =       df_Signal_Monthly["Signal kNN Norm"].shift(1)*df_Return["Payoff"]
    
    
    # Calculating Performance statistics for the strategy return series 
    wRetmu,wRetstd,sharpe,skew_portfolio,kurt_portfolio,max_m_loss      =       du.monthly_performance_stats(weightdRet)
    
    # Plotting Return/Cumulative returns and drawdowns using functions imported from data_util.py (utilities script)
    mdd                     =       du.maximumDrawdown(weightdRet)
    terminal                =       du.printingStatistics(weightdRet,wRetmu,wRetstd,sharpe,skew_portfolio,kurt_portfolio,max_m_loss,mdd)




"""""""""""""""""""""""""""""""""""""""END OF MAIN"""""""""""""""""""""""""""""""""""""""
