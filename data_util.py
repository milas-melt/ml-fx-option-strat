#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:40:05 2019

"""
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """

def printingStatistics(df,wRetmu,wRetstd,sharpe,skew_portfolio,kurt_portfolio,max_m_loss,mdd):
    
    df          = pd.DataFrame(df, index = df.index)
    dduration   = pd.DataFrame(df, index = df.index)
    returns     = pd.DataFrame(data = df)
    df          = 1 + df

    # Creating empty dataframes         
    data1 = pd.DataFrame(index  = df.index) 
    data2 = pd.DataFrame(index  = df.index) 

    # Calculating Cumulative Return series
    data1['Cumulative']     = df.cumprod()
    
    # Calculating High Watermark
    data1['HWM']            = data1['Cumulative'].cummax()
    
    # Calculating Drawdown
    data2['Drawdown']       = data1['Cumulative']/data1['HWM'] - 1
    data2['Returns']        = returns
        
    # Calculating maximum of all the DDs to calculate Max DD
    
    #Dissecting investment strategies in the cross section and time series 
        
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Returns & HWM')
    ax1.plot(df.index, data1, label = data1.columns)
    ax1.tick_params(axis='y')
    ax1.set_title('Cumulative Returns & HWM: ')

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Drawdowns & Returns')
    ax1.plot(df.index, data2,label =  ['Drawdowns','Returns'])
    ax1.tick_params(axis='y')
    ax1.set_title('Drawdowns & Returns: ')
    
    print("\n\n")
    print("Printing Strategy Statistics:")
    print("\n")

    print("The Returns of the Strategy is " + str(round(wRetmu*100,4))+"%")
    print("The Volatility of the Strategy is " + str(round(wRetstd*100,4))+"%")
    print("The Sharpe Ratio of the Strategy is " + str(round(sharpe,4)))
    print("The Maximum weekly Loss of the Strategy is " + str(round(max_m_loss*100,4))+"%")
    print("The Skew of the Strategy is " + str(round(skew_portfolio,4)))
    print("The Kurtosis of the Strategy is " + str(round(kurt_portfolio,4)))
    print("The Terminal Value of the INDEX is " + str(round(data1.Cumulative.iloc[-1],6)))  
    print("The Max Drawdown of the INDEX is " + str(round(mdd*100,4)) + " %")
    terminal = round(data1.Cumulative.iloc[-1],6)
    
    
    return terminal


""" ******************************************************************************************************* """
""" ******************************************************************************************************* """

def maximumDrawdown(ret_series):
    ret_series      = np.array(ret_series)
    ret_series      = 1 + ret_series
    cum_ret         = np.cumprod(ret_series)
    mdd             = 0
    peak            = cum_ret[0]
    
    for x in np.array(cum_ret):
        if x > peak:
            peak    =   x
            
        dd  = (peak - x) / peak
        
        if dd > mdd:
            mdd     =   dd
            
    return mdd  



def monthly_performance_stats(weightdRet):
    # Calculating Strategy Average Return
    wRetmu                  =       weightdRet.mean() * 12
    
    # Calculating Strategy Standard Deviation
    wRetstd                 =       weightdRet.std(ddof=1) * (12 ** 0.5)
    
    # Calculating Strategy Sharpe ratio
    sharpe                  =       wRetmu/wRetstd
    
    # Calculating Strategy Skew
    skew_portfolio          =       skew(weightdRet)
    
    # Calculating Strategy Kurtosis
    kurt_portfolio          =       kurtosis(weightdRet)
    
    # Calculating Maximum loss incurred across all the holding periods
    max_m_loss              =       weightdRet.min()
    
    
    return wRetmu, wRetstd, sharpe, skew_portfolio, kurt_portfolio, max_m_loss