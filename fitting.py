# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 23:11:48 2023

@author: LENOVO
"""

import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import scipy.optimize as opt
from sklearn.cluster import KMeans
from scipy.stats import norm
import seaborn as sns
from scipy.optimize import curve_fit
import itertools as iter

indicator1 = ["EN.ATM.CO2E.PC","EN.ATM.NOXE.KT.CE"]
country_code = ['CAN','CHN','AUS','IND','RUS','FRA','DEU','KWT','ITA']
def read(indicator,country_code):
     df = wb.data.DataFrame(indicator, country_code, mrv=30)
     return df
 
    
path = "World Indicator Repository.csv"
data = read(indicator1, country_code)
data.columns = [i.replace('YR','') for i in data.columns]
data=data.stack().unstack(level=1)
data.index.names = ['Country', 'Year']
data.columns
data = data.reset_index()
data

data.drop(['EN.ATM.NOXE.KT.CE'], axis = 1, inplace = True)
data

data["Year"] = pd.to_numeric(data["Year"])
def norm_df(df):
    y = df.iloc[:,2:]
    df.iloc[:,2:] = (y-y.min())/ (y.max() - y.min()) 
    return df

dt_norm = norm_df(data)
dt_norm






