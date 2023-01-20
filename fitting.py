"""

Program to perform clustering and fitting in given datasets

"""
# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wbgapi as wb
from sklearn.cluster import KMeans
import seaborn as sns
import scipy.optimize as opt
from scipy.optimize import curve_fit
import itertools as iter


# taking the list of required indicator IDs
indicator = ["EN.ATM.CO2E.PC","EN.ATM.NOXE.KT.CE"]
# taking the list of country code of selected countries
country_code = ['CAN','FRA','CHN','DEU','GBR','IND']


# reading the data into a dataframe
def read_dat(indicator,country_code):
    """
    Parameters
    ----------
    indicator : list of required indicator IDs
    country_code : list of country codes of selected countries
    Returns
    -------
    dataf : dataframe 
    """
    dataf = wb.data.DataFrame(indicator, country_code, mrv=30)
    return dataf


# Normalizing the data
def norm_df(dataf):
    """
    Parameters
    ----------
    df : dataframe with original data
    Returns
    -------
    df : dataframe with normalized data
    """
    y = dataf.iloc[:,2:]
    dataf.iloc[:,2:] = (y-y.min())/ (y.max() - y.min())
    return dataf


# fitting function
def funct(x, a, b, c):
    """
    Parameters
    ----------
    x : a variable
    a,b,c : parameters to be fitted
    """
    return a*x**2+b*x+c


# calculate the error ranges
def error_ranges(x, func, param, sigma):
    """
    Parameters
    ----------
    x,func,param, sigma : parameters to calculate error range
    Returns
    -------
    lower,upper : lower and upper ranges of error
    """
    lower = func(x, *param)
    upper = lower
    uplow = []
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
    pmix = list(iter.product(*uplow))
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
    return lower, upper


# reading data to dataframe using the read_dat function
dat= read_dat(indicator, country_code)

# Replacing 'YR' with new index names 
dat.columns = [i.replace('YR','') for i in dat.columns]
dat=dat.stack().unstack(level=1)
dat.index.names = ['Country', 'Year']
dat = dat.reset_index()
#dat = dat.fillna(0)
dat.drop(['EN.ATM.NOXE.KT.CE'], axis = 1, inplace = True)
dat["Year"] = pd.to_numeric(dat["Year"])

# Normalised dataframe
dt_norm = norm_df(dat)
df_fit = dt_norm.drop('Country', axis = 1)
# using k-means clustering
k = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(df_fit)
# Plotting clusters of different countries based on CO2 emission
sns.scatterplot(data=dt_norm, x="Country", y="EN.ATM.CO2E.PC", hue=k.labels_)
plt.ylabel("CO2 Emission")
plt.title("CO2 Emission Rate ")
plt.legend()
plt.show()
plt.savefig("cluster.png")


