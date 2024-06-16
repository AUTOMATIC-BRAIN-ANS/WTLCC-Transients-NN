# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:34:21 2022

@author: agnie
"""
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
import statistics as st
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller


SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\DATABASE_TBI_USK\BRS_COPY"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura"





# Function to read file
def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    if delete_column != None:
        del dataset[delete_column]
                  
    return dataset

#Function to get number of files with specific extension:
def counter_files(path, extension):
    list_dir = []    
    list_dir = os.listdir(path)
    count = 0
    for file in list_dir:
        if file.endswith(extension):
            count+=1
    return  count


#Function to get list of files with specific extension:
def get_list_files(path,extension):
    directory = os.fsencode(path)
    list_files = []
    only_files_name = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(extension):
           list_files.append(path + '/' +filename)
           only_files_name.append(filename)
    return list_files, only_files_name


#function to find extreme values
def outliers(data, ex):
    total_cols=len(data.axes[1])
    
    for i in range (0, total_cols):
        kolumna = data.iloc[:,i]
                   
        q1 = kolumna.quantile(q =0.25)
        q3 = kolumna.quantile(q = 0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-ex*iqr
        fence_high = q3+ex*iqr
        df_out = kolumna.loc[(kolumna < fence_low )|(kolumna > fence_high)]
        kolumna[df_out.index] = None
        
    return data

def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled



def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
   """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))



#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)



results_iteration = []
for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset = read_file(files_list[counter_files], separator = ',', decimal_sign = '.')
    dataset = dataset.rename(columns={"icp[mmHg]": "ICP", "abp[mmHg]": "ABP", "art": "ABP", "art[mmHg]": "ABP",
                                      "ABP_BaroIndex": "BRS", "brs": "BRS", "ART_BaroIndex": "BRS", "ART": "ABP"})
    dataset.drop(['DateTime','ABP'], axis=1, inplace=True)
    dataset=outliers(dataset,3)
    
    
    #######drop out values of BRS-ICP where are np.nan, i.e. in both
    ######BRS and ICP needs to be data
    binary_mask_brs = dataset['BRS']>0
    binary_mask_icp = dataset['ICP']>0
    
    binary_mask_all = binary_mask_brs*binary_mask_icp
    
    dataset_all=dataset[binary_mask_all]
    
    ################graph to show raw data, only without NaN's
    f,ax=plt.subplots(figsize=(10,5))
    ax.plot(dataset)
    #ax.set(xlabel='Time',ylabel='mm Hg or ms/mm Hg')
    #ax.set(title=f"Scipy computed Pearson r: {r} and p-value: {p}")
    
    
    
   
    ###############co 10 sekund + mediana ruchoma co 5 minut
    overall_pearson_r = dataset['BRS'].corr(dataset['ICP'])
    overall_pearson_r2 = dataset['BRS'].shift(11).corr(dataset['ICP'])
    
        
    r, p = stats.pearsonr(dataset.dropna()['BRS'], dataset.dropna()['ICP'])
    
   
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    print(f"Pandas computed Pearson r: {overall_pearson_r2}")
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    
    f,ax=plt.subplots(figsize=(7,3))
    dataset.rolling(window=1,center=True).median().plot(ax=ax)
    ax.set(xlabel='Time',ylabel='mm Hg or ms/mm Hg')
    ax.set(title=f"Scipy computed Pearson r: {r} and p-value: {p}")
    
    
    
    
    #########wizualizacja bez nanów (liczy się i tak bez nanów)
    datasetR = dataset.dropna()
    f,ax=plt.subplots(figsize=(10,3))
    datasetR.rolling(window=30,center=True).median().plot(ax=ax)
    ax.set(xlabel='Time',ylabel='mm Hg or ms/mm Hg')
    ax.set(title=f"12Scipy computed Pearson r: {r} and p-value: {p}")
    
    
    
    
    
    #########interpolated
    dataset = dataset.dropna()
    filled_b = interpolate_gaps(dataset.iloc[:,0], limit=2)
    filled_i = interpolate_gaps(dataset.iloc[:,1], limit=2)
    
    
    dataset2 = pd.DataFrame(filled_b)
    dataset3 = pd.DataFrame(filled_i)
    frames = [dataset2,dataset3]
    result = pd.concat(frames,axis=1)
    result.columns = ['ABP_BaroIndex', 'ICP']

    
    f,ax=plt.subplots(figsize=(10,3))
    result.rolling(window=30,center=True).median().plot(ax=ax)
    ax.set(xlabel='Time',ylabel='mm Hg or ms/mm Hg')
    ax.set(title=f"Scipy computed Pearson r: {r} and p-value: {p}")
    
    
    f,(ax1, ax2)=plt.subplots(2,1,figsize=(10,3))
    ax1.plot(result.iloc[:,0].rolling(window=360,center=True).median())
    ax2.plot(result.iloc[:,1].rolling(window=360,center=True).median())
    
   
    ################################
    # Set window size to compute moving window synchrony.
    r_window_size = 360
    
    # Compute rolling window synchrony
    rolling_r = result['ABP_BaroIndex'].rolling(window=r_window_size, center=True).corr(result['ICP'])
    f,ax=plt.subplots(2,1,figsize=(14,6),sharex=True)
    result.rolling(window=360,center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Frame',ylabel='Smiling Evidence')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Frame',ylabel='Pearson r')
    plt.suptitle("Smiling data and rolling window correlation")
    
    ##############wraping###########3
    
    
    
    BRS = result['ABP_BaroIndex']
    ICP = result['ICP']
    frames = 360
   
    rs = [crosscorr(ICP,BRS, lag) for lag in range(-int(frames),int(frames+1))]
    rs2 = []
    for i in range(0,len(rs)):
        rs2.append(rs[i]*1)
        
   
    offset = np.floor(len(rs2)/2)-(np.argmax(rs2))
    f,ax=plt.subplots(figsize=(14,3))
    ax.plot(rs2)
    ax.axvline(np.ceil(len(rs)/2),color='k',linestyle='--',label='Center')
    
    ax.axvline(np.argmax(rs2),color='r',linestyle='--',label='Peak synchrony')
    ax.set(title=f'Offset = {offset/6:.2f} minutes\n BRS vs. ICP', xlim=[0,721],xlabel='Offset',ylabel='Pearson r')
    ax.set_xticks([60, 160, 260, 360, 460, 560, 660])
    ax.set_xticklabels([-60, -40, -20, 0, 20, 40, 60]);
    plt.legend()
    
    
    BRS = result['ABP_BaroIndex']
    ICP = result['ICP']
    frames = 60
   
    rs = [crosscorr(ICP,BRS, lag) for lag in range(-int(frames),int(frames+1))]
    rs2 = []
    for i in range(0,len(rs)):
        rs2.append(rs[i]*1)
        
   
    offset = np.floor(len(rs2)/2)-(np.argmax(rs2))
    f,ax=plt.subplots(figsize=(14,3))
    ax.plot(rs2)
    ax.axvline(np.floor(len(rs)/2),color='k',linestyle='--',label='Center')
    
    ax.axvline(np.argmax(rs2),color='r',linestyle='--',label='Peak synchrony')
    ax.set(title=f'Offset = {offset/6:.2f} minutes\n BRS vs. ICP', xlim=[0,61],xlabel='Offset',ylabel='Pearson r')
    ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax.set_xticklabels([-9, -6, -3, 0, 3, 6, 9])
    plt.legend()
    
    
    
 ################
 
 # Windowed time lagged cross correlation

frames = 60
no_splits = 30
samples_per_split = result.shape[0]/no_splits
rss=[]
for t in range(0, no_splits):
    d1 = result['ICP'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
    d2 = result['ABP_BaroIndex'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
    rs = [crosscorr(d1,d2, lag) for lag in range(-int(frames),int(frames+1))]
    rss.append(rs)


rss = pd.DataFrame(rss)
rss2 = pd.DataFrame(rss)






f,ax = plt.subplots(figsize=(10,5))
sns.heatmap(rss2,cmap='RdBu_r',ax=ax)
ax.set(title=f'Windowed Time Lagged Cross Correlation',xlim=[0,121], xlabel='Offset',ylabel='Window epochs')
ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax.set_xticklabels([-9, -6, -3, 0, 3, 6, 9])








stat_test = adfuller(result['ABP_BaroIndex'])
print('ADF Statistic: %f' % stat_test[0])
print('p-value: %f' % stat_test[1])