import os
import pandas as pd
import numpy as np

SOURCE_FILE_PATH = "E:/Projekty/Sonata_repo/Data/Wieloparametrowe/raw_data"
EXTENSION_FILE = ".csv"






# Function to read file
def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    print(file_name)
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
    try:
   

        #recall of Function to read File:               
        dataset = read_file(files_list[counter_files], separator = ';', decimal_sign = ',')
        # dataset = dataset.rename(columns={"icp[mmHg]": "ICP", "abp[mmHg]": "ABP", "art": "ABP", "art[mmHg]": "ABP",
        #                                   "ABP_BaroIndex": "BRS", "brs": "BRS", "ART_BaroIndex": "BRS", "ART": "ABP"})
        dataset=outliers(dataset,3)
        
        
        
        #check if size of recording is at least 3 days:
        #averaging in window = 60 seconds; 1point = 1 minute
        #5days = 5*24*60 = 7200
        
        dataset = dataset.dropna()
        filled_ICP = interpolate_gaps(dataset['ICP'], limit=2)
        filled_BRS = interpolate_gaps(dataset['BRS'], limit=2)
        dataset_ICP = pd.DataFrame(filled_ICP)
        dataset_BRS = pd.DataFrame(filled_BRS)
        frames = [dataset_ICP,dataset_BRS]
        result = pd.concat(frames,axis=1)
        result.columns = ['ICP', 'BRS']
        
        
        if result.shape[0] >= 7200:
        
                    
        
            
            for l in range (0, 7200, 60 ):
                result_t = result.loc[l:l+4319,:]
                result_t = pd.DataFrame(result_t.values, columns=result_t.columns).reset_index(drop=True)
            
        
                frames = 60 #zawsze +/- 60 minut
                no_splits = 30
                samples_per_split = result_t.shape[0]/no_splits
                rss=[]
                for t in range(0, no_splits):
                    d1 = result_t['ICP'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                    d2 = result_t['BRS'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
                    rs = [crosscorr(d1,d2, lag) for lag in range(-int(frames),int(frames+1))]
                    rss.append(rs)
            
            
                rss = pd.DataFrame(rss)
                
            
            
                #f,ax = plt.subplots(figsize=(20,10))
                #fig = sns.heatmap(rss,cmap='RdBu_r',ax=ax)
                #ax.set(title=f'Windowed Time Lagged Cross Correlation',xlim=[0,121], xlabel='Offset',ylabel='Window epochs')
                #ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
                #ax.set_xticklabels([-9, -6, -3, 0, 3, 6, 9])
            
                
                #fig.figure.savefig("C:\Moje_dokumenty\Po_doktoracie_2022_2023\ML_ANS\WYNIKI\out_{}_{}.png".format(counter_files,l)) 
                
                filename = "E:/Projekty/Sonata_repo/Data/Wieloparametrowe/test/out_{}_{}.pkl".format(counter_files,l) # generate a different file name for each DataFrame
                rss.to_pickle(filename)
                
                
                
        else:
            print("jestem krótszy")
    except:
        print("błąd")
        continue




