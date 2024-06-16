import numpy as np
import torch
from tqdm import tqdm
import pickle as pkl
from pathlib import Path
import pandas as pd
from Utilities.transforms import get_transforms
import hashlib


def outliers(data, ex=3):
    total_cols=len(data.columns)
    for i in range (0, total_cols):
        kolumna = data.iloc[:,i]      
        q1 = kolumna.quantile(q=0.25)
        q3 = kolumna.quantile(q=0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-ex*iqr
        fence_high = q3+ex*iqr
        df_out = kolumna.loc[(kolumna < fence_low )|(kolumna > fence_high)]
        data.iloc[df_out.index, i] = np.nan
        
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

class MatrixData(torch.utils.data.Dataset):
    def __init__(self, train, config, transforms=None):
        path_to_files = Path(config["path_to_datasets"])
        path_to_label_csv = Path(config["label_csv"])
        self.label_csv = pd.read_csv(path_to_label_csv, sep=";", decimal=",")
        self.label_key = config["label_key"]
        self.time_column = config["time_column"]
        self.saving = config["saving"]
        
        data, filenames = self.load_dataset(config, path_to_files, train)
        self.patients = []
        labels = []
        self.data = []
        # print(filenames)
        # print(label_csv[config["label_key"]].values)
        left_out_patients = []
        for dt, filename in zip(data, filenames):
            if filename not in  self.label_csv[self.label_key].values:
                if filename not in left_out_patients:
                    print(f"File {filename} not in label csv")
                    left_out_patients.append(filename)
            else:
                filename_label =  self.label_csv[ self.label_csv[self.label_key] == filename][config["target_key"]].values[0]
                if filename_label == 1 or filename_label == 0:
                    labels.append(filename_label)
                    self.patients.append(filename)
                    self.data.append(dt)
                else:
                    if filename not in left_out_patients:
                        left_out_patients.append(filename)
                        print(f"File {filename} has label {filename_label} with empty target ({config['target_key']})")
        self.data = np.array(self.data)
        if train:
            print("Loaded train data with shape {}".format(self.data.shape))
            print(f"Left out {len(left_out_patients)} patients")
        else:
            print("Loaded val data with shape {}".format(self.data.shape))
            print(f"Left out {len(left_out_patients)} patients")
        self.labels = torch.tensor(labels, dtype=torch.float32)    

        #validate all data is finite
        if np.isnan(self.data).any():
            print("Data contains NaNs")
        #validate all labels are finite
        if np.isnan(self.labels).any():
            print("Labels contain NaNs")
        
        if config["pad"] is not None:
            _, h, w = self.data.shape
            #find closest h divisible by pad
            h_pad = (config["pad"] - h % config["pad"]) % config["pad"]
            w_pad = (config["pad"] - w % config["pad"]) % config["pad"]
            pad_list = [[0,0],[h_pad, 0],[w_pad,0]]
            self.data = np.pad(self.data, pad_width=pad_list, mode="constant", constant_values=0)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.transforms = transforms

    def __getitem__(self, index):
        return {"data": self.data[index], "label": self.labels[index], "patient": self.patients[index]}

    def __len__(self):
        return len(self.data)

    def get_dataset_name(self, config, train):
        suffix = ""
        for key in config["raw_to_processed_config"]:
            if isinstance(config["raw_to_processed_config"][key], list):
                suffix += "-" + key + "=" + ",".join([str(x) for x in config["raw_to_processed_config"][key]])
            else:
                suffix += "-" + key + "=" + str(config["raw_to_processed_config"][key])
                
        hash_obj = hashlib.sha256(suffix.encode('utf-8'))
        suffix = str(hash_obj.hexdigest())
        if train:
            return config["name"] + suffix + "_train.pkl"
        return config["name"] + suffix + "_test.pkl"
    
    def remove_outliers(self, dataframe, signals, no_sigma=3):
        '''Removes outliers from the dataframe.
        Returns the dataframe without outliers.
        '''
        for signal in signals:
            mean = dataframe[signal].mean()
            std = dataframe[signal].std()
            dataframe = dataframe[abs(dataframe[signal] - mean) < no_sigma * std]

        return dataframe


    def dataframe_to_slices(self, dataframe, signals, no_nans=4, min_length=180):
        '''Splits the dataframe signals information into slices 
        where both of the signals contain at maximum no_nans NaNs 
        and the length of the slice is at least min_length.
        Returns a list of slices.
        '''
        #fill nans with pandas method
        dataframe = dataframe.fillna(method="ffill", limit=no_nans)
        if len(dataframe) == 0:
            return None

        mask = np.ones(len(dataframe), dtype=bool)
        for signal in signals:
            mask = mask & ~np.isnan(dataframe[signal])
        mask = np.array(mask)

        #find indices of the start and end of each fully filled slice
        #make sure to take into account the beggining and the end of the mask
        starts = np.where(np.diff(mask.astype(int)) == 1)[0] + 1
        ends = np.where(np.diff(mask.astype(int)) == -1)[0] + 1
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        if mask[-1]:
            ends = np.append(ends, len(mask))

        #check if the starts and ends are equal length
        if len(starts) != len(ends):
            raise ValueError("Starts and ends are not equal length")
        
        #create slices
        slices = [(start, end) for start, end in zip(starts, ends) if end - start >= min_length]

        if len(slices) == 0:
            return None
        
        signal_slices = [
            {signal: np.array(dataframe[signal][start:end].values) for signal in signals} for start, end in slices
        ]
        return signal_slices
        
    def filter_days(self, dataframe, days=3, start=False):
        ''' Filters the dataframe to contain only the first or last days.
        each sample is 1 minute long.
        '''
        if start:
            return dataframe.iloc[:days*24*60]
        else:
            return dataframe.iloc[-days*24*60:]
        
    def filter_craniecotmy(self, patient, dataframe, local_config):
        '''Removes the patient if the craniectomy was performed with no date specified.
        if the date is specified, cuts the dataframe to the date of the craniectomy.
        '''
        if local_config["remove_craniectomy"]:
            cr_key = local_config["craniectomy_key"]
            date_keys = local_config["craniectomy_date_keys"]
            patient_metadata = self.label_csv.query(f"{self.label_key} == '{patient}'")
            if len(patient_metadata) == 0:
                return dataframe
            if patient_metadata[cr_key].values[0] == 1:
                date = patient_metadata[date_keys[0]].values[0]
                hour = patient_metadata[date_keys[1]].values[0]

                if pd.isnull(date) or pd.isnull(hour):
                    return None
                unix_datetime = pd.to_datetime(date + " " + hour, format="%d.%m.%Y %H:%M").timestamp()
                dataframe = dataframe[dataframe[self.time_column] < unix_datetime]
        return dataframe

    def preprocess_file(self, file, local_config, patient):
        self.craniectomy_flag = False
        file = self.filter_craniecotmy(patient, file, local_config)
        if file is None:
            self.craniectomy_flag = True
            return None
        
        file = self.filter_days(file, days=local_config["days"], start=local_config["first_days"])
        file = self.remove_outliers(file, local_config["columns"], no_sigma=3)
        signal_slices = self.dataframe_to_slices(file, local_config["columns"], no_nans=5, min_length=local_config["correlation_len"])

        return signal_slices


    def create_dataset(self, dataset_name, path_to_datasets, train, config):
        path_to_files = Path(config["source_files"])
        if train:
            path_to_files = path_to_files / "train"
        else:
            path_to_files = path_to_files / "test"
        
        all_csvs = list(path_to_files.rglob("*.csv"))
        local_config = config["raw_to_processed_config"]
        columns_to_load = local_config["columns"] + [self.time_column]
        whole_data = []
        files = []
        for file in tqdm(all_csvs):
            file_data = pd.read_csv(file, sep=";", decimal=",")
            
            signal_names = local_config["columns"]
            #check if columns are in file
            signal_missing = False
            for signal in signal_names:
                if signal not in file_data.columns:
                    print(f"File {file.name} does not contain signal {signal}")
                    signal_missing = True
            if signal_missing:
                continue

            file_data = file_data[columns_to_load]

            file_data = self.preprocess_file(file_data, local_config, file.name)
            if file_data is None:
                if self.craniectomy_flag:
                    print(f"File {file.name} was removed due to craniectomy")
                else:
                    print(f"File {file.name} does not contain any slices of length {local_config['correlation_len']}")
                continue
            
            frames = local_config["no_frames"]
            rss = []
            for slc in file_data:
                for start in range(0, len(slc[signal_names[0]]) - local_config["correlation_len"], local_config["correlation_step"]):
                    samples_per_split = 2*frames + 1
                    corr_matrix = np.zeros((local_config["time_shifts"], 2*frames))
                    part_signal_1 = slc[signal_names[0]][start:start + local_config["correlation_len"]]
                    mid_point = len(part_signal_1)//2

                    for i in range(0, local_config["time_shifts"]):
                        part_signal_2 = slc[signal_names[1]][start + i*samples_per_split:start + (i+1)*samples_per_split]
                        if len(part_signal_2) != samples_per_split:
                            break
                        corr_start = mid_point - frames
                        corr_end = mid_point + frames
                        corr_matrix[i, :] = np.correlate(part_signal_1, part_signal_2, mode="full")[corr_start:corr_end]

                    whole_data.append(corr_matrix)
                    files.append(file.name.split(" - ")[0])
            # print(len(whole_data))


        if self.saving:
            data_to_save = {"data": whole_data, "filenames": files}
            with open(path_to_datasets / dataset_name, "wb") as f:
                pkl.dump(data_to_save, f)

        return whole_data, files


            

    def _load_created_dataset(self, dataset_name, path_to_files):
        with open(path_to_files / dataset_name, "rb") as f:
            data = pkl.load(f)
        return data["data"], data["filenames"]

    def load_dataset(self, config, path_to_datasets, train):
        dataset_name = self.get_dataset_name(config, train)
        print(f"Loading dataset {dataset_name}")
        if self._dataset_exists(dataset_name, path_to_datasets) and self.saving:
            data, filenames = self._load_created_dataset(dataset_name, path_to_datasets)
        else:
            print("Did not find dataset, creating")
            data, filenames = self.create_dataset(dataset_name, path_to_datasets, train, config)
        return data, filenames

    def _dataset_exists(self, dataset_name, path_to_datasets):
        if (path_to_datasets / dataset_name).exists():
            return True
        return False
    
def get_md(train, config):
    transform = get_transforms(config)
    full_dataset = MatrixData(train,
                              config=config,
                              transforms=transform
                              )
    return full_dataset