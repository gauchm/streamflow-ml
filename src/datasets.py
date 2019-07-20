import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset
from src import load_data

class RdrsDataset(Dataset):
    def __init__(self, rdrs_vars, seq_len, seq_steps, date_start, date_end, conv_scalers=None, fc_scalers=None):
        self.date_start = date_start
        self.date_end = date_end
        self.seq_len = seq_len
        self.seq_steps = seq_steps
        self.samples_per_date = []
        self.conv_scalers = conv_scalers
        self.fc_scalers = fc_scalers
        
        rdrs_data, rdrs_var_names, rdrs_time_index = load_data.load_rdrs_forcings(as_grid=True)
        rdrs_data = rdrs_data[:,rdrs_vars,:,:]
        self.n_conv_vars = len(rdrs_vars)
        self.conv_height = rdrs_data.shape[2]
        self.conv_width = rdrs_data.shape[3]
        
        data_runoff = load_data.load_discharge_gr4j_vic()
        data_runoff = data_runoff[~pd.isna(data_runoff['runoff'])]
        data_runoff = data_runoff.join(pd.get_dummies(data_runoff['date'].dt.month, prefix='month'))
        data_runoff = data_runoff[(data_runoff['date'] >= self.date_start) & (data_runoff['date'] <= self.date_end)]
        gauge_info = pd.read_csv('../data/gauge_info.csv')[['ID', 'Lat', 'Lon']]
        data_runoff = pd.merge(data_runoff, gauge_info, left_on='station', right_on='ID').drop('ID', axis=1)
        
        # sort by date then station so that consecutive values have increasing dates (except for station cutoffs)
        # this way, a stateful lstm can be fed date ranges.
        data_runoff = data_runoff.sort_values(by=['station', 'date']).reset_index(drop='True')
        self.data_runoff = data_runoff[['date', 'station', 'runoff']]
        
        # conv input shape: (samples, seq_len, variables, height, width)
        # But: x_conv is the same for all stations for the same date, so we only generate #dates samples 
        #      and feed them multiple times (as many times as we have stations for that date)
        self.x_conv = np.zeros((len(data_runoff['date'].unique()), seq_len, self.n_conv_vars, self.conv_height, self.conv_width))       
        self.dates = np.sort(data_runoff['date'].unique())
        i = 0
        for date in self.dates:
            # For each day that is to be predicted, cut out a sequence that ends with that day's 23:00 and is seq_len long
            end_of_day_index = rdrs_time_index[rdrs_time_index == date].index.values[0] + 23
            self.x_conv[i,:,:,:,:] = rdrs_data[end_of_day_index - (self.seq_len * self.seq_steps) : end_of_day_index : self.seq_steps]
            i += 1

        # Scale training data
        if self.conv_scalers is None:
            self.conv_scalers = list(preprocessing.StandardScaler() for _ in range(self.x_conv.shape[2]))
        for i in range(self.n_conv_vars):
            self.x_conv[:,:,i,:,:] = np.nan_to_num(self.conv_scalers[i].fit_transform(self.x_conv[:,:,i,:,:].reshape((-1, 1)))
                                                   .reshape(self.x_conv[:,:,i,:,:].shape))

        self.x_fc = data_runoff.drop(['date', 'station', 'runoff'], axis=1).to_numpy()
        self.n_fc_vars = self.x_fc.shape[1]
        if self.fc_scalers is None:
            self.fc_scalers = list(preprocessing.StandardScaler() for _ in range(self.x_fc.shape[1]))
        for i in range(self.n_fc_vars):
            self.x_fc[:,i] = np.nan_to_num(self.fc_scalers[i].fit_transform(self.x_fc[:,i].reshape((-1,1))).reshape(self.x_fc[:,i].shape))
        
        self.x_conv = torch.from_numpy(self.x_conv).float()
        self.x_fc = torch.from_numpy(self.x_fc).float()
        self.y = torch.from_numpy(data_runoff['runoff'].to_numpy()).float()
        
    def __getitem__(self, index):
        date_of_index = self.data_runoff.iloc[index]['date']
        x_conv_index = np.argmax(self.dates == date_of_index)
        
        return {'x_conv': self.x_conv[x_conv_index,:,:,:,:], 
                'x_fc': self.x_fc[index,:],
                'y': self.y[index]}
    
    def __len__(self):
        return self.y.shape[0]
    
    
class StatefulBatchSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, batch_size):
        self.batch_size = batch_size
        self.num_batches = len(data_source) // batch_size
        
        if len(data_source) % batch_size != 0:
            print('Warning: Data source length not divisible by batch_size')
        
    def __iter__(self):
        batch = []
        for i in range(self.num_batches):
            for j in range(self.batch_size):
                batch.append(i + j * self.num_batches)
            yield batch
            batch = []
        
    def __len__(self):
        return self.num_batches
