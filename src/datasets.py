import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset, IterableDataset
from src import load_data, utils

class RdrsDataset(Dataset):
    def __init__(self, rdrs_vars, seq_len, seq_steps, date_start, date_end, conv_scalers=None, fc_scalers=None, include_months=True):
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
        if include_months:
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
        fc_var_names = data_runoff.drop(['date', 'station', 'runoff'], axis=1).columns
        vars_to_scale = list(c for c in fc_var_names if not c.startswith('month'))
        self.n_fc_vars = len(fc_var_names)
        if self.fc_scalers is None:
            self.fc_scalers = list(preprocessing.StandardScaler() for _ in range(self.n_fc_vars))
        for i in range(self.n_fc_vars):
            if fc_var_names[i] in vars_to_scale:
                to_transform = data_runoff[fc_var_names[i]].to_numpy().reshape((-1,1))
                self.x_fc[:,i] = np.nan_to_num(self.fc_scalers[i].fit_transform(to_transform).reshape(self.x_fc[:,i].shape))
        
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

    
class RdrsGridDataset(Dataset):
    """ 
    RDRS dataset where target is a spatial grid of streamflow values.
    If include_simulated_streamflow, the returned items will be dicts y -> actual gauge stations, y_sim -> simulated gauges.
    Else, the dict will only contain y.
    
    If out_lats and out_lons are not specified, will use RDRS grid as output resolution.
    Specifying out_lats/lons allows for different resolution of geophysical (non-time-series) inputs.
    
    If upsample=None, will not upsample. If 'input', will upsample RDRS data. 
    If 'output', will not upsample, but generate upsampling map for ConvLSTM output.
    """
    def __init__(self, rdrs_vars, seq_len, seq_steps, date_start, date_end, conv_scalers=None, exclude_stations=[], aggregate_daily=None, include_months=False, include_simulated_streamflow=False, out_lats=None, out_lons=None, upsample=None):
        self.date_start = date_start
        self.date_end = date_end
        self.seq_len = seq_len
        self.seq_steps = seq_steps
        self.conv_scalers = conv_scalers
        self.include_simulated_streamflow = include_simulated_streamflow
        
        rdrs_data, rdrs_var_names, rdrs_time_index, lats, lons = load_data.load_rdrs_forcings(as_grid=True, include_lat_lon=True)
        rdrs_data = rdrs_data[:,rdrs_vars,:,:]
        self.n_conv_vars = len(rdrs_vars)
        self.conv_height = rdrs_data.shape[2]
        self.conv_width = rdrs_data.shape[3]
        
        self.in_lats = lats
        self.in_lons = lons
        self.out_lats = self.in_lats if out_lats is None else np.tile(out_lats,len(out_lons)).reshape(len(out_lons),-1).T
        self.out_lons = self.in_lons if out_lons is None else np.tile(out_lons,len(out_lats)).reshape(len(out_lats),-1)
        self.upsample_map_rows, self.upsample_map_cols = None, None
        if upsample is not None:
            if out_lats is not None and out_lons is not None:
                print('Creating upsampling map to quickly upsample during training/testing')
                self.upsample_map_rows, self.upsample_map_cols = utils.map_to_geophysical_coords(self.in_lats, self.in_lons, 
                                                                                                 self.out_lats, self.out_lons)
            else:
                raise Exception('Need out_lat/out_lon for upsampling.')
        
        if aggregate_daily is not None:
            self.n_conv_vars += sum(1 for agg in aggregate_daily if agg == 'minmax')
            rdrs_time_index_daily = pd.Series(pd.date_range(rdrs_time_index.min().date(), rdrs_time_index.max().date(), freq='D'))
            rdrs_daily = np.zeros((len(rdrs_time_index), self.n_conv_vars, self.conv_height, self.conv_width))
            for j in range(len(rdrs_time_index_daily)):
                day_indices = rdrs_time_index[rdrs_time_index.dt.date == rdrs_time_index_daily.dt.date[j]].index.values
                i_new = 0
                for i in range(len(rdrs_vars)):                
                    day_data = rdrs_data[day_indices,i,:,:]
                    if aggregate_daily[i] == 'sum':
                        rdrs_daily[j,i_new,:,:] = day_data.sum(axis=0)
                    elif aggregate_daily[i] == 'minmax':
                        rdrs_daily[j,i_new,:,:] = day_data.min(axis=0)
                        rdrs_daily[j,i_new + 1,:,:] = day_data.max(axis=0)
                        i_new += 1
                    else:
                        raise Exception('Invalid aggregation method')
                    i_new += 1
            rdrs_time_index = rdrs_time_index_daily
            rdrs_data = rdrs_daily
        
        data_runoff = load_data.load_discharge_gr4j_vic()
        data_runoff = data_runoff[~pd.isna(data_runoff['runoff'])]
        data_runoff = data_runoff[(data_runoff['date'] >= self.date_start) & (data_runoff['date'] <= self.date_end)]
        gauge_info = pd.read_csv('../data/gauge_info.csv')[['ID', 'Lat', 'Lon']]
        data_runoff = pd.merge(data_runoff, gauge_info, left_on='station', right_on='ID').drop('ID', axis=1)
        data_runoff = data_runoff.sort_values(by=['date']).reset_index(drop='True')
            
        if len(exclude_stations) > 0:
            self.data_runoff_all_stations = data_runoff[['date', 'station', 'runoff']].copy()
            data_runoff = data_runoff[~data_runoff['station'].isin(exclude_stations)].reset_index(drop=True)
        
        self.data_runoff = data_runoff[['date', 'station', 'runoff']]
        
        # get station to (row, col) mapping in output lat/lons
        self.station_to_index = {}
        for station in data_runoff['station'].unique():
            station_lat = data_runoff[data_runoff['station'] == station]['Lat'].values[0]
            station_lon = data_runoff[data_runoff['station'] == station]['Lon'].values[0]
            # find nearest cell
            station_idx = np.argmin(np.square(self.out_lats - station_lat) + np.square(self.out_lons - station_lon))
            station_row = station_idx // self.out_lons.shape[1]
            station_col = station_idx % self.out_lons.shape[1]
            self.station_to_index[station] = (station_row, station_col)
        
        if include_months:
            self.n_conv_vars += 12
        
        # conv input shape: (samples, seq_len, variables, height, width)
        self.x_conv = np.zeros((len(data_runoff['date'].unique()), seq_len, self.n_conv_vars, self.conv_height, self.conv_width))       
        self.dates = np.sort(data_runoff['date'].unique())
        i = 0
        for date in self.dates:
            # For each day that is to be predicted, cut out a sequence that ends with that day's 23:00 and is seq_len long
            if aggregate_daily is None:
                end_of_day_index = rdrs_time_index[rdrs_time_index == date].index.values[0] + 23
            else:
                end_of_day_index = rdrs_time_index[rdrs_time_index == date].index.values[0]
            
            date_data = rdrs_data[end_of_day_index - (self.seq_len * self.seq_steps) : end_of_day_index : self.seq_steps]
            if upsample == 'input':
                date_data = utils.upsample_to_geophysical_input(date_data, self.upsample_map_rows, self.upsample_map_cols, fill_value=np.nan)
            if include_months:
                month_dummies = np.zeros((self.seq_len, 12, self.conv_height, self.conv_width))
                for j in range(self.seq_len):
                    day_index = end_of_day_index - (self.seq_len * self.seq_steps) + (j * self.seq_steps)
                    month = rdrs_time_index[day_index].month
                    month_dummies[j,month - 1,:,:] = 1
                date_data = np.concatenate([date_data, month_dummies], axis=1)
            self.x_conv[i,:,:,:,:] = date_data
            i += 1

        # Scale training data
        if self.conv_scalers is None:
            self.conv_scalers = list(preprocessing.StandardScaler() for _ in range(self.x_conv.shape[2]))
        for i in range(self.n_conv_vars - (include_months * 12)):
            self.x_conv[:,:,i,:,:] = np.nan_to_num(self.conv_scalers[i].fit_transform(self.x_conv[:,:,i,:,:].reshape((-1, 1)))
                                                   .reshape(self.x_conv[:,:,i,:,:].shape))

        if include_simulated_streamflow:
            self.simulated_streamflow, self.outlet_to_row_col = load_data.load_simulated_streamflow()
            # Create a tensor of shape (#days, height, width) of target values (only the cells of virtual gauges get populated)
            self.y_sim = torch.zeros((len(self.dates), self.out_lats.shape[0], self.out_lats.shape[1]))
             # Mask is True iff this cell is a subbasin's outlet
            self.mask_sim = torch.zeros((self.out_lats.shape[0], self.out_lats.shape[1]), dtype=torch.bool)
            for subbasin in self.simulated_streamflow['subbasin'].unique():
                row, col = self.outlet_to_row_col[subbasin]
                self.mask_sim[row, col] = True
                subbasin_streamflow = self.simulated_streamflow[self.simulated_streamflow['subbasin']==subbasin].set_index('date')
                for i in range(len(self.dates)):
                    if self.dates[i] in subbasin_streamflow.index:
                        self.y_sim[i, row, col] = subbasin_streamflow.loc[self.dates[i], 'simulated_streamflow']
            
        self.x_conv = torch.from_numpy(self.x_conv).float()
        # Create a tensor of shape (#days, height, width) of target values (only those cells where we have stations get populated)
        self.y = torch.zeros((len(self.dates), self.out_lats.shape[0], self.out_lats.shape[1]))
        # Mask is True iff we have an actual streamflow value for this cell at this date
        self.mask = torch.zeros((len(self.dates), self.out_lats.shape[0], self.out_lats.shape[1]), dtype=torch.bool)
        for station in data_runoff['station'].unique():
            row, col = self.station_to_index[station]
            station_runoff = data_runoff[data_runoff['station']==station].set_index('date')
            for i in range(len(self.dates)):
                if self.dates[i] in station_runoff.index:
                    self.mask[i, row, col] = True
                    self.y[i, row, col] = station_runoff.loc[self.dates[i], 'runoff']
                    
    def __getitem__(self, index):
        if self.include_simulated_streamflow:
            return {'x_conv': self.x_conv[index,:,:,:,:], 'y': self.y[index], 'mask': self.mask[index],
                    'y_sim': self.y_sim[index], 'mask_sim': self.mask_sim}
        else:
            return {'x_conv': self.x_conv[index,:,:,:,:], 'y': self.y[index], 'mask': self.mask[index]}
    
    def __len__(self):
        return self.y.shape[0]

    
class GeophysicalGridDataset(IterableDataset):
    """ Dataset of geophysical gridded inputs. """
    def __init__(self, dem=True, landcover=True, soil=True, groundwater=True, min_lat=None, max_lat=None, min_lon=None, max_lon=None, landcover_types=None, scalers=None):

        self.scalers = scalers
        
        min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = load_data.get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
        self.item = np.zeros((0,min_lat_idx - max_lat_idx,max_lon_idx - min_lon_idx))
        
        if dem:
            self.dem = load_data.load_dem(min_lat, max_lat, min_lon, max_lon)
            self.item = np.concatenate([self.item, self.dem.reshape((1,self.dem.shape[0], self.dem.shape[1]))], axis=0)
        if landcover:
            self.landcover, self.landcover_legend = load_data.load_landcover(landcover_types, min_lat, max_lat, min_lon, max_lon)
            self.item = np.concatenate([self.item, self.landcover], axis=0)
        if soil:
            self.soil, self.soil_legend = load_data.load_soil(min_lat, max_lat, min_lon, max_lon)
            self.item = np.concatenate([self.item, self.soil], axis=0)
        if groundwater:
            self.groundwater = load_data.load_groundwater(min_lat, max_lat, min_lon, max_lon)
            self.item = np.concatenate([self.item, self.groundwater.reshape((1, self.groundwater.shape[0], self.groundwater.shape[1]))], axis=0)
            
        # Scale training data
        if self.scalers is None:
            self.scalers = list(preprocessing.StandardScaler() for _ in range(self.item.shape[0]))
        for i in range(self.item.shape[0]):
            self.item[i,:,:] = np.nan_to_num(self.scalers[i].fit_transform(self.item[i,:,:].reshape((-1, 1)))\
                                                                 .reshape(self.item[i,:,:].shape))
            
        lats, lons = load_data.load_dem_lats_lons()
        self.lats = np.tile(lats,len(lons)).reshape(len(lons),-1).T[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]
        self.lons = np.tile(lons,len(lats)).reshape(len(lats),-1)[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx]
        
    def __iter__(self):
        while True:
            yield self.item
