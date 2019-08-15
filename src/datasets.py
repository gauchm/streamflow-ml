import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
import netCDF4 as nc
from torch.utils.data import Dataset, IterableDataset
from src import load_data, utils

module_dir = os.path.dirname(os.path.abspath(__file__))


class RdrsDataset(Dataset):
    """RDRS dataset where target is a list of streamflow values per gauging station.
    
    Attributes:
        date_start, date_end: First and last date in the dataset
        dates: list of dates in the dataset
        seq_len, seq_steps: Length and step-size (in hours) of RDRS input variable history for each sample
        conv_scalers, fc_scalers: sklearn.preprocessing-scalers used to scale time-series and non-time-series input data.
        n_conv_vars, n_fc_vars: Number of time-series and non-time-series input features.
        conv_height, conv_width: Number of rows and columns of the time-series variables
        data_runoff: pd.DataFrame of target streamflow values per date and station
        x_conv, x_fc: Torch tensors of time-series and non-time-series input data
        y: Torch tensor of target streamflow values
    """
    def __init__(self, rdrs_vars, seq_len, seq_steps, date_start, date_end, conv_scalers=None, fc_scalers=None, include_months=True):
        """Initializes RdrsDataset.
        
        Args:
            rdrs_vars (list(int)): List of RDRS variables to use.
            seq_len: Length of history of RDRS values to include in the samples.
            seq_steps: Step-size of the history, in hours.
            date_start: First date for which the dataset will provide target streamflow values.
            date_end: Last date for which the dataset will provide target streamflow values.
            conv_scalers: If provided, will use this list of sklearn.preprocessing scalers to scale the time-series input data.
            fc_scalers: If provided, will use this list of sklearn.preprocessing scalers to scale the non-time-series input data.
            include_months (bool): If True, the time-series input data will contain one-hot-encoded months as features.
        """
        self.date_start = date_start
        self.date_end = date_end
        self.seq_len = seq_len
        self.seq_steps = seq_steps
        self.conv_scalers = conv_scalers
        self.fc_scalers = fc_scalers
        
        rdrs_data, rdrs_var_names, rdrs_time_index = load_data.load_rdrs_forcings(as_grid=True)
        rdrs_data = rdrs_data[:,rdrs_vars,:,:]
        self.n_conv_vars = len(rdrs_vars)
        self.conv_height = rdrs_data.shape[2]
        self.conv_width = rdrs_data.shape[3]
        
        data_streamflow = load_data.load_discharge_gr4j_vic()
        data_streamflow = data_streamflow[~pd.isna(data_streamflow['runoff'])]
        if include_months:
            data_streamflow = data_streamflow.join(pd.get_dummies(data_streamflow['date'].dt.month, prefix='month'))
        data_streamflow = data_streamflow[(data_streamflow['date'] >= self.date_start) & (data_streamflow['date'] <= self.date_end)]
        gauge_info = pd.read_csv(module_dir + '/../data/gauge_info.csv')[['ID', 'Lat', 'Lon']]
        data_streamflow = pd.merge(data_streamflow, gauge_info, left_on='station', right_on='ID').drop('ID', axis=1)
        
        # sort by date then station so that consecutive values have increasing dates (except for station cutoffs)
        # this way, a stateful lstm can be fed date ranges.
        data_streamflow = data_streamflow.sort_values(by=['station', 'date']).reset_index(drop='True')
        self.data_runoff = data_streamflow[['date', 'station', 'runoff']]
        
        # conv input shape: (samples, seq_len, variables, height, width)
        # But: x_conv is the same for all stations for the same date, so we only generate #dates samples 
        #      and feed them multiple times (as many times as we have stations for that date)
        self.x_conv = np.zeros((len(data_streamflow['date'].unique()), seq_len, self.n_conv_vars, self.conv_height, self.conv_width))       
        self.dates = np.sort(data_streamflow['date'].unique())
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

        self.x_fc = data_streamflow.drop(['date', 'station', 'runoff'], axis=1).to_numpy()
        fc_var_names = data_streamflow.drop(['date', 'station', 'runoff'], axis=1).columns
        vars_to_scale = list(c for c in fc_var_names if not c.startswith('month'))
        self.n_fc_vars = len(fc_var_names)
        if self.fc_scalers is None:
            self.fc_scalers = list(preprocessing.StandardScaler() for _ in range(self.n_fc_vars))
        for i in range(self.n_fc_vars):
            if fc_var_names[i] in vars_to_scale:
                to_transform = data_streamflow[fc_var_names[i]].to_numpy().reshape((-1,1))
                self.x_fc[:,i] = np.nan_to_num(self.fc_scalers[i].fit_transform(to_transform).reshape(self.x_fc[:,i].shape))
        
        self.x_conv = torch.from_numpy(self.x_conv).float()
        self.x_fc = torch.from_numpy(self.x_fc).float()
        self.y = torch.from_numpy(data_streamflow['runoff'].to_numpy()).float()
        
    def __getitem__(self, index):
        """Fetches a sample of input/target values.
        
        Args:
            index (int): Index of the sample.
        Returns:
            A dict containing: 'x_conv' -> tensor of shape (#timesteps, #n_conv_vars, #rows, #cols), 'x_fc' -> tensor of shape (n_fc_vars), 'y' -> Tensor of target streamflow values.
        """
        date_of_index = self.data_runoff.iloc[index]['date']
        x_conv_index = np.argmax(self.dates == date_of_index)
        
        return {'x_conv': self.x_conv[x_conv_index,:,:,:,:], 
                'x_fc': self.x_fc[index,:],
                'y': self.y[index]}
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.y.shape[0]
    
    
class StatefulBatchSampler(torch.utils.data.Sampler):
    """Sampler for stateful LSTMs.
    
    This sampler divides the dataset into batch_size chunks, and in iteration i it returns item i from each chunk.
    This makes sure that the LSTM gets consecutive input samples.
    """
    def __init__(self, data_source, batch_size):
        """Initializes the sampler.
        
        Args:
            data_source: Dataset for which to fetch values.
            batch_size: Batch size of the LSTM.
        """
        self.batch_size = batch_size
        self.num_batches = len(data_source) // batch_size
        
        if len(data_source) % batch_size != 0:
            print('Warning: Data source length not divisible by batch_size')
        
    def __iter__(self):
        """Yields a batch of input indices.
        
        Yields:
            A list of indices into data_source.
        """
        batch = []
        for i in range(self.num_batches):
            for j in range(self.batch_size):
                batch.append(i + j * self.num_batches)
            yield batch
            batch = []
        
    def __len__(self):
        """Returns the number of batches in the data_source."""
        return self.num_batches

    
class RdrsGridDataset(Dataset):
    """RDRS dataset where target is a spatial grid of streamflow values.
    
    Attributes:
        date_start, date_end: First and last date in the dataset
        dates: list of dates in the dataset
        seq_len, seq_steps: Length and step-size (in hours) of RDRS input variable history for each sample
        conv_scalers: sklearn.preprocessing-scalers used to scale time-series and non-time-series input data.
        include_simulated_streamflow: Determines whether the dataset contains simulated streamflow for virtual gauges of all subbasins.
        n_conv_vars: Number of time-series input features.
        conv_height, conv_width: Number of rows and columns of the time-series variables
        data_runoff: pd.DataFrame of target streamflow values per date and station, excluding values for exclude_stations (if those are specified)
        data_runoff_all_stations: pd.DataFrame of target streamflow values per date and station, including values for exclude_stations (if those are specified)
        simulated_streamflow: If include_simulated_streamflow, this stores the simulated streamflow for each date and subbasin.
        in_lats, in_lons: 2-d Latitude and longitude arrays of the RDRS data, in rotated lat/lon CRS.
        rdrs_target_lats, rdrs_target_lons: If resample_rdrs, these 2d-arrays will store the resampled RDRS lats/lons. Else, identical to in_lats/in_lons.
        out_lats/out_lons: 2-d arrays of lat/lon values of the output values.
        station_to_index: dict mapping each station to its (row, col) in the output grid.
        outlet_to_row_col: dict mapping each subbasin to its (row, col) in the output grid.
        x_conv: Torch tensors of time-series and non-time-series input data
        y: Torch tensor of target streamflow values
        y_means, y_sim_means: tensor of shape (#rows, #cols) containing the mean (simulated) target values for this station/subbasin.
        mask: bool tensor of shape (#dates, #rows, #cols), True iff we have an actual streamflow value for this cell at this date
        mask_sim: bool tensor of shape (#dates, #rows, #cols), True iff this cell is outlet of its subbasin.
    """
    def __init__(self, rdrs_vars, seq_len, seq_steps, date_start, date_end, conv_scalers=None, exclude_stations=[], aggregate_daily=None, 
                 include_months=False, include_simulated_streamflow=False, resample_rdrs=False, out_lats=None, out_lons=None):
        """Initializes the dataset.
        
        Args:
            rdrs_vars (list(int)): List of RDRS variables to use.
            seq_len: Length of history of RDRS values to include in the samples.
            seq_steps: Step-size of the history, in hours.
            date_start: First date for which the dataset will provide target streamflow values.
            date_end: Last date for which the dataset will provide target streamflow values.
            conv_scalers: If provided, will use this list of sklearn.preprocessing scalers to scale the time-series input data.
            exclude_stations (list or None): If provided, will exclude these stations from the dataset.
            aggregate_daily (list or None): If provided, will aggregate each RDRS time-series variable to daily values as specified. Allowed values are 'minmax', 'sum'
            include_months (bool): If True, the time-series input data will contain one-hot-encoded months as features.
            include_simulated_streamflow (bool): If True, the returned item dicts will contain both 'y' -> actual gauge stations, 'y_sim' -> simulated gauges. 
                                                 Else, the dict will not contain 'y_sim'.
            resample_rdrs: If True, will resample RDRS from rotated lat/lon to lat/lon values.
            out_lats, out_lons: If not specified, will use RDRS grid as output resolution. Else, the target value resolution will be in these coordinates.
        """
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
        self.rdrs_target_lats = self.in_lats
        self.rdrs_target_lons = self.in_lons
        if resample_rdrs:
            landcover_nc = nc.Dataset(module_dir + '/../data/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_erie.nc', 'r')
            landcover_nc.set_auto_mask(False)
            landcover_lats = landcover_nc['lat'][:][::-1]
            landcover_lons = landcover_nc['lon'][:]
            landcover_nc.close()

            rdrs_target_lats = landcover_lats[::int(np.ceil(len(landcover_lats) / rdrs_data.shape[2]))]
            rdrs_target_lons = landcover_lons[::int(np.ceil(len(landcover_lons) / rdrs_data.shape[3]))]
            self.rdrs_target_lats = np.tile(rdrs_target_lats,len(rdrs_target_lons)).reshape(len(rdrs_target_lons),-1).T
            self.rdrs_target_lons = np.tile(rdrs_target_lons,len(rdrs_target_lats)).reshape(len(rdrs_target_lats),-1)
            self.rdrs_resample_maps = utils.map_to_coords(self.in_lats, self.in_lons, self.rdrs_target_lats, self.rdrs_target_lons)
        
        self.out_lats = self.rdrs_target_lats if out_lats is None else np.tile(out_lats,len(out_lons)).reshape(len(out_lons),-1).T
        self.out_lons = self.rdrs_target_lons if out_lons is None else np.tile(out_lons,len(out_lats)).reshape(len(out_lats),-1)
        
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
        
        data_streamflow = load_data.load_discharge_gr4j_vic()
        data_streamflow = data_streamflow[~pd.isna(data_streamflow['runoff'])]
        data_streamflow = data_streamflow[(data_streamflow['date'] >= self.date_start) & (data_streamflow['date'] <= self.date_end)]
        gauge_info = pd.read_csv(module_dir + '/../data/gauge_info.csv')[['ID', 'Lat', 'Lon']]
        data_streamflow = pd.merge(data_streamflow, gauge_info, left_on='station', right_on='ID').drop('ID', axis=1)
        data_streamflow = data_streamflow.sort_values(by=['date']).reset_index(drop='True')
            
        if len(exclude_stations) > 0:
            self.data_runoff_all_stations = data_streamflow[['date', 'station', 'runoff']].copy()
            data_streamflow = data_streamflow[~data_streamflow['station'].isin(exclude_stations)].reset_index(drop=True)
        
        self.data_runoff = data_streamflow[['date', 'station', 'runoff']]
        
        # get station to (row, col) mapping in output lat/lons
        self.station_to_index = {}
        for station in data_streamflow['station'].unique():
            station_lat = data_streamflow[data_streamflow['station'] == station]['Lat'].values[0]
            station_lon = data_streamflow[data_streamflow['station'] == station]['Lon'].values[0]
            # find nearest cell
            station_idx = np.argmin(np.square(self.out_lats - station_lat) + np.square(self.out_lons - station_lon))
            station_row = station_idx // self.out_lons.shape[1]
            station_col = station_idx % self.out_lons.shape[1]
            self.station_to_index[station] = (station_row, station_col)
        
        if include_months:
            self.n_conv_vars += 12
        
        # conv input shape: (samples, seq_len, variables, height, width)
        self.x_conv = np.zeros((len(data_streamflow['date'].unique()), seq_len, self.n_conv_vars, self.conv_height, self.conv_width))       
        self.dates = np.sort(data_streamflow['date'].unique())
        i = 0
        for date in self.dates:
            # For each day that is to be predicted, cut out a sequence that ends with that day's 23:00 and is seq_len long
            if aggregate_daily is None:
                end_of_day_index = rdrs_time_index[rdrs_time_index == date].index.values[0] + 23
            else:
                end_of_day_index = rdrs_time_index[rdrs_time_index == date].index.values[0]
            
            date_data = rdrs_data[end_of_day_index - (self.seq_len * self.seq_steps) : end_of_day_index : self.seq_steps]
            if resample_rdrs:
                date_data = utils.resample_by_map(date_data, *self.rdrs_resample_maps, fill_value=np.nan)
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
            simulated_streamflow, self.outlet_to_row_col = load_data.load_simulated_streamflow(self.out_lats[:,0], self.out_lons[0])
            self.simulated_streamflow = simulated_streamflow[(simulated_streamflow['date'] >= self.date_start) & \
                                                             (simulated_streamflow['date'] <= self.date_end)]
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
            self.y_sim_means = self.y_sim.mean(dim=0)  # Used for NSE calculation
            
        self.x_conv = torch.from_numpy(self.x_conv).float()
        # Create a tensor of shape (#days, height, width) of target values (only those cells where we have stations get populated)
        self.y = torch.zeros((len(self.dates), self.out_lats.shape[0], self.out_lats.shape[1]))
        # Mask is True iff we have an actual streamflow value for this cell at this date
        self.mask = torch.zeros((len(self.dates), self.out_lats.shape[0], self.out_lats.shape[1]), dtype=torch.bool)
        for station in data_streamflow['station'].unique():
            row, col = self.station_to_index[station]
            station_streamflow = data_streamflow[data_streamflow['station']==station].set_index('date')
            for i in range(len(self.dates)):
                if self.dates[i] in station_streamflow.index:
                    self.mask[i, row, col] = True
                    self.y[i, row, col] = station_streamflow.loc[self.dates[i], 'runoff']
        self.y_means = self.y.mean(dim=0)  # Used for NSE calculation
                    
    def __getitem__(self, index):
        """Fetches a sample of input/target values.
        
        Args:
            index (int): Index of the sample.
        Returns:
            A dict containing: 'x_conv' -> tensor of shape (#timesteps, #n_conv_vars, #rows, #cols), 'y' -> Tensor of target streamflow values at gauging stations,
                               'mask' -> bool tensor of shape (#rows, #cols), True iff this cell is a station that has a target value for sample no. index,
                               'y_sim' -> tensor of target simulated streamflow values at subbasin outlets.
        """
        if self.include_simulated_streamflow:
            return {'x_conv': self.x_conv[index,:,:,:,:], 'y': self.y[index], 'mask': self.mask[index], 'y_sim': self.y_sim[index]}
        else:
            return {'x_conv': self.x_conv[index,:,:,:,:], 'y': self.y[index], 'mask': self.mask[index]}
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.y.shape[0]

    
class GeophysicalGridDataset(IterableDataset):
    """Dataset of geophysical gridded inputs. 
    
    Attributes:
        item: tensor of shape (#geophysical_vars, #rows, #cols)
        dem: tensor of DEM data
        landcover, landcover_legend: tensor of landcover data, according legend
        soil, soil_legend: tensor of soil data, according legend
        groundwater: tensor of groundwater data
        scalers: list of scalers used to scale the geophysical variables
        lats, lons: 2d-arrays of latitudes/longitudes of the grid cells
        shape: shape of the dataset.
    """
    def __init__(self, dem=True, landcover=True, soil=True, groundwater=True, min_lat=None, max_lat=None, min_lon=None, max_lon=None, landcover_types=None):
        """Initializes the dataset.
        
        Args:
            dem, landcover, soil, groundwater (bool): Indicate if the corresponding variable(s) are included in the dataset.
            min_lat, max_lat, min_lon, max_lon: Restrict returned dataset to the bounding box of these coordinates.
            landcover_types: list of landcover types to include if landcover is True. If None, will include all landtypes.
        """
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
            
        self.scalers = list(preprocessing.StandardScaler() for _ in range(self.item.shape[0]))
        for i in range(self.item.shape[0]):
            self.item[i,:,:] = np.nan_to_num(self.scalers[i].fit_transform(self.item[i,:,:].reshape((-1, 1)))\
                                                                 .reshape(self.item[i,:,:].shape))
        self.item = torch.from_numpy(self.item).float()
        
        lats, lons = load_data.load_dem_lats_lons()
        self.lats = np.tile(lats,len(lons)).reshape(len(lons),-1).T[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx].copy()
        self.lons = np.tile(lons,len(lats)).reshape(len(lats),-1)[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx].copy()
        
        self.shape = self.item.shape
        
    def __iter__(self):
        """Yields the geophysical dataset."""
        while True:
            yield self.item
