import pandas as pd
import numpy as np
import os
import netCDF4 as nc
from datetime import datetime, timedelta
import pickle
import dill
import torch

module_dir = os.path.dirname(os.path.abspath(__file__))

landcover_legend = {1: 'Temperate or sub-polar needleleaf forest',
        2:  'Sub-polar taiga needleleaf forest',
        3:  'Tropical or sub-tropical broadleaf evergreen forest', 
        4:  'Tropical or sub-tropical broadleaf deciduous forest',
        5:  'Temperate or sub-polar broadleaf deciduous forest',
        6:  'Mixed forest',
        7:  'Tropical or sub-tropical shrubland',
        8:  'Temperate or sub-polar shrubland',
        9:  'Tropical or sub-tropical grassland',
        10: 'Temperate or sub-polar grassland',
        11: 'Sub-polar or polar shrubland-lichen-moss',
        12: 'Sub-polar or polar grassland-lichen-moss',
        13: 'Sub-polar or polar barren-lichen-moss',
        14: 'Wetland',
        15: 'Cropland',
        16: 'Barren lands',
        17: 'Urban',
        18: 'Water',
        19: 'Snow and Ice'}


def load_discharge_gr4j_vic():
    """Loads observed discharge for gauging stations in GRIP-E objective 1 & 2.
    
    Returns:
        A pd.DataFrame with columns [date, station, runoff], where 'runoff' contains the streamflow.
    """
    dir = module_dir + '/../data/ObservedDischarge_GR4J+VIC'
    data_streamflow = pd.DataFrame(columns=['date','runoff', 'station'])
    
    for f in os.listdir(dir):
        if not f.endswith('.rvt'):
            continue
        data = pd.read_csv(os.path.join(dir, f), skiprows=2, skipfooter=1, index_col=False, header=None, names=['runoff'], na_values='-1.2345', engine='python')
        
        data['date'] = pd.date_range('2010-01-01', periods=len(data), freq='D')
        data['station'] = f[11:-4]
        data['runoff'] = data['runoff'].astype(float)
        data_streamflow = data_streamflow.append(data, ignore_index=True, sort=True)
    
    return data_streamflow


def load_simulated_streamflow(lats=None, lons=None):
    """Loads VIC-GRU+Raven streamflow simulation for all subbasins.
    
    Loads streamflow as obtained by simulating runoff with the VIC-GRU phsyically-based model and routing using the Raven routing model.
    The simulations are based on subbasins, so this method returns one streamflow value per date and subbasin.
    Additionally, this method creates a mapping from subbasin to a grid cell in either the provided grid or the grid obtained from load_dem_lat_lons().
    
    Args:
        lats (np.ndarray): 1-dimensional array of latitudes of the grid
        lons (np.ndarray): 1-dimensional array of longitudes of the grid
    Returns:
        A pd.DataFrame of simulations for each subbasin
        A dict mapping each subbasin id to (row, col) in the grid of lat and lon values.
    """
    simulation_per_subbasin = pd.read_csv(module_dir + '/../data/GRIP-E_Hydrographs_VIC-GRU_test2013+2014_allSubbasins.csv').drop(['time', 'hour', 'precip [mm/day]'], axis=1)
    subid_to_gauge = pd.read_csv(module_dir + '/../data/VIC-GRU_subid2gauge.csv').set_index('SubId')[['ID', 'Name']].rename({'ID': 'StationID', 'Name': 'StationName'}, axis=1)
    
    # Load actual gauge stations
    streamflow_stations = load_discharge_gr4j_vic()['station'].unique()
    subid_to_gauge = subid_to_gauge[subid_to_gauge['StationID'].isin(streamflow_stations)]
    subbasin_lat_lons = pd.read_csv(module_dir + '/../data/simulations_shervan/GRIP-E_subbasin_outlet_info_20190812.csv').set_index('SubId')[['Outlet_lon', 'Outlet_lat']]

    simulation_per_subbasin = simulation_per_subbasin.rename(lambda c: int(c[3:].replace(' [m3/s]', '')) if c not in ['date', 'hour', 'precip [mm/day]'] else c, axis=1)
    simulation_per_subbasin = simulation_per_subbasin.set_index('date').transpose().unstack().reset_index().rename({'level_0': 'date', 'level_1': 'subbasin', 0: 'simulated_streamflow'}, axis=1)
    simulation_per_subbasin['date'] = pd.to_datetime(simulation_per_subbasin['date'])
    simulation_per_subbasin['date'] = simulation_per_subbasin['date'] - timedelta(days=1)  # Need to lag raven output back by one day.
    simulation_per_subbasin = simulation_per_subbasin.join(subbasin_lat_lons, on='subbasin')
    simulation_per_subbasin = simulation_per_subbasin.join(subid_to_gauge, how='left', on='subbasin')

    # Create mapping subbasin -> (row, col)
    if lats is None:
        lats, lons = load_dem_lats_lons()
    subbasin_outlet_to_index = {}
    for subbasin in simulation_per_subbasin['subbasin'].unique():
        outlet_lat, outlet_lon = subbasin_lat_lons.loc[subbasin, ['Outlet_lat', 'Outlet_lon']]
        # find nearest cell
        outlet_row = np.argmin(np.square(lats - outlet_lat))
        outlet_col = np.argmin(np.square(lons - outlet_lon))
        subbasin_outlet_to_index[subbasin] = (outlet_row, outlet_col)
    
    return simulation_per_subbasin, subbasin_outlet_to_index

    
def load_rdrs_forcings(as_grid=False, include_lat_lon=False):
    """Loads RDRS forcings.
    
    Loads hourly meteorological RDRS forcings. Note that the forcings are in rotated latitude/longitude CRS.
    
    Args:
        as_grid (bool): If False, will flatten returned forcing rows and columns into columns. 
        include_lat_lon (bool): If True and as_grid is True, will additionally return latitudes and longitudes of the forcing dataset.
    Returns:
        If not as_grid: A pd.DataFrame with dates as index and one column per variable and RDRS grid cell
        If as_grid: 
            A np.ndarray of shape (#timesteps, #vars, #rows, #cols) of forcing data
            A list of length #vars of variable names
            A pd.date_range of length #timesteps, and (if specified) lat and lon arrays)
            If include_lat_lon: An array of length #rows of latitudes and an array of length #cols of longitudes in rotated lat/lon CRS.
    """
    forcing_variables = ['RDRS_FB_SFC', 'RDRS_FI_SFC', 'RDRS_HU_40m', 'RDRS_P0_SFC', 'RDRS_PR0_SFC', 'RDRS_TT_40m', 'RDRS_UVC_40m', 'RDRS_WDC_40m']
    rdrs_nc = nc.Dataset(module_dir + '/../data/RDRS_CaPA24hr_forcings_final.nc', 'r')
    
    if as_grid:
        time_steps, nrows, ncols = rdrs_nc[forcing_variables[0]].shape
        rdrs_data = np.zeros((time_steps, len(forcing_variables), nrows, ncols))
        for i in range(len(forcing_variables)):
            rdrs_data[:,i,:,:] = rdrs_nc[forcing_variables[i]][:]
        
        if include_lat_lon:
            return_values = rdrs_data, forcing_variables, pd.Series(pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H')), rdrs_nc['lat'][:], rdrs_nc['lon'][:]
        else:
            return_values = rdrs_data, forcing_variables, pd.Series(pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H'))
        rdrs_nc.close()
        return return_values
    else:
        rdrs_data = pd.DataFrame(index=pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H')) # Using 7:00 because forcings are UTC, while streamflow is local time

        for var in forcing_variables:
            var_data = pd.DataFrame(rdrs_nc[var][:].reshape(43825,34*39))
            var_data.columns = [var + '_' + str(c) for c in var_data.columns]
            rdrs_data.reset_index(drop=True, inplace=True)
            rdrs_data = rdrs_data.reset_index(drop=True).join(var_data.reset_index(drop=True))
        rdrs_data.index = pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H')

        rdrs_nc.close()
        return rdrs_data


def pickle_results(name, results, time_stamp):
    """Pickles prediction results to disk.
    
    Writes a pd.DataFrame with columns [date, station, actual, prediction] to pickle/results/
    
    Args:
        name (str): Name of the model
        results: Either a list of (station_name, prediction-DataFrame, actual-DataFrame) or a tuple of (dict(station_name -> prediction-DataFrame), dict(station_name -> actual-DataFrame)).
                    The prediction-DataFrame is expected to have a column 'runoff' with predicted values.
        time_stamp: Time stamp of model run, used for unique identification.
    Returns:
        A string with the name of the saved file.
    """
    result_df = None
    if isinstance(results, list):
        result_list = results
    elif isinstance(results, tuple):
        result_list = []
        for station, predict in results[0].items():
            result_list.append((station, predict, results[1][station]))
    elif isinstance(results, pd.DataFrame):
        result_df = results
    else:
        raise Exception('invalid result format')
        
    if result_df is None:
        result_df = pd.DataFrame()
        for result in result_list:
            station, prediction, actual = result
            df = prediction[['runoff']].rename(columns={'runoff': 'prediction'})
            df['actual'] = actual
            df['station'] = station
            result_df = result_df.append(df.reset_index())
    
    file_name = '{}_{}.pkl'.format(name, time_stamp)
    pickle.dump(result_df, open(module_dir + '/../pickle/results/' + file_name, 'wb'))
    
    return file_name


def pickle_model(name, model, station, time_stamp, model_type='torch'):
    """Pickles a model to disk.
    
    Writes the passed model to pickle/models/
    
    Args:
        name (str): Name of the model.
        model: The model to save, e.g. an sklearn, XGBoost or torch model.
        station (str): A string describing which station the model is trained for.
        time_stamp: Time stamp of model run, used for unique identification.
        model_type (str): If 'torch', will use torch.save to pickle the model. If 'torch.dill', will use torch.save(...,pickle_module=dill). 
                          If 'xgb' or other, will use pickle.dump.
    """
    file_name = module_dir + '/../pickle/models/{}_{}_{}.pkl'.format(name, station, time_stamp)
    if model_type == 'torch':
        torch.save(model, file_name)
    elif model_type == 'torch.dill':
        torch.save(model, file_name, pickle_module=dill)
    elif model_type == 'xgb':
        pickle.dump(model, open(file_name, 'wb'))
    else:
        pickle.dump(model, open(file_name, 'wb'))
    print('Saved model as', file_name)
    
    
def save_model_with_state(name, epoch, model, optimizer, time_stamp, use_dill=False):
    """Pickles a PyTorch model including state.
    
    Writes the passed model, epoch and optimizer to pickle/models/, including their states.
    
    Args:
        name (str): Name of the model.
        epoch (int): Training epoch.
        model: PyTorch model to save.
        optimizer: Optimizer used in training.
        time_stamp: Time stamp of model run, used for unique identification.
        use_dill: If True, will use dill instead of pickle to pickle the model. Needed if a model contains lambdas, which pickle can't pickle.
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, module_dir + '/../pickle/models/{}_{}_state.pkl'.format(name, time_stamp))
    torch.save(optimizer, module_dir + '/../pickle/models/{}_{}_optimizer.pkl'.format(name, time_stamp))
    if use_dill:
        pickle_model(name, model, 'allStations', time_stamp, model_type='torch.dill')
    else:
        pickle_model(name, model, 'allStations', time_stamp, model_type='torch')
    
    
def load_model_and_state(name, time_stamp, device, optimizer=None, use_dill=False):
    """Load a model and state from disk.
    
    Loads a model, optimizer and epoch as saved by save_model_with_state.
    
    Args:
        name (str): Name of the model.
        time_stamp: Time stamp of model run, used for unique identification.
        device: Device on which to load the model, e.g. a GPU device
        optimizer: Torch optimizer to initialize with saved state
        use_dill (bool, default False): If True, will load the model using dill instead of pickle.
    Returns:
        A tuple (model, optimizer, epoch) or (model, epoch), depending on whether an optimizer was passed.
    """
    model_path = module_dir + '/../pickle/models/{}_allStations_{}.pkl'.format(name, time_stamp)
    if use_dill:
        model = torch.load(model_path, pickle_module=dill, map_location=device)
    else:
        model = torch.load(model_path, map_location=device)
    state = torch.load(module_dir + '/../pickle/models/{}_{}_state.pkl'.format(name, time_stamp), map_location=device)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        return model, optimizer, state['epoch']
    return model, state['epoch']


def load_train_test_gridded_dividedStreamflow():
    """Fast loading of train and test data of hourly streamflow.
    
    Load train and test data from HDF5 for predictions on gridded forcings with streamflow divided into hourly data.
    If no HDF5 file exists yet, it is created for fast future loading.
    
    Returns:
        A dict mapping stations to pd.DataFrames of hourly streamflow.
    """
    file_name = module_dir + '/../data/train_test/gridded_dividedStreamflow.h5'
    if not os.path.isfile(file_name):
        history = 7 * 24
        data_streamflow = load_discharge_gr4j_vic()
        station_cell_mapping = get_station_cell_mapping()
        rdrs_data = load_rdrs_forcings()
        month_onehot = pd.get_dummies(data_streamflow['date'].dt.month, prefix='month', columns=['month'])
        data_streamflow = data_streamflow.join(month_onehot)

        for station in data_streamflow['station'].unique():
            station_data = data_streamflow[data_streamflow['station'] == station].set_index('date')
            station_cell_ids = 39 *( station_cell_mapping[station_cell_mapping['station'] == station]['col'] - 1) \
                + (station_cell_mapping[station_cell_mapping['station'] == station]['row'] - 1)
            station_rdrs = rdrs_data.filter(regex='_(' + '|'.join(map(lambda x: str(x), station_cell_ids)) + ')$', axis=1)

            if any(station_data['runoff'].isna()):
                station_data = station_data[~pd.isna(station_data['runoff'])]
                print('Station', station, 'had NA streamflow values')

            station_data = station_data.resample('1H').ffill()
            station_data['runoff'] = station_data['runoff'] / 24
            station_data = station_data.join(station_rdrs)
            for i in range(1, history + 1):
                station_data = station_data.join(station_rdrs.shift(i, axis=0), rsuffix='_-{}'.format(i))
                
            station_data.to_hdf(file_name, 'station_' + station, complevel=5)
            
    return read_station_data_dict(file_name)


def load_train_test_gridded_aggregatedForcings(include_all_forcing_vars=False, include_all_cells=False):
    """Fast loading of train and test data of forcings aggreagted to daily values.
    
    Load train and test data from HDF5 for predictions on gridded forcings, aggregated into days.
    If no HDF5 file exists yet, it is created for fast future loading.
    
    Args:
        include_all_forcing_vars (bool): If True, will return min/max-aggregation for all variables and sum-aggregation for precipitation. Else, will return min/max-temperature and sum-precipitation.
        include_all_cells (bool): If False, will only return cells belonging to the station's subwatershed
    Returns:
        A dict mapping stations to pd.DataFrames of day-aggregated forcings.
    """
    file_name = module_dir + '/../data/train_test/gridded_aggregatedForcings{}{}.h5'.format('_all_vars' if include_all_forcing_vars else '', 
                                                                                            '_all_cells' if include_all_cells else '')
    if not os.path.isfile(file_name):
        history = 7
        data_streamflow = load_discharge_gr4j_vic()
        station_cell_mapping = get_station_cell_mapping()
        
        rdrs_data = load_rdrs_forcings()
        resampled = rdrs_data.resample('D')
        rdrs_daily = resampled.sum().join(resampled.min(), lsuffix='_sum', rsuffix='_min').join(resampled.max().rename(lambda c: c + '_max', axis=1))
        month_onehot = pd.get_dummies(data_streamflow['date'].dt.month, prefix='month', columns=['month'])
        data_streamflow = data_streamflow.join(month_onehot)
        
        for station in data_streamflow['station'].unique():
            station_data = data_streamflow[data_streamflow['station'] == station].set_index('date')
            if include_all_cells:
                station_cell_ids = ['\w*']
            else:
                station_cell_ids = 39 * (station_cell_mapping[station_cell_mapping['station'] == station]['col'] - 1) \
                    + (station_cell_mapping[station_cell_mapping['station'] == station]['row'] - 1)

            if not include_all_forcing_vars:
                # For temperature use min/max aggregation. Precipitation: sum. solar fluxes, pressure & humidity don't seem to help (at least with min/max/sum)
                regex = '((RDRS_TT_40m)_({0})_(min|max)|(RDRS_PR0_SFC)_({0})_sum)$'.format('|'.join(map(lambda x: str(x), station_cell_ids)))
            else:
                regex = '(_({0})_(min|max)|(RDRS_PR0_SFC)_({0})_sum)$'.format('|'.join(map(lambda x: str(x), station_cell_ids)))
            station_rdrs = rdrs_daily.filter(regex=regex, axis=1)
            if any(station_data['runoff'].isna()):
                station_data = station_data[~pd.isna(station_data['runoff'])]
                print('Station', station, 'had NA streamflow values')

            station_data = station_data.join(station_rdrs)
            for i in range(1, history + 1):
                station_data = station_data.join(station_rdrs.shift(i, axis=0), rsuffix='_-{}'.format(i))

            station_data.to_hdf(file_name, 'station_' + station, complevel=5)
            
    return read_station_data_dict(file_name)


def load_train_test_lstm():
    """Fast loading of train and test data of LSTM input.
    
    Load train and test data for LSTM from HDF5. If no HDF5 file exists yet, it is created for fast future loading.
    
    Returns:
        A dict mapping stations to pd.DataFrames with hourly RDRS variables for the station's subwatershed. Also includes one-hot-encoded months.
    """
    file_name = module_dir + '/../data/train_test/lstm.h5'
    if not os.path.isfile(file_name):
        data_streamflow = load_discharge_gr4j_vic()
        # For each station, read which grid cells belong to its subwatershed
        station_cell_mapping = get_station_cell_mapping()
        rdrs_data = load_rdrs_forcings()

        for station in data_streamflow['station'].unique():
            station_cell_ids = 39 * (station_cell_mapping[station_cell_mapping['station'] == station]['col'] - 1) \
                + (station_cell_mapping[station_cell_mapping['station'] == station]['row'] - 1)
            station_rdrs = rdrs_data.filter(regex='_(' + '|'.join(map(lambda x: str(x), station_cell_ids)) + ')$', axis=1)

            month_onehot = pd.get_dummies(station_rdrs.index.month, prefix='month')
            month_onehot.index = station_rdrs.index
            station_rdrs = station_rdrs.join(month_onehot)

            station_rdrs.to_hdf(file_name, 'station_' + station)
    
    return read_station_data_dict(file_name)


def get_station_cell_mapping():
    """For each station, reads which RDRS grid cells belong to its subwatershed.
    
    Returns:
        A pd.DataFrame with one row per RDRS grid cell, mapping each cell to a station.
    """
    return pd.read_csv(module_dir + '/../data/station_cell_mapping.csv', skiprows=1, names=['station', 'lat', 'lon', 'row', 'col', 'area'])


def read_station_data_dict(file_name):
    """Helper method to read HDF5 files of per-station data stores.
    
    Returns:
        A dict mapping each station to its pd.DataFrame in the file.
    """
    station_data_dict = {}
    with pd.HDFStore(file_name,  mode='r') as store:
        for station in store.keys():
            station_name = station[9:]
            station_data_dict[station_name] = store[station]
            
    return station_data_dict


def load_landcover_reduced(values_to_use=None):
    """Loads landcover data, downsampled to RDRS resolution.
    
    Loads landcover data, cropped out for lake erie watershed, and reduces resolution to RDRS data shape.
    
    Args:
        values_to_use (list(int) or None): If None, returns all landtypes. Else, returns only the specified types.
    Returns:
        A np.ndarray of shape (#landtypes, #rows, #cols), where the first dimension is the averaged amount of this landtype in cell (row, col), 
        A list of the names of the returned landtypes
    """
    rdrs_data, rdrs_vars, rdrs_dates = load_rdrs_forcings(as_grid=True)
    landcover_nc = nc.Dataset(module_dir + '/../data/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_erie.nc', 'r')
    landcover_fullres = np.array(landcover_nc['Band1'][:])[::-1,:]

    if values_to_use is None:
        values_to_use = landcover_legend.keys()

    pixels_per_row = (landcover_fullres.shape[0] // rdrs_data.shape[2]) + 1
    pixels_per_col = (landcover_fullres.shape[1] // rdrs_data.shape[3]) + 1

    landcover_reduced = np.zeros((len(values_to_use), rdrs_data.shape[2], rdrs_data.shape[3]))
    for row in range(landcover_reduced.shape[1]):
        for col in range(landcover_reduced.shape[2]):
            landcover_cell = landcover_fullres[row*pixels_per_row:(row+1)*pixels_per_row, col*pixels_per_col:(col+1)*pixels_per_col]
            non_zero_pixels_per_cell = (landcover_cell.flatten() != 0).sum()
            i = 0
            for k in landcover_legend.keys():
                if k not in values_to_use:
                    continue
                if non_zero_pixels_per_cell == 0:
                    landcover_reduced[i, row, col] = 0.0
                else:
                    landcover_reduced[i, row, col] = np.float((landcover_cell == k).sum()) / non_zero_pixels_per_cell
                i += 1
                
    landcover_nc.close()                
    return landcover_reduced, list(landcover_legend[i] for i in values_to_use)


def load_landcover(values_to_use=None, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """Loads landcover data at 30" resolution.
    
    Loads landcover data, cropped out for lake erie watershed, and reduces resolution to RDRS data shape.
    When first called, will create a NetCDF file with the data at 30" for faster future loading.
    
    Args:
        values_to_use (list(int) or None): If None, returns all landtypes. Else, returns only the specified types.
        min_lat, max_lat, min_lon, max_lon: Restrict returned array to the bounding box of these coordinates.
    Returns:
        A np.ndarray of shape (#landtypes, #rows, #cols), where the first dimension is the averaged amount of this landtype in cell (row, col), 
        A list of the names of the returned landtypes
    """
    filename = module_dir + '/../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_30sec.nc'
    
    if not os.path.isfile(filename):
        import gdal
        landcover_nc = nc.Dataset(module_dir + '/../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90.nc', 'r')
        landcover = landcover_nc['Band1'][:].filled(np.nan)
        dem_lats, dem_lons = load_dem_lats_lons()
        
        landcover_30sec_nc = nc.Dataset(module_dir + '/../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_30sec.nc', 'w')
        landcover_30sec_nc.setncattr('Conventions', 'CF-1.6')
        landcover_30sec_nc.createDimension('lat')
        landcover_30sec_nc.createDimension('lon')
        landcover_30sec_nc.createVariable('crs', 'S1')
        for attr in landcover_nc['crs'].ncattrs():
            landcover_30sec_nc['crs'].setncattr(attr, landcover_nc['crs'].getncattr(attr))
        landcover_30sec_nc.createVariable('lat', np.float64, dimensions=('lat'))
        landcover_30sec_nc.createVariable('lon', np.float64, dimensions=('lon'))
        landcover_30sec_nc['lat'][:] = dem_lats
        landcover_30sec_nc['lon'][:] = dem_lons
        for attr in landcover_nc['lat'].ncattrs():
            landcover_30sec_nc['lat'].setncattr(attr, landcover_nc['lat'].getncattr(attr))
        for attr in landcover_nc['lon'].ncattrs():
            landcover_30sec_nc['lon'].setncattr(attr, landcover_nc['lon'].getncattr(attr))
        
        # gdal.Warp can only resample one band at a time. Hence, resample each landtype separately and successively merge into _30sec.nc.
        for i, landtype in landcover_legend.items():
            print(landtype)
            landcover_temp_nc = nc.Dataset(module_dir + '/../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_temp.nc', 'w')
            landcover_temp_nc.createDimension('lat')
            landcover_temp_nc.createDimension('lon')
            landcover_temp_nc.createVariable('crs', 'S1')
            landcover_temp_nc.setncattr('Conventions', 'CF-1.6')
            for attr in landcover_nc['crs'].ncattrs():
                landcover_temp_nc['crs'].setncattr(attr, landcover_nc['crs'].getncattr(attr))
            landcover_temp_nc.createVariable('lat', np.float64, dimensions=('lat'))
            landcover_temp_nc.createVariable('lon', np.float64, dimensions=('lon'))
            landcover_temp_nc['lat'][:] = landcover_nc['lat'][:]
            landcover_temp_nc['lon'][:] = landcover_nc['lon'][:]
            for attr in landcover_nc['lat'].ncattrs():
                landcover_temp_nc['lat'].setncattr(attr, landcover_nc['lat'].getncattr(attr))
            for attr in landcover_nc['lon'].ncattrs():
                landcover_temp_nc['lon'].setncattr(attr, landcover_nc['lon'].getncattr(attr))


            varname = 'landtype_{}'.format(i)
            landcover_temp_nc.createVariable(varname, np.float, dimensions=('lat', 'lon'))
            landcover_temp_nc[varname][:] = (landcover == i).astype(np.float)
            landcover_temp_nc.close()

            gdal_temp = gdal.Open(module_dir + '/../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_temp.nc')
            warp_options = gdal.WarpOptions(format='netCDF', xRes=0.008333333333333333333, yRes=0.008333333333333333333, resampleAlg='average')
            print('Warping...')
            gdal.Warp(module_dir + '/../data/geophysical/landcover/landtype_temp.nc'.format(varname), gdal_temp, options=warp_options)
            print('Warping complete.')
            landtype_temp = nc.Dataset(module_dir + '/../data/geophysical/landcover/landtype_temp.nc', 'r')
            landcover_30sec_nc.createVariable(varname, 'f', dimensions=('lat', 'lon'))
            landcover_30sec_nc[varname][:] = landtype_temp['Band1'][:]
            landcover_30sec_nc[varname].setncattr('landtype', landtype)

            landtype_temp.close()
            os.remove(module_dir + '/../data/geophysical/landcover/landtype_temp.nc')
            os.remove(module_dir + '/../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_temp.nc')
        
        landcover_nc.close()
        landcover_30sec_nc.close()
    
    if values_to_use is None:
        values_to_use = list(landcover_legend.keys())
    
    landcover_nc = nc.Dataset(module_dir + '/../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_30sec.nc', 'r')
    landcover = np.zeros((0, landcover_nc['lat'].shape[0], landcover_nc['lon'].shape[0]))
    legend = []
    for i in range(len(values_to_use)):
        landcover_i = landcover_nc['landtype_{}'.format(values_to_use[i])][:][::-1,:].filled(np.nan)
        if landcover_i.sum() == 0:
            continue
        landcover = np.concatenate([landcover, landcover_i.reshape((1,landcover.shape[1],landcover.shape[2]))], axis=0)
        legend.append(landcover_legend[values_to_use[i]])
    
    landcover_nc.close()
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return landcover[:,max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx].copy(), legend


def load_dem(min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """Loads digital elevation map at 30" resolution.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Restrict returned array to the bounding box of these coordinates.
    Returns:
        A np.ndarray of shape (#rows, #cols) with the DEM information.
    """
    dem_nc = nc.Dataset(module_dir + '/../data/geophysical/dem/hydrosheds_n40-45w75-90_30sec.nc', 'r')
    dem = dem_nc['Band1'][:][::-1,:].filled(np.nan)

    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return dem[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx].copy()

def load_groundwater(min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """Loads groundwater data at 30" resolution.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Restrict returned array to the bounding box of these coordinates.
    Returns:
        A np.ndarray of shape (#rows, #cols) with the groundwater table depth.
    """
    groundwater_nc = nc.Dataset(module_dir + '/../data/geophysical/groundwater/N_America_model_wtd_v2_n40-45w75-90.nc', 'r')
    groundwater = groundwater_nc['Band1'][:][::-1,:].filled(np.nan)
    
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return groundwater[max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx].copy()

def load_soil(min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """Loads soil data at 30" resolution.
    
    The soil data contains information on the sand and clay content for eight soil layers.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Restrict returned array to the bounding box of these coordinates.
    Returns:
        A np.ndarray of shape (2 * 8, #rows, #cols), where the first 4 entries are sand and the last 4 entries are clay information.
    """
    soiltypes = ['SAND', 'CLAY']
    soil_nc = nc.Dataset(module_dir + '/../data/geophysical/soil/SAND1_n40-45w75-90.nc', 'r')
    soil = np.zeros((len(soiltypes) * 8, soil_nc['lat'].shape[0], soil_nc['lon'].shape[0]))
    soil_nc.close()

    # Each nc file contains 4 soil layers; per soil type there are 2 nc files.
    soil_legend = []
    for i in range(len(soiltypes)):
        for j in [1,2]:
            soil_nc = nc.Dataset(module_dir + '/../data/geophysical/soil/{}{}_n40-45w75-90.nc'.format(soiltypes[i], j), 'r')
            for layer in range(1,5):
                soil[i*8 + ((j-1)*4 + layer-1)] = soil_nc['Band{}'.format(layer)][:][::-1,:]\
                        .astype(np.float).filled(np.nan) / 100.0
                soil_legend.append('{}-layer{}'.format(soiltypes[i], (j-1)*4 + layer))
            soil_nc.close()
    
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return soil[:,max_lat_idx:min_lat_idx,min_lon_idx:max_lon_idx].copy(), soil_legend


def load_dem_lats_lons():
    """Returns latitudes and longitudes of the 30" DEM dataset.
    
    Note that latitudes are decreasing with index.
    
    Returns:
        A 1-dimensional np.ndarray of latitudes,
        A 1-dimensional np.ndarray of longitudes
    """
    dem_nc = nc.Dataset(module_dir + '/../data/geophysical/dem/hydrosheds_n40-45w75-90_30sec.nc', 'r')
    dem_nc.set_auto_mask(False)
    lats = dem_nc['lat'][:][::-1]
    lons = dem_nc['lon'][:]
    dem_nc.close()
    
    return lats, lons


def get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon):
    """Calculates indices to subset a dataset to the given bounding box, assuming the 30" DEM grid.
    
    Returns indices to split the 30" datasets such that they only contain lats/lons within the specified bounding box.
    Note that min_lat_idx will be larger than max_lat_idx, because latitudes decrease with index.
    
    Args:
        min_lat, max_lat, min_lon, max_lon: Bounding box coordinates.
    Returns:
        min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx: Indices into latitude and longitude arrays to restrict to the specified coordinates.
    """
    lats, lons = load_dem_lats_lons()
    lats = lats[::-1]  # lats go from high to low
    
    if min_lat is None:
        min_lat = lats.min()
    if max_lat is None:
        max_lat = lats.max()
    if min_lon is None:
        min_lon = lons.min()
    if max_lon is None:
        max_lon = lons.max()
        
    if min_lat > lats.max() or max_lat < lats.min() or min_lon > lons.max() or max_lon < lons.min():
        raise Exception('Empty lat/lon selection.')

    min_lat_idx = len(lats) - (lats >= min_lat).argmax() - 1
    max_lat_idx = len(lats) - (lats >= max_lat).argmax() - 1 if max_lat <= lats.max() else 0
    min_lon_idx = (lons >= min_lon).argmax()
    max_lon_idx = (lons >= max_lon).argmax() if max_lon < lons.max() else len(lons)
    
    return min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx
    