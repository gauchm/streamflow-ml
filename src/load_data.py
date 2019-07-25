import pandas as pd
import numpy as np
import os
import netCDF4 as nc
from datetime import datetime, timedelta
import pickle
import torch


def load_discharge_gr4j_vic():
    """
    Loads observed discharge for GR4J-Raven and VIC from disk.
    """
    dir = '../data/ObservedDischarge_GR4J+VIC'  # Read runoff observations
    data_runoff = pd.DataFrame(columns=['date','runoff', 'station'])
    
    for f in os.listdir(dir):
        if not f.endswith('.rvt'):
            continue
        data = pd.read_csv(os.path.join(dir, f), skiprows=2, skipfooter=1, index_col=False, header=None, names=['runoff'], na_values='-1.2345', engine='python')
        
        data['date'] = pd.date_range('2010-01-01', periods=len(data), freq='D')
        data['station'] = f[11:-4]
        data['runoff'] = data['runoff'].astype(float)
        data_runoff = data_runoff.append(data, ignore_index=True, sort=True)
    
    return data_runoff


def load_rdrs_forcings(as_grid=False, include_lat_lon=False):
    """
    Loads RDRS gridded forcings from disk. 
    If not as_grid, will flatten rows and columns into columns. 
    Else, will return tuple (array of shape (#timesteps, #vars, #rows, #cols), list of variable names, date range of length #timesteps, and (if specified) lat and lon arrays)
    """
    forcing_variables = ['RDRS_FB_SFC', 'RDRS_FI_SFC', 'RDRS_HU_40m', 'RDRS_P0_SFC', 'RDRS_PR0_SFC', 'RDRS_TT_40m', 'RDRS_UVC_40m', 'RDRS_WDC_40m']
    rdrs_nc = nc.Dataset('../data/RDRS_CaPA24hr_forcings_final.nc', 'r')
    
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
        rdrs_data = pd.DataFrame(index=pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H')) # Using 7:00 because forcings are UTC, while runoff is local time

        for var in forcing_variables:
            var_data = pd.DataFrame(rdrs_nc[var][:].reshape(43825,34*39))
            var_data.columns = [var + '_' + str(c) for c in var_data.columns]
            rdrs_data.reset_index(drop=True, inplace=True)
            rdrs_data = rdrs_data.reset_index(drop=True).join(var_data.reset_index(drop=True))
        rdrs_data.index = pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H')

        rdrs_nc.close()
        return rdrs_data


def pickle_results(name, results, time_stamp):
    """ 
    results: 
      a) list of (station_name, prediction, actual) or
      b) tuple of (dict(station_name -> prediction), dict(station_name -> actual))
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
    pickle.dump(result_df, open('../pickle/results/' + file_name, 'wb'))
    
    return file_name


def pickle_model(name, model, station, time_stamp, model_type='torch'):
    file_name = '../pickle/models/{}_{}_{}.pkl'.format(name, station, time_stamp)
    if model_type == 'torch':
        torch.save(model, file_name)
    elif model_type == 'xgb':
        pickle.dump(model, open(file_name, 'wb'))
    else:
        pickle.dump(model, open(file_name, 'wb'))
    print('Saved model as', file_name)


def load_train_test_gridded_dividedStreamflow():
    """
    Load train and test data from HDF5 for predictions on gridded forcings with streamflow divided into hourly data.
    If no HDF5 file exists yet, it is created.
    """
    file_name = '../data/train_test/gridded_dividedStreamflow.h5'
    if not os.path.isfile(file_name):
        history = 7 * 24
        data_runoff = load_discharge_gr4j_vic()
        station_cell_mapping = get_station_cell_mapping()
        rdrs_data = load_rdrs_forcings()
        month_onehot = pd.get_dummies(data_runoff['date'].dt.month, prefix='month', columns=['month'])
        data_runoff = data_runoff.join(month_onehot)

        for station in data_runoff['station'].unique():
            station_data = data_runoff[data_runoff['station'] == station].set_index('date')
            station_cell_ids = 39 * station_cell_mapping[station_cell_mapping['station'] == station]['col'] \
                + station_cell_mapping[station_cell_mapping['station'] == station]['row']
            station_rdrs = rdrs_data.filter(regex='_(' + '|'.join(map(lambda x: str(x), station_cell_ids)) + ')$', axis=1)

            if any(station_data['runoff'].isna()):
                station_data = station_data[~pd.isna(station_data['runoff'])]
                print('Station', station, 'had NA runoff values')

            station_data = station_data.resample('1H').ffill()
            station_data['runoff'] = station_data['runoff'] / 24
            station_data = station_data.join(station_rdrs)
            for i in range(1, history + 1):
                station_data = station_data.join(station_rdrs.shift(i, axis=0), rsuffix='_-{}'.format(i))
                
            station_data.to_hdf(file_name, 'station_' + station, complevel=5)
            
    return read_station_data_dict(file_name)


def load_train_test_gridded_aggregatedForcings(include_all_forcing_vars=False, include_all_cells=False):
    """
    Load train and test data from HDF5 for predictions on gridded forcings, aggregated into days.
    If no HDF5 file exists yet, it is created.
    
    if include_all_forcing_vars, will return min/max-aggregation for all variables and sum-aggregation for precipitation.
    else, will return min/max-temperature and sum-precipitation.
    
    if not include_all_cells, will only return cells belonging to the station's subwatershed
    """
    file_name = '../data/train_test/gridded_aggregatedForcings{}{}.h5'.format('_all_vars' if include_all_forcing_vars else '', 
                                                                              '_all_cells' if include_all_cells else '')
    if not os.path.isfile(file_name):
        history = 7
        data_runoff = load_discharge_gr4j_vic()
        station_cell_mapping = get_station_cell_mapping()
        
        rdrs_data = load_rdrs_forcings()
        resampled = rdrs_data.resample('D')
        rdrs_daily = resampled.sum().join(resampled.min(), lsuffix='_sum', rsuffix='_min').join(resampled.max().rename(lambda c: c + '_max', axis=1))
        month_onehot = pd.get_dummies(data_runoff['date'].dt.month, prefix='month', columns=['month'])
        data_runoff = data_runoff.join(month_onehot)
        
        for station in data_runoff['station'].unique():
            station_data = data_runoff[data_runoff['station'] == station].set_index('date')
            if include_all_cells:
                station_cell_ids = ['\w*']
            else:
                station_cell_ids = 39 * station_cell_mapping[station_cell_mapping['station'] == station]['col'] \
                    + station_cell_mapping[station_cell_mapping['station'] == station]['row']

            if not include_all_forcing_vars:
                # For temperature use min/max aggregation. Precipitation: sum. solar fluxes, pressure & humidity don't seem to help (at least with min/max/sum)
                regex = '((RDRS_TT_40m)_({0})_(min|max)|(RDRS_PR0_SFC)_({0})_sum)$'.format('|'.join(map(lambda x: str(x), station_cell_ids)))
            else:
                regex = '(_({0})_(min|max)|(RDRS_PR0_SFC)_({0})_sum)$'.format('|'.join(map(lambda x: str(x), station_cell_ids)))
            station_rdrs = rdrs_daily.filter(regex=regex, axis=1)
            if any(station_data['runoff'].isna()):
                station_data = station_data[~pd.isna(station_data['runoff'])]
                print('Station', station, 'had NA runoff values')

            station_data = station_data.join(station_rdrs)
            for i in range(1, history + 1):
                station_data = station_data.join(station_rdrs.shift(i, axis=0), rsuffix='_-{}'.format(i))

            station_data.to_hdf(file_name, 'station_' + station, complevel=5)
            
    return read_station_data_dict(file_name)


def load_train_test_lstm():
    """
    Load train and test data for LSTM from HDF5. If no HDF5 file exists yet, it is created.
    """
    file_name = '../data/train_test/lstm.h5'
    if not os.path.isfile(file_name):
        data_runoff = load_discharge_gr4j_vic()
        # For each station, read which grid cells belong to its subwatershed
        station_cell_mapping = get_station_cell_mapping()
        rdrs_data = load_rdrs_forcings()

        for station in data_runoff['station'].unique():
            station_cell_ids = 39 * station_cell_mapping[station_cell_mapping['station'] == station]['col'] \
                + station_cell_mapping[station_cell_mapping['station'] == station]['row']
            station_rdrs = rdrs_data.filter(regex='_(' + '|'.join(map(lambda x: str(x), station_cell_ids)) + ')$', axis=1)

            month_onehot = pd.get_dummies(station_rdrs.index.month, prefix='month')
            month_onehot.index = station_rdrs.index
            station_rdrs = station_rdrs.join(month_onehot)

            station_rdrs.to_hdf(file_name, 'station_' + station)
    
    return read_station_data_dict(file_name)


def get_station_cell_mapping():
    """ For each station, read which grid cells belong to its subwatershed """
    return pd.read_csv('../data/station_cell_mapping.csv', skiprows=1, names=['station', 'lat', 'lon', 'row', 'col', 'area'])


def read_station_data_dict(file_name):
    station_data_dict = {}
    with pd.HDFStore(file_name,  mode='r') as store:
        for station in store.keys():
            station_name = station[9:]
            station_data_dict[station_name] = store[station]
            
    return station_data_dict


def load_landcover_reduced(values_to_use=None):
    """
    Load landcover data and reduce resolution to rdrs data shape.
    if values_to_use is None, returns all land types. Else only the ones specified.
    Returns (array of shape (#landtypes, rows, cols), where the first dimension is the averaged amount of this landtype in cell (row, col), legend)
    """
    rdrs_data, rdrs_vars, rdrs_dates = load_rdrs_forcings(as_grid=True)
    landcover_nc = nc.Dataset('../data/landcover_erie.nc', 'r')
    landcover_fullres = np.array(landcover_nc['Band1'][:])[::-1,:]
    
    legend = {1: 'Temperate or sub-polar needleleaf forest',
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

    if values_to_use is None:
        values_to_use = legend.keys()

    pixels_per_row = (landcover_fullres.shape[0] // rdrs_data.shape[2]) + 1
    pixels_per_col = (landcover_fullres.shape[1] // rdrs_data.shape[3]) + 1

    landcover_reduced = np.zeros((len(values_to_use), rdrs_data.shape[2], rdrs_data.shape[3]))
    for row in range(landcover_reduced.shape[1]):
        for col in range(landcover_reduced.shape[2]):
            landcover_cell = landcover_fullres[row*pixels_per_row:(row+1)*pixels_per_row, col*pixels_per_col:(col+1)*pixels_per_col]
            non_zero_pixels_per_cell = (landcover_cell.flatten() != 0).sum()
            i = 0
            for k in legend.keys():
                if k not in values_to_use:
                    continue
                if non_zero_pixels_per_cell == 0:
                    landcover_reduced[i, row, col] = 0.0
                else:
                    landcover_reduced[i, row, col] = np.float((landcover_cell == k).sum()) / non_zero_pixels_per_cell
                i += 1
                
    landcover_nc.close()                
    return landcover_reduced, list(legend[i] for i in values_to_use)