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
        data = pd.read_csv(os.path.join(dir, f), skiprows=2, skipfooter=1, index_col=False, header=None, names=['runoff'], na_values='-1.2345')
        data['date'] = pd.date_range('2010-01-01', periods=len(data), freq='D')
        data['station'] = f[11:-4]
        data['runoff'] = data['runoff'].astype(float)
        data_runoff = data_runoff.append(data, ignore_index=True)
        
    return data_runoff


def load_rdrs_forcings():
    """
    Loads RDRS gridded forcings from disk.
    """
    forcing_variables = ['RDRS_FB_SFC', 'RDRS_FI_SFC', 'RDRS_HU_40m', 'RDRS_P0_SFC', 'RDRS_PR0_SFC', 'RDRS_TT_40m', 'RDRS_UVC_40m', 'RDRS_WDC_40m']
    rdrs_nc = nc.Dataset('../data/RDRS_CaPA24hr_forcings_final.nc', 'r')
    
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
    if isinstance(results, list):
        result_list = results
    elif isinstance(results, tuple):
        result_list = []
        for station, predict in results[0].items():
            result_list.append((station, predict, results[1][station]))
    else:
        raise Exception('invalid result format')
        
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


def load_train_test_gridded_aggregatedForcings():
    """
    Load train and test data from HDF5 for predictions on gridded forcings, aggregated into days.
    If no HDF5 file exists yet, it is created.
    """
    file_name = '../data/train_test/gridded_aggregatedForcings.h5'
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
            station_cell_ids = 39 * station_cell_mapping[station_cell_mapping['station'] == station]['col'] \
                + station_cell_mapping[station_cell_mapping['station'] == station]['row']

            # For temperature use min/max aggregation. Precipitation: sum. solar fluxes, pressure & humidity don't seem to help (at least with min/max/sum)
            regex = '((RDRS_TT_40m)_({0})_(min|max)|(RDRS_PR0_SFC)_({0})_sum)$'.format('|'.join(map(lambda x: str(x), station_cell_ids)))
            station_rdrs = rdrs_daily.filter(regex=regex, axis=1)
            if any(station_data['runoff'].isna()):
                station_data = station_data[~pd.isna(station_data['runoff'])]
                print('Station', station, 'had NA runoff values')

            station_data = station_data.join(station_rdrs)
            for i in range(1, history + 1):
                station_data = station_data.join(station_rdrs.shift(i, axis=0), rsuffix='_-{}'.format(i))

            station_data.to_hdf(file_name, 'station_' + station)
            
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