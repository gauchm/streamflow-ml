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
            station_cell_ids = 39 *( station_cell_mapping[station_cell_mapping['station'] == station]['col'] - 1) \
                + (station_cell_mapping[station_cell_mapping['station'] == station]['row'] - 1)
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
            station_cell_ids = 39 * (station_cell_mapping[station_cell_mapping['station'] == station]['col'] - 1) \
                + (station_cell_mapping[station_cell_mapping['station'] == station]['row'] - 1)
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
    Load landcover data, cropped out for lake erie watershed, and reduce resolution to rdrs data shape.
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


def load_landcover(values_to_use=None, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """
    Load landcover data with 30" resolution.
    If values_to_use is None, returns all land types. Else only the ones specified.
    If min/max lat/lon is specified, will only return the specified sub-area.
    Returns (array of shape (#landtypes, rows, cols), where the first dimension is the averaged amount of this landtype in cell (row, col), legend)
    """
    filename = '../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_30sec.nc'
    
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
    
    if not os.path.isfile(filename):
        import gdal
        dem_nc = nc.Dataset('../data/geophysical/dem/hydrosheds_n40-45w75-90_30sec.nc', 'r')
        landcover_nc = nc.Dataset('../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90.nc', 'r')
        landcover = landcover_nc['Band1'][:].filled(np.nan)
        
        landcover_30sec_nc = nc.Dataset('../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_30sec.nc', 'w')
        landcover_30sec_nc.setncattr('Conventions', 'CF-1.6')
        landcover_30sec_nc.createDimension('lat')
        landcover_30sec_nc.createDimension('lon')
        landcover_30sec_nc.createVariable('crs', 'S1')
        for attr in landcover_nc['crs'].ncattrs():
            landcover_30sec_nc['crs'].setncattr(attr, landcover_nc['crs'].getncattr(attr))
        landcover_30sec_nc.createVariable('lat', np.float64, dimensions=('lat'))
        landcover_30sec_nc.createVariable('lon', np.float64, dimensions=('lon'))
        landcover_30sec_nc['lat'][:] = dem_nc['lat'][:]
        landcover_30sec_nc['lon'][:] = dem_nc['lon'][:]
        for attr in landcover_nc['lat'].ncattrs():
            landcover_30sec_nc['lat'].setncattr(attr, landcover_nc['lat'].getncattr(attr))
        for attr in landcover_nc['lon'].ncattrs():
            landcover_30sec_nc['lon'].setncattr(attr, landcover_nc['lon'].getncattr(attr))
        
        dem_nc.close()
        
        # gdal.Warp can only resample one band at a time. Hence, resample each landtype separately and successively merge into _30sec.nc.
        for i, landtype in landcover_legend.items():
            print(landtype)
            landcover_temp_nc = nc.Dataset('../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_temp.nc', 'w')
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

            gdal_temp = gdal.Open('../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_temp.nc')
            warp_options = gdal.WarpOptions(format='netCDF', xRes=0.008333333333333333333, yRes=0.008333333333333333333, resampleAlg='average')
            print('Warping...')
            gdal.Warp('../data/geophysical/landcover/landtype_temp.nc'.format(varname), gdal_temp, options=warp_options)
            print('Warping complete.')
            landtype_temp = nc.Dataset('../data/geophysical/landcover/landtype_temp.nc', 'r')
            landcover_30sec_nc.createVariable(varname, 'f', dimensions=('lat', 'lon'))
            landcover_30sec_nc[varname][:] = landtype_temp['Band1'][:]
            landcover_30sec_nc[varname].setncattr('landtype', landtype)

            landtype_temp.close()
            os.remove('../data/geophysical/landcover/landtype_temp.nc')
            os.remove('../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_temp.nc')
        
        landcover_nc.close()
        landcover_30sec_nc.close()
    
    if values_to_use is None:
        values_to_use = list(landcover_legend.keys())
    
    landcover_nc = nc.Dataset('../data/geophysical/landcover/NA_NALCMS_LC_30m_LAEA_mmu12_urb05_n40-45w75-90_30sec.nc', 'r')
    landcover = np.zeros((0, landcover_nc['lat'].shape[0], landcover_nc['lon'].shape[0]))
    legend = []
    for i in range(len(values_to_use)):
        landcover_i = landcover_nc['landtype_{}'.format(values_to_use[i])][:].filled(np.nan)
        if landcover_i.sum() == 0:
            continue
        landcover = np.concatenate([landcover, landcover_i.reshape((1,landcover.shape[1],landcover.shape[2]))], axis=0)
        legend.append(landcover_legend[values_to_use[i]])
    
    landcover_nc.close()
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return landcover[:,min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx], legend


def load_dem(min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """
    Load DEM at 30" resolution.
    If min/max lat/lon is specified, will only return the specified sub-area.
    """
    dem_nc = nc.Dataset('../data/geophysical/dem/hydrosheds_n40-45w75-90_30sec.nc', 'r')
    dem = dem_nc['Band1'][:].filled(np.nan)

    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return dem[min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]

def load_groundwater(min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """
    Load groundwater data at 30" resolution.
    If min/max lat/lon is specified, will only return the specified sub-area.
    """
    groundwater_nc = nc.Dataset('../data/geophysical/groundwater/N_America_model_wtd_v2_n40-45w75-90.nc', 'r')
    groundwater = groundwater_nc['Band1'][:].filled(np.nan)
    
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return groundwater[min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx]

def load_soil(min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    """
    Load soil data at 30" resolution.
    If min/max lat/lon is specified, will only return the specified sub-area.
    """
    soiltypes = ['SAND', 'CLAY']
    soil_nc = nc.Dataset('../data/geophysical/soil/SAND1_n40-45w75-90.nc', 'r')
    soil = np.zeros((len(soiltypes) * 8, soil_nc['lat'].shape[0], soil_nc['lon'].shape[0]))
    soil_nc.close()

    # Each nc file contains 4 soil layers; per soil type there are 2 nc files.
    soil_legend = []
    for i in range(len(soiltypes)):
        for j in [1,2]:
            soil_nc = nc.Dataset('../data/geophysical/soil/{}{}_n40-45w75-90.nc'.format(soiltypes[i], j), 'r')
            for layer in range(1,5):
                soil[i*8 + ((j-1)*4 + layer-1)] = soil_nc['Band{}'.format(layer)][:].astype(np.float).filled(np.nan) / 100.0
                soil_legend.append('{}-layer{}'.format(soiltypes[i], (j-1)*4 + layer))
            soil_nc.close()
    
    min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx = get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon)
    return soil[:,min_lat_idx:max_lat_idx,min_lon_idx:max_lon_idx], soil_legend


def get_bounding_box_indices(min_lat, max_lat, min_lon, max_lon):
    """
    Returns indices to split the 30" datasets such that they only contain lats/lons within the specified bounding box.
    """
    dem_nc = nc.Dataset('../data/geophysical/dem/hydrosheds_n40-45w75-90_30sec.nc', 'r')
    lats = dem_nc['lat'][:]
    lons = dem_nc['lon'][:]
    
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

    min_lat_idx = (lats >= min_lat).argmax()
    max_lat_idx = (lats <= max_lat).argmin() if max_lat < lats.max() else len(lats)
    min_lon_idx = (lons >= min_lon).argmax()
    max_lon_idx = (lons <= max_lon).argmin() if max_lon < lons.max() else len(lons)
    dem_nc.close()
    
    return min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx
    