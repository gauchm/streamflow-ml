import pandas as pd
import os
import netCDF4 as nc
from datetime import datetime, timedelta


def load_discharge_gr4j_vic():
    """
    Loads observed discharge for GR4J-Raven and VIC from disk.
    """
    dir = 'ObservedDischarge_GR4J+VIC'  # Read runoff observations
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
    rdrs_nc = nc.Dataset('RDRS_CaPA24hr_forcings_final.nc', 'r')
    
    rdrs_data = pd.DataFrame(index=pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H')) # Using 7:00 because forcings are UTC, while runoff is local time
    
    for var in forcing_variables:
        var_data = pd.DataFrame(rdrs_nc[var][:].reshape(43825,34*39))
        var_data.columns = [var + '_' + str(c) for c in var_data.columns]
        rdrs_data.reset_index(drop=True, inplace=True)
        rdrs_data = rdrs_data.reset_index(drop=True).join(var_data.reset_index(drop=True))
    rdrs_data.index = pd.date_range('2010-01-01 7:00', '2015-01-01 7:00', freq='H')
    
    rdrs_nc.close()
    return rdrs_data