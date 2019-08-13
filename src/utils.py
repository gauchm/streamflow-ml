import numpy as np
import pandas as pd
import torch


def add_border(data, style, fill_value=np.nan):
    """
    Add a one-pixel border to the last two dimension of the lowres image to avoid artefacts when rescaling.
    style determines filling algorithm, either 'extrapolate', or 'fill_value'
    """
    new_shape = list(s for s in data.shape)
    new_shape[-1] += 2
    new_shape[-2] += 2
    data_with_border = torch.full(new_shape, fill_value, dtype=data.dtype) if isinstance(data,torch.Tensor) \
                    else np.full(new_shape, fill_value, dtype=data.dtype)
    data_with_border[...,1:-1,1:-1] = data
    if style == 'extrapolate':
        data_with_border[...,1:-1,0] = data[:,0] + (data[...,:,0] - data[...,:,1])
        data_with_border[...,1:-1,-1] = data[:,-1] + (data[...,:,-1] - data[...,:,-2])
        data_with_border[...,0,:] = data_with_border[...,1,:] + (data_with_border[...,1,:] - data_with_border[...,2,:])
        data_with_border[...,-1,:] = data_with_border[...,-2,:] + (data_with_border[...,-2,:] - data_with_border[...,-3,:])
    elif style == 'fill_value':
        data_with_border[...,[0,-1],[0,-1]] = fill_value
    else:
        raise Exception('Unsupported style')
    return data_with_border


def map_to_coords(source_lats, source_lons, target_lats, target_lons):
    """
    Generates a mapping from target cell to (row, col) in the source grid plus a 1-pixel border.
    """
    source_lats_with_border, source_lons_with_border = add_border(source_lats, 'extrapolate'), add_border(source_lons, 'extrapolate')
    
    if len(target_lats.shape) == 1:
        target_lats = np.tile(target_lats,len(target_lons)).reshape(len(target_lons),-1).T
        target_lons = np.tile(target_lons,len(target_lats)).reshape(len(target_lats),-1)
    
    resample_map_rows = np.zeros(target_lats.shape, dtype=int)
    resample_map_cols = np.zeros(target_lats.shape, dtype=int)
    for i in range(target_lats.shape[0]):
        for j in range(target_lats.shape[1]):
            target_lat, target_lon = target_lats[i,j], target_lons[i,j]
            closest_source_cell = np.argmin(np.square(source_lats_with_border - target_lat) + 
                                            np.square(source_lons_with_border - target_lon))
            resample_map_rows[i,j] = closest_source_cell // source_lats_with_border.shape[1]
            resample_map_cols[i,j] = closest_source_cell % source_lats_with_border.shape[1]
    
    return resample_map_rows, resample_map_cols


def resample_by_map(lowres_data, resample_map_rows, resample_map_cols, fill_value=np.nan):
    """
    Resamples last two dimensions of lowres_data based on resample_map, adding the 1-pixel border to lowres_data to avoid artifacts.
    """
    return add_border(lowres_data, 'fill_value', fill_value=fill_value)[..., resample_map_rows, resample_map_cols]
