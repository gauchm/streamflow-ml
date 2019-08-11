import numpy as np
import pandas as pd
import torch


def add_border(data, style, fill_value=np.nan):
    """
    Add a one-pixel border to the last two dimension of the lowres image to avoid artefacts when upscaling.
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


def map_to_geophysical_coords(lowres_lats, lowres_lons, geophys_lats, geophys_lons):
    """
    Generates a mapping from high-resolution cell to (row, col) in a low-resolution grid with a 1-pixel border.
    """
    lowres_lats_with_border, lowres_lons_with_border = add_border(lowres_lats, 'extrapolate'), add_border(lowres_lons, 'extrapolate')
    
    if len(geophys_lats.shape) == 1:
        geophys_lats = np.tile(geophys_lats,len(geophys_lons)).reshape(len(geophys_lons),-1).T
        geophys_lons = np.tile(geophys_lons,len(geophys_lats)).reshape(len(geophys_lats),-1)
    
    upsample_map_rows = np.zeros(geophys_lats.shape, dtype=int)
    upsample_map_cols = np.zeros(geophys_lats.shape, dtype=int)
    for i in range(geophys_lats.shape[0]):
        for j in range(geophys_lats.shape[1]):
            highres_lat, highres_lon = geophys_lats[i,j], geophys_lons[i,j]
            closest_lowres_cell = np.argmin(np.square(lowres_lats_with_border - highres_lat) + 
                                            np.square(lowres_lons_with_border - highres_lon))
            upsample_map_rows[i,j] = closest_lowres_cell // lowres_lats_with_border.shape[1]
            upsample_map_cols[i,j] = closest_lowres_cell % lowres_lats_with_border.shape[1]
    
    return upsample_map_rows, upsample_map_cols


def upsample_to_geophysical_input(lowres_data, upsample_map_rows, upsample_map_cols, fill_value=np.nan):
    """
    Upsamples last two dimensions of lowres_data based on upsample_map, adding the 1-pixel border to lowres_data to avoid artifacts.
    """
    return add_border(lowres_data, 'fill_value', fill_value=fill_value)[..., upsample_map_rows, upsample_map_cols]
