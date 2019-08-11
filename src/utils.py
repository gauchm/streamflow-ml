import numpy as np
import pandas as pd
import torch


def add_border(coords, style):
    """
    Add a one-pixel border to the lowres image to avoid artefacts when upscaling.
    style determines filling algorithm, either 'extrapolate', or 'zeros'
    """
    coords_with_border = np.zeros((coords.shape[0] + 2, coords.shape[1] + 2))
    coords_with_border[1:-1,1:-1] = coords
    if style == 'extrapolate':
        coords_with_border[1:-1,0] = coords[:,0] + (coords[:,0] - coords[:,1])
        coords_with_border[1:-1,-1] = coords[:,-1] + (coords[:,-1] - coords[:,-2])
        coords_with_border[0,:] = coords_with_border[1,:] + (coords_with_border[1,:] - coords_with_border[2,:])
        coords_with_border[-1,:] = coords_with_border[-2,:] + (coords_with_border[-2,:] - coords_with_border[-3,:])
    elif style == 'zeros':
        coords_with_border[[0,-1],[0,-1]] = 0
    else:
        raise Exception('Unsupported style')
    return coords_with_border


def map_to_geophysical_coords(lowres_lats, lowres_lons, geophys_lats, geophys_lons):
    """
    Generates a mapping from high-resolution cell to (row, col) in a low-resolution grid with a 1-pixel border.
    """
    lowres_lats_with_border, lowres_lons_with_border = add_border(lowres_lats, 'extrapolate'), add_border(lowres_lons, 'extrapolate')
    
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


def upsample_to_geophysical_input(lowres_data, upsample_map_rows, upsample_map_cols):
    """
    Upsamples lowres_data based on upsample_map, adding the 1-pixel border to lowres_data to avoid artifacts.
    """
    return add_border(lowres_data, 'zeros')[upsample_map_rows, upsample_map_cols]