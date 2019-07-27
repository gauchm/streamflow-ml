import pandas as pd
import numpy as np
import netCDF4 as nc
import os
from datetime import datetime, timedelta
import pickle
import torch
from torch import nn
import matplotlib.pyplot as plt
from src import load_data


def visualize_kernels(conv_layer, input_channel_names, num_out_channels, figsize=(16,12)):
    conv_input_weights = conv_layer.weight[:num_out_channels,:len(input_channel_names)].detach()
    vmin, vmax = conv_input_weights.min(), conv_input_weights.max()
    f, ax = plt.subplots(conv_input_weights.shape[0], conv_input_weights.shape[1], sharex=True, sharey=True, figsize=figsize)
    for i in range(conv_input_weights.shape[0]):
        for j in range(conv_input_weights.shape[1]):
            ax[i,j].imshow(conv_input_weights[i,j], cmap='Greys', vmin=vmin, vmax=vmax)
            ax[-1,j].set_xlabel(input_channel_names[j])
    ax[conv_input_weights.shape[0]//2,0].set_ylabel('hidden channels')
    plt.tight_layout()
    
    
def visualize_hidden_channels(hidden_layers, sample_no, is_convlstm=False, figsize=(20,7)):
    # Visualize conv_lstm hidden layer channels
    channel_dim = 2 if is_convlstm else 1
    max_channels = max(h.shape[channel_dim] for h in hidden_layers)
    f, ax = plt.subplots(len(hidden_layers), max_channels, sharex=True, sharey=True, figsize=figsize)
    f.suptitle('Sample {}'.format(sample_no))
    for layer_idx in range(len(hidden_layers)):
        ax[layer_idx,0].set_ylabel('hidden layer {}'.format(layer_idx))
        if is_convlstm:
            layer_last_output = hidden_layers[layer_idx][:,-1].detach()
        else:
            layer_last_output = hidden_layers[layer_idx].detach()
        for i in range(layer_last_output.shape[1]):
            ax[layer_idx,i].imshow(layer_last_output[sample_no,i], cmap='Greys')
        _ = list(f.delaxes(ax[layer_idx,i]) for i in range(hidden_layers[layer_idx].shape[channel_dim], max_channels))
    plt.tight_layout()
    
    
def get_mask_from_rdrs():
    """ Returns boolean mask showing where RDRS is nan """
    
    rdrs_data, _, _ = load_data.load_rdrs_forcings(as_grid=True)
    return np.isnan(rdrs_data[0,0])