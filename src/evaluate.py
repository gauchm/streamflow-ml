import pandas as pd
from matplotlib import pyplot as plt
import hydroeval
from sklearn import metrics
from datetime import datetime, timedelta
import torch
from torch import nn


def evaluate_hourly(station_name, prediction, actual, plot=False):
    """Calculates NSE for hourly predictions by aggregating to daily predictions and then comparing to actual streamflow.
    
    This method will clip predictions to be >= 0.
    
    Args:
        station_name (str): Name of the evaluated station.
        prediction: Series of predictions
        actual: Series of target values
        plot (bool): If True, will plot the clipped predicted and actual values.
    Returns:
        NSE when clipped before aggregation,
        NSE when clipped after aggregation
    """
    actual_daily = actual.resample('D').sum()
    
    predict_daily_clipped_before = prediction.clip(0).resample('D').sum()
    predict_daily = prediction.resample('D').sum()
    return (evaluate_daily(station_name, predict_daily_clipped_before, actual_daily, plot=False),
            evaluate_daily(station_name, predict_daily, actual_daily, plot=plot))
    

def evaluate_daily(station_name, prediction, actual, plot=False, writer=None, clip=True, group=None):
    """Calculates NSE for daily streamflow predictions. 
    
    Args:
        station_name (str): Name of the evaluated station or subbasin.
        prediction: Series of predictions
        actual: Series of target values
        plot (bool): If True, will plot the predicted and actual values.
        writer: If not None, will write the plot to tensorboard.
        clip (bool): Whether to clip predictions to be >= 0.
        group (str or None): Whether the station/subbasin is in the train/test/validation set
    Returns:
        NSE of predictions,
        MSE of predictions
    """
    predict_clipped = prediction.copy()
    if clip:
        predict_clipped = prediction.clip(0)
    nse_clip = hydroeval.evaluator(hydroeval.nse, predict_clipped.to_numpy(), actual.to_numpy())[0]
    mse_clip = metrics.mean_squared_error(actual.to_numpy(), predict_clipped.to_numpy())
    
    title = station_name + ': NSE ' + str(nse_clip) + ', MSE: '+ str(mse_clip)
    if group is not None:
        title += ' (' + group + ')'
    if writer is not None:
        writer.add_scalar('NSE_' + station_name, nse_clip, 0)
        f = plt.figure(figsize=(17,4))
        plt.title(title)
        plt.plot(actual, label='Target')
        plt.plot(predict_clipped, label='Prediction')
        plt.grid()
        plt.legend()
        writer.add_figure(station_name, f, 0, True)
    elif plot:
        f = plt.figure(figsize=(17,4))
        plt.title(title)
        plt.plot(actual, label='Target')
        plt.plot(predict_clipped, label='Prediction')
        plt.grid()
        plt.legend()
    return nse_clip, mse_clip


class NSELoss(nn.Module):
    """NSE loss function."""
    def __init__(self):
        """See base class."""
        super(NSELoss, self).__init__()
        
    def forward(self, prediction, target, means=None):
        """Calculates NSE loss of prediction.
        
        Args:
            prediction: tensor of predictions of shape (#batch, #subbasins)
            target: tensor of target values
            means: If specified, will use these mean target values in the normalization factor. Can be used to supply per-subbasin mean values. Else, will use mean batch target value.
        Returns:
            Mean NSE of the prediction.
        """
        if means is None:
            means = target.mean(dim=0)
        nses = torch.sum(torch.pow(prediction - target, 2), dim=0) / torch.sum(torch.pow(target - means, 2), dim=0)
        return nses.mean()