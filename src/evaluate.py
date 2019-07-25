import pandas as pd
from matplotlib import pyplot as plt
import hydroeval
from sklearn import metrics
from datetime import datetime, timedelta
import torch
from torch import nn


def evaluate_hourly(station_name, prediction, actual, plot=False):
    """
    Calculate NSE for hourly predictions by aggregating to daily predictions and then comparing to actual streamflow.
    Returns (NSE when clipped before aggregation, NSE when clipped after aggregation).
    Plot is always for clipping after aggregation.
    """
    actual_daily = actual.resample('D').sum()
    
    predict_daily_clipped_before = prediction.clip(0).resample('D').sum()
    predict_daily = prediction.resample('D').sum()
    return (evaluate_daily(station_name, predict_daily_clipped_before, actual_daily, plot=False),
            evaluate_daily(station_name, predict_daily, actual_daily, plot=plot))
    

def evaluate_daily(station_name, prediction, actual, plot=False, writer=None, clip=True):
    """
    Calculate NSE for daily streamflow prediction. If `writer` is not None, will write plot to tensorboard.
    """
    predict_clipped = prediction.copy()
    if clip:
        predict_clipped = prediction.clip(0)
    nse_clip = hydroeval.evaluator(hydroeval.nse, predict_clipped.to_numpy(), actual.to_numpy())[0]
    mse_clip = metrics.mean_squared_error(actual.to_numpy(), predict_clipped.to_numpy())
    
    if writer is not None:
        writer.add_scalar('NSE_' + station_name, nse_clip, 0)
        f = plt.figure(figsize=(17,4))
        plt.title(station_name + ': NSE ' + str(nse_clip) + ', MSE: '+ str(mse_clip))
        plt.plot(actual, label='Target')
        plt.plot(predict_clipped, label='Prediction')
        plt.grid()
        plt.legend()
        writer.add_figure(station_name, f, 0, True)
    elif plot:
        f = plt.figure(figsize=(17,4))
        plt.title(station_name + ': NSE ' + str(nse_clip) + ', MSE: '+ str(mse_clip))
        plt.plot(actual, label='Target')
        plt.plot(predict_clipped, label='Prediction')
        plt.grid()
        plt.legend()
    return nse_clip, mse_clip


class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()
        
    def forward(self, prediction, target):
        nses = torch.sum(torch.pow(prediction - target, 2), dim=0) / torch.sum(torch.pow(target - target.mean(dim=0), 2), dim=0)
        return nses.mean()