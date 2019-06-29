import pandas as pd
from matplotlib import pyplot as plt
import hydroeval
from sklearn import metrics
from datetime import datetime, timedelta


def evaluate_hourly(station_name, prediction, actual, clip_before_aggregate=False, plot=False):
    """
    Calculate NSE for hourly predictions by aggregating to daily predictions and then comparing to actual streamflow.
    """
    actual_daily = actual.resample('D').sum()
    
    if clip_before_aggregate:
        predict = prediction.clip(0)
    else:
        predict = prediction
    predict_daily = predict.resample('D').sum()
    
    return evaluate_daily(station_name, predict_daily, actual_daily, plot=plot)
    

def evaluate_daily(station_name, prediction, actual, plot=False, writer=None):
    """
    Calculate NSE for daily streamflow prediction. If `writer` is not None, will write plot to tensorboard.
    """
    predict_clipped = prediction.copy()
    predict_clipped = prediction.clip(0)
    nse_clip = hydroeval.evaluator(hydroeval.nse, predict_clipped.to_numpy(), actual.to_numpy())[0]
    
    if writer is not None:
        writer.add_scalar('NSE_' + station_name, nse_clip, 0)
        f = plt.figure(figsize=(17,4))
        plt.title(station_name + ': NSE ' + str(nse_clip))
        plt.plot(actual, label='Test')
        plt.plot(predict_clipped, label='Prediction')
        plt.legend()
        writer.add_figure(station_name, f, 0, True)
    elif plot:
        f = plt.figure(figsize=(17,4))
        plt.title(station_name + ': NSE ' + str(nse_clip))
        plt.plot(actual, label='Test')
        plt.plot(predict_clipped, label='Prediction')
        plt.legend()
    return nse_clip