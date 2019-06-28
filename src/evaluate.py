import pandas as pd
from matplotlib import pyplot as plt
import hydroeval
from sklearn import metrics
from datetime import datetime, timedelta


def evaluate_hourly(station_name, prediction, actual, plot=False):
    actual_daily = actual.resample('D').sum()

    predict = prediction.clip(0)
    predict_daily = predict.resample('D').sum()
    
    return evaluate_daily(station_name, predict_daily, actual_daily, plot=plot)
    

def evaluate_daily(station_name, prediction, actual, plot=False):
    predict_clipped = prediction.copy()
    predict_clipped = prediction.clip(0)
    nse_clip = hydroeval.evaluator(hydroeval.nse, predict_clipped.to_numpy(), actual.to_numpy())[0]
    
    if plot:
        plt.figure(figsize=(17,4))
        plt.title(station_name)
        plt.plot(actual, label='Test')
        plt.plot(predict_clipped, label='Prediction')
        plt.legend()
    return nse_clip