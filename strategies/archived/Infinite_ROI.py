import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import arrow
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI
from statistics import mean
from cachetools import TTLCache
from scipy.optimize import curve_fit


class Schism(IStrategy):
    """
    Strategy Configuration Items
    """
    timeframe = '5m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.05,
        "10": 0.025,
        "20": 0.015,
        "30": 0.01,
        "720": 0.005,
        "1440": 0
    }
    
    dynamic_roi = {
        'enabled': True,
        'type': 'exponential',  # linear, exponential, or connect
        'decay-rate': 0.015,    # bigger is faster, recommended to graph f(t) = start-pct * e(-rate*t)      
        'decay-time': 1440,     # amount of time to reach zero, only relevant for linear decay
        'start': 0.10,          # starting percentage
        'end': 0,               # ending percentage 
    }

    """
    Informative Pair Definitions
    """
    def informative_pairs(self):

        return informative_pairs

    """
    Indicator Definitions
    """ 
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    """
    Buy Trigger Signals
    """
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    """
    Sell Trigger Signals
    """
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        return dataframe


    """
    Override for default Freqtrade ROI table functionality
    """
    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        dynamic_roi = self.dynamic_roi

        # if the dynamic_roi dict is defined and enabled, do it, otherwise fallback to default functionality
        if dynamic_roi and dynamic_roi['enabled']:
            # linear decay: f(t) = start - (rate * t)
            if dynamic_roi['type'] == 'linear':
                rate = (dynamic_roi['start'] - dynamic_roi['end']) / dynamic_roi['decay-time']
                min_roi = max(dynamic_roi['end'], dynamic_roi['start'] - (rate * trade_dur))
            # exponential decay: f(t) = start * e^(-rate*t)
            elif dynamic_roi['type'] == 'exponential':
                min_roi = max(dynamic_roi['end'], dynamic_roi['start'] * np.exp(-dynamic_roi['decay-rate']*trade_dur))
            elif dynamic_roi['type'] == 'connect':
                # connect the points in the defined table with lines
                past_roi = list(filter(lambda x: x <= trade_dur, self.minimal_roi.keys()))
                next_roi = list(filter(lambda x: x >  trade_dur, self.minimal_roi.keys()))
                if not past_roi:
                    return None, None
                current_entry = max(past_roi)
                # next entry
                if not next_roi:
                    return current_entry, self.minimal_roi[current_entry]
                # use the slope-intercept formula between the two points in the roi table we are between
                else:
                    next_entry = min(next_roi)
                    # y = mx + b
                    x1 = current_entry
                    x2 = next_entry
                    y1 = self.minimal_roi[current_entry]
                    y2 = self.minimal_roi[next_entry]
                    m = (y1-y2)/(x1-x2)
                    b = (x1*y2 - x2*y1)/(x1-x2)
                    min_roi = (m * trade_dur) + b
            else:
                min_roi = 0

            return trade_dur, min_roi

        else:
            # Get highest entry in ROI dict where key <= trade-duration
            roi_list = list(filter(lambda x: x <= trade_dur, self.minimal_roi.keys()))
            if not roi_list:
                return None, None
            roi_entry = max(roi_list)
            return roi_entry, self.minimal_roi[roi_entry]
  