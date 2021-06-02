from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, DatetimeIndex, merge
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy  # noqa


class BinHV27(IStrategy):

    timeframe = '5m'

    buy_params = {
        'adx1': 25,
        'emarsi1': 20,
        'adx2': 30,
        'emarsi2': 20,
        'adx3': 35,
        'emarsi3': 20,
        'adx4': 30,
        'emarsi4': 25
    }

    sell_params = {
        'emarsi1': 75,
        'adx2': 30,
        'emarsi2': 80,
        'emarsi3': 75,
    }

    minimal_roi = {
        "0": 0.05,
        "10": 0.025,
        "20": 0.015,
        "30": 0.01,
        "720": 0.005,
        "1440": 0
    }

    custom_minimal_roi = {
        'enabled': True,
        'decay-type': 'exp',  # linear (lin) or exponential (exp)
        'decay-rate': 0.015,  # bigger is faster, recommended to graph f(t) = start-pct * e(-rate*t)      
        'decay-time': 1440,   # amount of time to reach zero, only relevant for linear decay
        'start-pct': 0.10,    # starting percentage
        'end-pct': 0,         # ending percentage 
        'fit-type': 'poly'    # best-fit a shape to the standard ROI table points using a polynomial (linear) or exponential shape
    }

    stoploss = -0.50

    # Probably don't change these
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = numpy.nan_to_num(ta.RSI(dataframe, timeperiod=5))
        dataframe['emarsi'] = numpy.nan_to_num(ta.EMA(dataframe['rsi'], timeperiod=5))

        dataframe['adx'] = numpy.nan_to_num(ta.ADX(dataframe))

        dataframe['minusdi'] = numpy.nan_to_num(ta.MINUS_DI(dataframe))
        dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(dataframe['minusdi'], timeperiod=25))
        dataframe['plusdi'] = numpy.nan_to_num(ta.PLUS_DI(dataframe))
        dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(dataframe['plusdi'], timeperiod=5))

        dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=60))
        dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=120))
        dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=120))
        dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=240))

        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & ((dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']

        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe['slowsma'].shift().gt(dataframe['slowsma'].shift(2))

        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())
        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            dataframe['slowsma'].gt(0) &
            dataframe['close'].lt(dataframe['highsma']) &
            dataframe['close'].lt(dataframe['lowsma']) &
            dataframe['minusdi'].gt(dataframe['minusdiema']) &
            dataframe['rsi'].ge(dataframe['rsi'].shift()) &
            (
              (
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(params['adx1']) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(params['emarsi1'])
              ) |
              (
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(params['adx2']) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(params['emarsi2'])
              ) |
              (
                ~dataframe['continueup'] &
                dataframe['adx'].gt(params['adx3']) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(params['emarsi3'])
              ) |
              (
                dataframe['continueup'] &
                dataframe['adx'].gt(params['adx4']) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(params['emarsi4'])
              )
            ),
            'buy'] = 1

        return dataframe


    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (
              (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                (dataframe['close'].gt(dataframe['lowsma']) | dataframe['close'].gt(dataframe['highsma'])) &
                dataframe['highsma'].gt(0) &
                dataframe['bigdown']
              ) |
              (
                ~dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                (dataframe['emarsi'].ge(params['emarsi1']) | dataframe['close'].gt(dataframe['slowsma'])) &
                dataframe['bigdown']
              ) |
              (
                ~dataframe['preparechangetrendconfirm'] &
                dataframe['close'].gt(dataframe['highsma']) &
                dataframe['highsma'].gt(0) &
                dataframe['adx'].gt(params['adx2']) &
                dataframe['emarsi'].ge(params['emarsi2']) &
                dataframe['bigup']
              ) |
              (
                dataframe['preparechangetrendconfirm'] &
                ~dataframe['continueup'] &
                dataframe['slowingdown'] &
                dataframe['emarsi'].ge(params['emarsi3']) &
                dataframe['slowsma'].gt(0)
              ) |
              (
                dataframe['preparechangetrendconfirm'] &
                dataframe['minusdi'].lt(dataframe['plusdi']) &
                dataframe['close'].gt(dataframe['lowsma']) &
                dataframe['slowsma'].gt(0)
              )
            ),
            'sell'] = 1

        return dataframe

    """
    Override for default Freqtrade ROI table functionality
    """
    def poly(x, a, b):
        return a + (x * b)

    def exp(x, a, b, c):
        return a * np.exp(-b * x) + c

    def make_estimator(func, data):
        xs, ys = zip(*sorted(data.items()))
        popt, pcov = curve_fit(func, xs, ys)
        return lambda x: func(x, *popt)

    def min_roi_reached_entry(self, trade_dur: int) -> Tuple[Optional[int], Optional[float]]:
        custom_roi = self.custom_minimal_roi

        # if the custom_roi dict is defined and enabled, do it, otherwise fallback to default functionality
        if custom_roi and custom_roi['enabled']:
            # linear decay: f(t) = start - (rate * t)
            if custom_roi['decay-type'] == 'lin':
                rate = (custom_roi['start-pct'] - custom_roi['end-pct']) / custom_roi['decay-time']
                min_roi = max(custom_roi['end-pct'], custom_roi['start-pct'] - (rate * trade_dur))
            # exponential decay: f(t) = start * e^(-rate*t) - c
            elif custom_roi['decay-type'] == 'exp':
                min_roi = max(custom_roi['end-pct'], custom_roi['start-pct'] * np.exp(-custom_roi['decay-rate']*time))
            # polynomial or exponential best-fit to points in defined roi table
            elif custom_roi['decay-type'] == 'fit':
                roi = self.minimal_roi
                estimator = self.make_estimator(custom_roi['fit-type'], roi)
                min_roi = estimator(trade_dur)
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