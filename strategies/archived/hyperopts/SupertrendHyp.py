# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real  # noqa

from freqtrade.optimize.hyperopt_interface import IHyperOpt

# --------------------------------
# Add your lib to import here
import talib.abstract as ta  # noqa
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SupertrendHyp(IHyperOpt):
    """
    Only used in the buy/sell methods when --spaces does not include buy or sell
    Should put previously best optimized values here so they are used during ROI/stoploss/etc.
    """
    # Buy hyperspace params:
    buy_params = {
        'buy-method': 'both', # sar, supertrend, both
        'buy-price': 'ohlc4' # open, close, hl2, hlc3, ohlc4
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-method': 'both', # sar, supertrend, both
        'sell-price': 'ohlc4' # open, close, hl2, hlc3, ohlc4
    }

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Categorical(['sar','supertrend','both'], name='buy-method'),
            Categorical(['open','close','hl2','hlc3','ohlc4'], name='buy-price')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
            Categorical(['sar','supertrend','both'], name='sell-method'),
            Categorical(['open','close','hl2','hlc3','ohlc4'], name='sell-price')
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            conditions = []

            if params['buy-method'] == 'supertrend' or params['buy-method'] == 'both':
                conditions.append(dataframe[params['buy-price']] > dataframe['supertrend'])
            
            if params['buy-method'] == 'sar' or params['buy-method'] == 'both':
                conditions.append(dataframe[params['buy-price']] > dataframe['sar'])

            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1

            return dataframe

        return populate_buy_trend

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            
            conditions = []

            if params['sell-method'] == 'supertrend' or params['sell-method'] == 'both':
                conditions.append(dataframe[params['sell-price']] < dataframe['supertrend'])
            
            if params['sell-method'] == 'sar' or params['sell-method'] == 'both':
                conditions.append(dataframe[params['sell-price']] < dataframe['sar'])

            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

            return dataframe

        return populate_sell_trend


    @staticmethod
    def generate_roi_table(params: Dict) -> Dict[int, float]:

        roi_table = {}
        roi_table[0] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5'] + params['roi_p6']
        roi_table[params['roi_t6']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4'] + params['roi_p5']
        roi_table[params['roi_t6'] + params['roi_t5']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3'] + params['roi_p4']
        roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4']] = params['roi_p1'] + params['roi_p2'] + params['roi_p3']
        roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3']] = params['roi_p1'] + params['roi_p2']
        roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2']] = params['roi_p1']
        roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0

        return roi_table

    @staticmethod
    def roi_space() -> List[Dimension]:

        return [
            Integer(1, 5, name='roi_t6'),
            Integer(1, 10, name='roi_t5'),
            Integer(1, 15, name='roi_t4'),
            Integer(15, 20, name='roi_t3'),
            Integer(20, 25, name='roi_t2'),
            Integer(25, 60, name='roi_t1'),

            Real(0.005, 0.10, name='roi_p6'),
            Real(0.005, 0.07, name='roi_p5'),
            Real(0.005, 0.05, name='roi_p4'),
            Real(0.005, 0.025, name='roi_p3'),
            Real(0.005, 0.01, name='roi_p2'),
            Real(0.003, 0.007, name='roi_p1'),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        conditions = []

        if params['buy-method'] == 'supertrend' or params['buy-method'] == 'both':
            conditions.append(dataframe[params['buy-price']] > dataframe['supertrend'])
        
        if params['buy-method'] == 'sar' or params['buy-method'] == 'both':
            conditions.append(dataframe[params['buy-price']] > dataframe['sar'])

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []

        if params['sell-method'] == 'supertrend' or params['sell-method'] == 'both':
            conditions.append(dataframe[params['sell-price']] < dataframe['supertrend'])
        
        if params['sell-method'] == 'sar' or params['sell-method'] == 'both':
            conditions.append(dataframe[params['sell-price']] < dataframe['sar'])

        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        return dataframe