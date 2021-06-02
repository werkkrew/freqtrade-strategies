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

class HacklemostOpt(IHyperOpt):

    """
    Only used in the buy/sell methods when --spaces does not include buy or sell
    Should put previously best optimized values here so they are used during ROI/stoploss/etc.
    """
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.02514,
        'bbdelta-tail': 1.19782,
        'buy-rmi': 1,
        'closedelta-close': 0.01671,
        'volume': 9
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-adx': 5
    }

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Integer(0, 50, name='buy-rmi'),
            Real(0.0005, 0.03, name='bbdelta-close'),
            Real(0.0005, 0.03, name='closedelta-close'),
            Real(0.7, 1.2, name='bbdelta-tail'),
            Integer(5, 40, name='volume')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
            Integer(0, 50, name='sell-adx'),
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            conditions = []

            conditions.append(
                (dataframe['rmi'] >= params['buy-rmi']) &
                (dataframe['rmi'].gt(dataframe['rmi'].shift(1))) &
                (dataframe['ha_close'].gt(dataframe['sar'])) &
                (dataframe['bb1-delta'].gt(dataframe['ha_close'] * params['bbdelta-close'])) &
                (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close'])) &
                (dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail'])) &
                (dataframe['ha_close'].lt(dataframe['ema_slow'])) &
                (dataframe['volume'].lt(dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )

            conditions.append(dataframe['volume'].gt(0))

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1

                return dataframe

        return populate_buy_trend

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            
            dataframe.loc[
                (
                    (dataframe['adx'] < params['sell-adx']) &
                    (dataframe['tema'].lt(dataframe['tema'].shift(1))) &
                    (dataframe['rmi'].lt(dataframe['rmi'].shift(1))) & 
                    (dataframe['rmi'].shift(1).gt(dataframe['rmi'].shift(2))) &
                    (dataframe['ha_close'].lt(dataframe['sar']))
                ),
                'sell'
            ] = 1

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
        roi_table[params['roi_t6'] + params['roi_t5'] + params['roi_t4'] + params['roi_t3'] + params['roi_t2'] + params['roi_t1']] = 0.0025

        return roi_table

    @staticmethod
    def roi_space() -> List[Dimension]:
                                                # min  / max      min  /  max
        return [                                #        0    :  0.100 / 0.200
            Integer(1, 20, name='roi_t6'),      # 1   -> 20   :  0.050 / 0.100 
            Integer(10, 20, name='roi_t5'),     # 11  -> 40   :  0.030 / 0.050
            Integer(10, 20, name='roi_t4'),     # 21  -> 60   :  0.015 / 0.030
            Integer(15, 30, name='roi_t3'),     # 36  -> 90   :  0.010 / 0.015
            Integer(45, 90, name='roi_t2'),     # 81  -> 180  :  0.003 / 0.005
            Integer(90, 180, name='roi_t1'),    # 171 -> 360  :  0.0025 (should be 0 but I changed it above.)

            Real(0.05, 0.10, name='roi_p6'),
            Real(0.02, 0.05, name='roi_p5'),
            Real(0.015, 0.020, name='roi_p4'),
            Real(0.005, 0.015, name='roi_p3'),
            Real(0.007, 0.01, name='roi_p2'),
            Real(0.003, 0.005, name='roi_p1'),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        conditions = []

        conditions.append(
            (dataframe['rmi'] >= params['buy-rmi']) &
            (dataframe['rmi'].gt(dataframe['rmi'].shift(1))) &
            (dataframe['ha_close'].gt(dataframe['sar'])) &
            (dataframe['bb1-delta'].gt(dataframe['ha_close'] * params['bbdelta-close'])) &
            (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close'])) &
            (dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail'])) &
            (dataframe['ha_close'].lt(dataframe['ema_slow'])) &
            (dataframe['volume'].lt(dataframe['volume_mean_slow'].shift(1) * params['volume']))
        )
  
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (
                (dataframe['adx'] < params['sell-adx']) &
                (dataframe['tema'].lt(dataframe['tema'].shift(1))) &
                (dataframe['rmi'].lt(dataframe['rmi'].shift(1))) & 
                (dataframe['rmi'].shift(1).gt(dataframe['rmi'].shift(2))) &
                (dataframe['ha_close'].lt(dataframe['sar']))
            ),
            'sell'
        ] = 1

        return dataframe