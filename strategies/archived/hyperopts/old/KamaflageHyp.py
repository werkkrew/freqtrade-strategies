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

class KamaflageHyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Real(-1.00, 1.00, name='macd'),
            Integer(-1.00, 1.00, name='macdhist'),
            Integer(40, 90, name='rmi')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return []

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            conditions = []
  
            conditions.append(dataframe['kama-3'] > dataframe['kama-21'])
            conditions.append(dataframe['macd'] > dataframe['macdsignal'])
            conditions.append(dataframe['macd'] > params['macd'])
            conditions.append(dataframe['macdhist'] > params['macdhist'])
            conditions.append(dataframe['rmi'] > dataframe['rmi'].shift())
            conditions.append(dataframe['rmi'] > params['rmi'])
            conditions.append(dataframe['volume'] < (dataframe['volume_ma'] * 20))

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
            """
            no sell signal
            """
            dataframe['sell'] = 0
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

