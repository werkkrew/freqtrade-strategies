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

class StinkfistHyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Integer(5, 70, name='mp'),
            Real(0.70, 0.99, name='inf-pct-adr')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return []

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            inf_timeframe = '1h'
            conditions = []

            # Primary buy triggers
            conditions.append(
                # informative timeframe conditions
                (dataframe['close'] <= dataframe[f"3d_low_{inf_timeframe}"] + (params['inf-pct-adr'] * dataframe[f"adr_{inf_timeframe}"])) &
                # default timeframe conditions
                (dataframe['mp'] < params['mp']) &
                (dataframe['streak-roc'] > dataframe['pcc-lowerband']) &
                (dataframe['mac'] == 1)
            )

            # Anything below here applies to persisting and new buy signal
            conditions.append(dataframe['volume'].gt(0))

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1
                return dataframe

        return populate_buy_trend

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
                                                # min  / max      min  /  max
        return [                                #        0    :  0.100 / 0.205
            Integer(1, 20, name='roi_t6'),      # 1   -> 20   :  0.050 / 0.105 
            Integer(10, 20, name='roi_t5'),     # 11  -> 40   :  0.030 / 0.055
            Integer(10, 20, name='roi_t4'),     # 21  -> 60   :  0.015 / 0.035
            Integer(15, 30, name='roi_t3'),     # 36  -> 90   :  0.010 / 0.020
            Integer(264, 630, name='roi_t2'),   # 300 -> 720  :  0.005 / 0.010
            Integer(420, 720, name='roi_t1'),   # 720 -> 1440 :  0

            Real(0.05, 0.10, name='roi_p6'),
            Real(0.02, 0.05, name='roi_p5'),
            Real(0.015, 0.020, name='roi_p4'),
            Real(0.005, 0.015, name='roi_p3'),
            Real(0.005, 0.01, name='roi_p2'),
            Real(0.005, 0.01, name='roi_p1'),
        ]

    @staticmethod
    def stoploss_space() -> List[Dimension]:

        return [
            Real(-0.99, -0.01, name='stoploss'),
        ]

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe['sell'] = 0
        return dataframe