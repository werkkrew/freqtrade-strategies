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

class SchismHyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Integer(15, 70, name='rmi-slow'),
            Integer(10, 50, name='rmi-fast'),
            Integer(10, 70, name='mp'),
            Integer(10, 70, name='inf-rsi')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return []

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            inf_timeframe = '4h'
            conditions = []

            conditions.append(
                (dataframe[f"rsi_{inf_timeframe}"] < params['inf-rsi']) &
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['rmi-slow']) &
                (dataframe['rmi-fast'] <= params['rmi-fast']) &
                (dataframe['mp'] <= params['mp'])
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
        """
        Define the sell strategy parameters to be used by Hyperopt.
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            no sell signal
            """
            dataframe['sell'] = 0
            return dataframe

        return populate_sell_trend

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe['sell'] = 0
        return dataframe