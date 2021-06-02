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

"""
author@: werkkrew
github@: https://github.com/werkkrew/freqtrade-strategies

Hyperopt for my KAMACCIRSI strategy.

Optimizes:
- Period for indicators within specified ranges
- RSI and Stoch lower band location
- TEMA trigger method

Default Ranges for dynamic indicator periods:

rsiStart = 5
rsiEnd = 30
temaStart = 5
temaEnd = 50

Default ranges for lower bands:
RSI: 10-50
Stoch: 10-50

Default Stochasitc Periods:
fastkPeriod = 14
slowkPeriod = 3
slowdPeriod = 3
"""

# Ranges for dynamic indicator periods
rsiStart = 5
rsiEnd = 30
temaStart = 5
temaEnd = 50

# Settings for Stochastic
# These do not get optimized
fastkPeriod = 14
slowkPeriod = 3
slowdPeriod = 3

class StochRSITEMAHyperopt(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        """
        Ranges and Options for BUY strategy
        """
        return [
            Integer(10, 50, name='stoch-lower-band'),
            Integer(rsiStart, rsiEnd, name='rsi-period'),
            Integer(10, 50, name='rsi-lower-band')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        """
        Ranges and Options for SELL strategy
        """
        return [
            Integer(temaStart, temaEnd, name='tema-period'),
            Categorical(['close', 'both', 'average'], name='tema-trigger')
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use.
            """
            conditions = []

            conditions.append(
                dataframe[f"rsi({params['rsi-period']})"] > params['rsi-lower-band'])
            conditions.append(qtpylib.crossed_above(
                dataframe['stoch-slowd'], params['stoch-lower-band']))
            conditions.append(qtpylib.crossed_above(
                dataframe['stoch-slowk'], params['stoch-lower-band']))
            conditions.append(qtpylib.crossed_above(
                dataframe['stoch-slowk'], dataframe['stoch-slowd']))

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

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
            Sell strategy Hyperopt will build and use.
            """

            conditions = []
            if params.get('tema-trigger') == 'close':
                conditions.append(
                    dataframe['close'] < dataframe[f"tema({params['tema-period']})"])
            if params.get('tema-trigger') == 'both':
                conditions.append((dataframe['close'] < dataframe[f"tema({params['tema-period']})"]) & (
                    dataframe['open'] < dataframe[f"tema({params['tema-period']})"]))
            if params.get('tema-trigger') == 'average':
                conditions.append(
                    ((dataframe['close'] + dataframe['open']) / 2) < dataframe[f"tema({params['tema-period']})"])

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

            return dataframe

        return populate_sell_trend

