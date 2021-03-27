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

class LateralusHyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Real(0.0005, 0.03, name='bbdelta-close'),
            Real(0.0005, 0.02, name='closedelta-close'),
            Real(0.0005, 0.02, name='close-bblower'),
            Real(0.5, 1.0, name='bbdelta-tail'),
            Integer(1, 30, name='volume')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return []

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            inf_timeframe = '1h'
            conditions = []

            conditions.append(
                (
                    (dataframe[f"ema-fast_{inf_timeframe}"] > dataframe[f"ema-slow_{inf_timeframe}"]) &
                    (dataframe[f"macd_{inf_timeframe}"] > dataframe[f"macdsignal_{inf_timeframe}"])
                ) &
                ((      
                        (dataframe['bb1-delta'] > (dataframe['close'] * params['bbdelta-close'])) &
                        (dataframe['closedelta'] > (dataframe['close'] * params['closedelta-close'])) &
                        (dataframe['tail'] < (dataframe['bb1-delta'] * params['bbdelta-tail'])) &
                        (dataframe['close'] < dataframe['lower-bb1'].shift()) &
                        (dataframe['close'] <= dataframe['close'].shift())
                ) |
                (       
                        (dataframe['close'] < dataframe['ema-slow']) &
                        (dataframe['close'] < params['close-bblower'] * dataframe['lower-bb2']) &
                        (dataframe['volume'] < (dataframe['volume-mean-slow'].shift(1) * params['volume']))
                ))
            )

            conditions.append(dataframe['volume'].gt(0))

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'buy'] = 1
                return dataframe

        return populate_buy_trend

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        no sell signal
        """
        dataframe['sell'] = 0
        return dataframe



