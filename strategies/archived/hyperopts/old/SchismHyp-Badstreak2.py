from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

from freqtrade.optimize.hyperopt_interface import IHyperOpt

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SchismHyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Integer(10, 50, name='mp'),
            Integer(10, 70, name='inf-rsi-1'),
            Integer(10, 70, name='inf-rsi-2'),
            Integer(10, 40, name='rmi-above'),
            Integer(50, 80, name='rmi-below'),
            Integer(10, 70, name='xinf-stake-rmi'),
            Integer(50, 90, name='xtf-stake-rsi'),
            Integer(10, 70, name='xtf-fiat-rsi')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return []

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            inf_timeframe = '4h'
            stake_currency = 'USD'
            fiat = 'USD'
            conditions = []

            conditions.append(
                (
                    (dataframe[f"rsi_{inf_timeframe}"] < params['inf-rsi-1']) &
                    (dataframe['mp'] < params['mp']) &
                    (dataframe['streak-roc'] > dataframe['pcc-lowerband']) &
                    (dataframe['mac'] == 1)
                ) | 
                (
                    (dataframe[f"rsi_{inf_timeframe}"] < params['inf-rsi-2']) &
                    (dataframe['rmi-up-trend'] == 1) &
                    (dataframe['rmi'] < params['rmi-below']) &
                    (dataframe['rmi'] > params['rmi-above'])
                )
            )

            if stake_currency in ('BTC', 'ETH'):
                conditions.append(
                    (dataframe[f"{stake_currency}_rsi"] > params['xtf-stake-rsi']) | 
                    (dataframe[f"{fiat}_rsi"] < params['xtf-fiat-rsi'])
                )
                conditions.append(dataframe[f"{stake_currency}_rmi_{inf_timeframe}"] < params['xinf-stake-rmi'])

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

            dataframe['sell'] = 0
            return dataframe

        return populate_sell_trend

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sell'] = 0
        return dataframe