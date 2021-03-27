from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

from freqtrade.optimize.hyperopt_interface import IHyperOpt

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Schism2Hyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Integer(15, 70, name='rmi-slow'),
            Integer(10, 50, name='rmi-fast'),
            Integer(10, 70, name='mp'),
            Integer(10, 70, name='inf-rsi'),
            Real(0.70, 0.99, name='inf-pct-adr'),
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

            inf_timeframe = '1h'
            stake_currency = 'USD'
            fiat = 'USD'
            conditions = []

            conditions.append(
                (dataframe[f"rsi_{inf_timeframe}"] >= params['inf-rsi']) &
                (dataframe['close'] <= dataframe[f"3d_low_{inf_timeframe}"] + (params['inf-pct-adr'] * dataframe[f"adr_{inf_timeframe}"])) &
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['rmi-slow']) &
                (dataframe['rmi-fast'] <= params['rmi-fast']) &
                (dataframe['mp'] <= params['mp'])
            )

            if stake_currency in ('BTC', 'ETH'):
                conditions.append(
                    (dataframe[f"{stake_currency}_rsi"] < params['xtf-stake-rsi']) | 
                    (dataframe[f"{fiat}_rsi"] > params['xtf-fiat-rsi'])
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