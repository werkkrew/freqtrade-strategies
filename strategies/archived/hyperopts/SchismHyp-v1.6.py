from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

from freqtrade.optimize.hyperopt_interface import IHyperOpt

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Schism6Hyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            # Base Timeframe
            Integer(15, 70, name='base-rmi-slow'),
            Integer(10, 50, name='base-rmi-fast'),
            Integer(10, 70, name='base-mp'),
            # Informative Timeframe
            Integer(10, 70, name='inf-rsi'),
            Categorical(['lower', 'upper', 'both', 'none'], name='inf-guard'),
            Real(0.70, 0.99, name='inf-pct-adr-top'),
            Real(0.01, 0.20, name='inf-pct-adr-bot'),
            # Exra BTC/ETH Stakes
            Integer(10, 70, name='xtra-inf-stake-rmi'),
            Integer(50, 90, name='xtra-base-stake-rsi'),
            Integer(10, 70, name='xtra-base-fiat-rsi')
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

            if params['inf-guard'] == 'upper' or params['inf-guard'] == 'both':
                conditions.append(
                    (dataframe['close'] <= dataframe[f"3d_low_{inf_timeframe}"] + 
                    (params['inf-pct-adr-top'] * dataframe[f"adr_{inf_timeframe}"]))
                )

            if params['inf-guard'] == 'lower' or params['inf-guard'] == 'both':
                conditions.append(
                    (dataframe['close'] >= dataframe[f"3d_low_{inf_timeframe}"] + 
                    (params['inf-pct-adr-bot'] * dataframe[f"adr_{inf_timeframe}"]))
                )

            conditions.append(
                (dataframe[f"rsi_{inf_timeframe}"] >= params['inf-rsi']) &
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['base-rmi-slow']) &
                (dataframe['rmi-fast'] <= params['base-rmi-fast']) &
                (dataframe['mp'] <= params['base-mp'])
            )

            if stake_currency in ('BTC', 'ETH'):
                conditions.append(
                    (dataframe[f"{stake_currency}_rsi"] < params['xtra-base-stake-rsi']) | 
                    (dataframe[f"{fiat}_rsi"] > params['xtra-base-fiat-rsi'])
                )
                conditions.append(dataframe[f"{stake_currency}_rmi_{inf_timeframe}"] < params['xtra-inf-stake-rmi'])

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