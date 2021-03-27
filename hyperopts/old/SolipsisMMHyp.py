from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

from freqtrade.optimize.hyperopt_interface import IHyperOpt

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SolipsisMMHyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            # Base Timeframe
            Categorical(['up', 'down'], name='base-trend'),
            Categorical([True, False], name='base-pmax-enable'),
            Categorical(['up', 'down'], name='base-pmax'),
            Integer(10, 90, name='base-rmi-slow'),
            Integer(10, 90, name='base-rmi-fast'),
            Integer(10, 90, name='base-mp'),
            Categorical([1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0], name='base-fib-level'),
            # Informative Timeframe
            Integer(10, 70, name='inf-rsi'),
            Categorical(['lower', 'upper', 'both', 'none'], name='inf-guard'),
            Real(0.70, 0.99, name='inf-pct-adr-top'),
            Real(0.01, 0.20, name='inf-pct-adr-bot'),
            Categorical([1.0, 0.786, 0.618, 0.5, 0.382, 0.236, 0.0], name='inf-fib-level'),
            # Exra BTC/ETH Stakes
            Integer(10, 70, name='xtra-inf-stake-rmi'),
            Integer(10, 70, name='xtra-base-stake-rsi'),
            Integer(10, 70, name='xtra-base-fiat-rsi')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
            Real(0.10, 0.90, name='sell-rmi-drop'),
            Categorical([True, False], name='sell-pmax-enable'),
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            conditions = []

            inf_timeframe = '1h'
            stake_currency = 'USD'
            custom_fiat = 'USD'

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
                (dataframe[f"fib-ret_{inf_timeframe}"] >= params['inf-fib-level'])
            )

            if params['base-pmax-enable'] == True:
                conditions.append((dataframe['pmax-trend'] == params['base-pmax']))

            if params['base-trend'] == 'down':
                conditions.append(
                    (dataframe['fib-ret'] <= params['base-fib-level']) &
                    (dataframe['rmi-dn-trend'] == 1) &
                    (dataframe['rmi-slow'] >= params['base-rmi-slow']) &
                    (dataframe['rmi-fast'] <= params['base-rmi-fast']) &
                    (dataframe['mp'] <= params['base-mp'])
                )

            elif params['base-trend'] == 'up':
                conditions.append(
                    (dataframe['fib-ret'] >= params['base-fib-level']) &
                    (dataframe['rmi-up-trend'] == 1) &
                    (dataframe['rmi-slow'] <= params['base-rmi-slow']) &
                    (dataframe['rmi-fast'] >= params['base-rmi-fast']) &
                    (dataframe['mp'] >= params['base-mp'])
                )

            if stake_currency in ('BTC', 'ETH'):
                conditions.append(
                    (dataframe[f"{stake_currency}_rsi"] < params['xtra-base-stake-rsi']) | 
                    (dataframe[f"{custom_fiat}_rsi"] > params['xtra-base-fiat-rsi'])
                )
                conditions.append(dataframe[f"{stake_currency}_rmi_{inf_timeframe}"] < params['xtra-inf-stake-rmi'])

            conditions.append(dataframe['buy_signal'])
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
            conditions = []

            rmi_drop = dataframe['rmi-max'] - (dataframe['rmi-max'] * params['sell-rmi-drop'])
            conditions.append(
                (dataframe['rmi-dn-trend'] == 1) &
                (qtpylib.crossed_below(dataframe['rmi-slow'], rmi_drop)) &
                (dataframe['volume'].gt(0))
            )

            if params['sell-pmax-enable'] == True:
                conditions.append((dataframe['pmax-trend'] == 'down'))

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

            # dataframe['sell'] = 0
            return dataframe

        return populate_sell_trend

    # If not optimizing the sell space assume it is disabled entirely.
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sell'] = 0
        return dataframe