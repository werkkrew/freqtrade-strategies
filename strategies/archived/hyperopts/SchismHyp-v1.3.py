from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

from freqtrade.optimize.hyperopt_interface import IHyperOpt

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Schism3Hyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Categorical(['min', 'max', 'mean', 'median'], name='bounce-price'),
            Integer(1, 10, name='bounce-lookback'),  #  1, 10
            Integer(20, 50, name='down-inf-rsi'),    # 20, 50
            Integer(20, 50, name='down-rmi-slow'),   # 20, 50
            Integer(10, 50, name='down-rmi-fast'),   # 10, 50
            Integer(10, 70, name='down-mp'),         # 10, 70
            Integer(30, 70, name='up-inf-rsi'),      # 30, 70
            Integer(10, 70, name='xinf-stake-rmi'),  # 10, 70
            Integer(50, 90, name='xtf-stake-rsi'),   # 50, 90
            Integer(10, 70, name='xtf-fiat-rsi')     # 10, 70
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

            dataframe['bounce-pending'] = np.where(
                (dataframe[f"rsi_{inf_timeframe}"] >= params['down-inf-rsi']) &
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['down-rmi-slow']) &
                (dataframe['rmi-fast'] <= params['down-rmi-fast']) &
                (dataframe['mp'] <= params['down-mp']),
                1,0
            )

            dataframe['bounce-price'] = np.where(
                dataframe['bounce-pending'] == 1, 
                dataframe['close'], 
                getattr(dataframe['close'].rolling(params['bounce-lookback'], min_periods=1), params['bounce-price'])()
            )

            dataframe['bounce-range'] = np.where(dataframe['bounce-pending'].rolling(params['bounce-lookback'], min_periods=1).sum() >= 1,1,0) 

            conditions.append(
                (dataframe[f"rsi_{inf_timeframe}"] >= params['up-inf-rsi']) &
                (dataframe['bounce-range'] == 1) &
                (dataframe['rmi-up-trend'] == 1) &
                (dataframe['close'] >= dataframe['bounce-price'])
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