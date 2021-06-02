from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

from freqtrade.optimize.hyperopt_interface import IHyperOpt

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class SolipsisHyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            # Base Timeframe
            Integer(20, 70, name='base-rmi-slow'),
            Integer(10, 50, name='base-rmi-fast'),
            Integer(10, 50, name='base-mp'),
            # Informative Timeframe
            Categorical(['lower', 'upper', 'both', 'none'], name='inf-guard'),
            Real(0.70, 0.99, name='inf-pct-adr-top'),
            Real(0.01, 0.20, name='inf-pct-adr-bot'),
            # Extra BTC/ETH Stakes
            Integer(10, 70, name='xtra-inf-stake-rmi'),
            Integer(10, 70, name='xtra-base-stake-rmi'),
            Integer(10, 70, name='xtra-base-fiat-rmi'),
            # Extra BTC/STAKE if not in whitelist
            Integer(10, 70, name='xbtc-base-rmi'),
            Integer(10, 70, name='xbtc-inf-rmi')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
            Real(0.10, 0.90, name='sell-rmi-drop')
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            conditions = []

            inf_timeframe = '1h'
            stake_currency = 'USD'
            custom_fiat = 'USD'
            btc_in_whitelist = False

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
                (dataframe['rmi-dn-trend'] == 1) &
                (dataframe['rmi-slow'] >= params['base-rmi-slow']) &
                (dataframe['rmi-fast'] <= params['base-rmi-fast']) &
                (dataframe['mp'] <= params['base-mp'])
            )

            if stake_currency in ('BTC', 'ETH'):
                conditions.append(
                    (dataframe[f"{stake_currency}_rmi"] < params['xtra-base-stake-rmi']) | 
                    (dataframe[f"{custom_fiat}_rmi"] > params['xtra-base-fiat-rmi'])
                )
                conditions.append(dataframe[f"{stake_currency}_rmi_{inf_timeframe}"] < params['xtra-inf-stake-rmi'])
            else:
                if btc_in_whitelist == False:
                    conditions.append(
                        (dataframe['BTC_rmi'] < params['xbtc-base-rmi']) &
                        (dataframe[f"BTC_rmi_{inf_timeframe}"] > params['xbtc-inf-rmi'])
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

        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            conditions = []

            rmi_drop = dataframe['rmi-max'] - (dataframe['rmi-max'] * params['sell-rmi-drop'])
            conditions.append(
                (dataframe['rmi-dn-trend'] == 1) &
                (qtpylib.crossed_below(dataframe['rmi-slow'], rmi_drop)) &
                (dataframe['volume'].gt(0))
            )

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