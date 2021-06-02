from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from skopt.space import Categorical, Dimension, Integer, Real

from freqtrade.optimize.hyperopt_interface import IHyperOpt

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

class Solipsis5Hyp(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            # Base Timeframe
            Integer(5, 15, name='base-yolo'),
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
        return []

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

            # YOLO ON THE BASE
            conditions.append(dataframe['yolo'] > params['base-yolo'])

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

            dataframe['sell'] = 0
            return dataframe

        return populate_sell_trend

    # If not optimizing the sell space assume it is disabled entirely.
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sell'] = 0
        return dataframe

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