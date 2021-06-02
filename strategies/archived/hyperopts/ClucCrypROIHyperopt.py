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

class ClucCrypROIHyperopt(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Real(0.0005, 0.02, name='bbdelta-close'),
            Real(0.0005, 0.02, name='closedelta-close'),
            Real(0.7, 1.0, name='bbdelta-tail'),
            Real(0.0005, 0.02, name='close-bblower'),
            Integer(15, 40, name='volume'),
            Integer(10, 50, name='adx'),
            Integer(30, 70, name='aroon')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
            Real(0.95, 1.2, name='sell-bbmiddle-close'),
            Integer(10, 50, name='sell-adx'),
            Integer(30, 70, name='sell-aroon')
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            
            conditions = []

            if 'stake-fiat-adx' in dataframe.columns and 'coin-fiat-adx' in dataframe.columns:
                conditions.append(
                    ((
                        (dataframe['stake-fiat-adx'] > params['adx']) & 
                        (dataframe['stake-fiat-aroon-down'] > params['aroon'])
                    ) | (
                        (dataframe['stake-fiat-adx'] < params['adx']) 
                    )) & ((
                        (dataframe['coin-fiat-adx'] > params['adx']) & 
                        (dataframe['coin-fiat-aroon-up'] > params['aroon'])
                    ) | (
                        (dataframe['coin-fiat-adx'] < params['adx']) 
                    ))
                )

            conditions.append(
                (      
                    dataframe['bb1-delta'].gt(dataframe['close'] * params['bbdelta-close']) &
                    dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                    dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail']) &
                    dataframe['close'].lt(dataframe['lower-bb1'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
                ) |
                (       
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < params['close-bblower'] * dataframe['lower-bb2']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
                )
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

            if 'stake-fiat-adx' in dataframe.columns and 'coin-fiat-adx' in dataframe.columns:
                conditions.append(
                    ((
                        (dataframe['stake-fiat-adx'] > params['sell-adx']) & 
                        (dataframe['stake-fiat-aroon-up'] > params['sell-aroon'])
                    ) | (
                        (dataframe['stake-fiat-adx'] < params['sell-adx']) 
                    )) & ((
                        (dataframe['coin-fiat-adx'] > params['sell-adx']) & 
                        (dataframe['coin-fiat-aroon-down'] > params['sell-aroon'])
                    ) | (
                        (dataframe['coin-fiat-adx'] < params['sell-adx']) 
                    ))
                )

            conditions.append((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['mid-bb2'])
            conditions.append(dataframe['ema_fast'].gt(dataframe['close']))
            conditions.append(dataframe['volume'].gt(0))

            if conditions:
                dataframe.loc[
                    reduce(lambda x, y: x & y, conditions),
                    'sell'] = 1

            return dataframe

        return populate_sell_trend

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

        return [
            Integer(1, 5, name='roi_t6'),
            Integer(1, 10, name='roi_t5'),
            Integer(1, 15, name='roi_t4'),
            Integer(15, 20, name='roi_t3'),
            Integer(20, 25, name='roi_t2'),
            Integer(25, 60, name='roi_t1'),

            Real(0.005, 0.10, name='roi_p6'),
            Real(0.005, 0.07, name='roi_p5'),
            Real(0.005, 0.05, name='roi_p4'),
            Real(0.005, 0.025, name='roi_p3'),
            Real(0.005, 0.01, name='roi_p2'),
            Real(0.003, 0.007, name='roi_p1'),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        conditions = []

        if 'stake-fiat-adx' in dataframe.columns and 'coin-fiat-adx' in dataframe.columns:
            conditions.append(
                ((
                    (dataframe['stake-fiat-adx'] > params['adx']) & 
                    (dataframe['stake-fiat-aroon-down'] > params['aroon'])
                ) | (
                    (dataframe['stake-fiat-adx'] < params['adx']) 
                )) & ((
                    (dataframe['coin-fiat-adx'] > params['adx']) & 
                    (dataframe['coin-fiat-aroon-up'] > params['aroon'])
                ) | (
                    (dataframe['coin-fiat-adx'] < params['adx']) 
                ))
            )

        conditions.append(
            (      
                dataframe['bb1-delta'].gt(dataframe['close'] * params['bbdelta-close']) &
                dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail']) &
                dataframe['close'].lt(dataframe['lower-bb1'].shift()) &
                dataframe['close'].le(dataframe['close'].shift())
            ) |
            (       
                (dataframe['close'] < dataframe['ema_slow']) &
                (dataframe['close'] < params['close-bblower'] * dataframe['lower-bb2']) &
                (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )
        )

        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        conditions = []

        if 'stake-fiat-adx' in dataframe.columns and 'coin-fiat-adx' in dataframe.columns:
            conditions.append(
                ((
                    (dataframe['stake-fiat-adx'] > params['sell-adx']) & 
                    (dataframe['stake-fiat-aroon-up'] > params['sell-aroon'])
                ) | (
                    (dataframe['stake-fiat-adx'] < params['sell-adx']) 
                )) & ((
                    (dataframe['coin-fiat-adx'] > params['sell-adx']) & 
                    (dataframe['coin-fiat-aroon-down'] > params['sell-aroon'])
                ) | (
                    (dataframe['coin-fiat-adx'] < params['sell-adx']) 
                ))
            )

        conditions.append((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['mid-bb2'])
        conditions.append(dataframe['ema_fast'].gt(dataframe['close']))
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

class ClucCrypROIHyperopt_ETH(ClucCrypROIHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'adx': 46,
        'aroon': 51,
        'bbdelta-close': 0.01775,
        'bbdelta-tail': 0.84013,
        'close-bblower': 0.01096,
        'closedelta-close': 0.01068,
        'volume': 15
    }
	
    # Sell hyperspace params:
    sell_params = {
		'sell-adx': 39, 
		'sell-aroon': 65, 
		'sell-bbmiddle-close': 0.98656
    }

class ClucCrypROIHyperopt_BTC(ClucCrypROIHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'adx': 47,
        'aroon': 34,
        'bbdelta-close': 0.01957,
        'bbdelta-tail': 0.86961,
        'close-bblower': 0.00257,
        'closedelta-close': 0.01381,
        'volume': 27
    }

    # Sell hyperspace params:
    sell_params = {
		'sell-adx': 20, 
		'sell-aroon': 63, 
		'sell-bbmiddle-close': 0.95563
    }