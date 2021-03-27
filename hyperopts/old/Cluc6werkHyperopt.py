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

class Cluc6werkHyperopt(IHyperOpt):

    """
    Only used in the buy/sell methods when --spaces does not include buy or sell
    Should put previously best optimized values here so they are used during ROI/stoploss/etc.
    OVERRIDE THESE AT THE BOTTOM FOR SPECIFIC STAKES
    """
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00309,
        'bbdelta-tail': 0.83332,
        'close-bblower': 0.0019,
        'closedelta-close': 0.01889,
        'adx': 30,
        'aroon': 50,
        'volume': 22
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 1.00125,
        'sell-fisher': -1.37474,
        'sell-adx': 30,
        'sell-aroon': 50
    }

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
                    (
                        (
                            (dataframe['stake-fiat-adx'] > params['adx']) & 
                            (dataframe['stake-fiat-aroon-down'] > params['aroon'])
                        ) |
                        (
                            (dataframe['stake-fiat-adx'] < params['adx']) 
                        )
                    ) &
                    (
                        (
                            (dataframe['coin-fiat-adx'] > params['adx']) & 
                            (dataframe['coin-fiat-aroon-up'] > params['aroon'])
                        ) |
                        (
                            (dataframe['coin-fiat-adx'] < params['adx']) 
                        )
                    )
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
                    (
                        (
                            (dataframe['stake-fiat-adx'] > params['sell-adx']) & 
                            (dataframe['stake-fiat-aroon-up'] > params['sell-aroon'])
                        ) |
                        (
                            (dataframe['stake-fiat-adx'] < params['sell-adx']) 
                        )
                    ) &
                    (
                        (
                            (dataframe['coin-fiat-adx'] > params['sell-adx']) & 
                            (dataframe['coin-fiat-aroon-down'] > params['sell-aroon'])
                        ) |
                        (
                            (dataframe['coin-fiat-adx'] < params['sell-adx']) 
                        )
                    )
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
            Integer(1, 15, name='roi_t6'),
            Integer(1, 45, name='roi_t5'),
            Integer(1, 90, name='roi_t4'),
            Integer(45, 120, name='roi_t3'),
            Integer(45, 180, name='roi_t2'),
            Integer(90, 300, name='roi_t1'),

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
                (
                    (
                        (dataframe['stake-fiat-adx'] > params['adx']) & 
                        (dataframe['stake-fiat-aroon-down'] > params['aroon'])
                    ) |
                    (
                        (dataframe['stake-fiat-adx'] < params['adx']) 
                    )
                ) &
                (
                    (
                        (dataframe['coin-fiat-adx'] > params['adx']) & 
                        (dataframe['coin-fiat-aroon-up'] > params['aroon'])
                    ) |
                    (
                        (dataframe['coin-fiat-adx'] < params['adx']) 
                    )
                )
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
                (
                    (
                        (dataframe['stake-fiat-adx'] > params['sell-adx']) & 
                        (dataframe['stake-fiat-aroon-up'] > params['sell-aroon'])
                    ) |
                    (
                        (dataframe['stake-fiat-adx'] < params['sell-adx']) 
                    )
                ) &
                (
                    (
                        (dataframe['coin-fiat-adx'] > params['sell-adx']) & 
                        (dataframe['coin-fiat-aroon-down'] > params['sell-aroon'])
                    ) |
                    (
                        (dataframe['coin-fiat-adx'] < params['sell-adx']) 
                    )
                )
            )

        conditions.append((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['mid-bb2'])
        conditions.append(dataframe['ema_fast'].gt(dataframe['close']))
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1

        return dataframe

class Cluc6werkHyperopt_ETH(Cluc6werkHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'adx': 41,
        'aroon': 31,
        'bbdelta-close': 0.00888,
        'bbdelta-tail': 0.91161,
        'close-bblower': 0.00511,
        'closedelta-close': 0.0181,
        'volume': 30
    }
	
    # Sell hyperspace params:
    sell_params = {
		'sell-adx': 15, 
		'sell-aroon': 63, 
		'sell-bbmiddle-close': 0.97701
    }

class Cluc6werkHyperopt_BTC(Cluc6werkHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'adx': 50,
        'aroon': 30,
        'bbdelta-close': 0.00621,
        'bbdelta-tail': 0.90672,
        'close-bblower': 0.0058,
        'closedelta-close': 0.01988,
        'volume': 34
    }	
	
    # Sell hyperspace params:
    sell_params = {
		'sell-adx': 17, 
		'sell-aroon': 36, 
		'sell-bbmiddle-close': 0.98147
    }	