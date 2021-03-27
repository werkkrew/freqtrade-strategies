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

class Cluc4werkHyperopt(IHyperOpt):

    """
    Only used in the buy/sell methods when --spaces does not include buy or sell
    Should put previously best optimized values here so they are used during ROI/stoploss/etc.
    OVERRIDE THESE AT THE BOTTOM FOR SPECIFIC STAKES
    """

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00793,
        'bbdelta-tail': 0.83802,
        'close-bblower': 0.0034,
        'closedelta-close': 0.00613,
        'rocr-1h': 0.64081,
        'volume': 21
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-bbmiddle-close': 0.97703
    }

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Real(0.5, 1.0, name='rocr-1h'),
            Real(0.0005, 0.02, name='bbdelta-close'),
            Real(0.0005, 0.02, name='closedelta-close'),
            Real(0.7, 1.0, name='bbdelta-tail'),
            Real(0.0005, 0.02, name='close-bblower'),
            Integer(15, 40, name='volume')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
            Real(0.97, 1.1, name='sell-bbmiddle-close')
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the buy strategy parameters to be used by Hyperopt.
        """
        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Buy strategy Hyperopt will build and use.
            """
            dataframe.loc[
                (
                    dataframe['rocr_1h'].gt(params['rocr-1h'])
                ) &
                ((      
                        dataframe['lower'].shift().gt(0) &
                        dataframe['bbdelta'].gt(dataframe['close'] * params['bbdelta-close']) &
                        dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                        dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail']) &
                        dataframe['close'].lt(dataframe['lower'].shift()) &
                        dataframe['close'].le(dataframe['close'].shift())
                ) |
                (       
                        (dataframe['close'] < dataframe['ema_slow']) &
                        (dataframe['close'] < params['close-bblower'] * dataframe['bb_lowerband']) &
                        (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
                )),
                'buy'
            ] = 1
            return dataframe

        return populate_buy_trend

    @staticmethod
    def sell_strategy_generator(params: Dict[str, Any]) -> Callable:
        """
        Define the sell strategy parameters to be used by hyperopt
        """
        def populate_sell_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:
            
            dataframe.loc[
                (dataframe['high'].le(dataframe['high'].shift(1))) &
                (dataframe['high'].shift(1).le(dataframe['high'].shift(2))) &
                (dataframe['close'].le(dataframe['close'].shift(1))) &
                ((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
                (dataframe['volume'] > 0)
                ,
                'sell'
            ] = 1

            """
            dataframe.loc[
                #(dataframe['high'].le(dataframe['high'].shift(1))) &
                #(dataframe['close'] > dataframe['bb_middleband']) &
                (qtpylib.crossed_above((dataframe['close'] * params['sell-bbmiddle-close']),dataframe['bb_middleband'])) &
                #(qtpylib.crossed_above(dataframe['close'],dataframe['bb_middleband'])) &
                (dataframe['volume'] > 0)
                ,
                'sell'
            ] = 1
            """
            
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
            Integer(1, 300, name='roi_t6'),
            Integer(1, 300, name='roi_t5'),
            Integer(1, 300, name='roi_t4'),
            Integer(1, 300, name='roi_t3'),
            Integer(1, 300, name='roi_t2'),
            Integer(1, 300, name='roi_t1'),

            Real(0.001, 0.005, name='roi_p6'),
            Real(0.001, 0.005, name='roi_p5'),
            Real(0.001, 0.005, name='roi_p4'),
            Real(0.001, 0.005, name='roi_p3'),
            Real(0.0001, 0.005, name='roi_p2'),
            Real(0.0001, 0.005, name='roi_p1'),
        ]

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr-1h'])
            ) &
            ((      
                    dataframe['lower'].shift().gt(0) &
                    dataframe['bbdelta'].gt(dataframe['close'] * params['bbdelta-close']) &
                    dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close']) &
                    dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail']) &
                    dataframe['close'].lt(dataframe['lower'].shift()) &
                    dataframe['close'].le(dataframe['close'].shift())
            ) |
            (       
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < params['close-bblower'] * dataframe['bb_lowerband']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )),
            'buy'
        ] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (dataframe['high'].le(dataframe['high'].shift(1))) &
            (dataframe['high'].shift(1).le(dataframe['high'].shift(2))) &
            (dataframe['close'].le(dataframe['close'].shift(1))) &
            ((dataframe['close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        """
        dataframe.loc[
            #(dataframe['high'].le(dataframe['high'].shift(1))) &
            #(dataframe['close'] > dataframe['bb_middleband']) &
            (qtpylib.crossed_above((dataframe['close'] * params['sell-bbmiddle-close']),dataframe['bb_middleband'])) &
            #(qtpylib.crossed_above(dataframe['close'],dataframe['bb_middleband'])) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1
        """
        return dataframe

class Cluc4werkHyperopt_ETH(Cluc4werkHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00793,
        'bbdelta-tail': 0.83802,
        'close-bblower': 0.0034,
        'closedelta-close': 0.00613,
        'rocr-1h': 0.64081,
        'volume': 21
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-bbmiddle-close': 0.97703
    }

class Cluc4werkHyperopt_BTC(Cluc4werkHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00793,
        'bbdelta-tail': 0.83802,
        'close-bblower': 0.0034,
        'closedelta-close': 0.00613,
        'rocr-1h': 0.64081,
        'volume': 21
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-bbmiddle-close': 0.97703
    }

class Cluc4werkHyperopt_USD(Cluc4werkHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.00793,
        'bbdelta-tail': 0.83802,
        'close-bblower': 0.0034,
        'closedelta-close': 0.00613,
        'rocr-1h': 0.64081,
        'volume': 21
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-bbmiddle-close': 0.97703
    }