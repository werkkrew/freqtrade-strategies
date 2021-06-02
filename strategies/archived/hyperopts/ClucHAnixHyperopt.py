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

class ClucHAnixHyperopt(IHyperOpt):

    """
    Only used in the buy/sell methods when --spaces does not include buy or sell
    Should put previously best optimized values here so they are used during ROI/stoploss/etc.
    OVERRIDE THESE AT THE BOTTOM FOR SPECIFIC STAKES
    """

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01965,
        'bbdelta-tail': 0.95089,
        'close-bblower': 0.00799,
        'closedelta-close': 0.00556,
        'rocr-1h': 0.54904
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-fisher': 0.38414, 
        'sell-bbmiddle-close': 1.07634
    }

    @staticmethod
    def indicator_space() -> List[Dimension]:
        return [
            Real(0.5, 1.0, name='rocr-1h'),
            Real(0.0005, 0.02, name='bbdelta-close'),
            Real(0.0005, 0.02, name='closedelta-close'),
            Real(0.7, 1.0, name='bbdelta-tail'),
            Real(0.0005, 0.02, name='close-bblower')
            #,            Integer(15, 40, name='volume')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:
        return [
            Real(0.1, 0.5, name='sell-fisher'),
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
                        (dataframe['lower'].shift().gt(0)) &
                        (dataframe['bbdelta'].gt(dataframe['ha_close'] * params['bbdelta-close'])) &
                        (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close'])) &
                        (dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail'])) &
                        (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                        (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
                ) |
                (       
                        (dataframe['ha_close'] < dataframe['ema_slow']) &
                        (dataframe['ha_close'] < params['close-bblower'] * dataframe['bb_lowerband']) 
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
                (dataframe['fisher'] > params['sell-fisher']) &
                (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
                (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
                (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
                (dataframe['ema_fast'] > dataframe['ha_close']) &
                ((dataframe['ha_close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
                (dataframe['volume'] > 0)
                ,
                'sell'
            ] = 1

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

        dataframe.loc[
            (
                dataframe['rocr_1h'].gt(params['rocr-1h'])
            ) &
            ((      
                    (dataframe['lower'].shift().gt(0)) &
                    (dataframe['bbdelta'].gt(dataframe['ha_close'] * params['bbdelta-close'])) &
                    (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close'])) &
                    (dataframe['tail'].lt(dataframe['bbdelta'] * params['bbdelta-tail'])) &
                    (dataframe['ha_close'].lt(dataframe['lower'].shift())) &
                    (dataframe['ha_close'].le(dataframe['ha_close'].shift()))
            ) |
            (       
                    (dataframe['ha_close'] < dataframe['ema_slow']) &
                    (dataframe['ha_close'] < params['close-bblower'] * dataframe['bb_lowerband']) 
            )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (dataframe['fisher'] > params['sell-fisher']) &
            (dataframe['ha_high'].le(dataframe['ha_high'].shift(1))) &
            (dataframe['ha_high'].shift(1).le(dataframe['ha_high'].shift(2))) &
            (dataframe['ha_close'].le(dataframe['ha_close'].shift(1))) &
            (dataframe['ema_fast'] > dataframe['ha_close']) &
            ((dataframe['ha_close'] * params['sell-bbmiddle-close']) > dataframe['bb_middleband']) &
            (dataframe['volume'] > 0)
            ,
            'sell'
        ] = 1

        return dataframe


class ClucHAnixHyperopt_ETH(ClucHAnixHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01566,
        'bbdelta-tail': 0.8478,
        'close-bblower': 0.00998,
        'closedelta-close': 0.00614,
        'rocr-1h': 0.61579,
        'volume': 27
    }
	
    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 1.02894, 
		'sell-fisher': 0.38414
    }

class ClucHAnixHyperopt_BTC(ClucHAnixHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01192,
        'bbdelta-tail': 0.96183,
        'close-bblower': 0.01212,
        'closedelta-close': 0.01039,
        'rocr-1h': 0.53422,
        'volume': 27
    }
	
    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.98016, 
		'sell-fisher': 0.38414
    }

class ClucHAnixHyperopt_USD(ClucHAnixHyperopt):

    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01806,
        'bbdelta-tail': 0.85912,
        'close-bblower': 0.01158,
        'closedelta-close': 0.01466,
        'rocr-1h': 0.51901,
        'volume': 26
    }
	
    # Sell hyperspace params:
    sell_params = {
        'sell-bbmiddle-close': 0.96094, 
        'sell-fisher': 0.38414
    }