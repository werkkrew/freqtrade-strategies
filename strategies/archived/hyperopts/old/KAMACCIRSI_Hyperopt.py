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

"""
author@: werkkrew
github@: https://github.com/werkkrew/freqtrade-strategies

Hyperopt for my KAMACCIRSI strategy.

Optimizes:
- Period for indicators within specified ranges
- Should RSI and/or CCI be enabled
    - Cross points for RSI / CCI on both buy and sell side separately
- Should KAMA use a crossing point or slope

Default ranges for cross points:
RSI:
    - Buy: 0-100
    - Sell: 0-100
CCI: 
    - Buy: 0-200
    - Sell: -200-0
"""

class KAMACCIRSI_new_Hyperopt(IHyperOpt):

    @staticmethod
    def indicator_space() -> List[Dimension]:

        return [
            Integer(100, 200, name='cci-limit'),
            Integer(40, 90, name='rsi-limit'),           
            Categorical([True, False], name='cci-enabled'),
            Categorical([True, False], name='rsi-enabled'),
            Categorical(['cross', 'slope'], name='kama-trigger')
        ]

    @staticmethod
    def sell_indicator_space() -> List[Dimension]:

        return [
            Integer(-200, -100, name='sell-cci-limit'),
            Integer(40, 90, name='sell-rsi-limit'), 
            Categorical([True, False], name='sell-cci-enabled'),
            Categorical([True, False], name='sell-rsi-enabled'),
            Categorical(['cross', 'slope'], name='sell-kama-trigger')
        ]

    @staticmethod
    def buy_strategy_generator(params: Dict[str, Any]) -> Callable:

        def populate_buy_trend(dataframe: DataFrame, metadata: dict) -> DataFrame:

            conditions = []
            # Guards
            if params.get('rsi-enabled'):
                conditions.append(
                    (
                        qtpylib.crossed_above(dataframe['rsi'], params['rsi-limit']) |
                        qtpylib.crossed_above(dataframe['rsi'].shift(1), params['rsi-limit']) |
                        qtpylib.crossed_above(dataframe['rsi'].shift(2), params['rsi-limit'])
                    ) &
                    (
                        dataframe['rsi'].gt(params['rsi-limit'])
                    )
                )
            if params.get('cci-enabled'):
                conditions.append(
                    (
                        qtpylib.crossed_above(dataframe['cci'], params['cci-limit']) |
                        qtpylib.crossed_above(dataframe['cci'].shift(1), params['cci-limit']) |
                        qtpylib.crossed_above(dataframe['cci'].shift(2), params['cci-limit'])
                    ) &
                    (
                        dataframe['cci'].gt(params['cci-limit'])
                    )
                )

            # Triggers
            if params.get('kama-trigger') == 'cross':
                conditions.append(qtpylib.crossed_above(dataframe['kama-short'], dataframe['kama-long']))
            if params.get('kama-trigger') == 'slope':
                conditions.append(dataframe['kama-long'] > dataframe['kama-long'].shift(1))

            # Check that volume is not 0
            conditions.append(dataframe['volume'] > 0)

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
            # Guards
            if params.get('sell-rsi-enabled'):
                conditions.append(
                    (
                        qtpylib.crossed_below(dataframe['rsi'], params['sell-rsi-limit']) |
                        qtpylib.crossed_below(dataframe['rsi'].shift(1), params['sell-rsi-limit']) |
                        qtpylib.crossed_below(dataframe['rsi'].shift(2), params['sell-rsi-limit'])
                    ) &
                    (
                        dataframe['rsi'].lt(params['sell-rsi-limit'])
                    )
                )
            if params.get('sell-cci-enabled'):
                conditions.append(
                    (
                        qtpylib.crossed_below(dataframe['cci'], params['sell-cci-limit']) |
                        qtpylib.crossed_below(dataframe['cci'].shift(1), params['sell-cci-limit']) |
                        qtpylib.crossed_below(dataframe['cci'].shift(2), params['sell-cci-limit'])
                    ) &
                    (
                        dataframe['cci'].lt(params['sell-cci-limit'])
                    )
                )

            # Triggers
            if params.get('sell-kama-trigger') == 'cross':
                conditions.append(qtpylib.crossed_below(dataframe['kama-short'], dataframe['kama-long']))
            if params.get('sell-kama-trigger') == 'slope':
                conditions.append(dataframe['kama-long'] < dataframe['kama-long'].shift(1))

            # Check that the candle had volume
            conditions.append(dataframe['volume'] > 0)

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