# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

# --- Do not remove these libs ---
from functools import reduce
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class KAMACCIRSI_new(IStrategy):
    """
    author@: werkkrew
    github@: https://github.com/werkkrew/freqtrade-strategies

    Strategy using 3 indicators with fully customizable parameters and full hyperopt support
    including indicator periods as well as cross points.

    There is nothing groundbreaking about this strategy, how it works, or what it does. 
    It was mostly an experiment for me to learn Freqtrade strategies and hyperopt development.

    Default hyperopt defined parameters below were done on 60 days of data from Kraken against 20 BTC pairs
    using the SharpeHyperOptLoss loss function.

    Suggestions and improvements are welcome!

    Supports selling via strategy, as well as ROI and Stoploss/Trailing Stoploss

    Indicators Used:
    KAMA "Kaufman Adaptive Moving Average" (Short Duration)
    KAMA (Long Duration)
    CCI "Commodity Channel Index"
    RSI "Relative Strength Index"

    Buy Strategy:
        kama-cross OR kama-slope 
            kama-short > kama-long
            kama-long-slope > 1
        cci-enabled? 
            cci > X 
        rsi-enabled?
            rsi > Y 

    Sell Strategy:
        kama-cross OR kama-slope 
            kama-short < kama-long
            kama-long-slope < 1
        cci-enabled?
            cci < A
        rsi-enabled?
            rsi < B
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 2

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """

    #*   22/50:      1 trades. 0/0/1 Wins/Draws/Losses. Avg profit  -7.18%. Median profit  -7.18%. Total profit -0.00007190 BTC (  -7.18Σ%). Avg duration   0.0 min. Objective: 1.88236

    # Buy hyperspace params:
    buy_params = {
        'cci-enabled': True,
        'cci-limit': 187,
        'kama-trigger': 'cross',
        'rsi-enabled': True,
        'rsi-limit': 78
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-cci-enabled': True,
        'sell-cci-limit': -124,
        'sell-kama-trigger': 'cross',
        'sell-rsi-enabled': False,
        'sell-rsi-limit': 40
    }

    # ROI table:
    minimal_roi = {
        "0": 0.06674,
        "14": 0.05142,
        "59": 0.01717,
        "137": 0
    }

    # Stoploss:
    stoploss = -0.11084

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.19192
    trailing_stop_positive_offset = 0.25216
    trailing_only_offset_is_reached = False


    """
    END HYPEROPT
    """

    timeframe = '5m'

    # Make sure these match or are not overridden in config
    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.0
    ignore_roi_if_buy_signal = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    # Set this to the highest period value in the indicator_params dict or highest of the ranges in the hyperopt settings (default: 72)
    startup_candle_count: int = 72
    
    """
    Populate all of the indicators we need (note: indicators are separate for buy/sell)
    """
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Dynamic TA indicators
        Used so hyperopt can optimized around the period of various indicators
        """
        dataframe['kama-short'] = ta.KAMA(dataframe, timeperiod=5)
        dataframe['kama-long'] = ta.KAMA(dataframe, timeperiod=20)

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=21)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

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

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

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
    