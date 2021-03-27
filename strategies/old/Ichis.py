# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------
import pandas as pd  # noqa
pd.options.mode.chained_assignment = None  # default='warn'

import technical.indicators as ftt
#from technical.util import resample_to_interval, resampled_merge

from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair

import numpy as np


class ichi(IStrategy):

    # Optimal timeframe for the strategy
    timeframe = '1h'

    #startup_candle_count = 120
    process_only_new_candles = False

    # no ROI
    minimal_roi = {
        "0": 0.05,
        "30": 0.04,
        "60": 0.03,
        "90": 0.025

    }

    # Stoploss:
    stoploss = -0.01
    trailing_stop = True
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True
    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        displacement = 30
        ichimoku = ftt.ichimoku(dataframe, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=displacement
            )

        #dataframe['chikou_span'] = ichimoku['chikou_span']

        # cross indicators
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']

        # cloud, green a > b, red a < b
        #dataframe['senkou_a'] = ichimoku['senkou_span_a']
        #dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        #dataframe['cloud_green'] = ichimoku['cloud_green'] * 1
        #dataframe['cloud_red'] = ichimoku['cloud_red'] * -1

        # DANGER ZONE START

        # The cloud is normally shifted into the future visually, but it's based on present data.
        # So in this case it should be ok to look at the "future" (which is actually the present)
        # by shifting it back by displacement.
        #dataframe['future_green'] = ichimoku['cloud_green'].shift(-displacement).fillna(0).astype('int') * 2

        # The chikou_span is shifted into the past, so we need to be careful not to read the
        # current value.  But if we shift it forward again by displacement it should be safe to use.
        # We're effectively "looking back" at where it normally appears on the chart.
        #dataframe['chikou_high'] = (
        #        (dataframe['chikou_span'] > dataframe['senkou_a']) &
        #        (dataframe['chikou_span'] > dataframe['senkou_b'])
        #    ).shift(displacement).fillna(0).astype('int')

        # DANGER ZONE END

        dataframe['go_long'] = (
                (dataframe['tenkan_sen'] > dataframe['kijun_sen']) &
                (dataframe['close'] > dataframe['leading_senkou_span_a']) &
                (dataframe['close'] > dataframe['leading_senkou_span_b']) 
                #&
                #(dataframe['future_green'] > 0) &
                #(dataframe['chikou_high'] > 0)
                ).astype('int') * 3

        def SSLChannels(dataframe, length = 7, mode='sma'):
            df = dataframe.copy()
            df['ATR'] = ta.ATR(df, timeperiod=14)
            df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
            df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
            df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
            df['hlv'] = df['hlv'].ffill()
            df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
            df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
            return df['sslDown'], df['sslUp']

        ssl = SSLChannels(dataframe, 10)
        dataframe['sslDown'] = ssl[0]
        dataframe['sslUp'] = ssl[1]
        
        dataframe['max'] = dataframe['high'].rolling(3).max()      
        dataframe['min'] = dataframe['low'].rolling(6).min()       
        dataframe['upper'] = np.where(dataframe['max'] > dataframe['max'].shift(),1,0)      
        dataframe['lower'] = np.where(dataframe['min'] < dataframe['min'].shift(),1,0)      
        dataframe['up_trend'] = np.where(dataframe['upper'].rolling(5, min_periods=1).sum() != 0,1,0)      
        dataframe['dn_trend'] = np.where(dataframe['lower'].rolling(5, min_periods=1).sum() != 0,1,0)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['go_long'], 0)) &
            (dataframe['sslUp'] > dataframe['sslDown']) &
            (dataframe['up_trend'] == 1)
        ,
        'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (qtpylib.crossed_above(dataframe['sslDown'], dataframe['sslUp'])) &
                (
                (qtpylib.crossed_below(dataframe['tenkan_sen'], dataframe['kijun_sen']))
                | 
                (qtpylib.crossed_below(dataframe['close'], dataframe['kijun_sen']))
                )
                                
        ,
        'sell'] = 1

        return dataframe