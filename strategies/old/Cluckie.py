import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from datetime import datetime
from freqtrade.persistence import Trade

class Cluckie(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """
     # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.01118,
        'bbdelta-tail': 0.88481,
        'close-bblower': 0.00396,
        'closedelta-close': 0.01232,
        'fisher': -0.8167,
        'volume': 29
    }

    # Sell hyperspace params:
    sell_params = {
        'sell-adx': 70,
        'sell-fisher': 0.95954
    }

    # ROI table:
    minimal_roi = {
        "0": 100
    }

    # Stoploss:
    stoploss = -0.015

    """
    END HYPEROPT
    """
    
    timeframe = '5m'


    # Make sure these match or are not overridden in config
    use_sell_signal = True
    #sell_profit_only = True
    #sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True

    def check_buy_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['bids'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > order['price'] * 1.01:
            return True
        return False


    def check_sell_timeout(self, pair: str, trade: Trade, order: dict, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel sell order if price is more than 1% below the order.
        if current_price < order['price'] * 0.99:
            return True
        return False

    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Set Up Bollinger Bands
        upper_bb1, mid_bb1, lower_bb1 = ta.BBANDS(dataframe['close'], timeperiod=40)
        upper_bb2, mid_bb2, lower_bb2 = ta.BBANDS(qtpylib.typical_price(dataframe), timeperiod=20)

        # only putting some bands into dataframe as the others are not used elsewhere in the strategy
        dataframe['lower-bb1'] = lower_bb1
        dataframe['lower-bb2'] = lower_bb2
        dataframe['mid-bb2'] = mid_bb2
       
        dataframe['bb1-delta'] = (mid_bb1 - dataframe['lower-bb1']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()

        dataframe['ema_slow'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=24).mean()

        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher-rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)
        
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        dataframe['adx'] = ta.ADX(dataframe)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params

        dataframe.loc[
            (
                dataframe['fisher-rsi'].lt(params['fisher'])
            ) &
            ((      
                    (dataframe['bb1-delta'].gt(dataframe['close'] * params['bbdelta-close'])) &
                    (dataframe['closedelta'].gt(dataframe['close'] * params['closedelta-close'])) &
                    (dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail'])) &
                    (dataframe['close'].lt(dataframe['lower-bb1'].shift())) &
                    (dataframe['close'].le(dataframe['close'].shift())) &
                    (dataframe['tema'] > dataframe['tema'].shift(1)) 
            ) |
            (       
                    (dataframe['close'] < dataframe['ema_slow']) &
                    (dataframe['close'] < params['close-bblower'] * dataframe['lower-bb2']) &
                    (dataframe['volume'] < (dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )),
            'buy'
        ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (
                    (dataframe['adx'] > params['sell-adx']) &
                    (dataframe['tema'] > dataframe['mid-bb2']) &
                    (dataframe['tema'] < dataframe['tema'].shift(1)) &
                    (dataframe['fisher-rsi'].gt(params['sell-fisher']))
            )
            ,
            'sell'
        ] = 1

        return dataframe