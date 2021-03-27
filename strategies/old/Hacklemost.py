import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy as np
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI

class Hacklemost(IStrategy):

    """
    PASTE OUTPUT FROM HYPEROPT HERE
    """
    # Buy hyperspace params:
    buy_params = {
        'bbdelta-close': 0.02514,
        'bbdelta-tail': 1.19782,
        'buy-rmi': 1,
        'closedelta-close': 0.01671,
        'volume': 9
    }

    # Sell hyperspace params:
    sell_params = {
     'sell-adx': 5
    }

    # ROI table:
    minimal_roi = {
        "0": 0.14034,
        "1": 0.06487,
        "21": 0.03861,
        "36": 0.02318,
        "57": 0.01272,
        "126": 0.00494,
        "255": 0.0025
    }

    # Stoploss:
    stoploss = -0.31765

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.20185
    trailing_stop_positive_offset = 0.22421
    trailing_only_offset_is_reached = True


    """
    END HYPEROPT
    """
    
    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    sell_profit_offset = 0.01
    ignore_roi_if_buy_signal = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']

        # Set Up Bollinger Bands
        upper_bb1, mid_bb1, lower_bb1 = ta.BBANDS(dataframe['ha_close'], timeperiod=40)
        upper_bb2, mid_bb2, lower_bb2 = ta.BBANDS(qtpylib.typical_price(heikinashi), timeperiod=20)

        # only putting some bands into dataframe as the others are not used elsewhere in the strategy
        dataframe['lower-bb1'] = lower_bb1
        dataframe['lower-bb2'] = lower_bb2
        dataframe['mid-bb2'] = mid_bb2
       
        dataframe['bb1-delta'] = (mid_bb1 - dataframe['lower-bb1']).abs()
        dataframe['closedelta'] = (dataframe['ha_close'] - dataframe['ha_close'].shift()).abs()
        dataframe['tail'] = (dataframe['ha_close'] - dataframe['ha_low']).abs()

        dataframe['ema_slow'] = ta.EMA(dataframe['ha_close'], timeperiod=48)
        dataframe['volume_mean_slow'] = dataframe['volume'].rolling(window=24).mean()

        dataframe['rsi'] = ta.RSI(heikinashi, timeperiod=14)
        
        dataframe['tema'] = ta.TEMA(heikinashi, timeperiod=9)
        dataframe['adx'] = ta.ADX(heikinashi)
        dataframe['rmi'] = RMI(heikinashi)

        dataframe['sar'] = ta.SAR(heikinashi)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        active_trade = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()

        conditions = []

        if not active_trade:
            conditions.append(
                (dataframe['rmi'] >= params['buy-rmi']) &
                (dataframe['rmi'].gt(dataframe['rmi'].shift(1))) &
                (dataframe['ha_close'].gt(dataframe['sar'])) &
                (dataframe['bb1-delta'].gt(dataframe['ha_close'] * params['bbdelta-close'])) &
                (dataframe['closedelta'].gt(dataframe['ha_close'] * params['closedelta-close'])) &
                (dataframe['tail'].lt(dataframe['bb1-delta'] * params['bbdelta-tail'])) &
                (dataframe['ha_close'].lt(dataframe['ema_slow'])) &
                (dataframe['volume'].lt(dataframe['volume_mean_slow'].shift(1) * params['volume']))
            )
        else:
            conditions.append(dataframe['rmi'] >= params['buy-rmi'])
            conditions.append(dataframe['rmi'].gt(dataframe['rmi'].shift(1)))     

        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.sell_params

        dataframe.loc[
            (
                (dataframe['adx'] < params['sell-adx']) &
                (dataframe['tema'].lt(dataframe['tema'].shift(1))) &
                (dataframe['rmi'].lt(dataframe['rmi'].shift(1))) & 
                (dataframe['rmi'].shift(1).gt(dataframe['rmi'].shift(2))) &
                (dataframe['ha_close'].lt(dataframe['sar']))
            ),
            'sell'
        ] = 1

        return dataframe

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

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        ob = self.dp.orderbook(pair, 1)
        current_price = ob['asks'][0][0]
        # Cancel buy order if price is more than 1% above the order.
        if current_price > rate * 1.01:
            return False
        return True