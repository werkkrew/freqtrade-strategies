import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from datetime import datetime
from freqtrade.persistence import Trade
from technical.indicators import RMI
from statistics import mean

"""
TODO: 
    - Better buy signal.
    - Potentially leverage an external data source?
"""

class Lateralus(IStrategy):

    timeframe = '5m'
    inf_timeframe = '1h'

    buy_params = {
        'bbdelta-close': 0.00865,
        'bbdelta-tail': 0.82205,
        'close-bblower': 0.0063,
        'closedelta-close': 0.00697,
        'volume': 16
    }

    # ROI table:
    minimal_roi = {
        "0": 0.015
    }

    # Stoploss:
    stoploss = -0.085

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    #startup_candle_count: int = 55

    custom_trade_info = {}

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.inf_timeframe) for pair in pairs]
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Misc. calculations regarding existing open positions (reset on every loop iteration)
        self.custom_trade_info[metadata['pair']] = trade_data = {}
        trade_data['active_trade'] = trade_data['other_trades'] = False

        if self.config['runmode'].value in ('live', 'dry_run'):
            
            active_trade = Trade.get_trades([Trade.pair == metadata['pair'], Trade.is_open.is_(True),]).all()
            other_trades = Trade.get_trades([Trade.pair != metadata['pair'], Trade.is_open.is_(True),]).all()

            if active_trade:
                current_rate = self.get_current_price(metadata['pair'])
                active_trade[0].adjust_min_max_rates(current_rate)
                trade_data['active_trade']   = True
                trade_data['current_profit'] = active_trade[0].calc_profit_ratio(current_rate)
                trade_data['peak_profit']    = active_trade[0].calc_profit_ratio(active_trade[0].max_rate)

            if other_trades:
                trade_data['other_trades'] = True
                total_other_profit = tuple(trade.calc_profit_ratio(self.get_current_price(trade.pair)) for trade in other_trades)
                trade_data['avg_other_profit'] = mean(total_other_profit) 

        self.custom_trade_info[metadata['pair']] = trade_data

        # Set Up Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=10, stds=2.7)
        dataframe['lower-bb1'] = bollinger['lower']
        dataframe['mid-bb1'] = bollinger['mid']

        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=40, stds=1.1)
        dataframe['lower-bb2'] = bollinger2['lower']
        dataframe['mid-bb2'] = bollinger['mid']
       
        dataframe['bb1-delta'] = (dataframe['mid-bb1'] - dataframe['lower-bb1']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['low']).abs()
    
        # Set up other indicators
        dataframe['volume-mean-slow'] = dataframe['volume'].rolling(window=24).mean()
        dataframe['rmi-slow'] = RMI(dataframe, length=20, mom=5)
        dataframe['rmi-fast'] = RMI(dataframe, length=9, mom=3)
        dataframe['ema-slow'] = ta.EMA(dataframe, timeperiod=55)

        # Trend Calculations
        dataframe['max'] = dataframe['high'].rolling(6).max()      
        dataframe['min'] = dataframe['low'].rolling(6).min()       
        dataframe['upper'] = np.where(dataframe['max'] > dataframe['max'].shift(),1,0)      
        dataframe['lower'] = np.where(dataframe['min'] < dataframe['min'].shift(),1,0)      
        dataframe['up_trend'] = np.where(dataframe['upper'].rolling(4, min_periods=1).sum() != 0,1,0)      
        dataframe['dn_trend'] = np.where(dataframe['lower'].rolling(4, min_periods=1).sum() != 0,1,0)

        # Informative Timeframe Indicators
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_timeframe)

        informative['ema-fast'] = ta.EMA(informative, timeperiod=5)
        informative['ema-slow'] = ta.EMA(informative, timeperiod=21)

        inf_macd = ta.MACD(informative)
        informative['macd'] = inf_macd['macd']
        informative['macdsignal'] = inf_macd['macdsignal']
        informative['macdhist'] = inf_macd['macdhist']

        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, self.inf_timeframe, ffill=True)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        params = self.buy_params
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []

        # Persist a buy signal for existing trades to make use of ignore_roi_if_buy_signal = True
        # when this buy signal is not present a sell can happen according to ROI table
        if trade_data['active_trade']:
            if (trade_data['peak_profit'] > 0):
                conditions.append(trade_data['current_profit'] > (trade_data['peak_profit'] * 0.8))
            conditions.append(dataframe['rmi-slow'] >= 60)

        # Normal buy triggers that apply to new trades we want to enter
        else:
            conditions.append(
                (
                    (dataframe[f"ema-fast_{self.inf_timeframe}"] > dataframe[f"ema-slow_{self.inf_timeframe}"]) &
                    (dataframe[f"macd_{self.inf_timeframe}"] > dataframe[f"macdsignal_{self.inf_timeframe}"])
                ) &
                ((      
                        (dataframe['bb1-delta'] > (dataframe['close'] * params['bbdelta-close'])) &
                        (dataframe['closedelta'] > (dataframe['close'] * params['closedelta-close'])) &
                        (dataframe['tail'] < (dataframe['bb1-delta'] * params['bbdelta-tail'])) &
                        (dataframe['close'] < dataframe['lower-bb1'].shift()) &
                        (dataframe['close'] <= dataframe['close'].shift())
                ) |
                (       
                        (dataframe['close'] < dataframe['ema-slow']) &
                        (dataframe['close'] < params['close-bblower'] * dataframe['lower-bb2']) &
                        (dataframe['volume'] < (dataframe['volume-mean-slow'].shift(1) * params['volume']))
                ))
            )

        # applies to both new buys and persisting buy signal
        conditions.append(dataframe['volume'].gt(0))

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trade_data = self.custom_trade_info[metadata['pair']]
        conditions = []
        
        # if we are in an active trade for this pair
        if trade_data['active_trade']:
            # if we are at a loss, consider what the trend looks and preempt the stoploss
            conditions.append(
                (trade_data['current_profit'] < 0) &
                (trade_data['current_profit'] > self.stoploss) &  
                (dataframe['dn_trend'] == 1) &
                (qtpylib.crossed_below(dataframe['rmi-fast'], 50)) &
                (dataframe['volume'].gt(0))
            )

            # if there are other open trades in addition to this one, consider the average profit 
            # across them all (not including this one), don't sell if entire market is down big and wait for recovery
            if trade_data['other_trades']:
                conditions.append(trade_data['avg_other_profit'] >= -0.005)

        # the bot comes through this loop even when there isn't an open trade to sell
        # so we pass an impossible condiiton here because we don't want a sell signal 
        # clogging up the charts and not having one leads the bot to crash
        else:
            conditions.append(dataframe['volume'].lt(0))
                           
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'sell'] = 1
        
        return dataframe

    """
    Custom methods
    """
    def get_current_price(self, pair: str) -> float:
        ticker = self.dp.ticker(pair)
        current_price = ticker['last']

        return current_price

    """
    Price protection on trade entry and timeouts, built-in Freqtrade functionality
    https://www.freqtrade.io/en/latest/strategy-advanced/
    """
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


class Lateralus_Slow(Lateralus):

    timeframe = '1h'
    inf_timeframe = '4h'

    # ROI table:
    minimal_roi = {
        "0": 0.15,
        "10": 0.10,
        "20": 0.05,
        "30": 0.025,
        "60": 0.015,
        "120": 0.005
    }

    # Stoploss:
    stoploss = -0.085

