# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy, merge_informative_pair
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
import numpy as np
# --------------------------------

"""
Gettin' Moist
                                           
                               88                    
                               ""             ,d     
                                              88     
88,dPYba,,adPYba,   ,adPPYba,  88 ,adPPYba, MM88MMM  
88P'   "88"    "8a a8"     "8a 88 I8[    ""   88     
88      88      88 8b       d8 88  `"Y8ba,    88     
88      88      88 "8a,   ,a8" 88 aa    ]8I   88,    
88      88      88  `"YbbdP"'  88 `"YbbdP"'   "Y888  

v1
"""

class gettinMoist(IStrategy):

    minimal_roi = {
         "0": 100
    }

    # Stoploss:
    stoploss = -0.99

    timeframe = '5m'

    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    startup_candle_count: int = 72
    process_only_new_candles = False

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['color'] = dataframe['close'] > dataframe['open']
    
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=7)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=6)

        dataframe['primed'] = np.where(dataframe['color'].rolling(3).sum() == 3,1,0)
        dataframe['in-the-mood'] = dataframe['rsi'] > dataframe['rsi'].rolling(12).mean()
        dataframe['moist'] = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        dataframe['throbbing'] = dataframe['roc'] > dataframe['roc'].rolling(12).mean()
        dataframe['ready-to-go'] = np.where(dataframe['close'] > dataframe['open'].rolling(12).mean(), 1,0)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['primed']) &
                (dataframe['moist']) &
                (dataframe['throbbing']) &
                (dataframe['ready-to-go'])
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['sell'] = 0

        return dataframe

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
                    
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if current_profit > 0.01 and current_profit > last_candle['roc']:
            return 'nutted'

        if current_profit < -0.03 and current_profit < last_candle['roc']:
            return 'went_soft'

        return None