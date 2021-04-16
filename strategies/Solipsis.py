# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from Solipsis_v4 import Solipsis4 as Solipsis

"""
*** THIS IS YOUR LIVE TRADING COPY ***
*** THIS IS A CHILD STRATEGY, REQUIRES IMPORT OF PARENT STRATEGY **
*** CURRENTLY MAPS TO PARENT STRATEGY: Solipsis_v4 ***

LAST OPTIMIZED:

EXCHANGE: Kraken, full pairlists with some personally selected blacklisted coins

Stake      Date Optimized
----------|----------------
USD 
BTC 
ETH 


Optimization / Backtest Results for Current Live Instance Settings
USD
 


"""
# Sub-strategy with parameters specific to USD stake
class Solipsis_USD(Solipsis):

    timeframe = '5m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "1440": 0
    }

    use_dynamic_roi = True
    use_custom_stoploss = True

    stoploss = -0.10

"""
Optimization / Backtest Results for Current Live Instance Settings
BTC


"""
# Sub-strategy with parameters specific to BTC stake
class Solipsis_BTC(Solipsis):

    timeframe = '5m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "1440": 0
    }

    use_dynamic_roi = True
    use_custom_stoploss = True

    stoploss = -0.10
   
"""
Optimization / Backtest Results for Current Live Instance Settings
ETH


"""
# Sub-strategy with parameters specific to ETH stake
class Solipsis_ETH(Solipsis):

    timeframe = '5m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "1440": 0
    }

    use_dynamic_roi = True
    use_custom_stoploss = True

    stoploss = -0.10