# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from Solipsis_v3 import Solipsis3 as Solipsis

"""
*** THIS IS YOUR LIVE TRADING COPY ***
*** THIS IS A CHILD STRATEGY, REQUIRES IMPORT OF PARENT STRATEGY **
*** CURRENTLY MAPS TO PARENT STRATEGY: Solipsis_v3 ***

NOTE: Child strategy seems to have issues being used for hyperopt, I am not completely sure why.
      Call hyperopts using the parent strategy (e.g. --strategy Solipsis3 --hyperopt Solipsis3Hyp )
      The child strategy is best used for keeping the parent strategy clean when using custom_pair_params
      in live trading and backtest, but is not required.

Example for Custom Pair Params:

    custom_pair_params = [
        {
            'pairs': ('ABC/XYZ', 'DEF/XYZ'),
            'buy_params': {},
            'sell_params': {},
            'minimal_roi': {}
        }

LAST OPTIMIZED:

EXCHANGE: 

USD - 
BTC -
ETH - 

USD -
    Global:
    5m / 1h
    20210301- / Sharpe / DROI / CSTP
    
    Pairlists (Sorted/grouped by avg. profit in global backtests):

"""
# Sub-strategy with parameters specific to USD stake
class Solipsis_USD(Solipsis):

    timeframe = '5m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "360": 0.005,
        "720": 0
    }

    # Global for not in custom_pair_params
    buy_params = {
        'base-mp': 61,
        'base-rmi-fast': 65,
        'base-rmi-slow': 27,
        'inf-guard': 'lower',
        'inf-pct-adr-bot': 0.14002,
        'inf-pct-adr-top': 0.98205,
        'xbtc-base-rmi': 56,
        'xbtc-inf-rmi': 17,
        'xtra-base-fiat-rmi': 45,
        'xtra-base-stake-rmi': 69,
        'xtra-inf-stake-rmi': 27
    }

    # Global for not in custom_pair_params
    dynamic_roi = {
        'enabled': True,
        'profit-factor': 308,
        'rmi-start': 48,
        'rmi-end': 53,
        'grow-delay': 17,
        'grow-time': 770,
        'fallback': 'table',
        'min-roc-atr': 0.0075
    }

    # Global for not in custom_pair_params
    custom_stop = {
        'decay-time': 1058,
        'decay-delay': 131,
        'decay-start': -0.23,
        'decay-end': -0.043,
        'cur-min-diff': 0.03,
        'cur-threshold': -0.0056,
        'roc-bail': -0.01,
        'rmi-trend': 50,
        'bail-how': 'immediate',
        'pos-trail': False,
        'pos-threshold': 0.005,
        'pos-trail-dist': 0.015
    }

    stoploss = custom_stop['decay-start']
    use_custom_stoploss = True

    custom_pair_params = []

"""
BTC - 
    Global:
    15m / 4h
    20210301- / Sharpe / DROI / CSTP
    

    Pairlists (Sorted/grouped by avg. profit in global backtests):

"""
# Sub-strategy with parameters specific to BTC stake
class Solipsis_BTC(Solipsis):

    timeframe = '15m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "720": 0.005,
        "1440": 0
    }

    # Global for not in custom_pair_params
    buy_params = {
        'base-mp': 36,
        'base-rmi-fast': 45,
        'base-rmi-slow': 36,
        'inf-guard': 'none',
        'inf-pct-adr-bot': 0.08671,
        'inf-pct-adr-top': 0.95097,
        'xbtc-base-rmi': 55,
        'xbtc-inf-rmi': 20,
        'xtra-base-fiat-rmi': 22,
        'xtra-base-stake-rmi': 62,
        'xtra-inf-stake-rmi': 68
    }

    # Global for not in custom_pair_params
    dynamic_roi = {
        'enabled': True,
        'profit-factor': 400,
        'rmi-start': 11,
        'rmi-end': 72,
        'grow-delay': 137,
        'grow-time': 371,
        'fallback': 'roc',
        'min-roc-atr': 0.0025
    }

    # Global for not in custom_pair_params
    custom_stop = {
        'decay-time': 761,
        'decay-delay': 122,
        'decay-start': -0.23,
        'decay-end': -0.001,
        'cur-min-diff': 0.013,
        'cur-threshold': -0.027,
        'roc-bail': -0.01068,
        'rmi-trend': 36,
        'bail-how': 'immediate',
        'pos-trail': True,
        'pos-threshold': 0.003,
        'pos-trail-dist': 0.016
    }

    stoploss = custom_stop['decay-start']
    use_custom_stoploss = True

    custom_pair_params = []
   
"""
ETH - 
    Global:
    1h / 4h
    20210301- / Sharpe / DROI / CSTP
    

    Pairlists (Sorted/grouped by avg. profit in global backtests):

"""
# Sub-strategy with parameters specific to ETH stake
class Solipsis_ETH(Solipsis):

    timeframe = '15m'
    inf_timeframe = '1h'

    minimal_roi = {
        "0": 0.01,
        "720": 0.005,
        "1440": 0
    }

    # Global for not in custom_pair_params
    buy_params = {
        'base-mp': 60,
        'base-rmi-fast': 47,
        'base-rmi-slow': 20,
        'inf-guard': 'upper',
        'inf-pct-adr-bot': 0.17062,
        'inf-pct-adr-top': 0.7037,
        'xbtc-base-rmi': 27,
        'xbtc-inf-rmi': 69,
        'xtra-base-fiat-rmi': 32,
        'xtra-base-stake-rmi': 43,
        'xtra-inf-stake-rmi': 61
    }

    # Global for not in custom_pair_params
    dynamic_roi = {
        'enabled': True,
        'profit-factor': 448,
        'rmi-start': 6,
        'rmi-end': 60,
        'grow-delay': 20,
        'grow-time': 1158,
        'fallback': 'roc',
        'min-roc-atr': 0.0025
    }

    # Global for not in custom_pair_params
    custom_stop = {
        'decay-time': 1159,
        'decay-delay': 183,
        'decay-start': -0.25,
        'decay-end': -0.0225,
        'cur-min-diff': 0.0317,
        'cur-threshold': -0.04,
        'roc-bail': -0.034,
        'rmi-trend': 50,
        'bail-how': 'atr',
        'pos-trail': True,
        'pos-threshold': 0.0225,
        'pos-trail-dist': 0.01
    }

    stoploss = custom_stop['decay-start']
    use_custom_stoploss = True

    custom_pair_params = []