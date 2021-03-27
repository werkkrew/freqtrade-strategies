# Strategy specific imports, files must reside in same folder as strategy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from Solipsis_v3 import Solipsis3 as Solipsis

"""
*** THIS IS YOUR LIVE TRADING COPY ***
*** THIS IS A CHILD STRATEGY, REQUIRES IMPORT OF PARENT STRATEGY **
*** CURRENTLY MAPS TO PARENT STRATEGY: Solipsis_v3 ***

LAST OPTIMIZED:

EXCHANGE: Kraken, full pairlists with some personally selected blacklisted coins

USD - 03/25/2021
BTC - 03/27/2021
ETH - 03/27/2021

USD -
    Global:
    5m / 1h
    20210301- / Sharpe / DROI / CSTP
    1038 trades. 929/0/109 Wins/Draws/Losses. Avg profit   0.86%. Total profit  557.30976144 USD ( 889.87Σ%). Avg duration 436.9 min. Objective: -303.93860

    Pairlists (Sorted/grouped by avg. profit in global backtests):

    ('KNC/USD', 'OXT/USD', 'QTUM/USD', 'YFI/USD', 'BAL/USD', 'WAVES/USD', 'FIL/USD', 'ANT/USD', 'REPV1/USD', 'ETC/USD', 'MANA/USD', 'LSK/USD', 'CRV/USD', 'MLN/USD')
    143/1000:   1292 trades. 1168/13/111 Wins/Draws/Losses. Avg profit   0.74%. Total profit  586.33656072 USD ( 957.50Σ%). Avg duration 197.1 min. Objective: -402.24916

    ('ZEC/USD', 'REP/USD', 'GRT/USD', 'UNI/USD', 'FLOW/USD', 'DASH/USD', 'KEEP/USD', 'AAVE/USD', 'BCH/USD', 'ALGO/USD', 'KAVA/USD', 'BAT/USD', 'KSM/USD', 'ETH/USD', 'SC/USD', 'EOS/USD', 'XMR/USD', 'LTC/USD', 'NANO/USD', 'GNO/USD')
    81/704:   1744 trades. 1590/24/130 Wins/Draws/Losses. Avg profit   0.65%. Total profit  735.77585861 USD ( 1136.05Σ%). Avg duration 231.8 min. Objective: -535.37970

    ('EWT/USD', 'ADA/USD', 'ATOM/USD', 'XTZ/USD', 'LINK/USD', 'XLM/USD', 'TRX/USD', 'DOT/USD', 'OMG/USD', 'ICX/USD', 'OCEAN/USD', 'COMP/USD', 'STORJ/USD', 'SNX/USD')
    213/392:   1292 trades. 1130/4/158 Wins/Draws/Losses. Avg profit   0.46%. Total profit  332.13998403 USD ( 599.07Σ%). Avg duration 135.4 min. Objective: -322.25214


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

    custom_pair_params = [
        {
            'pairs': ('KNC/USD', 'OXT/USD', 'QTUM/USD', 'YFI/USD', 'BAL/USD', 'WAVES/USD', 'FIL/USD', 'ANT/USD', 'REPV1/USD', 'ETC/USD', 'MANA/USD', 'LSK/USD', 'CRV/USD', 'MLN/USD'),
            'buy_params': {
                'base-mp': 73,
                'base-rmi-fast': 53,
                'base-rmi-slow': 24,
                'inf-guard': 'none',
                'inf-pct-adr-bot': 0.02261,
                'inf-pct-adr-top': 0.89013,
                'xbtc-base-rmi': 63,
                'xbtc-inf-rmi': 10,
                'xtra-base-fiat-rmi': 42,
                'xtra-base-stake-rmi': 36,
                'xtra-inf-stake-rmi': 15
            },
            'dynamic_roi': {
                'enabled': True,
                'profit-factor': 372,
                'rmi-start': 40,
                'rmi-end': 52,
                'grow-delay': 23,
                'grow-time': 652,
                'fallback': 'roc',
                'min-roc-atr': 0.005
            },
            'custom_stop': {
                'decay-time': 1316,
                'decay-delay': 103,
                'decay-start': -0.28,
                'decay-end': -0.004,
                'cur-min-diff': 0.022,
                'cur-threshold': -0.035,
                'roc-bail': -0.045,
                'rmi-trend': 35,
                'bail-how': 'immediate',
                'pos-trail': False,
                'pos-threshold': 0.005,
                'pos-trail-dist': 0.015
            }
        },
        {
            'pairs': ('ZEC/USD', 'REP/USD', 'GRT/USD', 'UNI/USD', 'FLOW/USD', 'DASH/USD', 'KEEP/USD', 'AAVE/USD', 'BCH/USD', 'ALGO/USD', 'KAVA/USD', 'BAT/USD', 'KSM/USD', 'ETH/USD', 'SC/USD', 'EOS/USD', 'XMR/USD', 'LTC/USD', 'NANO/USD', 'GNO/USD'),
            'buy_params': {
                'base-mp': 63,
                'base-rmi-fast': 52,
                'base-rmi-slow': 22,
                'inf-guard': 'lower',
                'inf-pct-adr-bot': 0.05986,
                'inf-pct-adr-top': 0.76319,
                'xbtc-base-rmi': 68,
                'xbtc-inf-rmi': 17,
                'xtra-base-fiat-rmi': 51,
                'xtra-base-stake-rmi': 32,
                'xtra-inf-stake-rmi': 16
            },
            'dynamic_roi': {
                'enabled': True,
                'profit-factor': 204,
                'rmi-start': 14,
                'rmi-end': 75,
                'grow-delay': 85,
                'grow-time': 630,
                'fallback': 'roc',
                'min-roc-atr': 0.005
            },
            'custom_stop': {
                'decay-time': 934,
                'decay-delay': 47,
                'decay-start': -0.22,
                'decay-end': -0.039,
                'cur-min-diff': 0.0336,
                'cur-threshold': -0.003,
                'roc-bail': -0.01,
                'rmi-trend': 57,
                'bail-how': 'immediate',
                'pos-trail': False,
                'pos-threshold': 0.005,
                'pos-trail-dist': 0.015
            }
        },
        {
            'pairs': ('EWT/USD', 'ADA/USD', 'ATOM/USD', 'XTZ/USD', 'LINK/USD', 'XLM/USD', 'TRX/USD', 'DOT/USD', 'OMG/USD', 'ICX/USD', 'OCEAN/USD', 'COMP/USD', 'STORJ/USD', 'SNX/USD'),
            'buy_params': {
                'base-mp': 64,
                'base-rmi-fast': 48,
                'base-rmi-slow': 20,
                'inf-guard': 'upper',
                'inf-pct-adr-bot': 0.10287,
                'inf-pct-adr-top': 0.83482,
                'xbtc-base-rmi': 70,
                'xbtc-inf-rmi': 17,
                'xtra-base-fiat-rmi': 47,
                'xtra-base-stake-rmi': 18,
                'xtra-inf-stake-rmi': 17
            },
            'dynamic_roi': {
                'enabled': True,
                'profit-factor': 272,
                'rmi-start': 44,
                'rmi-end': 84,
                'grow-delay': 116,
                'grow-time': 877,
                'fallback': 'roc',
                'min-roc-atr': 0.005
            },
            'custom_stop': {
                'decay-time': 1081,
                'decay-delay': 27,
                'decay-start': -0.26,
                'decay-end': -0.035,
                'cur-min-diff': 0.016,
                'cur-threshold': -0.0358,
                'roc-bail': -0.032,
                'rmi-trend': 52,
                'bail-how': 'atr',
                'pos-trail': False,
                'pos-threshold': 0.005,
                'pos-trail-dist': 0.015
            }
        }
    ]

"""
BTC - 
    Global:
    15m / 4h
    20210301- / Sharpe / DROI / CSTP
    594/1000:   1254 trades. 1023/1/230 Wins/Draws/Losses. Avg profit   0.43%. Median profit   1.00%. Total profit  0.00605866 BTC ( 537.60Σ%). Avg duration 168.1 min. Objective: -338.21036

    Pairlists (Sorted/grouped by avg. profit in global backtests):

"""
# Sub-strategy with parameters specific to BTC stake
class Solipsis_BTC(Solipsis):

    timeframe = '15m'
    inf_timeframe = '4h'

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
    15m / 4h
    20210301- / Sharpe / DROI / CSTP
    

    Pairlists (Sorted/grouped by avg. profit in global backtests):

"""
# Sub-strategy with parameters specific to ETH stake
class Solipsis_ETH(Solipsis):

    timeframe = '15m'
    inf_timeframe = '4h'



    stoploss = custom_stop['decay-start']
    use_custom_stoploss = True

    custom_pair_params = []