# Freqtrade Strategies

Strategies for [Freqtrade](https://freqtrade.io), developed primarily in a partnership between @werkkrew and @JimmyNixx from the Freqtrade Discord.

Use these strategies at your own risk. We are not responsible for anything.

## Repostitory Structure

The root of the `strategies/` and `hyperopts/` folders contains my current version of the strategy and is likely what I run live. Anything in the `old/` directory has been deprecated and I no longer run/maintain these strategies in any way. These are here for posterity and may still be useful/valuable/profitable, or contain useful examples for inspiration.

## About Solipsis

Solipsis is the current strategy I run. It is an evolution of the `Schism` series found in the `old/` folder. The basic idea was to override the Freqtrade base ROI functionality to stimulate a sticking buy signal to exploit the use of `ignore_roi_if_buy_signal` and stay in trades that are in an upward trend past where the static ROI table would have sold. This evolved into several other things using active trade data.

The primary difference between `Schism` and `Solipsis` is that Schism used active trade data within the confines of the buy and sell signals to accomplish this, which did not work in any way with backtesting or hyperoptimization. The fundamental idea worked but was only possible to validate and tune in a live or dry-run which was very time consuming. Solipsis attempts to solve this problem by making the ROI table truly dynamic within the `minimal_roi` methods, and does the dynamic sell bailout work within the scope of the `custom_stoploss`.  This makes it possible to fully backtest and hyperopt the ideas, but does present some limitations (such as using data about other trades) that were used in `Schism`.

### TODO

- Continue to hunt for a better all around buy signal.
- Tweak ROI Trend Ride
  - Adjust pullback to be more dynamic, seems to get out a tad bit early in many cases.
  - Consider a way to identify very large/fast spikes when RMI has not yet reacted to stay in past ROI point.
- Further enchance and optimize custom stop loss
  - Continue to evaluate good circumstances to bail and sell vs hold on for recovery
  - Curent implementation seems to work pretty well but feel like there is room for improvement.
- Develop a PR to fully support hyperopting the custom_stoploss and dynamic_roi spaces?
- Make a container for my hyperopt fork available

### Features

- **Dynamic ROI**
  - Several options, initial idea was to ride trends past ROI in a similar way to trailing stoploss but using indicators.
  - Fallback choices includes table, roc, atr, and others.  Has the ability to set ROI table values dynamically based on indicator math.
- **Custom Stoploss**
  - Generally a vanilla implementation of Freqtrade custom stoploss but tries to do some clever things.  Uses indicator data. (Thanks @JoeSchr!)
- Dynamic informative indicators based on certain stake currences and whitelist contents.
  - If BTC/STAKE is not in whitelist, make sure to use that for an informative.
  - If your stake is BTC or ETH, use COIN/FIAT and BTC/FIAT as informatives.
- Ability to provide custom parameters on a per-pair or group of pairs basis, this includes buy/sell/minimal_roi/dynamic_roi/custom_stop settings, if one desired.
- Custom indicator file to keep primary strategy clean(ish).
  - Most (but not all) of what is in there is taken from freqtrade/technical with some slight modification, removes dependenacy on that import and allows for some customization without having to edit those files directly.
- Child strategy for stake specific settings and different settings for different instances, hoping to keep this strategy file relatively clutter-free from the extensive options especially when using per-pair settings.

### Notes and Recommendations

- For whatever reason, hyperopt will not run from the child strategy, so point at SolipsisX (where X is the current version) directly in your hyperopt command.
- If trading on a stablecoin or fiat stake (such as USD, EUR, USDT, etc.) is *highly recommended* that you remove BTC/STAKE from your whitelist as this strategy performs much better on alts when using BTC as an informative but does not buy any BTC itself.
- It is recommended to configure protections *if/as* you will use them in live and run *some* hyperopt/backtest with "--enable-protections" as this strategy will hit a lot of stoplosses so the stoploss protection is helpful to test. *However* - this option makes hyperopt very slow, so run your initial backtest/hyperopts without this option. Once you settle on a baseline set of options, do some final optimizations with protections on.
- It is *not* recommended to use freqtrades built-in trailing stop, nor to hyperopt for that.
- It is *highly* recommended to hyperopt this with '--spaces buy' only and at least 1000 total epochs several times. There are a lot of variables being hyperopted and it may take a lot of epochs to find the right settings.
- It is possible to hyperopt the custom stoploss and dynamic ROI settings, however a change to the freqtrade code is needed. I have done this in a fork on github and I use it personally, but this code will likely never get merged upstream so use with extreme caution. (https://github.com/werkkrew/freqtrade/tree/hyperopt)
- Hyperopt Notes:
  - Hyperopting buy/custom-stoploss/dynamic-roi together takes a LOT of repeat 1000 epoch runs to get optimal results.  There are a ton of variables moving around and often times the reported best epoch is not desirable.
  - Avoid hyperopt results with small avg. profit and avg. duration of < 60m (in my opinion.)
  - I find the best results come from SharpeHyperOptLoss
  - I personally re-run it until I find epochs with at least 0.5% avg profit and a 10:1 w/l ratio as my personal preference.
- It is *recommended* to leave this file untouched and do your configuration / optimizations from the child strategy Solipsis.py.

**Example of custom_pair_params format:**

```python
    custom_pair_params = [
        {
            'pairs': ('ABC/XYZ', 'DEF/XYZ'),
            'buy_params': {},
            'sell_params': {},
            'minimal_roi': {},
            'dynamic_roi': {},
            'custom_stop': {}
        }
    ]
```

## Changelog

**Note:** Comprehensive changelog has only been kept since v3, April 03, 2021 release.  Prior change details are summary only.

v1:
```
- Initial re-write of Schism v2.
- Moved sticking buy into ROI method
- Moved sell signal ideas into custom_stoploss
- Removed majority of 'populate_trades'
- Moved extranneous code and indicators out to custom_indicators helper file
- Added / Removed several indicators looking for a better buy signal, ultimately ended up keeping the primary Schism buy as it was.
```

v2:
```
- Fixed a significant bug in the custom_stoploss making part of it behave like a positive trail when it shouldn't have.
    - Added some other "modes"
- Misc. de-clutter of some things
- Adjusted some hyperopt ranges a bit
- Removed PMAX indicator. I found in almost every hyperopt it was coming out disabled. It did not seem to add enough value to warrant the performance cost.
- Removed Fibonacci Retracements, didn't seem to really make a difference.
- Removed the informative RSI indicator
- Removed the "uptrend" RMI stuff as well as the hyperopt to choose which is better (down trend was always better)
- Started some preliminary work to add BTC/STAKE on base and informative timeframes as an informative pair if it is not in whitelist.
    - Based on conversation with cyberd3vil it might be smart to blacklist BTC/USDT from trading and use it exclusively as an informative
        for trading on alts.
    - Added logic such that if BTC/STAKE is not in the whitelist, strategy automatically adds a couple additional indicators and conditions.

```

v3:
```
- Made dynamic_roi and custom_stoploss fully compliant with backtesting and hyperopt and validated results.
- Numerous bugfixes from previous versions

April 03, 2021:
- Moved header documentation into readme
- Fixed bug in 'get_pair_params' for minimal_roi per_pair settings

```





![Where Lambo?](misc/wherelambo.jpg)
