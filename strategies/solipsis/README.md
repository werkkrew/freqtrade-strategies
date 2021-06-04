# Solipsis

**Solipsis is no longer the strategy I am using and will likely no longer be maintained.**

**THE DEFAULT PARAMETERS ARE INTENTIONALLY BAD, HYPEROPTING THIS STRATEGY IS CONSISTERED REQUIRED.  SEE NOTES BELOW.**

## About Solipsis

Solipsis is an evolution of the `Schism` series found in the `old/` folder. The basic idea was to override the Freqtrade base ROI functionality to stimulate a sticking buy signal to exploit the use of `ignore_roi_if_buy_signal` and stay in trades that are in an upward trend past where the static ROI table would have sold without having to design a really strong sell signal. This evolved into several other things using active trade data.

The primary difference between `Schism` and `Solipsis` is that Schism used active trade data within the confines of the buy and sell signals to accomplish this, which did not work in any way with backtesting or hyperoptimization. The fundamental idea worked but was only possible to validate and tune in a live or dry-run which was very time consuming. Solipsis attempts to solve this problem by making the ROI table truly dynamic within the `minimal_roi` methods, and does the dynamic sell bailout work within the scope of the `custom_stoploss`. This makes it possible to fully backtest and hyperopt the ideas, but does present some limitations (such as using data about other trades) that were used in `Schism` as well as the usual limitations backtesting only being able to use OHLC data presents.

### TODO

- Continue to improve buy signal. Ideas are welcome!
- Continue to optimize and innovate around dynamic ROI idea (now custom_sell!)
- Further enchance and optimize custom stop loss (now part of custom_sell!)
  - Continue to evaluate good circumstances to bail and sell vs hold on for recovery
  - Curent implementation seems to work pretty well but feel like there is room for improvement.
- Consider some factors for how "strong" the buy signal is and consider a way to evaluate informative pairs/timeframes for how careful we should be when entering. For example, if BTC oracle says to be cautious, require a stronger buy signal for entry?

### Features

- **Custom Sell** *(previously Dynamic ROI)*
  - In profit the idea is to override the static ROI table when there is a trend to allow for larger profits.
  - In loss the idea is to behave like a "dynamic bailout" that preempts the stoploss when certain criteria are met.
- **Dynamic informative indicators** based on certain stake currences and whitelist contents.
  - If BTC/STAKE is not in whitelist, make sure to use that for an informative. (This is important for strategy performance)
  - If your stake is BTC or ETH, use COIN/FIAT and BTC/FIAT as informatives.
- **Custom indicator file** to keep primary strategy clean(ish).
- **Child strategy** for stake specific settings and different settings for different instances, hoping to keep this strategy file relatively clutter-free from the extensive options available. I personally use the child strategy to also document the current optimization results the strategy is running off of so I can reference them easily in subsequent optimization runs.

## Overview and Theory

- **Buying**
  - Base pair uses a "downtrend" buy where it expects a bounce from a lower resistance level. This uses several indicators including [RMI](https://www.tradingview.com/script/kwIt9OgQ-Relative-Momentum-Index/), [Momentum Pinball](https://www.tradingview.com/script/fBpVB1ez-Momentum-Pinball-Indicator/), [PCC](https://www.tradingview.com/script/6wwAWXA1-MA-Streak-Change-Channel/), and [MA Streak](https://www.tradingview.com/script/Yq1z7cIv-MA-Streak-Can-Show-When-a-Run-Is-Getting-Long-in-the-Tooth/).
  - Base pair uses an upper guard by calculating the "Average Day Range" and not buying when markets are near daily peaks where we anticipate resistance.
  - Informative pair on BTC/STAKE (e.g. BTC/USD) where we only buy alt-coins when BTC is doing well. Theoretically many alts perform well, or even better when BTC is doing poorly but more often than not the entire market follows the performance of BTC/USD and so we try to only make trades when BTC is looking bullish.
    - Based on my testing when removing BTC/STAKE from the whitelist and using it as an informative provides a significant positive impact on strategy performance. *This feature can be disabled simply by having BTC/STAKE in your whitelist.*
  - When stake is BTC or ETH, lets say we have XLM/BTC in our whitelist and our chosen stablecoin/fiat is USD. In this case XLM/USD and BTC/USD will be used as informative pairs for the XLM/BTC trade. Idea is if XLM/USD is strong, or BTC/USD is weak, then XLM/BTC is likely to go up. Conversely, if BTC/USD is very strong and XLM/USD is weak or neutral, then XLM/BTC is likely to go down. 

- **Custom Sell** *(combined previous Dynamic ROI and Custom Stoploss)*
  - Logic is enclosed in the `custom_sell` method but the goal is to behave like a dynamic ROI table in line with my previous `dynamic_roi` implementation.
  - **In Profit** (formerly Dynamic ROI):
    - Fundamentally, we do not use the standard `minimal_roi` table at all (by setting "0": 100) and instead use the `custom_sell` method combined with defined ROI values and trends to identify when to sell.
    - You can define how the base ROI value works by setting the type to either `static`, `decay` or `step`. This is the effective "minimum_roi" value at any given time which will still be overridden in a trend, producing the "dynamic" nature of the signal.
      - `static` will use the `csell_roi_start` value as the minimum_roi value and never change
      - `decay` will decay in a linear manner between `csell_roi_start` to `csell_roi_end` over `csell_roi_time`.
      - `step` will start at `csell_roi_start` and then become `csell_roi_end` after `csell_roi_time`
    - You can disable the custom_sell by setting `use_custom_sell` to `False` and putting a standard `minimal_roi` table in the config.
    - Again, this will function similar to the previous dynamic_roi, meaning:
      - The `csell_roi_type` is effectively setting the `minimum_roi` just like the table, if there is a trend, we will not sell. If there is not a trend, we will fall back to this value to decide if we sell or not.
      - Similarly, if we are in a trend, but the price pulls back from the max too much (as defined by our variables), we will also sell.
      - Outside of a trend, the static, decaying, or stepped `minimal_roi` value is how a sell is determined.
    - `custom_sell` will produce the following sell reasons to backtesting and telegram (a sell happened because...):
      - `trend_roi` - There was a trend we were in and it has ended, ROI point was met and we sold (no pullback)
      - `notrend_roi` - There was no trend and our desired ROI point was met
      - `intrend_pullback_roi` - We are in a trend but the profit pulled back too far from the peak, min_roi was still met
      - `intrend_pullback_noroi` - We are in a trend but the profit pulled back too far from the peak, min_roi was NOT met
  - **In Loss** (formerly Custom Stoploss):
    - The stoploss is static as defined usually but a sell will happen in a loss, prior to the stoploss, if certain conditions are met. This is able to be hyperopted. 
      - If **profit is below our defined threshold**, and some conditions are met, sell at a loss, for example:
        - Depending on the `bail_how` setting, we will sell out of what is considered a bad trade based on either the `sroc` value, or a `timeout` value. In the case of the timeout it is also possible to override the timeout in the case of a positive trend. The idea there is that if the time has passed, but the trade is on its way back up, don't bail out unless the trend disappears.

## Notes and Recommendations

- The default parameters are bad on purpose. If you backtest this strategy without hyperopting it around the buy/sell spaces, it will look bad.
- If using a standard looking `minimal_roi` dict, instead of the one I have provided, sells **can** and **will** happen as per standard freqtrade ROI functionality and the `custom_sell` will very likely not do anything. This is fine if this is your desired functionality, just be aware of it. Using a standard `minimal_roi` table is **not** recommended.
- If trading on a stablecoin or fiat stake (such as USD, EUR, USDT, etc.) it is *highly recommended* that you remove BTC/STAKE from your whitelist as this strategy performs much better on alts when using BTC as an informative but does not buy any BTC itself.
- It is recommended to configure protections *if/as* you will use them in live and run *some* hyperopt/backtest with "--enable-protections" as this strategy will hit a lot of stoplosses (as we use it like a sell) so the stoploss protection is helpful to test. *However* - this option makes hyperopt very slow, so run your initial backtest/hyperopts without this option. Once you settle on a baseline set of options, do some final optimizations with protections on.
- It is *probably not* recommended to use freqtrades built-in trailing stop, nor to hyperopt for that, although feel free to experiment.
- It is *highly* recommended to hyperopt this with '--spaces buy sell' only and at least 1000 total epochs several times. There are a lot of variables being hyperopted and it may take a lot of epochs to find the right settings.
- Hyperopt Notes:
  - Just food for thought, hyperopting buy and custom-sell together takes a LOT of repeat 1000 epoch runs to get optimal results. There are a ton of variables moving around and often times the **reported best epoch is not always desirable**, so make sure to look through the results and not just assume the "best" one is actually the best.
  - Avoid hyperopt results with very small avg. profit and avg. duration of < 60m (in my opinion.) - This is best done by setting --min-trades to something as to prevent the loss function from going toward very small numbers of trades that inflate the loss function result.
  - The hyperopt loss function you use will have a massive impact on how aggressive the strategy is. I find `Sharpe` to lean towards pretty aggresive buys and `SortinoDaily` to produce the most conservative results. 
  - I personally re-run it until I find epochs with at least 0.5% avg profit and a 10:1 w/l ratio as my personal preference.
- It is *recommended* to leave the base strategy file untouched and do your configuration/paste hyperopt params into the child strategy Solipsis.py.
- You can adjust how aggressive/conservative the buy is by changing/restricting the hyperopt ranges, for example:
  - Setting the minimum `base_ma_streak` value > 1 dramatically reduces the # of trades
  - Changing optimize to False on the `base_trigger` and `xbtc_guard` and using default of `pcc` and `strict` respectively
- You might consider playing with some of the indicator periods defined, especially the `kama`, `mastreak`, and `pcc` as they have a rather dramatic impact.

## Changelog

**Note:** Comprehensive changelog has only been kept since v3, April 03, 2021 release.  Prior change details are summary only.

### v5

May 16, 2021 (version 5.2.1):
- **Custom Sell**
  - Fixed a bug with one of the variables.

May 14, 2021 (version 5.2):
- **General**
  - Slight modification to hyperopt params (added load=True)
  - Added @rk to credits because he is the best!
- **Custom Sell**
  - Issue: https://github.com/freqtrade/freqtrade/issues/4920 has been addressed and code has been updated without workaround, should work as designed once it is merged to develop.
  - Moved the "bailout" and "timeout" (sells for loss) portion of the custom sell back into `custom_stoploss`
    - Changed `sell_profit_only` to true now that the sell for loss is back in `custom_stoploss`
  - Minor tweak to in how `custom_sell` sets `had-trend` back to false.
  - Changed `ignore_roi_if_buy_signal` to true because it probably can't hurt?

May 13, 2021 (version 5.1.4):
- **General**
  - It seems there is an issue with hyperopt when calling it from a child strategy. As such, it is recommended to call the strategy directly during hyperopt (e.g. --strategy Solipsis5), you can still put the found parameters into the child strategy for live/dry run if you prefer and this is still my recommended way to configure the strategy. Especially if running multiple instances.
- **Custom Sell**
  - As per issue: https://github.com/freqtrade/freqtrade/issues/4920 current_profit has not been behaving how I expected. As such I have implemented a (hopefully) temporary workaround to provide more reliable/realistic profit values during backtesting/hyperopting until an upstream solution is implemented.

May 11, 2021 (version 5.1.3):
- **General**
  - Updated `custom_sell` dataframe access to use new method.

May 9, 2021 (version 5.1.2):
- **General**
  - Fixed a bug in how custom_sell was using the dataframe, the current state of how to do this upstream in freqtrade is in flux, I went back to the Solipsis v4 way of getting the dataframe in custom_sell until the devs decide on a consistent method going forward.
  - **Note:** As per the previous note, `custom_sell` behavior feels a bit sus to me at the moment, I would use v5 with extreme caution.
- **Indicators/ROI**
  - Changed `candle` trend type to look for green candles (close>open) rather than close > previous close

May 7, 2021 (version 5.1.1):
- **General**
  - Changed default for `base_trigger` from `optimize=False` to `optimize=True`

May 5, 2021 (version 5.1):
- **Bugfixes**
  - Fixed bug causing hyperopt output to not align correctly with a backtest using the outputted parameters.
  - Removed the buy/sell timeout things as some people reported issues with them and they arent critical.
- **Note**
  - I am not 100% convinced the new `custom_sell` is working as intended. Use with caution and keep checking for updates to both freqtrade and this strategy as I try to work through some things I am finding as odd, as well as general improvements and optimizations.

April 29, 2021 (version 5.0):
- **General Changes**
  - **ATTENTION** The strategy will *appear* to work fine even if you are not on the latest develop branch which includes the `custom_sell` functionality. Unless you see custom sell reasons in the sell table after backtesting, you need to make sure you pull the latest develop code to your installation or pull the latest :develop container!
  - I know v4 wasn't out for very long before this jump but the new `custom_sell` method in current freqtrade develop created an opportunity to change/improve a lot of things.
  - Updated the child strategy such that you can call hyperopt from it now without errors.
  - Removed the "default" `buy_params` and `sell_params` to force users to hyperopt prior to testing. This strategy **requires** hyperopting and there is no one-size-fits-all set of parameters. I wanted to discourage people from downloading this, backtesting with my base set of params and complaining.
    - I also hope this might encourage folks to keep their specific optimized settings in the child-strategy and use that.
  - Changed out `dynamic_roi` and `custom_stoploss` to use the new `custom_sell` feature in the latest freqtrade develop branch, functionally it should be very similar but is more compliant / supported by freqtrade as I am no longer overloading the roi methods. This should be a much better path forward.
    - Consequently, this means you **must** be on the latest freqtrade develop branch on your local installation or container.
  - Added a version tag in the header of the strategy for sub-releases, (e.g. 4.1, 4.2, etc.), major versions will still be held in the file name. This will help me/us know what version you are using as it evolves.
  - Dataframe is now natively accessible in `custom_sell` and `custom_stoploss` so the code used to force that behavior has been removed.
  - The SROC comparission is now done using the percentage value as the indicator uses, rather than dividing by 100.
- **Dynamic ROI**
  - Dynamic ROI as it was, has been removed in favor of the new `custom_sell` method. 
    - See the **custom_sell** area in the "Overview and Theory" section for additional details.
- **Custom Stoploss**
  - Custom Stoploss as it was, has been removed in favor of the new `custom_sell` method.
    - See the **custom_sell** area in the "Overview and Theory" section for addtional details.

### v4

April 18, 2021:
- **Indicator / Signal Changes**
  - Changed all instances of T3 indicator to KAMA. T3 seemed to perform slightly better but requires such a huge amount of history that it causes errors in dry-run/live.

April 16, 2021:
- **General Changes**
  - Updated README quite a bit to hopefully (probably not) cut down on questions.
  - Added a RESULTS file to show some example results of the strategy.
  - Hyperopt file and my hyperopt fork are no longer needed, **latest develop branch of freqtrade required**.
    - Taking advantage of new "HyperStrategy" feature set means no more editing of hyperopt file and a clean way to hyperopt custom_stoploss and dynamic_roi without my forked hacks!
    - custom_stoploss and dynamic_roi are hyperopted when using the 'sell' space, (e.g. include --spaces sell or --spaces buy sell)
  - Cleaned up custom_indicators to remove some things I know I will definitely not be using.
  - **Removed per-pair parameters support.** 
    - Keeping this working with the new changes to hyperopt would have been *some* amount of effort and I am personally not using this feature so I decided to just get rid of it. If you *really* want to use it, you can use the example code from Solipsis v3 to re-create it.
    - After several rounds of extensive testing I found it very difficult to group pairs in any way that gave better results than the global hyperopted settings.
  - Removed helper methods for updating price/trade data to simplify the contents of the file.
- **Indicator / Signal Changes**
  - **Base Pair / Base Timeframe:**
    - Changed the way the buy signal works a fair bit, counting occurrences of RMI downward trend, etc.
    - Added MA Streak + PCC to the buy signal on base pair/base timeframe
    - Added two triggers buy triggers with a hyperopt setting to choose which trigger to use
  - **Base Pair / Informative Timeframe:**
    - Changed upper ADR guard from 3d_low - 1d_high to 1d_low - 1d_high -> significant performance improvement
    - Removed lower ADR guard. 90% of the time hyperopt decided to use upper only, the lower guard did not seem to have a big impact and removing it reduces the variables needed to optimize.
  - **BTC/STAKE Informative Pair:**
    - Removed informative timeframe from BTC/STAKE indicators. Base timeframe alone provides enough signals for our purposes.
    - Adjusted base timeframe RMI period to be much longer (14 -> 55).
    - Added a long T3 (144) to BTC/STAKE to stay out of the market when BTC is very bearish.
      - Can be overridden by base pair being above T3 (233) so that if a particular pair is doing very well despite BTC it can be traded.
    - BTC protection type is chose by hyperopt between `strict` (BTC guard only), `lazy` (BTC guard with base pair override), `none` (disable the BTC informative)
  - **ETH and BTC as Stake Informatives:**
    - Removed informative timeframe from BTC and ETH stake indicators. Base timeframe alone provides enough signals for our purposes.

- **Custom Stoploss Changes**
  - Almost completely stripped the previous custom_stoploss and dramatically simplified it.
  - Got rid of positive trail options, in theory this is completely at odds with/redundant to the dynamic ROI feature set.
  - Removed decay, stoploss is now a static stoploss unless something prompts us to get out earlier.
  - Custom Stoploss is now exclusively a "dynamic bailout" intended to bail-out before the defined static stoploss point based on some factors:
    - If current_profit is below the threshold value, allow a bailout.  Bailout if some combination of the following are true:
      - Rate of Change (ROC) is below the defined value.
      - Amount of allowed time in trade has passed.

- **Dynamic ROI Changes**
  - Completely re-wrote dynamic ROI.  It is now much simpler with far fewer options, highlights:
    - No more options in terms of the fallback, it will now simply use the ROI table, or look for a trend. Dynamic ROI based on indicators seems to have potential but my implementation was not good. May look to re-implement this later with different indicators.
    - Profit-pullback based on RMI and the profit_factor has been removed in favor of a static pullback value, similar to how a trailing stoploss would function.
    - Got rid of RMI threshold growth over time.
    - Added a hyperoptable option to enable/disable the peak pullback feature and if it should strictly respect the ROI table or not when a pullback occurs.
  - Added options for what type of trend to use when holding onto a trade, this is part of the hyperopt for dynamic roi, currently choices are: `rmi`, `ssl`, `candle` or `any`.
    - RMI uses the direction of the RMI indicator (3 out of 5 increasing), SSL uses SSL Channels with ATR, candle simply looks to see if the rolling window of 3 out of the last 5 candles have closed higher than the one before, any will consider it a trend if any of the above conditions are met.

### v3

April 03, 2021:
- Moved header documentation into readme
- Fixed bug in 'get_pair_params' for minimal_roi per_pair settings

Initial Release:
- Made dynamic_roi and custom_stoploss fully compliant with backtesting and hyperopt and validated results.
- Numerous bugfixes from previous versions
- Tons of changes that I failed to document

### v2

- Fixed a significant bug in the custom_stoploss making part of it behave like a positive trail when it shouldn't have.
  - Added some other "modes"
- Misc. de-clutter of some things
- Adjusted some hyperopt ranges a bit
- Removed PMAX indicator. I found in almost every hyperopt it was coming out disabled. It did not seem to add enough value to warrant the performance cost.
- Removed Fibonacci Retracements, didn't seem to really make a difference.
- Removed the informative RSI indicator
- Removed the "uptrend" RMI stuff as well as the hyperopt to choose which is better (down trend was always better)
- Started some preliminary work to add BTC/STAKE on base and informative timeframes as an informative pair if it is not in whitelist.
  - Based on conversation with cyberd3vil it might be smart to blacklist BTC/USDT from trading and use it exclusively as an informative for trading on alts.
  - Added logic such that if BTC/STAKE is not in the whitelist, strategy automatically adds a couple additional indicators and conditions.

### v1

- Initial re-write of Schism v2.
- Moved sticking buy into ROI method
- Moved sell signal ideas into custom_stoploss
- Removed majority of 'populate_trades'
- Moved extranneous code and indicators out to custom_indicators helper file
- Added / Removed several indicators looking for a better buy signal, ultimately ended up keeping the primary Schism buy as it was.