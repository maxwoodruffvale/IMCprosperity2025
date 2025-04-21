import math
from statistics import NormalDist

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.api as sm
import numpy as np

# for the submit data
# files = ["./data/evaluation.csv"]


# for the data bottle (vouchers 9500, 9750, 10000)
files = [
         # "./data/new_round-2-island-data-bottle 1/prices_round_2_day_-1.csv",
         # "./data/round-3-island-data-bottle 1/prices_round_3_day_0.csv",
         # "./data/round-4-island-data-bottle/prices_round_4_day_1.csv",
         "./data/round-5-island-data-bottle/prices_round_5_day_2.csv",
         "./data/round-5-island-data-bottle/prices_round_5_day_3.csv",
         "./data/round-5-island-data-bottle/prices_round_5_day_4.csv",
         ]

# for the data bottle (vouchers 10250, 10500)
# other_files = ["./data/round-3-island-data-bottle 2/prices_round_3_day_0.csv",
#          "./data/round-3-island-data-bottle 2/prices_round_3_day_1.csv",
#          "./data/round-3-island-data-bottle 2/prices_round_3_day_2.csv",
#          ]

dfs = []

for file in files:
    tmp_df = pd.read_csv(file)
    dfs.append(tmp_df)

df = pd.concat(dfs, ignore_index=True)
df["global_timestamp"] = (df["day"] - 2) * 1_000_000 + df["timestamp"]

df.sort_values("global_timestamp", inplace=True)

df.drop(columns=["mid_price", "profit_and_loss", "timestamp"], inplace=True, errors="ignore")

bid_cols = [col for col in df.columns if col.startswith("bid_price_")]
ask_cols = [col for col in df.columns if col.startswith("ask_price_")]

bid_vol_cols = [col for col in df.columns if col.startswith("bid_volume_")]
ask_vol_cols = [col for col in df.columns if col.startswith("ask_volume_")]

def get_lowest_bid(row):
    bids = [row[b] for b in bid_cols if pd.notnull(row[b])]
    if not bids:
        return None
    return min(bids)

def get_highest_ask(row):
    asks = [row[a] for a in ask_cols if pd.notnull(row[a])]
    if not asks:
        return None
    return max(asks)

df["lowest_bid"] = df.apply(get_lowest_bid, axis=1)
df["highest_ask"] = df.apply(get_highest_ask, axis=1)
df["fair_value"] = (df["lowest_bid"] + df["highest_ask"]) / 2.0

df_test = df[df["product"] == "PICNIC_BASKET1"]
df_test.reset_index(drop=True, inplace = True)
df_test["returns"] = df_test["fair_value"].pct_change()

returns = df_test.drop(columns = [col for col in df_test.columns if col not in ["returns", "global_timestamp"]], errors = "ignore")

trade_files = [
    "./data/round-5-island-data-bottle/trades_round_5_day_2.csv",
    "./data/round-5-island-data-bottle/trades_round_5_day_3.csv",
    "./data/round-5-island-data-bottle/trades_round_5_day_4.csv",
]

dfs = []

day = 0
for file in trade_files:
    temp_df = pd.read_csv(file)
    temp_df["global_timestamp"] = day * 1_000_000 + temp_df["timestamp"]
    dfs.append(temp_df)
    day += 1

trade_df = pd.concat(dfs, ignore_index=True)
trade_df.sort_values("global_timestamp", inplace=True)

trade_df.drop(columns=["currency", "timestamp"], inplace=True, errors="ignore")

def visualize_trader_pair_results(df, results, trader1, trader2, returns_col):
    """
    Visualize regression results for a specific trader pair.

    Parameters:
        df (pd.DataFrame): Merged DataFrame with trader volume and returns
        results (dict): Regression results
        trader1 (str): First trader's name
        trader2 (str): Second trader's name
        returns_col (str): Name of the returns column
    """
    # 1. Time series plot of volumes and returns
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot volumes
    ax1.plot(df['global_timestamp'], df['buy_volume'], label=f'{trader1} buys from {trader2}', color='green')
    ax1.plot(df['global_timestamp'], -df['sell_volume'], label=f'{trader1} sells to {trader2}', color='red')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'Trading Volume Between {trader1} and {trader2}')
    ax1.set_ylabel('Volume')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot returns
    ax2.plot(df['global_timestamp'], df[returns_col], label='Returns', color='blue')
    ax2.set_title('Product Returns')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Returns')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{trader1}_{trader2}_volumes_returns.png')

    # 2. Coefficients by lag period
    periods = list(results.keys())
    buy_coefs = [results[p]['buy_model'].params['buy_volume'] for p in periods]
    sell_coefs = [results[p]['sell_model'].params['sell_volume'] for p in periods]
    net_coefs = [results[p]['net_model'].params['net_volume'] for p in periods]

    plt.figure(figsize=(10, 6))
    plt.plot(periods, buy_coefs, 'o-', label=f'{trader1} buys from {trader2}', color='green')
    plt.plot(periods, sell_coefs, 'o-', label=f'{trader1} sells to {trader2}', color='red')
    plt.plot(periods, net_coefs, 'o-', label='Net volume', color='purple')
    plt.title(f'Regression Coefficients by Future Period ({trader1}-{trader2})')
    plt.xlabel('Future Period')
    plt.ylabel('Coefficient')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{trader1}_{trader2}_coefficients.png')

    # 3. P-values by lag period
    buy_pvals = [results[p]['buy_model'].pvalues['buy_volume'] for p in periods]
    sell_pvals = [results[p]['sell_model'].pvalues['sell_volume'] for p in periods]
    net_pvals = [results[p]['net_model'].pvalues['net_volume'] for p in periods]

    plt.figure(figsize=(10, 6))
    plt.semilogy(periods, buy_pvals, 'o-', label=f'{trader1} buys from {trader2}', color='green')
    plt.semilogy(periods, sell_pvals, 'o-', label=f'{trader1} sells to {trader2}', color='red')
    plt.semilogy(periods, net_pvals, 'o-', label='Net volume', color='purple')
    plt.axhline(y=0.05, color='black', linestyle='--', label='p=0.05', alpha=0.7)
    plt.title(f'P-values by Future Period ({trader1}-{trader2})')
    plt.xlabel('Future Period')
    plt.ylabel('P-value (log scale)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{trader1}_{trader2}_pvalues.png')

    # 4. Summary statistics table for best model
    best_period = max(periods, key=lambda p: max(
        results[p]['buy_model'].rsquared,
        results[p]['sell_model'].rsquared,
        results[p]['net_model'].rsquared
    ))

    best_results = results[best_period]
    print(f"\nBest model results for period {best_period}:")
    print("\nBuy Volume Model:")
    print(best_results['buy_model'].summary().tables[1])
    print("\nSell Volume Model:")
    print(best_results['sell_model'].summary().tables[1])
    print("\nNet Volume Model:")
    print(best_results['net_model'].summary().tables[1])
    print("\nBoth Volumes Model:")
    print(best_results['both_model'].summary().tables[1])

def analyze_specific_trader_pair(trade_df, returns_df, trader1, trader2, future_periods=5):
    # 1. Extract trades between the specified traders
    # When trader1 buys from trader2
    buy_trades = trade_df[(trade_df['buyer'] == trader1) & (trade_df['seller'] == trader2)]
    buy_volume = buy_trades.groupby('global_timestamp')['quantity'].sum().reset_index()
    buy_volume.columns = ['global_timestamp', 'buy_volume']

    # When trader1 sells to trader2
    sell_trades = trade_df[(trade_df['buyer'] == trader2) & (trade_df['seller'] == trader1)]
    sell_volume = sell_trades.groupby('global_timestamp')['quantity'].sum().reset_index()
    sell_volume.columns = ['global_timestamp', 'sell_volume']

    # 2. Merge with returns data
    # Create a base DataFrame with all timestamps
    timestamps = sorted(trade_df['global_timestamp'].unique())
    base_df = pd.DataFrame({'global_timestamp': timestamps})

    # Merge buy volume
    merged_df = pd.merge(base_df, buy_volume, on='global_timestamp', how='left')
    # Merge sell volume
    merged_df = pd.merge(merged_df, sell_volume, on='global_timestamp', how='left')
    # Merge returns
    merged_df = pd.merge(merged_df, returns_df, on='global_timestamp', how='left')

    # Fill NaN values with 0 (no trades)
    merged_df['buy_volume'] = merged_df['buy_volume'].fillna(0)
    merged_df['sell_volume'] = merged_df['sell_volume'].fillna(0)

    # Create net volume (buy - sell)
    merged_df['net_volume'] = merged_df['buy_volume'] - merged_df['sell_volume']

    # Get the returns column name
    returns_col = \
    [col for col in merged_df.columns if col not in ['global_timestamp', 'buy_volume', 'sell_volume', 'net_volume']][0]

    # 3. Create future returns columns
    for i in range(1, future_periods + 1):
        merged_df[f'{returns_col}_future_{i}'] = merged_df[returns_col].shift(-i)

    # Drop rows with NaN (end of the series)
    merged_df = merged_df.dropna()

    # 4. Run regressions for each future period
    results = {}

    for i in range(1, future_periods + 1):
        future_return = f'{returns_col}_future_{i}'

        # Regression with buy volume
        X_buy = sm.add_constant(merged_df['buy_volume'])
        model_buy = sm.OLS(merged_df[future_return], X_buy).fit()

        # Regression with sell volume
        X_sell = sm.add_constant(merged_df['sell_volume'])
        model_sell = sm.OLS(merged_df[future_return], X_sell).fit()

        # Regression with net volume
        X_net = sm.add_constant(merged_df['net_volume'])
        model_net = sm.OLS(merged_df[future_return], X_net).fit()

        # Regression with both buy and sell volume
        X_both = sm.add_constant(merged_df[['buy_volume', 'sell_volume']])
        model_both = sm.OLS(merged_df[future_return], X_both).fit()

        results[i] = {
            'period': i,
            'buy_model': model_buy,
            'sell_model': model_sell,
            'net_model': model_net,
            'both_model': model_both
        }

    # 5. Visualize results
    visualize_trader_pair_results(merged_df, results, trader1, trader2, returns_col)

    return merged_df, results

trader1 = "Camilla"
trader2 = "Caesar"

df, results = analyze_specific_trader_pair(trade_df, returns, trader1, trader2, future_periods=10)


# df_picnic1 = df[df["product"] == "PICNIC_BASKET1"].copy()
# df_picnic1.reset_index(drop=True, inplace=True)
# picnic1_ts = df_picnic1["fair_value"].astype(float)
# #
# df_picnic2 = df[df["product"] == "PICNIC_BASKET2"].copy()
# df_picnic2.reset_index(drop=True, inplace=True)
# picnic2_ts = df_picnic2["fair_value"].astype(float)
# #
# df_croissants = df[df["product"] == "CROISSANTS"].copy()
# df_croissants.reset_index(drop=True, inplace=True)
# croissants_ts = df_croissants["fair_value"].astype(float)
#
# df_jams = df[df["product"] == "JAMS"].copy()
# df_jams.reset_index(drop=True, inplace=True)
# jams_ts = df_jams["fair_value"].astype(float)
#
# df_djembes = df[df["product"] == "DJEMBES"].copy()
# df_djembes.reset_index(drop=True, inplace=True)
# djembes_ts = df_djembes["fair_value"].astype(float)
#
# synthetic1_ts = 6 * croissants_ts + 3 * jams_ts + djembes_ts
# #
# synthetic2_ts = 4 * croissants_ts + 2 * jams_ts
#
# synthetic3_ts = picnic1_ts - (3/2) * picnic2_ts
#
# window_size = 25

# diff1_ts = picnic1_ts - synthetic1_ts
# print(diff1_ts.mean())
# diff1_ts -= diff1_ts.mean()

# diff2_ts = picnic2_ts - synthetic2_ts
# print(diff2_ts.mean())
# diff2_ts -= diff2_ts.mean()

# diff3_ts = djembes_ts - synthetic3_ts
# print(diff3_ts.mean())
# diff3_ts -= diff3_ts.mean()

# std_ts = diff2_ts.rolling(window_size).std()
# z_ts = (diff2_ts) / std_ts

# plt.figure()
# diff3_ts.plot(title=f"Diff", color = "Green")
# plt.axhline(y = 60.7282625)
# z_ts.plot(color = "Red")
# plt.show()

# df_rock = df[df["product"] == "VOLCANIC_ROCK"].copy()
# df_rock.reset_index(drop = True, inplace = True)
# rock_ts = df_rock["fair_value"].astype(float)
#
# df_v9500 = df[df["product"] == "VOLCANIC_ROCK_VOUCHER_9500"].copy()
# df_v9500.reset_index(drop = True, inplace = True)
# v9500_ts = df_v9500["fair_value"].astype(float)
#
# df_v9750 = df[df["product"] == "VOLCANIC_ROCK_VOUCHER_9750"].copy()
# df_v9750.reset_index(drop = True, inplace = True)
# v9750_ts = df_v9750["fair_value"].astype(float)
#
# df_v10000 = df[df["product"] == "VOLCANIC_ROCK_VOUCHER_10000"].copy()
# df_v10000.reset_index(drop = True, inplace = True)
# v10000_ts = df_v10000["fair_value"].astype(float)
#
# df_v10250 = df[df["product"] == "VOLCANIC_ROCK_VOUCHER_10250"].copy()
# df_v10250.reset_index(drop = True, inplace = True)
# v10250_ts = df_v10250["fair_value"].astype(float)
#
# df_v10500 = df[df["product"] == "VOLCANIC_ROCK_VOUCHER_10500"].copy()
# df_v10500.reset_index(drop = True, inplace = True)
# v10500_ts = df_v10500["fair_value"].astype(float)
#
# def compute_tau(row):
#     """
#     final_day: The day when the option expires.
#     e.g. if your data is day=0..2, final_day=3 => means at day=3 the option is worthless (tau=0).
#     """
#     # day + fraction_of_day
#     current_day_frac = row["global_timestamp"] / 1000000
#     days_left = 8 - current_day_frac
#     # Convert to years:
#     tau = days_left / 365.0
#     return tau
#
# df_v9500["tau"] = df_v9500.apply(compute_tau, axis=1)
# df_v9750["tau"] = df_v9750.apply(compute_tau, axis=1)
# df_v10000["tau"] = df_v10000.apply(compute_tau, axis=1)
# df_v10250["tau"] = df_v10250.apply(compute_tau, axis=1)
# df_v10500["tau"] = df_v10500.apply(compute_tau, axis=1)
#
# def bs_call(S, K, T, r, vol):
#     d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
#     d2 = d1 - vol * np.sqrt(T)
#     N = NormalDist().cdf
#     return S * N(d1) - np.exp(-r * T) * K * N(d2)
#
# def bs_vega(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     n = NormalDist().pdf
#     return S * n(d1) * np.sqrt(T)
#
# def find_vol(target_value, S, K, T, r, *args):
#     MAX_ITERATIONS = 200
#     PRECISION = 1.0e-6
#     sigma = 0.3
#     for i in range(0, MAX_ITERATIONS):
#         price = bs_call(S, K, T, r, sigma)
#         vega = bs_vega(S, K, T, r, sigma)
#         diff = target_value - price  # our root
#         if (abs(diff) < PRECISION):
#             return sigma
#         sigma = sigma + diff/vega # f(x) / f'(x)
#     return sigma # value wasn't found, return best guess so far
#
# def compute_implied_vol(row, K):
#     S = row["fair_value_rock"]
#     tau = row["tau"]
#     r = 0.0
#     market_price = row["fair_value_voucher"]
#     iv = find_vol(market_price, S, K, tau, r)
#     return iv
#
# merged9500 = pd.merge(
#     df_v9500,
#     df_rock[["global_timestamp", "fair_value"]],
#     on="global_timestamp",
#     how="inner",
#     suffixes=("_voucher", "_rock")
# )
#
# merged9750 = pd.merge(
#     df_v9750,
#     df_rock[["global_timestamp", "fair_value"]],
#     on="global_timestamp",
#     how="inner",
#     suffixes=("_voucher", "_rock")
# )
#
# merged10000 = pd.merge(
#     df_v10000,
#     df_rock[["global_timestamp", "fair_value"]],
#     on="global_timestamp",
#     how="inner",
#     suffixes=("_voucher", "_rock")
# )
#
# merged10250 = pd.merge(
#     df_v10250,
#     df_rock[["global_timestamp", "fair_value"]],
#     on="global_timestamp",
#     how="inner",
#     suffixes=("_voucher", "_rock")
# )
#
# merged10500 = pd.merge(
#     df_v10500,
#     df_rock[["global_timestamp", "fair_value"]],
#     on="global_timestamp",
#     how="inner",
#     suffixes=("_voucher", "_rock")
# )

# merged9500["implied_vol"] = merged9500.apply(compute_implied_vol, args = (9500,), axis=1)
# merged9500 = merged9500[merged9500["implied_vol"] >= 0.15].copy()
# merged9750["implied_vol"] = merged9750.apply(compute_implied_vol, args = (9750,), axis=1)
# merged9750 = merged9750[merged9750["implied_vol"] >= 0.125].copy()
# merged10000["implied_vol"] = merged10000.apply(compute_implied_vol, args = (10000,), axis=1)
# merged10250["implied_vol"] = merged10250.apply(compute_implied_vol, args = (10250,), axis=1)
# merged10500["implied_vol"] = merged10500.apply(compute_implied_vol, args = (10500,), axis=1)
#
# window = 100
# merged9500["rolling_iv_mean"] = merged9500["implied_vol"].rolling(window).mean()
# merged9500["rolling_iv_std"] = merged9500["implied_vol"].rolling(window).std()
# merged9500["rolling_iv_z"] = (merged9500["implied_vol"] - merged9500["rolling_iv_mean"]) / merged9500["rolling_iv_std"]
# merged9750["rolling_iv_mean"] = merged9750["implied_vol"].rolling(window).mean()
# merged9750["rolling_iv_std"] = merged9750["implied_vol"].rolling(window).std()
# merged9750["rolling_iv_z"] = (merged9750["implied_vol"] - merged9750["rolling_iv_mean"]) / merged9750["rolling_iv_std"]
# merged10000["rolling_iv_mean"] = merged10000["implied_vol"].rolling(window).mean()
# merged10000["rolling_iv_std"] = merged10000["implied_vol"].rolling(window).std()
# merged10000["rolling_iv_z"] = (merged10000["implied_vol"] - merged10000["rolling_iv_mean"]) / merged10000["rolling_iv_std"]
# merged10250["rolling_iv_mean"] = merged10250["implied_vol"].rolling(window).mean()
# merged10250["rolling_iv_std"] = merged10250["implied_vol"].rolling(window).std()
# merged10250["rolling_iv_z"] = (merged10250["implied_vol"] - merged10250["rolling_iv_mean"]) / merged10250["rolling_iv_std"]
# merged10500["rolling_iv_mean"] = merged10500["implied_vol"].rolling(window).mean()
# merged10500["rolling_iv_std"] = merged10500["implied_vol"].rolling(window).std()
# merged10500["rolling_iv_z"] = (merged10500["implied_vol"] - merged10500["rolling_iv_mean"]) / merged10500["rolling_iv_std"]
#
plt.figure(figsize=(8, 6))

# plt.scatter(merged9500["global_timestamp"], merged9500["rolling_iv_mean"], label="mean", alpha=0.3)
# plt.scatter(merged9500["global_timestamp"], merged9500["implied_vol"], label="iv", alpha=0.3)
# plt.scatter(merged9500["global_timestamp"], merged9500["rolling_iv_z"], label="z", alpha=0.3)
# plt.scatter(merged9750["global_timestamp"], merged9750["rolling_iv_mean"], label="mean", alpha=0.3)
# plt.scatter(merged9750["global_timestamp"], merged9750["implied_vol"], label="iv", alpha=0.3)
# plt.scatter(merged9750["global_timestamp"], merged9750["rolling_iv_z"], label="z", alpha=0.3)
# plt.scatter(merged10000["global_timestamp"], merged10000["rolling_iv_mean"], label="mean", alpha=0.3)
# plt.scatter(merged10000["global_timestamp"], merged10000["implied_vol"], label="iv", alpha=0.3)
# plt.scatter(merged10000["global_timestamp"], merged10000["rolling_iv_z"], label="z", alpha=0.3)
# plt.scatter(merged10250["global_timestamp"], merged10250["rolling_iv_mean"], label="mean", alpha=0.3)
# plt.scatter(merged10250["global_timestamp"], merged10250["implied_vol"], label="iv", alpha=0.3)
# plt.scatter(merged10250["global_timestamp"], merged10250["rolling_iv_z"], label="z", alpha=0.3)
# plt.scatter(merged10500["global_timestamp"], merged10500["rolling_iv_mean"], label="mean", alpha=0.3)
# plt.scatter(merged10500["global_timestamp"], merged10500["implied_vol"], label="iv", alpha=0.3)
# plt.scatter(merged10500["global_timestamp"], merged10500["rolling_iv_z"], label="z", alpha=0.3)
#
# plt.title("Volcanic Rock Vouchers: Implied Vol vs. t")
# plt.xlabel("time")
# plt.ylabel("iv")
# plt.legend()
# plt.show()