import math
from statistics import NormalDist

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# for the submit data
# files = ["./data/evaluation.csv"]

# for the data bottle (vouchers 9500, 9750, 10000)
files = ["./data/round-3-island-data-bottle 1/prices_round_3_day_0.csv",
         "./data/round-3-island-data-bottle 1/prices_round_3_day_1.csv",
         "./data/round-3-island-data-bottle 1/prices_round_3_day_2.csv",
         ]

# files = ["./old data/2024_data_bottles copy/round-4-island-data-bottle/prices_round_4_day_1.csv",
#          "./old data/2024_data_bottles copy/round-4-island-data-bottle/prices_round_4_day_2.csv",
#          "./old data/2024_data_bottles copy/round-4-island-data-bottle/prices_round_4_day_3.csv",
#          ]

# for the data bottle (vouchers 10250, 10500)
# files = ["./data/round-3-island-data-bottle 2/prices_round_3_day_0.csv",
#          "./data/round-3-island-data-bottle 2/prices_round_3_day_1.csv",
#          "./data/round-3-island-data-bottle 2/prices_round_3_day_2.csv",
#          ]

dfs = []

for file in files:
    tmp_df = pd.read_csv(file)
    dfs.append(tmp_df)

df = pd.concat(dfs, ignore_index=True)
df["global_timestamp"] = (df["day"]) * 1_000_000 + df["timestamp"]
# df["global_timestamp"] = df["timestamp"]

df.sort_values("global_timestamp", inplace=True)

df.drop(columns=["day", "mid_price", "profit_and_loss", "timestamp"], inplace=True, errors="ignore")

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

df_rock = df[df["product"] == "VOLCANIC_ROCK"].copy()
df_rock.reset_index(drop = True, inplace = True)
rock_ts = df_rock["fair_value"].astype(float)

rock_ts = np.array(rock_ts)

df_voucher = df[df["product"] == "VOLCANIC_ROCK_VOUCHER_9500"].copy()
df_voucher.reset_index(drop = True, inplace = True)
voucher_ts = df_voucher["fair_value"].astype(float)

voucher_ts = np.array(voucher_ts)

# 9500: 832.461393116127
# 9750: 583.5033208446866
# 10000: 343.0775653477831
# 10250: 147.71419561585927
# print(np.nanmean(voucher_ts))

window_size = 50

df_rock["log_return"] = np.log(df_rock["fair_value"].shift(-1) / df_rock["fair_value"])

df_rock["realized_vol"] = (
    df_rock["log_return"]
    .rolling(window_size)
    .std()
    * np.sqrt(365) * np.sqrt(10000)
)

def compute_tau(row):
    """
    final_day: The day when the option expires.
    e.g. if your data is day=0..2, final_day=3 => means at day=3 the option is worthless (tau=0).
    """
    # day + fraction_of_day
    current_day_frac = row["global_timestamp"] / 1000000
    days_left = 7 - current_day_frac
    # Convert to years:
    tau = days_left / 365.0
    return tau

df_voucher["tau"] = df_voucher.apply(compute_tau, axis=1)

def bs_call(S, K, T, r, vol):
    d1 = (np.log(S/K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    N = NormalDist().cdf
    return S * N(d1) - np.exp(-r * T) * K * N(d2)

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    n = NormalDist().pdf
    return S * n(d1) * np.sqrt(T)

def find_vol(target_value, S, K, T, r, *args):
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-6
    sigma = 0.3
    for i in range(0, MAX_ITERATIONS):
        price = bs_call(S, K, T, r, sigma)
        vega = bs_vega(S, K, T, r, sigma)
        diff = target_value - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

def compute_implied_vol(row):
    S = row["fair_value_rock"]
    K = 9500
    tau = row["tau"]
    r = 0.0
    market_price = row["fair_value_voucher"]
    iv = find_vol(market_price, S, K, tau, r)
    return iv

merged = pd.merge(
    df_voucher,
    df_rock[["global_timestamp", "fair_value"]],
    on="global_timestamp",
    how="inner",
    suffixes=("_voucher", "_rock")
)

merged["implied_vol"] = merged.apply(compute_implied_vol, axis=1)

print(merged["implied_vol"].mean(), df_rock["realized_vol"].mean())

plt.figure()
plt.plot(merged["global_timestamp"], merged["implied_vol"], marker="o", linestyle="--")
plt.title("Implied Volatility for VOLCANIC_ROCK_VOUCHER_9500")
plt.xlabel("Global Timestamp")
plt.ylabel("Implied Vol (annualized)")
plt.show()