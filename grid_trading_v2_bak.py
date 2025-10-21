import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_crypto_data(data_path, symbol, start_time, end_time):
    """
    Load cryptocurrency data from CSV files organized by date and data type.

    Args:
        data_path: Path to the directory containing the raw data
        symbol: Trading symbol (e.g., "BTCUSDT")
        start_time: Start datetime string
        end_time: End datetime string

    Returns:
        pandas.DataFrame with OHLCV data and Open Time column
    """
    min_range = pd.date_range(start=start_time, end=end_time, freq="min")
    date_range = pd.date_range(start=start_time, end=end_time, freq="D")
    all_df = []

    for date in date_range:
        daily_data = {"Open": [], "High": [], "Low": [], "Close": [], "Volume": []}

        for data_type in ["open", "high", "low", "close", "volume"]:
            folder_str = Path(
                data_type, "1m", date.strftime("%Y/%m/%d"), f"{symbol}.csv"
            )
            with open(data_path / folder_str, newline="") as csvfile:
                reader = csv.reader(csvfile)
                rows = []
                for row in reader:
                    if not row:
                        continue
                    try:
                        rows.append(float(row[0]))
                    except ValueError:
                        continue
                daily_data[data_type.capitalize()] = rows

        if len(daily_data["Open"]) != 1440:
            print(
                f"Data missing for {date.strftime('%Y-%m-%d')}, {len(daily_data['Open'])} rows found."
            )

        for i in range(len(daily_data["Open"])):
            all_df.append(
                {
                    "Open": daily_data["Open"][i],
                    "High": daily_data["High"][i],
                    "Low": daily_data["Low"][i],
                    "Close": daily_data["Close"][i],
                    "Volume": daily_data["Volume"][i],
                }
            )

    print("Sample data:", all_df[0:10])

    df = pd.DataFrame(all_df)
    df["Open Time"] = min_range[: len(df)]

    return df


symbol = "BTCUSDT_spot"
data_path = Path("/home/delus/Documents/code/playground/crypto_visualization/raw_data")
start_time = "2025-01-01 00:00:00"
end_time = "2025-10-19 23:59:59"

# Load data using the function
df = load_crypto_data(data_path, "BTCUSDT", start_time, end_time)
print(df.head(5))
df = df[(df["Open Time"] >= start_time) & (df["Open Time"] <= end_time)]

lower_bound = 10000
upper_bound = 80000
fee_pct = 0.0008

results = []

grid_sizes = [round(x, 3) for x in list(np.arange(0.003, 0.101, 0.001))]


Grid_Sizes = []
Profit = []

for grid_size in grid_sizes:
    print(grid_size)
    grid_levels = []
    level = lower_bound
    while level <= upper_bound:
        grid_levels.append(level)
        level *= 1 + grid_size
    grid_levels.append(level)

    trade_count = 0

    first_close = df.iloc[0]["Open"]
    current_level = 0
    for i in range(len(grid_levels) - 1):
        if grid_levels[i] <= first_close < grid_levels[i + 1]:
            current_level = i
            break

    row_count = 0
    for index, row in df.iterrows():
        prices = [row["Open"], row["Low"], row["High"], row["Close"]]  # Capitalize
        if row_count != 0:
            prices[0] = prev

        for i in range(len(prices) - 1):
            start_price = prices[i]
            end_price = prices[i + 1]
            if start_price < end_price:
                while True:
                    if start_price <= grid_levels[current_level + 1] < end_price:
                        print(grid_levels[current_level + i])
                        current_level += 1
                        trade_count += 1
                    else:
                        break
            else:
                while True:
                    if end_price <= grid_levels[current_level - 1] < start_price:
                        print(grid_levels[current_level - 1])
                        current_level -= 1
                        trade_count += 1
                    else:
                        break

        prev = row["Close"]
        row_count += 1

    fake_count = 0
    open_price = df.iloc[0]["Open"]
    close_price = df.iloc[len(df) - 1]["Close"]
    if open_price < close_price:
        for level in grid_levels:
            if open_price <= level < close_price:
                fake_count += 1
    else:
        for level in grid_levels:
            if close_price <= level < open_price:
                fake_count += 1

    key_num = (trade_count - fake_count) / 2
    profit_per_grid = (
        100 / (len(grid_levels) - 1) * (grid_size - fee_pct * 2) * key_num
    )  # -1

    Grid_Sizes.append(grid_size)
    Profit.append(profit_per_grid)
    results.append(
        {
            "grid_size": grid_size,
            "grid_count": len(grid_levels) - 1,
            "trade_count": trade_count,
            "fake_count": fake_count,
            "arbitrage number": key_num,
            "profit from arbitrage": profit_per_grid,
        }
    )

results_df = pd.DataFrame(results)
results_df.to_csv(f"{symbol}_exp1.csv", index=False)
plt.plot(Grid_Sizes, Profit, "red")
plt.xlabel("Grid Size")
plt.ylabel("Profit per Grid")
plt.title("Grid & Profit")
plt.show()
