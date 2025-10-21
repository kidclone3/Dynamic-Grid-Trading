import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import jit


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


@jit(nopython=True)
def create_grid_levels(lower_bound, upper_bound, grid_size):
    """Create exponentially-spaced grid levels."""
    levels = [lower_bound]
    level = lower_bound
    while level <= upper_bound:
        level *= 1 + grid_size
        levels.append(level)
    return np.array(levels)


@jit(nopython=True)
def find_initial_level(grid_levels, first_price):
    """Find the initial grid level index for the starting price."""
    for i in range(len(grid_levels) - 1):
        if grid_levels[i] <= first_price < grid_levels[i + 1]:
            return i
    return 0


@jit(nopython=True)
def simulate_grid_trades(ohlc_data, grid_levels, initial_level):
    """
    Simulate grid trading on OHLC data.

    Args:
        ohlc_data: numpy array with shape (n, 4) containing [Open, Low, High, Close]
        grid_levels: numpy array of grid price levels
        initial_level: starting grid level index

    Returns:
        trade_count: total number of trades executed
    """
    trade_count = 0
    current_level = initial_level
    n_rows = ohlc_data.shape[0]

    prev_close = ohlc_data[0, 0]  # First open

    for row_idx in range(n_rows):
        # Build price path: Open -> Low -> High -> Close
        if row_idx == 0:
            prices = ohlc_data[row_idx]
        else:
            prices = np.array(
                [
                    prev_close,
                    ohlc_data[row_idx, 1],
                    ohlc_data[row_idx, 2],
                    ohlc_data[row_idx, 3],
                ]
            )

        # Process each price transition
        for i in range(3):
            start_price = prices[i]
            end_price = prices[i + 1]

            if start_price < end_price:
                # Price going up - check for sells
                while current_level < len(grid_levels) - 2:
                    if start_price <= grid_levels[current_level + 1] < end_price:
                        current_level += 1
                        trade_count += 1
                    else:
                        break
            else:
                # Price going down - check for buys
                while current_level > 0:
                    if end_price <= grid_levels[current_level - 1] < start_price:
                        current_level -= 1
                        trade_count += 1
                    else:
                        break

        prev_close = ohlc_data[row_idx, 3]

    return trade_count


@jit(nopython=True)
def calculate_fake_count(grid_levels, open_price, close_price):
    """Calculate non-profitable one-way trades."""
    fake_count = 0

    if open_price < close_price:
        for level in grid_levels:
            if open_price <= level < close_price:
                fake_count += 1
    else:
        for level in grid_levels:
            if close_price <= level < open_price:
                fake_count += 1

    return fake_count


def backtest_grid_strategy(df, lower_bound, upper_bound, fee_pct, grid_sizes):
    """
    Backtest grid trading strategy across multiple grid sizes.

    Args:
        df: DataFrame with OHLC data
        lower_bound: Lower price bound for grid
        upper_bound: Upper price bound for grid
        fee_pct: Trading fee percentage
        grid_sizes: List of grid sizes to test

    Returns:
        results_df: DataFrame with backtest results
    """
    # Convert OHLC data to numpy array for faster processing
    ohlc_data = df[["Open", "Low", "High", "Close"]].values
    first_open = ohlc_data[0, 0]
    last_close = ohlc_data[-1, 3]

    results = []

    for grid_size in grid_sizes:
        print(f"Testing grid size: {grid_size:.3f}")

        # Create grid levels
        grid_levels = create_grid_levels(lower_bound, upper_bound, grid_size)

        # Find initial level
        initial_level = find_initial_level(grid_levels, first_open)

        # Simulate trades
        trade_count = simulate_grid_trades(ohlc_data, grid_levels, initial_level)

        # Calculate fake count
        fake_count = calculate_fake_count(grid_levels, first_open, last_close)

        # Calculate profit metrics
        arbitrage_count = (trade_count - fake_count) / 2
        profit_per_grid = (
            100 / (len(grid_levels) - 1) * (grid_size - fee_pct * 2) * arbitrage_count
        )

        results.append(
            {
                "grid_size": grid_size,
                "grid_count": len(grid_levels) - 1,
                "trade_count": trade_count,
                "fake_count": fake_count,
                "arbitrage_number": arbitrage_count,
                "profit_from_arbitrage": profit_per_grid,
            }
        )

    return pd.DataFrame(results)


def plot_results(results_df, symbol):
    """Plot grid size vs profit."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        results_df["grid_size"],
        results_df["profit_from_arbitrage"],
        "red",
        linewidth=2,
    )
    plt.xlabel("Grid Size", fontsize=12)
    plt.ylabel("Profit from Arbitrage", fontsize=12)
    plt.title(f"{symbol} - Grid Size vs Profit", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{symbol}_traditional_grid_strategy.png")


if __name__ == "__main__":
    # Configuration
    symbol = "BTCUSDT_spot"
    data_path = Path(
        "/home/delus/Documents/code/playground/crypto_visualization/raw_data"
    )
    start_time = "2025-01-01 00:00:00"
    end_time = "2025-10-19 23:59:59"

    lower_bound = 10000
    upper_bound = 150000
    fee_pct = 0.0008

    # Load data
    print("Loading data...")
    df = load_crypto_data(data_path, "BTCUSDT", start_time, end_time)
    df = df[(df["Open Time"] >= start_time) & (df["Open Time"] <= end_time)]
    print(f"Loaded {len(df)} rows")

    # Generate grid sizes to test
    grid_sizes = np.arange(0.003, 0.101, 0.001).round(3)

    # Run backtest
    print("\nRunning backtest...")
    results_df = backtest_grid_strategy(
        df, lower_bound, upper_bound, fee_pct, grid_sizes
    )

    # Save results
    output_file = f"{symbol}_exp2.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Display summary
    print("\nTop 5 performing grid sizes:")
    print(
        results_df.nlargest(5, "profit_from_arbitrage")[
            ["grid_size", "grid_count", "trade_count", "profit_from_arbitrage"]
        ]
    )

    # Plot results
    plot_results(results_df, symbol)
