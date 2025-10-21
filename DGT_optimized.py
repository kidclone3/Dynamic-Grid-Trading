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
            try:
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
            except FileNotFoundError:
                print(f"File not found: {data_path / folder_str}")
                continue

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

    df = pd.DataFrame(all_df)
    df["Open Time"] = min_range[: len(df)]

    return df


@jit(nopython=True)
def calculate_grid_levels_centered(start_price, grid_size, grid_numbers_half):
    """Calculate grid levels centered around start price."""
    levels = np.zeros(2 * grid_numbers_half + 1)

    # Lower levels
    for i in range(grid_numbers_half):
        levels[grid_numbers_half - i - 1] = start_price / ((1 + grid_size) ** (i + 1))

    # Center level
    levels[grid_numbers_half] = start_price

    # Upper levels
    for i in range(grid_numbers_half):
        levels[grid_numbers_half + i + 1] = start_price * ((1 + grid_size) ** (i + 1))

    return levels


@jit(nopython=True)
def simulate_dynamic_grid_trading(
    ohlc_data, grid_size, grid_numbers_half, fee_pct, initial_capital=100.0
):
    """
    Simulate dynamic grid trading with grid rebalancing.

    Args:
        ohlc_data: numpy array [Open, Low, High, Close]
        grid_size: Grid spacing percentage
        grid_numbers_half: Half of grid levels (excluding center)
        fee_pct: Trading fee percentage
        initial_capital: Starting capital per grid

    Returns:
        Tuple of (USDT, BTC, total_money_input, grid_count, trade_count)
    """
    n_rows = ohlc_data.shape[0]
    grid_numbers = 2 * grid_numbers_half

    # Initialize first grid
    initial_price = ohlc_data[0, 0]
    grid_levels = calculate_grid_levels_centered(
        initial_price, grid_size, grid_numbers_half
    )
    lower_bound = grid_levels[0]
    upper_bound = grid_levels[-1]
    current_level = grid_numbers_half

    # Track portfolio
    USDT = 0.0
    BTC = 0.0
    wallet = 0.0
    money_input = initial_capital
    grid_count = 1
    trade_count = 0

    # Track price extremes
    max_price = initial_price
    min_price = initial_price

    prev_close = ohlc_data[0, 0]

    for row_idx in range(n_rows):
        # Build price path: Previous Close -> Low -> High -> Close
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

            max_price = max(max_price, end_price)
            min_price = min(min_price, end_price)

            # Price going up - sell
            if start_price < end_price:
                while current_level < grid_numbers:
                    if start_price <= grid_levels[current_level + 1] < end_price:
                        current_level += 1
                        trade_count += 1
                    else:
                        break

            # Price going down - buy
            else:
                while current_level > 0:
                    if end_price <= grid_levels[current_level - 1] < start_price:
                        current_level -= 1
                        trade_count += 1
                    else:
                        break

            # Check if price exceeded upper bound
            if end_price > upper_bound:
                USDT += initial_capital

                # Profit from upward movement
                upward_profit = (
                    (grid_numbers_half * (grid_numbers_half + 1) / 2)
                    * initial_capital
                    / grid_numbers
                    * (grid_size - fee_pct * 2)
                )

                # Profit from arbitrage
                arbitrage_profit = (
                    (trade_count - grid_numbers_half)
                    / 2
                    * initial_capital
                    / grid_numbers
                    * (grid_size - fee_pct * 2)
                )

                USDT += upward_profit + arbitrage_profit
                wallet += upward_profit + arbitrage_profit

                # Reset grid
                initial_price = end_price
                grid_levels = calculate_grid_levels_centered(
                    initial_price, grid_size, grid_numbers_half
                )
                lower_bound = grid_levels[0]
                upper_bound = grid_levels[-1]
                current_level = grid_numbers_half
                max_price = initial_price
                min_price = initial_price
                trade_count = 0
                grid_count += 1

            # Check if price exceeded lower bound
            elif end_price < lower_bound:
                # Count how many upper levels were touched
                count = 0
                for j in range(1, grid_numbers_half):
                    if max_price >= grid_levels[grid_numbers_half + j]:
                        count += 1
                    else:
                        break

                # Profit from upward movements
                upward_profit = (
                    (count * (count + 1))
                    * initial_capital
                    / grid_numbers
                    * (grid_size - fee_pct * 2)
                )

                # Profit from arbitrage
                arbitrage_profit = (
                    (trade_count - grid_numbers_half - count)
                    / 2
                    * initial_capital
                    / grid_numbers
                    * (grid_size - fee_pct * 2)
                )

                USDT += upward_profit + arbitrage_profit
                wallet += upward_profit + arbitrage_profit

                # Buy BTC at center
                BTC += (initial_capital / 2) / initial_price * (1 - fee_pct * 2)

                # Buy BTC at lower levels
                for j in range(grid_numbers_half):
                    BTC += (
                        (initial_capital / grid_numbers)
                        / grid_levels[j]
                        * (1 - fee_pct * 2)
                    )

                # Reset grid
                initial_price = end_price
                grid_levels = calculate_grid_levels_centered(
                    initial_price, grid_size, grid_numbers_half
                )
                lower_bound = grid_levels[0]
                upper_bound = grid_levels[-1]
                current_level = grid_numbers_half
                max_price = initial_price
                min_price = initial_price

                # Manage wallet for new grid
                if wallet >= initial_capital:
                    wallet -= initial_capital
                else:
                    money_input += initial_capital - wallet
                    wallet = 0.0

                trade_count = 0
                grid_count += 1

        prev_close = ohlc_data[row_idx, 3]

    # Final settlement
    close_price = ohlc_data[-1, 3]

    # Count touched upper levels
    count = 0
    for j in range(1, grid_numbers_half):
        if max_price >= grid_levels[grid_numbers_half + j]:
            count += 1
        else:
            break

    upward_profit = (
        (count * (count + 1))
        * initial_capital
        / grid_numbers
        * (grid_size - fee_pct * 2)
    )
    USDT += upward_profit

    # Count lower levels below close price
    count2 = 0
    for j in range(grid_numbers_half):
        if grid_levels[j] >= close_price:
            BTC += (initial_capital / grid_numbers) / grid_levels[j] * (1 - fee_pct * 2)
            count2 += 1

    # Final arbitrage profit
    arbitrage_profit = (
        (trade_count - count - count2)
        / 2
        * initial_capital
        / grid_numbers
        * (grid_size - fee_pct * 2)
    )
    USDT += arbitrage_profit

    # Remaining capital
    USDT += (grid_numbers_half - count2) * initial_capital / grid_numbers

    return USDT, BTC, money_input, grid_count, trade_count


def backtest_dynamic_grid(df, grid_sizes, grid_numbers_half_list, fee_pct=0.0008):
    """
    Backtest dynamic grid strategy across multiple parameter combinations.

    Args:
        df: DataFrame with OHLC data
        grid_sizes: List of grid sizes to test
        grid_numbers_half_list: List of half grid numbers to test
        fee_pct: Trading fee percentage

    Returns:
        results_df: DataFrame with backtest results
    """
    # Convert to numpy array
    ohlc_data = df[["Open", "Low", "High", "Close"]].values
    close_price = ohlc_data[-1, 3]

    results = []

    for grid_size in grid_sizes:
        for grid_numbers_half in grid_numbers_half_list:
            print(
                f"Testing grid_size={grid_size}, grid_numbers_half={grid_numbers_half}"
            )

            USDT, BTC, money_input, grid_count, trade_count = (
                simulate_dynamic_grid_trading(
                    ohlc_data, grid_size, grid_numbers_half, fee_pct
                )
            )

            total_value = USDT + BTC * close_price
            invested_capital = grid_count * 100

            # Calculate metrics
            profit = total_value - invested_capital
            profit_pct = (total_value / invested_capital * 100) - 100
            usdt_profit_pct = (USDT / invested_capital * 100) - 100
            real_profit_pct = (total_value / money_input * 100) - 100
            real_usdt_profit_pct = (USDT / money_input * 100) - 100

            # IRR calculation (assuming period is known)
            # You may need to adjust the time period (43 months in original)
            irr = ((total_value / money_input) ** (12 / 43) * 100) - 100

            results.append(
                {
                    "grid_size": grid_size,
                    "grid_numbers_half": grid_numbers_half,
                    "grid_count": grid_count,
                    "trade_count": trade_count,
                    "money": invested_capital,
                    "USDT": USDT,
                    "BTC": BTC,
                    "total_value": total_value,
                    "profit": profit,
                    "profit_percentage": profit_pct,
                    "USDT_profit_percentage": usdt_profit_pct,
                    "input_money": money_input,
                    "real_profit_percentage": real_profit_pct,
                    "real_USDT_profit_percentage": real_usdt_profit_pct,
                    "IRR": irr,
                }
            )

    return pd.DataFrame(results)


def plot_results(results_df, symbol):
    """Plot grid size vs profit for different grid numbers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Profit by grid size
    for grid_half in results_df["grid_numbers_half"].unique():
        data = results_df[results_df["grid_numbers_half"] == grid_half]
        ax1.plot(
            data["grid_size"],
            data["real_profit_percentage"],
            label=f"Grid Half: {grid_half}",
            linewidth=2,
        )

    ax1.set_xlabel("Grid Size", fontsize=12)
    ax1.set_ylabel("Real Profit %", fontsize=12)
    ax1.set_title(f"{symbol} - Grid Size vs Profit", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: IRR comparison
    for grid_half in results_df["grid_numbers_half"].unique():
        data = results_df[results_df["grid_numbers_half"] == grid_half]
        ax2.plot(
            data["grid_size"], data["IRR"], label=f"Grid Half: {grid_half}", linewidth=2
        )

    ax2.set_xlabel("Grid Size", fontsize=12)
    ax2.set_ylabel("IRR %", fontsize=12)
    ax2.set_title(f"{symbol} - Grid Size vs IRR", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.savefig(f"{symbol}_dgt_strategy_backtest_plots.png")

def main(symbol: str):
    # Load from folder structure (grid_trading_v2.py method)
    data_path = Path(
        "/home/delus/Documents/code/playground/crypto_visualization/raw_data"
    )
    start_time = "2025-01-01 00:00:00"
    end_time = "2025-10-20 23:59:00"
    print("Loading data...")
    df = load_crypto_data(data_path, symbol, start_time, end_time)
    df = df[(df["Open Time"] >= start_time) & (df["Open Time"] <= end_time)]

    print(f"Loaded {len(df)} rows")

    # Strategy parameters
    grid_sizes = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    grid_numbers_half_list = [1, 2, 3, 5, 7, 10]
    fee_pct = 0.0008

    # Run backtest
    print("\nRunning backtest...")
    results_df = backtest_dynamic_grid(df, grid_sizes, grid_numbers_half_list, fee_pct)

    # Save results
    output_file = f"results/{symbol}_spot_dgt_backtest_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Display top performers
    print("\nTop 5 performing configurations by real profit %:")
    print(
        results_df.nlargest(5, "real_profit_percentage")[
            [
                "grid_size",
                "grid_numbers_half",
                "real_profit_percentage",
                "IRR",
                "total_value",
            ]
        ]
    )

    print("\nTop 5 performing configurations by IRR:")
    print(
        results_df.nlargest(5, "IRR")[
            [
                "grid_size",
                "grid_numbers_half",
                "real_profit_percentage",
                "IRR",
                "total_value",
            ]
        ]
    )

    # Plot results
    plot_results(results_df, f"results/{symbol}_spot")

if __name__ == "__main__":
    # Configuration
    # symbol = "BTCUSDT"
    symbols = ["BTCUSDT", "ETHUSDT", "SUIUSDT", "SOLUSDT", "DOGEUSDT"]
    for symbol in symbols:
        main(symbol)

    
