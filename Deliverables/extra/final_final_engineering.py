import logging
import sys
import pandas as pd
import numpy as np
from datetime import time

# --- Configuration & Constants ---
MINUTE_DATA_CSV_PATH = "BKNG_minute_data.csv"
DAILY_DATA_CSV_PATH = "BKNG.csv"
OUTPUT_CSV_PATH = "BKNG_FINAL.CSV"
TRADING_MINUTES_PER_DAY = 390
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0
EPSILON = 1e-9  # Small constant to prevent division by zero
TRADING_DAYS_PER_YEAR = 252

# --- Setup Logging ---
# Configures logging to print informational messages to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# --- Helper & Feature Calculation Functions ---

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) for a given series.

    RSI is a momentum indicator that measures the magnitude of recent price
    changes to evaluate overbought or oversold conditions.

    Args:
        series: A pandas Series of prices (e.g., closing prices).
        window: The lookback period for the RSI calculation. Default is 14.

    Returns:
        A pandas Series containing the calculated RSI values.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / (avg_loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df_slice: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculates the Average True Range (ATR).

    ATR is a technical analysis indicator that measures market volatility by
    decomposing the entire range of an asset price for that period.

    Args:
        df_slice: A pandas DataFrame containing 'High', 'Low', and 'Close' columns.
        window: The lookback period for the ATR calculation. Default is 14.

    Returns:
        A pandas Series containing the calculated ATR values.
    """
    df_copy = df_slice.copy()
    numeric_cols = ["High", "Low", "Close"]
    for col in numeric_cols:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            logging.warning(
                f"Column '{col}' not numeric or found for ATR. "
                "ATR may be incorrect."
            )
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

    high_low = df_copy["High"] - df_copy["Low"]
    high_close_prev = abs(df_copy["High"] - df_copy["Close"].shift(1))
    low_close_prev = abs(df_copy["Low"] - df_copy["Close"].shift(1))

    tr_df = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    tr = tr_df.max(axis=1, skipna=False)
    atr = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return atr

def detect_events_updated(df_orig: pd.DataFrame) -> pd.DataFrame:
    """Labels each row as 'crash', 'dip', 'rally', or 'normal' based on future price changes.

    This function looks ahead in the data to identify significant price
    movements within defined time windows and magnitude thresholds.

    Args:
        df_orig: The input DataFrame with a 'Close' price column.

    Returns:
        The DataFrame with an added 'event' column labeling each time point.
    """
    df = df_orig.copy()
    df["event"] = "normal"

    # Define the characteristics for each event type
    event_categories = {
        "crash": {"min_magnitude": -0.03, "time_window": (3, 20)},
        "dip": {
            "min_magnitude": -0.01,
            "max_magnitude": -0.02999,
            "time_window": (2, 15),
        },
        "rally": {"min_magnitude": 0.02, "time_window": (3, 20)},
    }
    
    # Process events to ensure more specific events (like 'dip') are checked first
    sorted_events = sorted(
        event_categories.items(),
        key=lambda item: "max_magnitude" in item[1],
        reverse=True,
    )

    # Iterate through each event type and its configuration
    for event_name, config in sorted_events:
        current_price = df["Close"]
        min_win, max_win = config["time_window"]

        # Check for the event across its defined time window
        for i in range(min_win, max_win + 1):
            future_price = df["Close"].shift(-i)
            pct_change = (future_price - current_price) / (current_price + EPSILON)

            # Define the condition based on magnitude thresholds
            condition = pct_change <= config["min_magnitude"]
            if "max_magnitude" in config:
                condition &= pct_change >= config["max_magnitude"]
            
            # Apply the label where the condition is met
            event_mask = condition.fillna(False)
            df.loc[event_mask, "event"] = event_name

    return df

def load_and_prepare_minute_data(file_path: str) -> pd.DataFrame:
    """Loads minute-level data, filters for trading hours, and handles missing data.

    This function reads a CSV, filters out non-trading hours, creates a complete
    index for all trading minutes, and interpolates missing values.

    Args:
        file_path: The path to the minute-level data CSV file.

    Returns:
        A cleaned and prepared pandas DataFrame with minute-level data.
    """
    logging.info(f"Loading minute data from '{file_path}'...")
    try:
        df_minute = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
        df_minute.sort_index(inplace=True)
    except FileNotFoundError:
        logging.critical(f"CRITICAL: Minute data file not found: '{file_path}'. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"CRITICAL: Error loading minute CSV: {e}. Exiting.")
        sys.exit(1)

    # Filter data to include only official trading hours
    market_open = time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
    market_close = time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
    df_minute = df_minute[
        (df_minute.index.time >= market_open) & (df_minute.index.time <= market_close)
    ]
    logging.info(f"Filtered to trading hours. Records: {len(df_minute)}")

    if df_minute.empty:
        logging.critical("CRITICAL: DataFrame is empty after filtering. Exiting.")
        sys.exit(1)

    # Create a full trading index to identify and fill gaps
    unique_days = df_minute.index.normalize().unique()
    full_trading_index = pd.DatetimeIndex([])
    for day in unique_days:
        start_time = day.replace(hour=market_open.hour, minute=market_open.minute)
        end_time = day.replace(hour=market_close.hour, minute=market_close.minute)
        day_index = pd.date_range(start=start_time, end=end_time, freq="min")
        full_trading_index = full_trading_index.union(day_index)

    df_minute = df_minute.reindex(full_trading_index)

    # Interpolate missing price data and fill missing volume/transactions with 0
    cols_to_interpolate = ["Open", "High", "Low", "Close"]
    df_minute[cols_to_interpolate] = df_minute[cols_to_interpolate].interpolate(
        method="time", limit_direction="both"
    )
    df_minute["Volume"] = df_minute["Volume"].fillna(0)
    df_minute["Transactions"] = df_minute["Transactions"].fillna(0)

    logging.info("Reindexing and interpolation complete.")
    return df_minute

def main():
    """Main function to execute the full data processing and feature engineering pipeline."""
    # Load and preprocess the raw minute-level data
    df_minute = load_and_prepare_minute_data(MINUTE_DATA_CSV_PATH)

    # --- Daily Feature Integration ---
    try:
        logging.info(f"Loading daily data from '{DAILY_DATA_CSV_PATH}'...")
        df_daily = pd.read_csv(DAILY_DATA_CSV_PATH, parse_dates=["Date"], index_col="Date")
        df_daily.sort_index(inplace=True)
        df_daily = df_daily.rename(columns=lambda c: c.capitalize())

        # Calculate daily-level features
        df_daily["52w_high"] = df_daily["Close"].rolling(window=TRADING_DAYS_PER_YEAR).max()
        df_daily["52w_low"] = df_daily["Close"].rolling(window=TRADING_DAYS_PER_YEAR).min()
        df_daily["Volume_30D_avg"] = df_daily["Volume"].rolling(window=30).mean()
        df_daily["MA_100D_proxy"] = df_daily["Close"].rolling(window=100).mean()

        # Map daily features to the minute-level DataFrame
        features_to_map = ["52w_high", "52w_low", "Volume_30D_avg", "MA_100D_proxy"]
        df_minute = pd.merge(
            df_minute,
            df_daily[features_to_map],
            left_on=df_minute.index.date,
            right_index=True,
            how="left",
        )
        df_minute.index.name = "Date"
        logging.info("Daily features calculated and merged.")

    except FileNotFoundError:
        logging.warning("Daily data file not found. Daily features will be NaN.")
        for col in ["52w_high", "52w_low", "Volume_30D_avg", "MA_100D_proxy"]:
            df_minute[col] = np.nan

    # --- Minute-Level Feature Calculation ---
    logging.info("Calculating minute-level features and technical indicators...")

    # Calculate VWAP (Volume Weighted Average Price)
    df_minute["Typical_Price_Volume"] = (df_minute["Volume"] * (df_minute["High"] + df_minute["Low"] + df_minute["Close"]) / 3)
    df_minute["Cumulative_Typical_Price_Volume_Daily"] = df_minute.groupby(df_minute.index.date)["Typical_Price_Volume"].cumsum()
    df_minute["Cumulative_Volume_Daily"] = df_minute.groupby(df_minute.index.date)["Volume"].cumsum()
    df_minute["VWAP"] = df_minute["Cumulative_Typical_Price_Volume_Daily"] / (df_minute["Cumulative_Volume_Daily"] + EPSILON)

    # Calculate Moving Averages and Price-to-VWAP ratio
    df_minute["MA_50"] = df_minute["Close"].rolling(window="50min").mean()
    df_minute["MA_200"] = df_minute["Close"].rolling(window="200min").mean()
    df_minute["Close/VWAP"] = df_minute["Close"] / (df_minute["VWAP"] + EPSILON)

    # Calculate standard technical indicators (RSI, ATR, MACD)
    df_minute["RSI_14"] = calculate_rsi(df_minute["Close"], window=14)
    df_minute["ATR_14"] = calculate_atr(df_minute[["High", "Low", "Close"]], window=14)

    ema12 = df_minute["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_minute["Close"].ewm(span=26, adjust=False).mean()
    df_minute["MACD"] = ema12 - ema26
    df_minute["MACD_Signal"] = df_minute["MACD"].ewm(span=9, adjust=False).mean()

    # Calculate Realized Volatility over 5 and 30-minute windows
    annualization_5min = np.sqrt(TRADING_DAYS_PER_YEAR * (TRADING_MINUTES_PER_DAY / 5))
    annualization_30min = np.sqrt(TRADING_DAYS_PER_YEAR * (TRADING_MINUTES_PER_DAY / 30))
    df_minute["realized_vol_5min"] = (df_minute["Close"].pct_change().rolling("5min").std() * annualization_5min)
    df_minute["realized_vol_30min"] = (df_minute["Close"].pct_change().rolling("30min").std() * annualization_30min)

    # --- Time-Based Feature Calculation ---
    logging.info("Calculating time-based features...")
    df_minute["day_of_week"] = df_minute.index.dayofweek
    df_minute["month_of_year"] = df_minute.index.month

    market_open_dt = (pd.to_datetime(df_minute.index.date) + pd.to_timedelta(MARKET_OPEN_HOUR, unit="h") + pd.to_timedelta(MARKET_OPEN_MINUTE, unit="m"))
    df_minute["hours_since_open"] = (df_minute.index - market_open_dt).total_seconds() / 3600
    df_minute.loc[df_minute["hours_since_open"] < 0, "hours_since_open"] = 0

    # --- Event Detection ---
    logging.info("Detecting events (crash, dip, rally)...")
    df_minute = detect_events_updated(df_minute)
    logging.info(f"Event distribution:\n{df_minute['event'].value_counts(normalize=True)}")

    # --- Finalization and Cleanup ---
    logging.info("Finalizing processing and cleaning up...")
    
    # Drop intermediate columns used for VWAP calculation
    cols_to_drop = [
        "Typical_Price_Volume",
        "Cumulative_Typical_Price_Volume_Daily",
        "Cumulative_Volume_Daily",
    ]
    df_minute.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # Forward-fill and then back-fill to handle any remaining NaNs in feature columns
    feature_cols = [col for col in df_minute.columns if col != "event"]
    df_minute[feature_cols] = df_minute[feature_cols].ffill()
    df_minute[feature_cols] = df_minute[feature_cols].bfill()

    final_nan_count = df_minute.isnull().sum().sum()
    if final_nan_count == 0:
        logging.info("No NaNs remaining in the final DataFrame.")
    else:
        logging.warning(f"WARNING: {final_nan_count} NaNs remain in the DataFrame.")

    # Save the final DataFrame to a CSV file
    try:
        df_minute.to_csv(OUTPUT_CSV_PATH)
        logging.info(f"DataFrame successfully saved to '{OUTPUT_CSV_PATH}'")
    except Exception as e:
        logging.error(f"Error saving DataFrame to CSV: {e}")

if __name__ == "__main__":
    main()
