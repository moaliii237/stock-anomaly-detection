# create_artifacts.py (New Version)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle
import os
import logging

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# Paths are relative to the project root (e.g., .../Deliverables/project)
DATA_PATH = "src/project/app/BKNG_engineering.csv"
OUTPUT_PATH = "src/project/models/saved_models/preprocessing_artifacts.pkl"

# This feature list MUST EXACTLY MATCH the one in your ProductionAnomalyPredictor class
FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Transactions",
    "52w_high",
    "52w_low",
    "Volume_30D_avg",
    "MA_100D_proxy",
    "Price_30D_zscore",
    "Close/52w_high",
    "Close/52w_low",
    "MA_50",
    "MA_200",
    "Close/MA_50",
    "Close/MA_200",
    "intraday_amplitude",
    "overnight_gap",
    "VWAP",
    "Close/VWAP",
    "Volume/30D_avg",
    "Volume_intraday_zscore",
    "volume_acceleration",
    "high_volume_bar_60min_95p",
    "consec_high_volume_bars_5",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "ROC_5",
    "ROC_15",
    "ATR_14",
    "realized_vol_5min",
    "realized_vol_30min",
    "BB_MA_20",
    "BB_std_20",
    "BB_upper",
    "BB_lower",
    "BB_width",
    "MA_50_slope",
    "MA_200_slope",
    "tenkan_sen",
    "kijun_sen",
    "senkou_span_a",
    "senkou_span_b",
    "chikou_span",
    "days_since_ATH",
    "hours_since_open",
    "day_of_week",
    "month_of_year",
    "covid_period",
    "high_vol_regime_90q",
    "bull_market_proxy",
    "bear_market_proxy",
]

# Define the column that contains the anomaly type labels for Stage 2
TARGET_COLUMN = "event"
# --- End Configuration ---


def create_and_save_artifacts():
    """
    Loads training data, creates preprocessing artifacts compatible with the
    ProductionAnomalyPredictor, and saves them to a pickle file.
    """
    logging.info("Starting artifact creation...")

    # Load the training data
    try:
        data = pd.read_csv(DATA_PATH)
        logging.info(f"Data loaded successfully from '{DATA_PATH}'.")
    except FileNotFoundError:
        logging.critical(f"FATAL ERROR: Data file not found at '{DATA_PATH}'.")
        return

    # --- 1. Create a Single Master Scaler ---
    logging.info("Creating a single MinMaxScaler for all feature columns...")
    # Ensure all required feature columns exist in the dataframe
    missing_cols = [col for col in FEATURE_COLUMNS if col not in data.columns]
    if missing_cols:
        logging.critical(
            f"FATAL ERROR: The following required columns are missing from the data: {missing_cols}"
        )
        return

    # Select the features in the correct order and fit the scaler
    features_to_scale = data[FEATURE_COLUMNS]
    master_scaler = MinMaxScaler()
    master_scaler.fit(features_to_scale)
    logging.info("Single master scaler has been fitted successfully.")

    # --- 2. Get Anomaly Label Encoder Classes ---
    logging.info("Extracting anomaly type classes for the Stage 2 model...")
    if TARGET_COLUMN not in data.columns:
        logging.critical(
            f"FATAL ERROR: Target column '{TARGET_COLUMN}' not found in the data."
        )
        return

    # Drop missing values from the target column and find the unique classes
    valid_labels = data[TARGET_COLUMN].dropna()
    label_encoder = LabelEncoder()
    label_encoder.fit(valid_labels)
    anomaly_classes = list(label_encoder.classes_)
    logging.info(f"Detected anomaly classes: {anomaly_classes}")

    # --- 3. Combine Artifacts into the Required Structure ---
    preprocessing_artifacts = {
        "scaler": master_scaler,
        "anomaly_label_encoder_classes": anomaly_classes,
    }
    logging.info(
        "Final artifacts dictionary created with keys: 'scaler' and 'anomaly_label_encoder_classes'."
    )

    # --- 4. Save the Artifacts File ---
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(preprocessing_artifacts, f)
        logging.info(
            "\nSUCCESS! New 'preprocessing_artifacts.pkl' file has been created at:"
        )
        logging.info(f"'{os.path.abspath(OUTPUT_PATH)}'")
        logging.info("\nYou can now run your Flask app.")
    except Exception as e:
        logging.critical(f"FATAL ERROR: Failed to save artifacts file. Details: {e}")


if __name__ == "__main__":
    create_and_save_artifacts()
