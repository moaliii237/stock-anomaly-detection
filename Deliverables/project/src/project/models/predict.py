import logging
import pickle
import traceback
from datetime import datetime

import numpy as np

from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionAnomalyPredictor:
    """
    Production-ready anomaly predictor for real-time financial market surveillance.
    This version uses a predefined feature list for robustness and an efficient NumPy buffer for performance.
    """

    def __init__(self, models_path: str = "models/saved_models/"):
        """Initialize the production predictor by loading all saved artifacts."""
        self.models_path = models_path
        self.feature_buffer = None
        self.buffer_is_full = False
        # Counter to track how many points are in the buffer.
        self.points_in_buffer = 0

        try:
            self.stage1_model = load_model(f"{models_path}stage1_lstm.keras")
            self.stage2_model = load_model(f"{models_path}stage2_lstm.keras")

            # Load preprocessing artifacts
            with open(f"{models_path}preprocessing_artifacts.pkl", "rb") as f:
                self.preprocessing = pickle.load(f)

            # Use hardcoded feature list for robustness
            self.feature_columns = [
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
            self.n_features = len(self.feature_columns)

            # Use the single 'scaler' object, which is standard practice
            self.scaler = self.preprocessing.get("scaler")
            if self.scaler is None:
                raise ValueError(
                    "'scaler' object not found in preprocessing_artifacts.pkl."
                )

            # sequence length matches the buffer_size used in preprocessing
            self.sequence_length = 30

            # Initialize the efficient NumPy buffer
            self.feature_buffer = np.zeros(
                (self.sequence_length, self.n_features), dtype=np.float32
            )

            logger.info(
                "Production predictor (Hybrid Version) initialized successfully."
            )
            logger.info(f"Sequence length: {self.sequence_length}")
            logger.info(f"Number of features: {self.n_features}")

        except FileNotFoundError as e:
            logger.error(
                f"Model artifact not found: {e}. Ensure models are trained and saved in '{models_path}'."
            )
            raise e
        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            raise e

    def predict_single_point(self, new_data_point: dict):
        """Predict using the efficient NumPy buffer and a fixed feature list."""

        timestamp = new_data_point.get("Date", str(datetime.now()))
        try:
            # Create a new row of data in the exact order required by the model
            new_row = np.array(
                [new_data_point.get(col, 0) for col in self.feature_columns],
                dtype=np.float32,
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting new data point to numeric array: {e}")
            return {
                "status": "error",
                "message": "Invalid data format in new data point.",
            }

        # Efficiently update the buffer by rolling and replacing the last row
        self.feature_buffer = np.roll(self.feature_buffer, -1, axis=0)
        self.feature_buffer[-1, :] = new_row

        # Use the counter instead of checking for zeros to know when the
        # buffer is full.
        if not self.buffer_is_full:
            self.points_in_buffer += 1
            if self.points_in_buffer < self.sequence_length:
                return {
                    "status": "buffering",
                    "progress": f"{self.points_in_buffer}/{self.sequence_length}",
                }
            else:
                self.buffer_is_full = True
                logger.info("Rolling buffer is now full. Starting real predictions.")

        try:
            # Scale the entire buffer at once
            X_scaled = self.scaler.transform(self.feature_buffer)
            # Reshape for the LSTM model
            X_seq = X_scaled.reshape(1, self.sequence_length, self.n_features)

            # Stage 1: Anomaly Detection
            stage1_pred = self.stage1_model.predict(X_seq, verbose=0)
            # Correctly access the probability of the 'anomaly' class (usually at index 1)
            # This depends on how the label_encoder was fit during training.
            # Assuming 'normal' is 0 and 'anomaly' is 1. Check your training artifacts if this is wrong.
            anomaly_prob = (
                float(stage1_pred[0][1])
                if stage1_pred.shape[1] > 1
                else float(stage1_pred[0][0])
            )
            is_anomaly = anomaly_prob > 0.5

            # Stage 2: Anomaly Classification
            if is_anomaly:
                stage2_pred = self.stage2_model.predict(X_seq, verbose=0)
                anomaly_type_idx = np.argmax(stage2_pred[0])
                # The label encoders should be in your self.preprocessing artifact
                anomaly_types = self.preprocessing.get(
                    "anomaly_label_encoder_classes", ["crash", "dip", "rally"]
                )
                predicted_type = anomaly_types[anomaly_type_idx]
                confidence = float(np.max(stage2_pred[0]))
                type_probabilities = {
                    name: float(prob)
                    for name, prob in zip(anomaly_types, stage2_pred[0])
                }
            else:
                predicted_type = "normal"
                type_probabilities = None
                # Confidence when 'normal' is the inverse of anomaly probability
                confidence = 1 - anomaly_prob

            return {
                "status": "success",
                "timestamp": timestamp,
                "is_anomaly": is_anomaly,
                "anomaly_probability": anomaly_prob,
                "predicted_anomaly_type": predicted_type,
                "confidence": confidence,
                "type_probabilities": type_probabilities,
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
            return {"status": "error", "message": f"Prediction failed: {e}"}

    def reset_buffer(self):
        """Reset the rolling buffer."""
        if self.feature_columns:
            self.feature_buffer = np.zeros(
                (self.sequence_length, self.n_features), dtype=np.float32
            )
        else:
            self.feature_buffer = None
        self.buffer_is_full = False
        self.points_in_buffer = 0
        logger.info("Buffer reset")
