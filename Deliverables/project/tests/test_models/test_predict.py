import pytest
import numpy as np
import pandas as pd
import pickle
import os
from unittest.mock import Mock, patch, mock_open, MagicMock
from datetime import datetime

from src.project.models.predict import ProductionAnomalyPredictor


@pytest.fixture
def mock_preprocessing_artifacts():
    """Mock preprocessing artifacts that would be loaded from pickle."""
    return {
        "scaler": Mock(),
        "anomaly_label_encoder_classes": ["crash", "dip", "rally"],
    }


@pytest.fixture
def mock_models():
    """Mock trained models for stage1 and stage2."""
    stage1_model = Mock()
    stage2_model = Mock()

    # Mock stage1 predictions (binary: normal/anomaly)
    stage1_model.predict.return_value = np.array(
        [[0.3, 0.7]]
    )  # 70% anomaly probability

    # Mock stage2 predictions (multi-class: crash/dip/rally)
    stage2_model.predict.return_value = np.array(
        [[0.1, 0.8, 0.1]]
    )  # 80% dip probability

    return stage1_model, stage2_model


@pytest.fixture
def sample_data_point():
    """Sample financial data point for prediction."""
    return {
        "Date": "2024-01-15 10:30:00",
        "Open": 150.0,
        "High": 152.0,
        "Low": 149.0,
        "Close": 151.0,
        "Volume": 10000,
        "Transactions": 500,
        "52w_high": 180.0,
        "52w_low": 120.0,
        "Volume_30D_avg": 8000.0,
        "MA_100D_proxy": 148.0,
        "Price_30D_zscore": 0.5,
        "Close/52w_high": 0.84,
        "Close/52w_low": 1.26,
        "MA_50": 149.0,
        "MA_200": 145.0,
        "Close/MA_50": 1.01,
        "Close/MA_200": 1.04,
        "intraday_amplitude": 2.0,
        "overnight_gap": 0.5,
        "VWAP": 150.5,
        "Close/VWAP": 1.003,
        "Volume/30D_avg": 1.25,
        "Volume_intraday_zscore": 0.8,
        "volume_acceleration": 1.1,
        "high_volume_bar_60min_95p": 0,
        "consec_high_volume_bars_5": 1,
        "RSI_14": 55.0,
        "MACD": 0.5,
        "MACD_Signal": 0.3,
        "MACD_Hist": 0.2,
        "ROC_5": 1.5,
        "ROC_15": 3.2,
        "ATR_14": 2.1,
        "realized_vol_5min": 0.02,
        "realized_vol_30min": 0.05,
        "BB_MA_20": 150.0,
        "BB_std_20": 2.5,
        "BB_upper": 155.0,
        "BB_lower": 145.0,
        "BB_width": 10.0,
        "MA_50_slope": 0.1,
        "MA_200_slope": 0.05,
        "tenkan_sen": 150.5,
        "kijun_sen": 149.0,
        "senkou_span_a": 150.0,
        "senkou_span_b": 148.0,
        "chikou_span": 151.0,
        "days_since_ATH": 30,
        "hours_since_open": 2.5,
        "day_of_week": 1,
        "month_of_year": 1,
        "covid_period": 0,
        "high_vol_regime_90q": 0,
        "bull_market_proxy": 1,
        "bear_market_proxy": 0,
    }


class TestProductionAnomalyPredictorInit:
    """Test suite for ProductionAnomalyPredictor initialization."""

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_initialization_success(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
    ):
        """Test successful initialization of the predictor."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Initialize predictor
        predictor = ProductionAnomalyPredictor("test_models/")

        # Verify initialization
        assert predictor.models_path == "test_models/"
        assert predictor.stage1_model == stage1_model
        assert predictor.stage2_model == stage2_model
        assert predictor.scaler == mock_preprocessing_artifacts["scaler"]
        assert predictor.sequence_length == 30
        assert predictor.n_features == 55  # Fixed: Updated from 50 to 55
        assert predictor.buffer_is_full == False
        assert predictor.points_in_buffer == 0

        # Verify buffer initialization
        assert predictor.feature_buffer.shape == (
            30,
            55,
        )  # Fixed: Updated from 50 to 55
        assert predictor.feature_buffer.dtype == np.float32

        # Verify model loading calls
        mock_load_model.assert_any_call("test_models/stage1_lstm.keras")
        mock_load_model.assert_any_call("test_models/stage2_lstm.keras")

        # Verify preprocessing loading
        mock_file.assert_called_with("test_models/preprocessing_artifacts.pkl", "rb")
        mock_pickle_load.assert_called_once()

        # Verify logging
        mock_logger.info.assert_any_call(
            "Production predictor (Hybrid Version) initialized successfully."
        )

    @patch("src.project.models.predict.load_model")
    @patch("src.project.models.predict.logger")
    def test_initialization_model_not_found(self, mock_logger, mock_load_model):
        """Test initialization failure when model files are not found."""
        # Force FileNotFoundError
        mock_load_model.side_effect = FileNotFoundError("Model file not found")

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            ProductionAnomalyPredictor("nonexistent_models/")

        # Verify error logging
        mock_logger.error.assert_called()

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_initialization_missing_scaler(
        self, mock_logger, mock_pickle_load, mock_file, mock_load_model, mock_models
    ):
        """Test initialization failure when scaler is missing from preprocessing artifacts."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = {}  # Missing scaler

        with pytest.raises(ValueError, match="'scaler' object not found"):
            ProductionAnomalyPredictor("test_models/")

    @patch("src.project.models.predict.load_model")
    @patch("src.project.models.predict.logger")
    def test_initialization_general_exception(self, mock_logger, mock_load_model):
        """Test initialization failure with general exception."""
        # Force general exception
        mock_load_model.side_effect = Exception("General initialization error")

        with pytest.raises(Exception, match="General initialization error"):
            ProductionAnomalyPredictor("test_models/")

        # Verify error logging
        mock_logger.error.assert_called_with(
            "Error initializing predictor: General initialization error"
        )


class TestProductionAnomalyPredictorPrediction:
    """Test suite for prediction functionality."""

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_predict_single_point_buffering_phase(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test prediction during buffering phase (buffer not full)."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        predictor = ProductionAnomalyPredictor("test_models/")

        # Test first prediction (buffering)
        result = predictor.predict_single_point(sample_data_point)

        # Verify buffering response
        assert result["status"] == "buffering"
        assert result["progress"] == "1/30"
        assert predictor.points_in_buffer == 1
        assert not predictor.buffer_is_full

        # Test multiple buffering predictions
        for i in range(2, 30):
            result = predictor.predict_single_point(sample_data_point)
            assert result["status"] == "buffering"
            assert result["progress"] == f"{i}/30"
            assert predictor.points_in_buffer == i

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_predict_single_point_anomaly_detected(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test prediction when anomaly is detected."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Mock scaler transform - Fixed: Return correct shape (30, 55)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.random((30, 55))
        mock_preprocessing_artifacts["scaler"] = mock_scaler

        predictor = ProductionAnomalyPredictor("test_models/")

        # Fill buffer first
        for _ in range(30):
            predictor.predict_single_point(sample_data_point)

        # Now test actual prediction with anomaly detection
        result = predictor.predict_single_point(sample_data_point)

        # Verify successful prediction
        assert result["status"] == "success"
        assert result["is_anomaly"] == True  # 70% anomaly probability > 0.5
        assert result["anomaly_probability"] == 0.7
        assert (
            result["predicted_anomaly_type"] == "dip"
        )  # Highest probability from stage2
        assert result["confidence"] == 0.8  # Max probability from stage2
        assert "type_probabilities" in result
        assert result["type_probabilities"]["dip"] == 0.8
        assert "timestamp" in result

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_predict_single_point_normal_detected(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test prediction when normal behavior is detected."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        stage1_model.predict.return_value = np.array(
            [[0.8, 0.2]]
        )  # 20% anomaly probability
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Mock scaler transform - Fixed: Return correct shape (30, 55)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.random((30, 55))
        mock_preprocessing_artifacts["scaler"] = mock_scaler

        predictor = ProductionAnomalyPredictor("test_models/")

        # Fill buffer first
        for _ in range(30):
            predictor.predict_single_point(sample_data_point)

        # Test prediction with normal detection
        result = predictor.predict_single_point(sample_data_point)

        # Verify normal prediction
        assert result["status"] == "success"
        assert result["is_anomaly"] == False  # 20% anomaly probability < 0.5
        assert result["anomaly_probability"] == 0.2
        assert result["predicted_anomaly_type"] == "normal"
        assert result["confidence"] == 0.8  # 1 - 0.2
        assert result["type_probabilities"] is None

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_predict_single_point_invalid_data_format(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
    ):
        """Test prediction with invalid data format."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        predictor = ProductionAnomalyPredictor("test_models/")

        # Test with invalid data (non-numeric values)
        invalid_data = {"Date": "2024-01-15", "Open": "invalid_price"}
        result = predictor.predict_single_point(invalid_data)

        # Verify error response
        assert result["status"] == "error"
        assert "Invalid data format" in result["message"]

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_predict_single_point_missing_date(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test prediction when Date field is missing."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        predictor = ProductionAnomalyPredictor("test_models/")

        # Remove Date field
        data_without_date = sample_data_point.copy()
        del data_without_date["Date"]

        result = predictor.predict_single_point(data_without_date)

        # Should still work, using current timestamp
        assert result["status"] == "buffering"
        # Fixed: The actual function doesn't return timestamp during buffering phase
        # Only check that it doesn't crash

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_predict_single_point_prediction_exception(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test prediction failure during model inference."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Mock scaler to raise exception
        mock_scaler = Mock()
        mock_scaler.transform.side_effect = Exception("Scaling failed")
        mock_preprocessing_artifacts["scaler"] = mock_scaler

        predictor = ProductionAnomalyPredictor("test_models/")

        # Fill buffer first
        for _ in range(30):
            predictor.predict_single_point(sample_data_point)

        # Test prediction with scaling failure
        result = predictor.predict_single_point(sample_data_point)

        # Verify error response
        assert result["status"] == "error"
        assert "Prediction failed" in result["message"]

        # Verify error logging
        mock_logger.error.assert_called()

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    def test_predict_single_point_stage1_single_output(
        self,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        sample_data_point,
    ):
        """Test prediction when stage1 model has single output."""
        # Setup mocks with single output stage1 model
        stage1_model = Mock()
        stage2_model = Mock()
        stage1_model.predict.return_value = np.array([[0.7]])  # Single output
        stage2_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])

        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Mock scaler transform - Fixed: Return correct shape (30, 55)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.random((30, 55))
        mock_preprocessing_artifacts["scaler"] = mock_scaler

        predictor = ProductionAnomalyPredictor("test_models/")

        # Fill buffer and predict
        for _ in range(30):
            predictor.predict_single_point(sample_data_point)

        result = predictor.predict_single_point(sample_data_point)

        # Verify single output handling
        assert result["status"] == "success"
        assert result["anomaly_probability"] == 0.7


class TestProductionAnomalyPredictorBufferManagement:
    """Test suite for buffer management functionality."""

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_reset_buffer(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
    ):
        """Test buffer reset functionality."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        predictor = ProductionAnomalyPredictor("test_models/")

        # Modify buffer state
        predictor.buffer_is_full = True
        predictor.points_in_buffer = 25
        predictor.feature_buffer = np.ones((30, 55))  # Fixed: Updated from 50 to 55

        # Reset buffer
        predictor.reset_buffer()

        # Verify reset
        assert predictor.buffer_is_full == False
        assert predictor.points_in_buffer == 0
        assert np.all(predictor.feature_buffer == 0)
        assert predictor.feature_buffer.shape == (
            30,
            55,
        )  # Fixed: Updated from 50 to 55
        assert predictor.feature_buffer.dtype == np.float32

        # Verify logging
        mock_logger.info.assert_called_with("Buffer reset")

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    def test_reset_buffer_no_feature_columns(
        self,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
    ):
        """Test buffer reset when feature_columns is None."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        predictor = ProductionAnomalyPredictor("test_models/")

        # Simulate no feature columns
        predictor.feature_columns = None

        # Reset buffer
        predictor.reset_buffer()

        # Verify reset with None feature buffer
        assert predictor.feature_buffer is None
        assert predictor.buffer_is_full == False
        assert predictor.points_in_buffer == 0


class TestProductionAnomalyPredictorIntegration:
    """Integration tests for the full prediction pipeline."""

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    @patch("src.project.models.predict.logger")
    def test_full_prediction_pipeline(
        self,
        mock_logger,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test complete prediction pipeline from initialization to full buffer predictions."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Mock scaler transform - Fixed: Return correct shape (30, 55)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.random((30, 55))
        mock_preprocessing_artifacts["scaler"] = mock_scaler

        predictor = ProductionAnomalyPredictor("test_models/")

        # Test complete buffering phase
        for i in range(1, 30):
            result = predictor.predict_single_point(sample_data_point)
            assert result["status"] == "buffering"
            assert result["progress"] == f"{i}/30"

        # Buffer should now be full on next prediction
        result = predictor.predict_single_point(sample_data_point)
        assert predictor.buffer_is_full == True

        # Verify full prediction pipeline
        assert result["status"] == "success"
        assert result["is_anomaly"] == True
        assert result["predicted_anomaly_type"] == "dip"
        assert "confidence" in result
        assert "type_probabilities" in result

        # Test subsequent predictions (buffer already full)
        result2 = predictor.predict_single_point(sample_data_point)
        assert result2["status"] == "success"

        # Verify models were called correctly
        stage1_model.predict.assert_called()
        stage2_model.predict.assert_called()
        mock_scaler.transform.assert_called()

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    def test_buffer_rolling_mechanism(
        self,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test that buffer rolling mechanism works correctly."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        predictor = ProductionAnomalyPredictor("test_models/")

        # Fill buffer with distinct values to track rolling
        for i in range(30):
            data_point = sample_data_point.copy()
            data_point["Open"] = float(i)  # Use index as distinct value
            predictor.predict_single_point(data_point)

        # Verify buffer is filled correctly
        assert predictor.buffer_is_full == True
        assert predictor.feature_buffer[-1, 0] == 29.0  # Last value should be 29
        assert predictor.feature_buffer[0, 0] == 0.0  # First value should be 0

        # Add one more point to test rolling
        new_data = sample_data_point.copy()
        new_data["Open"] = 100.0
        predictor.predict_single_point(new_data)

        # Verify rolling: first value should now be 1, last should be 100
        assert predictor.feature_buffer[-1, 0] == 100.0  # New value at end
        assert predictor.feature_buffer[0, 0] == 1.0  # First value rolled

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    def test_alternative_anomaly_types(
        self,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test prediction with different anomaly types."""
        # Setup mocks with crash prediction
        stage1_model, stage2_model = mock_models
        stage2_model.predict.return_value = np.array(
            [[0.8, 0.1, 0.1]]
        )  # Crash prediction
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Mock scaler transform - Fixed: Return correct shape (30, 55)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.random((30, 55))
        mock_preprocessing_artifacts["scaler"] = mock_scaler

        predictor = ProductionAnomalyPredictor("test_models/")

        # Fill buffer and predict
        for _ in range(30):
            predictor.predict_single_point(sample_data_point)

        result = predictor.predict_single_point(sample_data_point)

        # Verify crash prediction
        assert result["predicted_anomaly_type"] == "crash"
        assert result["type_probabilities"]["crash"] == 0.8
        assert result["confidence"] == 0.8


class TestProductionAnomalyPredictorEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    def test_missing_features_in_data_point(
        self,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
    ):
        """Test prediction when some features are missing from data point."""
        # Setup mocks
        stage1_model, stage2_model = mock_models
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        predictor = ProductionAnomalyPredictor("test_models/")

        # Data point with only few features (others will be 0)
        minimal_data = {"Date": "2024-01-15", "Open": 150.0, "High": 152.0}

        result = predictor.predict_single_point(minimal_data)

        # Should still work with default values (0) for missing features
        assert result["status"] == "buffering"
        assert predictor.points_in_buffer == 1

    @patch("src.project.models.predict.load_model")
    @patch("builtins.open", new_callable=mock_open)
    @patch("src.project.models.predict.pickle.load")
    def test_exact_threshold_anomaly_probability(
        self,
        mock_pickle_load,
        mock_file,
        mock_load_model,
        mock_preprocessing_artifacts,
        mock_models,
        sample_data_point,
    ):
        """Test prediction when anomaly probability is exactly at threshold."""
        # Setup mocks with exactly 0.5 probability
        stage1_model, stage2_model = mock_models
        stage1_model.predict.return_value = np.array([[0.5, 0.5]])  # Exactly 50%
        mock_load_model.side_effect = [stage1_model, stage2_model]
        mock_pickle_load.return_value = mock_preprocessing_artifacts

        # Mock scaler transform - Fixed: Return correct shape (30, 55)
        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.random.random((30, 55))
        mock_preprocessing_artifacts["scaler"] = mock_scaler

        predictor = ProductionAnomalyPredictor("test_models/")

        # Fill buffer and predict
        for _ in range(30):
            predictor.predict_single_point(sample_data_point)

        result = predictor.predict_single_point(sample_data_point)

        # 0.5 should not be considered anomaly (> 0.5 required)
        assert result["is_anomaly"] == False
        assert result["predicted_anomaly_type"] == "normal"
