import pytest
import numpy as np
import pandas as pd
import os
from unittest.mock import Mock, patch, mock_open, call
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.project.data.preprocessing.data_processor import (
    LSTMDataProcessor,
    TwoStageLSTMDataProcessor,
)


@pytest.fixture
def mock_config():
    """Mock configuration object for LSTM data processor."""
    config = Mock()
    config.train_path = "data/train.csv"
    config.test_path = "data/test.csv"
    config.sequence_length = 30
    config.target_column = "event_type"
    config.non_feature_cols = ["Date", "event_type"]
    config.val_ratio = 0.7
    return config


@pytest.fixture
def sample_financial_data():
    """Sample financial time series data for testing."""
    dates = pd.date_range("2024-01-01 09:30:00", periods=100, freq="1min")
    data = {
        "Date": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": np.random.uniform(100, 200, 100),
        "High": np.random.uniform(150, 250, 100),
        "Low": np.random.uniform(50, 150, 100),
        "Close": np.random.uniform(100, 200, 100),
        "Volume": np.random.randint(1000, 100000, 100),
        "RSI_14": np.random.uniform(20, 80, 100),
        "MACD": np.random.uniform(-2, 2, 100),
        "event_type": ["normal"] * 70 + ["dip"] * 15 + ["crash"] * 10 + ["rally"] * 5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    """Sample test data with different structure for compatibility testing."""
    dates = pd.date_range("2024-02-01 09:30:00", periods=50, freq="1min")
    data = {
        "Date": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "Open": np.random.uniform(100, 200, 50),
        "High": np.random.uniform(150, 250, 50),
        "Low": np.random.uniform(50, 150, 50),
        "Close": np.random.uniform(100, 200, 50),
        "Volume": np.random.randint(1000, 100000, 50),
        "RSI_14": np.random.uniform(20, 80, 50),
        # Missing MACD column for compatibility testing
        "event_type": ["normal"] * 35 + ["dip"] * 10 + ["crash"] * 5,
    }
    return pd.DataFrame(data)


class TestLSTMDataProcessor:
    """Test suite for LSTMDataProcessor class."""

    def test_initialization(self, mock_config):
        """Test proper initialization of LSTMDataProcessor."""
        processor = LSTMDataProcessor(mock_config)

        assert processor.config == mock_config
        assert isinstance(processor.scaler, MinMaxScaler)
        assert isinstance(processor.label_encoder, LabelEncoder)

    @patch("pandas.read_csv")
    @patch("src.project.data.preprocessing.data_processor.logger")
    def test_load_and_validate_data_success(
        self,
        mock_logger,
        mock_read_csv,
        mock_config,
        sample_financial_data,
        sample_test_data,
    ):
        """Test successful data loading and validation."""
        mock_read_csv.side_effect = [sample_financial_data, sample_test_data]

        processor = LSTMDataProcessor(mock_config)

        with patch.object(processor, "_validate_data_compatibility") as mock_validate:
            df_train, df_test = processor.load_and_validate_data()

        # Verify data loading
        assert mock_read_csv.call_count == 2
        mock_read_csv.assert_any_call(mock_config.train_path)
        mock_read_csv.assert_any_call(mock_config.test_path)

        # Verify data processing
        assert isinstance(df_train.index, pd.DatetimeIndex)
        assert isinstance(df_test.index, pd.DatetimeIndex)
        assert df_train.index.is_monotonic_increasing
        assert df_test.index.is_monotonic_increasing

        # Verify validation was called
        mock_validate.assert_called_once()

        # Verify logging
        mock_logger.info.assert_any_call(f"Loaded training data: {df_train.shape}")
        mock_logger.info.assert_any_call(f"Loaded test data: {df_test.shape}")

    @patch("pandas.read_csv")
    @patch("src.project.data.preprocessing.data_processor.logger")
    def test_load_and_validate_data_file_error(
        self, mock_logger, mock_read_csv, mock_config
    ):
        """Test error handling when files cannot be loaded."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        processor = LSTMDataProcessor(mock_config)

        with pytest.raises(FileNotFoundError):
            processor.load_and_validate_data()

        mock_logger.error.assert_called_once_with("Error loading data: File not found")

    @patch("src.project.data.preprocessing.data_processor.logger")
    def test_validate_data_compatibility_missing_columns(
        self, mock_logger, mock_config, sample_financial_data, sample_test_data
    ):
        """Test data compatibility validation with missing columns."""
        processor = LSTMDataProcessor(mock_config)

        processor._validate_data_compatibility(sample_financial_data, sample_test_data)

        # Should log warning about missing MACD column in test data
        mock_logger.warning.assert_any_call("Columns missing in test: {'MACD'}")

    @patch("src.project.data.preprocessing.data_processor.logger")
    def test_validate_data_compatibility_no_issues(
        self, mock_logger, mock_config, sample_financial_data
    ):
        """Test data compatibility validation with no issues."""
        processor = LSTMDataProcessor(mock_config)

        # Use same data for both train and test
        processor._validate_data_compatibility(
            sample_financial_data, sample_financial_data
        )

        # Should not log any warnings
        mock_logger.warning.assert_not_called()

    def test_prepare_sequences_fit_scalers(self, mock_config, sample_financial_data):
        """Test sequence preparation with scaler fitting."""
        processor = LSTMDataProcessor(mock_config)

        X_seq, y_categorical = processor.prepare_sequences(
            sample_financial_data, fit_scalers=True
        )

        # Verify output shapes
        expected_samples = len(sample_financial_data) - mock_config.sequence_length
        expected_features = len(sample_financial_data.columns) - len(
            mock_config.non_feature_cols
        )

        assert X_seq.shape == (
            expected_samples,
            mock_config.sequence_length,
            expected_features,
        )
        assert y_categorical.shape[0] == expected_samples
        assert y_categorical.shape[1] == len(
            np.unique(sample_financial_data[mock_config.target_column])
        )

        # Verify one-hot encoding
        assert np.allclose(y_categorical.sum(axis=1), 1)  # Each row sums to 1

    def test_prepare_sequences_no_fit_scalers(self, mock_config, sample_financial_data):
        """Test sequence preparation without fitting scalers."""
        processor = LSTMDataProcessor(mock_config)

        # First fit the scalers
        processor.prepare_sequences(sample_financial_data, fit_scalers=True)

        # Then use them without fitting
        X_seq, y_categorical = processor.prepare_sequences(
            sample_financial_data, fit_scalers=False
        )

        # Should work without errors
        assert X_seq.shape[0] > 0
        assert y_categorical.shape[0] > 0

    def test_prepare_sequences_custom_label_encoder(
        self, mock_config, sample_financial_data
    ):
        """Test sequence preparation with custom label encoder."""
        processor = LSTMDataProcessor(mock_config)

        # Create custom label encoder
        custom_encoder = LabelEncoder()
        custom_encoder.fit(["custom_normal", "custom_anomaly"])

        # Modify data to use custom labels
        custom_data = sample_financial_data.copy()
        custom_data[mock_config.target_column] = ["custom_normal"] * 50 + [
            "custom_anomaly"
        ] * 50

        X_seq, y_categorical = processor.prepare_sequences(
            custom_data, fit_scalers=True, label_encoder=custom_encoder
        )

        # Verify custom encoder was used
        assert y_categorical.shape[1] == 2  # Two custom classes

    @patch("src.project.data.preprocessing.data_processor.logger")
    def test_prepare_sequences_error_handling(self, mock_logger, mock_config):
        """Test error handling in sequence preparation."""
        processor = LSTMDataProcessor(mock_config)

        # Create invalid data (empty DataFrame)
        invalid_data = pd.DataFrame()

        with pytest.raises(Exception):
            processor.prepare_sequences(invalid_data, fit_scalers=True)

        mock_logger.error.assert_called_once()

    def test_create_sequences(self, mock_config):
        """Test sequence creation from time series data."""
        processor = LSTMDataProcessor(mock_config)

        # Create sample data
        X = np.random.random((100, 5))  # 100 timesteps, 5 features
        y = np.random.randint(0, 3, 100)  # 3 classes

        X_seq, y_seq = processor._create_sequences(X, y)

        # Verify shapes
        expected_samples = 100 - mock_config.sequence_length
        assert X_seq.shape == (expected_samples, mock_config.sequence_length, 5)
        assert y_seq.shape == (expected_samples,)

        # Verify sequence integrity
        assert np.array_equal(X_seq[0], X[0 : mock_config.sequence_length])
        assert y_seq[0] == y[mock_config.sequence_length]

    def test_split_temporal_validation(self, mock_config, sample_financial_data):
        """Test temporal validation split."""
        processor = LSTMDataProcessor(mock_config)

        df_early, df_late = processor.split_temporal_validation(sample_financial_data)

        # Verify split ratio
        expected_split = int(len(sample_financial_data) * mock_config.val_ratio)
        assert len(df_early) == expected_split
        assert len(df_late) == len(sample_financial_data) - expected_split

        # Verify temporal order
        assert df_early.index.max() <= df_late.index.min()


class TestTwoStageLSTMDataProcessor:
    """Test suite for TwoStageLSTMDataProcessor class."""

    def test_initialization(self, mock_config):
        """Test proper initialization of TwoStageLSTMDataProcessor."""
        processor = TwoStageLSTMDataProcessor(mock_config)

        assert processor.config == mock_config
        assert isinstance(processor.scaler, MinMaxScaler)
        assert isinstance(processor.label_encoder, LabelEncoder)

    def test_prepare_binary_sequences(self, mock_config, sample_financial_data):
        """Test binary sequence preparation for Stage 1."""
        processor = TwoStageLSTMDataProcessor(mock_config)

        X_seq, y_binary_cat = processor.prepare_binary_sequences(
            sample_financial_data, fit_scalers=True
        )

        # Verify output shapes
        expected_samples = len(sample_financial_data) - mock_config.sequence_length
        assert X_seq.shape[0] == expected_samples
        assert y_binary_cat.shape == (expected_samples, 2)  # Binary classification

        # Verify binary encoding
        assert np.allclose(y_binary_cat.sum(axis=1), 1)  # Each row sums to 1

        # Check that normal class is encoded as 0, others as 1
        original_labels = sample_financial_data[mock_config.target_column].values[
            mock_config.sequence_length :
        ]
        normal_mask = original_labels == "normal"

        # Verify binary mapping
        binary_predictions = np.argmax(y_binary_cat, axis=1)
        assert np.all(binary_predictions[normal_mask] == 0)  # Normal -> 0
        assert np.all(binary_predictions[~normal_mask] == 1)  # Anomaly -> 1

    def test_prepare_binary_sequences_no_fit(self, mock_config, sample_financial_data):
        """Test binary sequence preparation without fitting scalers."""
        processor = TwoStageLSTMDataProcessor(mock_config)

        # First prepare with fitting
        processor.prepare_binary_sequences(sample_financial_data, fit_scalers=True)

        # Then prepare without fitting
        X_seq, y_binary_cat = processor.prepare_binary_sequences(
            sample_financial_data, fit_scalers=False
        )

        # Should work without errors
        assert X_seq.shape[0] > 0
        assert y_binary_cat.shape[1] == 2

    def test_prepare_anomaly_only_sequences(self, mock_config, sample_financial_data):
        """Test anomaly-only sequence preparation for Stage 2."""
        processor = TwoStageLSTMDataProcessor(mock_config)

        # First prepare binary sequences to fit the main label encoder
        processor.prepare_binary_sequences(sample_financial_data, fit_scalers=True)

        X_seq, y_seq = processor.prepare_anomaly_only_sequences(sample_financial_data)

        # Verify that only anomaly samples are included
        anomaly_count = len(
            sample_financial_data[
                sample_financial_data[mock_config.target_column] != "normal"
            ]
        )
        expected_samples = anomaly_count - mock_config.sequence_length

        if expected_samples > 0:
            assert X_seq.shape[0] == expected_samples
            assert hasattr(processor, "anomaly_label_encoder")

            # Verify anomaly classes
            expected_classes = set(
                sample_financial_data[mock_config.target_column].unique()
            ) - {"normal"}
            actual_classes = set(processor.anomaly_label_encoder.classes_)
            assert actual_classes == expected_classes

    def test_prepare_anomaly_only_sequences_no_anomalies(self, mock_config):
        """Test anomaly-only sequence preparation with no anomaly samples."""
        processor = TwoStageLSTMDataProcessor(mock_config)

        # Create data with only normal samples
        normal_only_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=50, freq="1min").strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "feature1": np.random.random(50),
                "feature2": np.random.random(50),
                "event_type": ["normal"] * 50,
            }
        )

        with pytest.raises(ValueError, match="No anomaly samples found for Stage 2"):
            processor.prepare_anomaly_only_sequences(normal_only_data)

    # FIXED: Corrected logging test
    def test_prepare_anomaly_only_sequences_logging(
        self, mock_config, sample_financial_data
    ):
        """Test logging in anomaly-only sequence preparation."""
        processor = TwoStageLSTMDataProcessor(mock_config)

        # First prepare binary sequences
        processor.prepare_binary_sequences(sample_financial_data, fit_scalers=True)

        with patch("src.project.data.preprocessing.data_processor.logger") as mock_logger:
            processor.prepare_anomaly_only_sequences(sample_financial_data)

            # Check that the expected logging calls were made
            # The actual log message is: "Stage 2 data: {len(df_anomaly)} anomaly samples from {len(df)} total"
            anomaly_count = len(
                sample_financial_data[sample_financial_data["event_type"] != "normal"]
            )
            expected_message = f"Stage 2 data: {anomaly_count} anomaly samples from {len(sample_financial_data)} total"

            mock_logger.info.assert_any_call(expected_message)


class TestIntegration:
    """Integration tests for the complete data processing pipeline."""

    def test_financial_data_realistic_scenario(self, mock_config):
        """Test with realistic financial market data scenario."""
        # Create realistic financial data
        dates = pd.date_range("2024-01-01 09:30:00", periods=1000, freq="1min")

        # Simulate market events
        market_events = []
        for i in range(1000):
            if i < 800:
                market_events.append("normal")
            elif i < 900:
                market_events.append("dip")
            elif i < 950:
                market_events.append("rally")
            else:
                market_events.append("crash")

        financial_data = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": np.random.uniform(150, 200, 1000),
                "High": np.random.uniform(180, 220, 1000),
                "Low": np.random.uniform(140, 180, 1000),
                "Close": np.random.uniform(150, 200, 1000),
                "Volume": np.random.randint(50000, 500000, 1000),
                "RSI_14": np.random.uniform(30, 70, 1000),
                "MACD": np.random.uniform(-1, 1, 1000),
                "ATR_14": np.random.uniform(1, 5, 1000),
                "event_type": market_events,
            }
        )

        processor = TwoStageLSTMDataProcessor(mock_config)

        # Test complete pipeline
        X_binary, y_binary = processor.prepare_binary_sequences(
            financial_data, fit_scalers=True
        )
        X_anomaly, y_anomaly = processor.prepare_anomaly_only_sequences(financial_data)

        # Verify realistic proportions
        normal_proportion = np.mean(np.argmax(y_binary, axis=1) == 0)
        assert 0.7 < normal_proportion < 0.9  # Most samples should be normal

        # Verify anomaly classes
        assert len(processor.anomaly_label_encoder.classes_) == 3  # dip, rally, crash


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_sequence_length(self, mock_config, sample_financial_data):
        """Test with minimal data that just meets sequence length requirements."""
        mock_config.sequence_length = 5

        # Create minimal dataset
        minimal_data = sample_financial_data.head(10)  # Just enough for 5 sequences

        processor = LSTMDataProcessor(mock_config)
        X_seq, y_categorical = processor.prepare_sequences(
            minimal_data, fit_scalers=True
        )

        # Should create exactly 5 sequences
        assert X_seq.shape[0] == 5
        assert y_categorical.shape[0] == 5

    # FIXED: Test actually handles what happens with insufficient data
    def test_insufficient_data_for_sequences(self, mock_config, sample_financial_data):
        """Test with insufficient data for sequence creation."""
        mock_config.sequence_length = 200  # More than available data

        processor = LSTMDataProcessor(mock_config)

        # Based on the _create_sequences implementation, when sequence_length > data length,
        # the range(sequence_length, len(X)) will be empty, resulting in empty arrays
        X_seq, y_categorical = processor.prepare_sequences(
            sample_financial_data, fit_scalers=True
        )

        # Should create empty sequences when there's insufficient data
        assert X_seq.shape[0] == 0
        assert y_categorical.shape[0] == 0

    def test_single_class_data(self, mock_config):
        """Test with data containing only one class."""
        single_class_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=50, freq="1min").strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "feature1": np.random.random(50),
                "feature2": np.random.random(50),
                "event_type": ["normal"] * 50,
            }
        )

        processor = LSTMDataProcessor(mock_config)
        X_seq, y_categorical = processor.prepare_sequences(
            single_class_data, fit_scalers=True
        )

        # Should handle single class scenario
        assert y_categorical.shape[1] == 1  # Only one class
        assert np.all(y_categorical == 1)  # All samples belong to the single class

    def test_empty_dataframe(self, mock_config):
        """Test with empty DataFrame."""
        empty_data = pd.DataFrame()

        processor = LSTMDataProcessor(mock_config)

        with pytest.raises(Exception):
            processor.prepare_sequences(empty_data, fit_scalers=True)

    def test_missing_target_column(self, mock_config, sample_financial_data):
        """Test with missing target column."""
        data_no_target = sample_financial_data.drop(columns=[mock_config.target_column])

        processor = LSTMDataProcessor(mock_config)

        with pytest.raises(KeyError):
            processor.prepare_sequences(data_no_target, fit_scalers=True)

    def test_non_numeric_features(self, mock_config):
        """Test handling of non-numeric features."""
        mixed_data = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=50, freq="1min").strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "numeric_feature": np.random.random(50),
                "string_feature": ["text"] * 50,  # Non-numeric feature
                "event_type": ["normal"] * 30 + ["dip"] * 20,
            }
        )

        processor = LSTMDataProcessor(mock_config)

        # Should handle or raise appropriate error for non-numeric data
        with pytest.raises(Exception):
            processor.prepare_sequences(mixed_data, fit_scalers=True)
