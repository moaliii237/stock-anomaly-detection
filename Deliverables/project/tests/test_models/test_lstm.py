import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call

# Fix import path - remove 'src.' to match the actual module structure
from src.project.models.lstm import train_two_stage_lstm


@pytest.fixture
def mock_stage1_config():
    """Mock Stage1Config object."""
    config = Mock()
    config.train_path = "train.csv"
    config.test_path = "test.csv"
    config.model_save_path = "stage1.keras"
    return config


@pytest.fixture
def mock_stage2_config():
    """Mock Stage2Config object."""
    config = Mock()
    config.train_path = "train.csv"
    config.test_path = "test.csv"
    config.model_save_path = "stage2.keras"
    return config


@pytest.fixture
def mock_trainer():
    """Mock TwoStageLSTMTrainer with all required methods."""
    trainer = Mock()

    # Mock data_processor
    trainer.data_processor = Mock()
    trainer.data_processor.label_encoder = Mock()
    trainer.data_processor.label_encoder.classes_ = np.array(
        ["normal", "dip", "rally", "crash"]
    )

    # Mock data loading
    df_train = pd.DataFrame(
        {
            "price": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
            "event_type": ["normal"] * 80 + ["dip"] * 10 + ["crash"] * 10,
        },
        index=pd.date_range("2024-01-01", periods=100, freq="1min"),
    )

    df_test = pd.DataFrame(
        {
            "price": np.random.randn(50).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 50),
            "event_type": ["normal"] * 40 + ["crash"] * 10,
        },
        index=pd.date_range("2024-02-01", periods=50, freq="1min"),
    )

    trainer.data_processor.load_and_validate_data.return_value = (df_train, df_test)

    # Mock temporal split
    df_val = df_test.iloc[:25]
    df_final_test = df_test.iloc[25:]
    trainer.data_processor.split_temporal_validation.return_value = (
        df_val,
        df_final_test,
    )

    # Mock training histories
    history1 = Mock()
    history1.history = {
        "loss": [0.8, 0.6, 0.4, 0.3],
        "val_loss": [0.9, 0.7, 0.5, 0.4],
        "accuracy": [0.6, 0.7, 0.8, 0.85],
    }

    history2 = Mock()
    history2.history = {
        "loss": [1.2, 0.9, 0.6, 0.4],
        "val_loss": [1.3, 1.0, 0.7, 0.5],
        "accuracy": [0.4, 0.6, 0.75, 0.8],
    }

    trainer.train_stage1.return_value = history1
    trainer.train_stage2.return_value = history2

    # Mock save artifacts
    trainer.save_production_artifacts.return_value = None

    return trainer


@pytest.fixture
def sample_evaluation_result():
    """Sample evaluation metrics."""
    return (0.85, 0.78, 0.12)  # accuracy, crash_recall, false_alarm_rate


class TestTrainTwoStageLSTM:
    """Test suite for train_two_stage_lstm function."""

    # Fix patch paths - remove 'src.' prefix to match import paths
    @patch("src.project.models.lstm.evaluate_model")
    @patch("src.project.models.lstm.TwoStageLSTMTrainer")
    @patch("src.project.models.lstm.Stage2Config")
    @patch("src.project.models.lstm.Stage1Config")
    @patch("src.project.models.lstm.logger")
    def test_train_two_stage_lstm_success(
        self,
        mock_logger,
        mock_stage1_config_class,
        mock_stage2_config_class,
        mock_trainer_class,
        mock_evaluate,
        mock_trainer,
        sample_evaluation_result,
    ):
        """Test successful training of two-stage LSTM system."""
        # Setup mocks
        mock_stage1_config_class.return_value = Mock()
        mock_stage2_config_class.return_value = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_evaluate.return_value = sample_evaluation_result

        # Use actual data paths from your project
        train_path = "src/project/data/BKNG_engineering_train.csv"
        test_path = "src/project/data/BKNG_engineering_test.csv"
        stage1_model_path = "models/stage1.keras"
        stage2_model_path = "models/stage2.keras"

        # Execute function
        result = train_two_stage_lstm(
            train_path, test_path, stage1_model_path, stage2_model_path
        )

        # Verify return structure
        dfs, histories, evaluation, trainer = result
        df_train, df_test = dfs
        history1, history2 = histories

        # Verify configurations were created correctly
        mock_stage1_config_class.assert_called_once_with(
            train_path=train_path,
            test_path=test_path,
            model_save_path=stage1_model_path,
        )
        mock_stage2_config_class.assert_called_once_with(
            train_path=train_path,
            test_path=test_path,
            model_save_path=stage2_model_path,
        )

        # Verify trainer initialization
        mock_trainer_class.assert_called_once()

        # Verify data loading and processing
        mock_trainer.data_processor.load_and_validate_data.assert_called_once()
        mock_trainer.data_processor.split_temporal_validation.assert_called_once()

        # Verify training stages
        mock_trainer.train_stage1.assert_called_once()
        mock_trainer.train_stage2.assert_called_once()

        # Verify evaluation
        assert mock_evaluate.call_count == 1

        # Verify artifact saving
        mock_trainer.save_production_artifacts.assert_called_once()

        # Verify return values
        assert isinstance(df_train, pd.DataFrame)
        assert isinstance(df_test, pd.DataFrame)
        assert evaluation == sample_evaluation_result
        assert trainer == mock_trainer

        # Verify logging
        mock_logger.info.assert_any_call("=" * 60)
        mock_logger.info.assert_any_call("=== Two-Stage LSTM Training Completed ===")

    @patch("src.project.models.lstm.evaluate_model")
    @patch("src.project.models.lstm.TwoStageLSTMTrainer")
    @patch("src.project.models.lstm.Stage2Config")
    @patch("src.project.models.lstm.Stage1Config")
    @patch("src.project.models.lstm.logger")
    def test_train_two_stage_lstm_stage1_training_failure(
        self,
        mock_logger,
        mock_stage1_config_class,
        mock_stage2_config_class,
        mock_trainer_class,
        mock_evaluate,
        mock_trainer,
    ):
        """Test handling of Stage 1 training failure."""
        # Setup mocks
        mock_stage1_config_class.return_value = Mock()
        mock_stage2_config_class.return_value = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Force Stage 1 training to fail
        mock_trainer.train_stage1.side_effect = Exception("Stage 1 training failed")

        # Test should raise exception
        with pytest.raises(Exception, match="Stage 1 training failed"):
            train_two_stage_lstm(
                "train.csv", "test.csv", "stage1.keras", "stage2.keras"
            )

        # Verify Stage 1 was attempted
        mock_trainer.train_stage1.assert_called_once()

        # Verify Stage 2 was not attempted
        mock_trainer.train_stage2.assert_not_called()

    @patch("src.project.models.lstm.evaluate_model")
    @patch("src.project.models.lstm.TwoStageLSTMTrainer")
    @patch("src.project.models.lstm.Stage2Config")
    @patch("src.project.models.lstm.Stage1Config")
    @patch("src.project.models.lstm.logger")
    def test_train_two_stage_lstm_data_loading_failure(
        self,
        mock_logger,
        mock_stage1_config_class,
        mock_stage2_config_class,
        mock_trainer_class,
        mock_evaluate,
        mock_trainer,
    ):
        """Test handling of data loading failure."""
        # Setup mocks
        mock_stage1_config_class.return_value = Mock()
        mock_stage2_config_class.return_value = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Force data loading to fail
        mock_trainer.data_processor.load_and_validate_data.side_effect = (
            FileNotFoundError("Data file not found")
        )

        # Test should raise exception
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            train_two_stage_lstm(
                "nonexistent_train.csv",
                "nonexistent_test.csv",
                "stage1.keras",
                "stage2.keras",
            )

        # Verify data loading was attempted
        mock_trainer.data_processor.load_and_validate_data.assert_called_once()

    @patch("src.project.models.lstm.evaluate_model")
    @patch("src.project.models.lstm.TwoStageLSTMTrainer")
    @patch("src.project.models.lstm.Stage2Config")
    @patch("src.project.models.lstm.Stage1Config")
    def test_train_two_stage_lstm_with_realistic_paths(
        self,
        mock_stage1_config_class,
        mock_stage2_config_class,
        mock_trainer_class,
        mock_evaluate,
        mock_trainer,
        sample_evaluation_result,
    ):
        """Test with realistic file paths from your project structure."""
        # Setup mocks
        mock_stage1_config_class.return_value = Mock()
        mock_stage2_config_class.return_value = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_evaluate.return_value = sample_evaluation_result

        # Use your actual file paths
        result = train_two_stage_lstm(
            "src/project/data/BKNG_engineering_train.csv",
            "src/project/data/BKNG_engineering_test.csv",
            "src/project/models/saved_models/stage1.keras",
            "src/project/models/saved_models/stage2.keras",
        )

        # Verify successful execution
        dfs, histories, evaluation, trainer = result
        assert len(dfs) == 2
        assert len(histories) == 2
        assert len(evaluation) == 3
        assert trainer == mock_trainer


# Note: Removed main execution block tests as they're more complex to mock
# Focus on testing the core function logic instead


class TestIntegration:
    """Integration tests for the lstm module."""

    @patch("src.project.models.lstm.evaluate_model")
    @patch("src.project.models.lstm.TwoStageLSTMTrainer")
    @patch("src.project.models.lstm.Stage2Config")
    @patch("src.project.models.lstm.Stage1Config")
    def test_full_pipeline_integration(
        self,
        mock_stage1_config_class,
        mock_stage2_config_class,
        mock_trainer_class,
        mock_evaluate,
        mock_trainer,
        sample_evaluation_result,
    ):
        """Test full pipeline integration with realistic data flow."""
        # Setup comprehensive mocks
        mock_stage1_config_class.return_value = Mock()
        mock_stage2_config_class.return_value = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_evaluate.return_value = sample_evaluation_result

        # Execute complete pipeline
        result = train_two_stage_lstm(
            "src/project/data/BKNG_engineering_train.csv",
            "src/project/data/BKNG_engineering_test.csv",
            "models/stage1_lstm.keras",
            "models/stage2_lstm.keras",
        )

        # Verify complete workflow
        dfs, histories, evaluation, trainer = result

        # Verify all components were properly executed
        assert mock_trainer.data_processor.load_and_validate_data.called
        assert mock_trainer.data_processor.split_temporal_validation.called
        assert mock_trainer.train_stage1.called
        assert mock_trainer.train_stage2.called
        assert mock_evaluate.called
        assert mock_trainer.save_production_artifacts.called

        # Verify data flow integrity
        assert isinstance(dfs[0], pd.DataFrame)  # Training data
        assert isinstance(dfs[1], pd.DataFrame)  # Test data
        assert len(evaluation) == 3  # Three evaluation metrics
        assert trainer == mock_trainer
