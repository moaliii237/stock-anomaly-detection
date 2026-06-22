import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.project.models.evaluate import evaluate_model, make_two_stage_predictions


@pytest.fixture
def mock_trainer():
    """Create a comprehensive mock trainer for testing."""
    trainer = Mock()

    # Mock stage1_model (binary classification)
    trainer.stage1_model = Mock()
    trainer.stage1_model.predict.return_value = np.array(
        [
            [0.8, 0.2],  # normal
            [0.3, 0.7],  # anomaly
            [0.9, 0.1],  # normal
            [0.2, 0.8],  # anomaly
            [0.7, 0.3],  # normal
        ]
    )

    # Mock stage2_model (anomaly type classification)
    trainer.stage2_model = Mock()
    trainer.stage2_model.predict.return_value = np.array(
        [[0.1, 0.8, 0.1], [0.2, 0.1, 0.7]]  # dip (index 1)  # crash (index 2)
    )

    # Mock data_processor
    trainer.data_processor = Mock()
    trainer.data_processor.label_encoder = Mock()
    trainer.data_processor.label_encoder.classes_ = np.array(
        ["normal", "dip", "rally", "crash"]
    )
    trainer.data_processor.prepare_sequences.return_value = (
        np.random.random((5, 10, 5)),  # X_test
        np.array(
            [  # y_test_original (one-hot encoded)
                [1, 0, 0, 0],  # normal
                [0, 1, 0, 0],  # dip
                [1, 0, 0, 0],  # normal
                [0, 0, 0, 1],  # crash
                [1, 0, 0, 0],  # normal
            ]
        ),
    )

    return trainer


@pytest.fixture
def sample_test_data():
    """Sample test DataFrame."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1min")
    df = pd.DataFrame(
        {
            "price": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
        },
        index=dates,
    )
    return df


class TestEvaluateModel:
    """Test suite for evaluate_model function."""

    @patch("src.project.models.evaluate.make_two_stage_predictions")
    @patch("src.project.models.evaluate.accuracy_score")
    @patch("src.project.models.evaluate.recall_score")
    @patch("src.project.models.evaluate.logger")
    def test_evaluate_model_success_case(
        self,
        mock_logger,
        mock_recall,
        mock_accuracy,
        mock_predictions,
        mock_trainer,
        sample_test_data,
    ):
        """Test successful evaluation with all metrics above thresholds."""
        # Setup mocks
        mock_predictions.return_value = (
            np.array([0, 1, 0, 3, 0]),  # final predictions
            np.array([0.2, 0.7, 0.1, 0.8, 0.3]),  # confidence
        )
        mock_accuracy.return_value = 0.85  # Above 0.8 threshold
        mock_recall.return_value = np.array([0.75])  # Above 0.7 threshold

        labels_mapping = ["normal", "dip", "rally", "crash"]

        result = evaluate_model(mock_trainer, sample_test_data, labels_mapping)

        # Verify results
        accuracy, crash_recall, false_alarm_rate = result
        assert accuracy == 0.85
        assert crash_recall == 0.75
        assert isinstance(false_alarm_rate, float)

        # Verify success message
        mock_logger.info.assert_any_call("SYSTEM READY")

    @patch("src.project.models.evaluate.make_two_stage_predictions")
    @patch("src.project.models.evaluate.accuracy_score")
    @patch("src.project.models.evaluate.recall_score")
    @patch("src.project.models.evaluate.logger")
    def test_evaluate_model_failure_case(
        self,
        mock_logger,
        mock_recall,
        mock_accuracy,
        mock_predictions,
        mock_trainer,
        sample_test_data,
    ):
        """Test evaluation with metrics below thresholds."""
        mock_predictions.return_value = (
            np.array([0, 0, 0, 0, 0]),  # all normal predictions
            np.array([0.1, 0.2, 0.1, 0.3, 0.2]),
        )
        mock_accuracy.return_value = 0.65  # Below 0.8 threshold
        mock_recall.return_value = np.array([0.5])  # Below 0.7 threshold

        result = evaluate_model(mock_trainer, sample_test_data, ["normal", "crash"])

        accuracy, crash_recall, false_alarm_rate = result
        assert accuracy == 0.65
        assert crash_recall == 0.5

        # Verify failure message
        mock_logger.warning.assert_any_call("NEEDS IMPROVEMENT")

    @patch("src.project.models.evaluate.make_two_stage_predictions")
    @patch("src.project.models.evaluate.accuracy_score")
    @patch("src.project.models.evaluate.recall_score")
    def test_evaluate_model_dict_labels_mapping(
        self,
        mock_recall,
        mock_accuracy,
        mock_predictions,
        mock_trainer,
        sample_test_data,
    ):
        """Test with dictionary labels mapping containing strings."""
        mock_predictions.return_value = (
            np.array([0, 1, 0, 2, 0]),
            np.array([0.2, 0.7, 0.1, 0.8, 0.3]),
        )
        mock_accuracy.return_value = 0.8
        mock_recall.return_value = np.array([0.7])

        labels_mapping = {0: "normal", 1: "dip", 2: "crash"}

        result = evaluate_model(mock_trainer, sample_test_data, labels_mapping)

        accuracy, crash_recall, false_alarm_rate = result
        assert accuracy == 0.8
        assert crash_recall == 0.7

    @patch("src.project.models.evaluate.make_two_stage_predictions")
    @patch("src.project.models.evaluate.accuracy_score")
    @patch("src.project.models.evaluate.recall_score")
    def test_evaluate_model_dict_numeric_labels(
        self,
        mock_recall,
        mock_accuracy,
        mock_predictions,
        mock_trainer,
        sample_test_data,
    ):
        """Test with dictionary labels mapping containing numeric values."""
        mock_predictions.return_value = (
            np.array([0, 1, 0, 2, 0]),
            np.array([0.2, 0.7, 0.1, 0.8, 0.3]),
        )
        mock_accuracy.return_value = 0.8
        mock_recall.return_value = np.array([0.7])

        labels_mapping = {0: 0, 1: 1, 2: 2}  # Numeric mapping

        result = evaluate_model(mock_trainer, sample_test_data, labels_mapping)

        accuracy, crash_recall, false_alarm_rate = result
        assert accuracy == 0.8
        assert crash_recall == 0.7

    @patch("src.project.models.evaluate.make_two_stage_predictions")
    @patch("src.project.models.evaluate.accuracy_score")
    @patch("src.project.models.evaluate.recall_score")
    def test_evaluate_model_no_crash_label(
        self,
        mock_recall,
        mock_accuracy,
        mock_predictions,
        mock_trainer,
        sample_test_data,
    ):
        """Test when 'crash' label is not present in class names."""
        mock_predictions.return_value = (
            np.array([0, 1, 0, 1, 0]),
            np.array([0.2, 0.7, 0.1, 0.8, 0.3]),
        )
        mock_accuracy.return_value = 0.8

        # Mock label encoder without crash
        mock_trainer.data_processor.label_encoder.classes_ = np.array(["normal", "dip"])

        result = evaluate_model(mock_trainer, sample_test_data, None)

        accuracy, crash_recall, false_alarm_rate = result
        assert accuracy == 0.8
        assert crash_recall == 0.0  # No crash label available

    @patch("src.project.models.evaluate.make_two_stage_predictions")
    def test_evaluate_model_exception_handling(
        self, mock_predictions, mock_trainer, sample_test_data
    ):
        """Test exception handling in evaluate_model."""
        # Force an exception
        mock_predictions.side_effect = Exception("Test exception")

        with patch("src.project.models.evaluate.logger") as mock_logger:
            result = evaluate_model(mock_trainer, sample_test_data, None)

        # Should return sentinel values
        assert result == (0.0, 0.0, 1.0)
        mock_logger.error.assert_called()

    @patch("src.project.models.evaluate.make_two_stage_predictions")
    @patch("src.project.models.evaluate.accuracy_score")
    @patch("src.project.models.evaluate.recall_score")
    def test_evaluate_model_zero_normal_samples(
        self,
        mock_recall,
        mock_accuracy,
        mock_predictions,
        mock_trainer,
        sample_test_data,
    ):
        """Test false alarm rate calculation with no normal samples."""
        mock_predictions.return_value = (
            np.array([1, 2, 1, 2, 1]),  # No normal (0) predictions
            np.array([0.7, 0.8, 0.7, 0.8, 0.7]),
        )
        mock_accuracy.return_value = 0.8
        mock_recall.return_value = np.array([0.7])

        # Mock y_true with no normal samples
        mock_trainer.data_processor.prepare_sequences.return_value = (
            np.random.random((5, 10, 5)),
            np.array(
                [  # All non-normal
                    [0, 1, 0, 0],  # dip
                    [0, 0, 0, 1],  # crash
                    [0, 1, 0, 0],  # dip
                    [0, 0, 0, 1],  # crash
                    [0, 1, 0, 0],  # dip
                ]
            ),
        )

        result = evaluate_model(mock_trainer, sample_test_data, None)

        accuracy, crash_recall, false_alarm_rate = result
        assert false_alarm_rate == 0  # Should be 0 when no normal samples


class TestMakeTwoStagePredictions:
    """Test suite for make_two_stage_predictions function."""

    def test_make_predictions_with_anomalies(self, mock_trainer):
        """Test two-stage predictions when anomalies are detected."""
        X_test = np.random.random((5, 10, 5))

        predictions, confidence = make_two_stage_predictions(
            mock_trainer, X_test, ["normal", "dip", "rally", "crash"]
        )

        # Verify shapes
        assert len(predictions) == 5
        assert len(confidence) == 5

        # Verify stage 1 was called
        mock_trainer.stage1_model.predict.assert_called_once()

        # Verify stage 2 was called (since anomalies detected)
        mock_trainer.stage2_model.predict.assert_called_once()

    def test_make_predictions_no_anomalies(self, mock_trainer):
        """Test when no anomalies are detected in stage 1."""
        # Mock stage1 to predict all normal
        mock_trainer.stage1_model.predict.return_value = np.array(
            [
                [0.9, 0.1],  # normal
                [0.8, 0.2],  # normal
                [0.7, 0.3],  # normal
            ]
        )

        X_test = np.random.random((3, 10, 5))

        predictions, confidence = make_two_stage_predictions(
            mock_trainer, X_test, ["normal", "crash"]
        )

        # All should be predicted as normal (0)
        assert all(pred == 0 for pred in predictions)

        # Stage 2 should not be called
        mock_trainer.stage2_model.predict.assert_not_called()

    def test_make_predictions_numpy_array_labels(self, mock_trainer):
        """Test with numpy array labels mapping."""
        X_test = np.random.random((5, 10, 5))
        labels_mapping = np.array(["normal", "dip", "rally", "crash"])

        predictions, confidence = make_two_stage_predictions(
            mock_trainer, X_test, labels_mapping
        )

        assert len(predictions) == 5
        assert len(confidence) == 5

    def test_make_predictions_numeric_dict_labels(self, mock_trainer):
        """Test with dictionary containing numeric labels."""
        X_test = np.random.random((5, 10, 5))
        labels_mapping = {0: 0, 1: 1, 2: 2}  # Numeric values

        predictions, confidence = make_two_stage_predictions(
            mock_trainer, X_test, labels_mapping
        )

        assert len(predictions) == 5
        assert len(confidence) == 5

    def test_make_predictions_no_labels_mapping(self, mock_trainer):
        """Test with no labels mapping provided."""
        X_test = np.random.random((5, 10, 5))

        predictions, confidence = make_two_stage_predictions(mock_trainer, X_test, None)

        assert len(predictions) == 5
        assert len(confidence) == 5
        # Should have anomaly types as 1, 2, etc. (adding 1 to stage2 outputs)

    def test_make_predictions_exception_handling(self, mock_trainer):
        """Test exception handling in make_two_stage_predictions."""
        # Force an exception in stage1 prediction
        mock_trainer.stage1_model.predict.side_effect = Exception("Test error")

        X_test = np.random.random((5, 10, 5))

        with pytest.raises(Exception):
            make_two_stage_predictions(mock_trainer, X_test, None)

    @patch("src.project.models.evaluate.logger")
    def test_make_predictions_logging(self, mock_logger, mock_trainer):
        """Test that appropriate logging messages are generated."""
        X_test = np.random.random((5, 10, 5))

        make_two_stage_predictions(mock_trainer, X_test, None)

        # Verify logging calls
        mock_logger.info.assert_any_call(
            "Stage 1 predictions: 2 anomalies out of 5 samples"
        )
        mock_logger.info.assert_any_call("Stage 2 classifying 2 detected anomalies")


class TestIntegration:
    """Integration tests for the evaluation module."""

    @patch("src.project.models.evaluate.accuracy_score")
    @patch("src.project.models.evaluate.recall_score")
    @patch("src.project.models.evaluate.logger")
    def test_full_evaluation_pipeline(
        self, mock_logger, mock_recall, mock_accuracy, mock_trainer, sample_test_data
    ):
        """Test the complete evaluation pipeline from start to finish."""
        mock_accuracy.return_value = 0.82
        mock_recall.return_value = np.array([0.75])

        labels_mapping = ["normal", "dip", "rally", "crash"]

        accuracy, crash_recall, false_alarm_rate = evaluate_model(
            mock_trainer, sample_test_data, labels_mapping
        )

        # Verify all components work together
        assert accuracy == 0.82
        assert crash_recall == 0.75
        assert isinstance(false_alarm_rate, float)
        assert 0 <= false_alarm_rate <= 1

        # Verify data processor was called
        mock_trainer.data_processor.prepare_sequences.assert_called_once()

        # Verify both models were used
        mock_trainer.stage1_model.predict.assert_called()
        mock_trainer.stage2_model.predict.assert_called()

    def test_edge_case_single_sample(self, mock_trainer):
        """Test evaluation with single sample."""
        # Modify mock for single sample
        mock_trainer.data_processor.prepare_sequences.return_value = (
            np.random.random((1, 10, 5)),
            np.array([[1, 0, 0, 0]]),  # Single normal sample
        )
        mock_trainer.stage1_model.predict.return_value = np.array([[0.9, 0.1]])

        df_single = pd.DataFrame(
            {"price": [100], "volume": [1000]}, index=[pd.Timestamp("2024-01-01")]
        )

        with (
            patch("src.project.models.evaluate.accuracy_score") as mock_acc,
            patch("src.project.models.evaluate.recall_score") as mock_rec,
        ):
            mock_acc.return_value = 1.0
            mock_rec.return_value = np.array([0.0])  # No crash samples

            result = evaluate_model(mock_trainer, df_single, None)

        accuracy, crash_recall, false_alarm_rate = result
        assert accuracy == 1.0
        assert crash_recall == 0.0
        assert isinstance(false_alarm_rate, float)

    def test_all_crash_predictions(self, mock_trainer):
        """Test scenario where all predictions are crashes."""
        # Mock to predict all as anomalies (crashes)
        mock_trainer.stage1_model.predict.return_value = np.array(
            [[0.1, 0.9], [0.2, 0.8], [0.1, 0.9]]  # anomaly  # anomaly  # anomaly
        )
        mock_trainer.stage2_model.predict.return_value = np.array(
            [
                [0.1, 0.1, 0.8],  # crash
                [0.1, 0.1, 0.8],  # crash
                [0.1, 0.1, 0.8],  # crash
            ]
        )

        X_test = np.random.random((3, 10, 5))

        predictions, confidence = make_two_stage_predictions(
            mock_trainer, X_test, ["normal", "dip", "crash"]
        )

        # All should be predicted as crashes
        assert all(pred != 0 for pred in predictions)  # Not normal
        assert len(predictions) == 3
