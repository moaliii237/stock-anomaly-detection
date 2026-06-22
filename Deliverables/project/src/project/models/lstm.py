import logging
from typing import Tuple, Any

import pandas as pd

from src.project.config.model_config import Stage1Config, Stage2Config
from src.project.models.evaluate import evaluate_model
from src.project.models.two_stage_classifier import TwoStageLSTMTrainer

logger = logging.getLogger(__name__)


def train_two_stage_lstm(
    train_path: str, test_path: str, stage1_model_path: str, stage2_model_path: str, stage1_config: Stage1Config = None, stage2_config: Stage2Config = None
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[Any, Any],
    Tuple[float, float, float],
    TwoStageLSTMTrainer,
]:
    """Train a complete two-stage LSTM system for financial anomaly detection.

    This function orchestrates the entire training pipeline for a two-stage LSTM
    anomaly detection system designed for financial time series data. The system
    uses a hierarchical approach where the first stage performs binary classification
    (normal vs anomaly) and the second stage classifies specific anomaly types.

    The two-stage approach is particularly effective for imbalanced datasets common
    in financial markets, where normal conditions significantly outnumber anomalous
    events. By separating the detection and classification tasks, each model can
    be optimized for its specific purpose.

    Training Pipeline:
    1. Initialize configurations for both stages
    2. Load and validate training and test datasets
    3. Split test data into validation and final test sets (temporal split)
    4. Train Stage 1: Binary anomaly detection model
    5. Train Stage 2: Multi-class anomaly classification model
    6. Evaluate the complete system on final test set
    7. Save production-ready artifacts

    Args:
        train_path: Path to the training dataset CSV file. Must contain datetime-indexed
                   time series data with features and target labels.
        test_path: Path to the test dataset CSV file. Should have the same structure
                  as the training data for proper validation.
        stage1_model_path: File path where the binary classification model will be saved.
                          Should use .keras extension for TensorFlow/Keras models.
        stage2_model_path: File path where the anomaly classification model will be saved.
                          Should use .keras extension for TensorFlow/Keras models.

    Returns:
        Tuple containing four elements:
            - (df_train, df_test): Tuple of loaded training and test DataFrames
            - (history1, history2): Tuple of training histories for both stages
            - evaluation: Tuple of (accuracy, crash_recall, false_alarm_rate) metrics
            - trainer: Trained TwoStageLSTMTrainer instance with fitted models

    Raises:
        FileNotFoundError: If training or test data files cannot be found.
        ValueError: If data validation fails or datasets are incompatible.
        Exception: If training fails for either stage or evaluation encounters errors.

    Note:
        The function automatically saves production artifacts including:
        - Trained models (stage1_lstm.keras, stage2_lstm.keras)
        - Preprocessing components (scalers, encoders)
        - Configuration objects
        - Model metadata

    Example:
        >>> dfs, histories, metrics, trainer = train_two_stage_lstm(
        ...     "data/train.csv",
        ...     "data/test.csv",
        ...     "models/stage1.keras",
        ...     "models/stage2.keras"
        ... )
        >>> accuracy, crash_recall, false_alarms = metrics
        >>> print(f"System accuracy: {accuracy:.1%}")
    """
    # Initialize configurations
    if stage1_config is None:
        stage1_config = Stage1Config(
            train_path=train_path, test_path=test_path, model_save_path=stage1_model_path
        )

    if stage2_config is None:
        stage2_config = Stage2Config(
            train_path=train_path, test_path=test_path, model_save_path=stage2_model_path
        )

    # Load data
    trainer = TwoStageLSTMTrainer(stage1_config, stage2_config)
    df_train, df_test = trainer.data_processor.load_and_validate_data()
    df_val, df_final_test = trainer.data_processor.split_temporal_validation(df_test)

    # Train both stages
    history1 = trainer.train_stage1(df_train, df_val)
    history2 = trainer.train_stage2(df_train, df_val)

    evaluation = evaluate_model(
        trainer,
        df_final_test,
        labels_mapping=trainer.data_processor.label_encoder.classes_,
    )

    trainer.save_production_artifacts()

    # Log final results
    logger.info("=" * 60)
    logger.info("=== Two-Stage LSTM Training Completed ===")
    logger.info(f"Stage 1 Training History: {history1.history}")
    logger.info(f"Stage 2 Training History: {history2.history}")
    logger.info(f"Stage 1 Model saved to: {stage1_model_path}")
    logger.info(f"Stage 2 Model saved to: {stage2_model_path}")
    logger.info(f"Evaluation Results: {evaluation}")
    logger.info("=== Training Completed ===")
    logger.info("=" * 60)

    return (df_train, df_test), (history1, history2), evaluation, trainer


if __name__ == "__main__":
    """Main execution block for standalone training script.

    When executed as a standalone script, this block demonstrates how to train
    a two-stage LSTM system using example file paths. The example is configured
    for financial market data (BKNG - Booking Holdings) with typical financial
    anomaly detection scenarios.

    The script performs a complete training cycle and evaluates the system twice:
    once during training (on validation set) and once on the final test set to
    ensure robust performance assessment.

    File Structure Expected:
    - ../data/BKNG_engineering_train.csv: Training dataset
    - ../data/BKNG_engineering_test.csv: Test dataset
    - saved_models/: Directory for saving trained models

    Example Output:
        Train Data Shape: (8000, 15)
        Test Data Shape: (2000, 15)
        Overall Accuracy: 85.2%
        Crash Detection Rate: 78.9%
        False Alarm Rate: 7.3%
        SYSTEM READY
    """
    # Example usage
    train_path = "../data/BKNG_engineering_train.csv"
    test_path = "../data/BKNG_engineering_test.csv"
    stage1_model_path = "saved_models/stage1_lstm.keras"
    stage2_model_path = "saved_models/stage2_lstm.keras"

    dfs, histories, evaluation, trainer = train_two_stage_lstm(
        train_path, test_path, stage1_model_path, stage2_model_path
    )

    df_train, df_test = dfs
    history1, history2 = histories
    logger.info(f"Train Data Shape: {df_train.shape}")
    logger.info(f"Test Data Shape: {df_test.shape}")

    evaluation = evaluate_model(
        trainer, df_test, labels_mapping=trainer.data_processor.label_encoder.classes_
    )
