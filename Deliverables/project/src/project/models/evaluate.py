import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, recall_score

logger = logging.getLogger(__name__)


def evaluate_model(
    trainer, df_test, labels_mapping: Dict = None
) -> Tuple[float, float, float]:
    """Comprehensive evaluation function for two-stage LSTM anomaly detection models.

    This function evaluates a trained two-stage LSTM model on test data, computing
    key performance metrics that are critical for anomaly detection systems in
    financial markets. It handles both numeric and string label mappings and
    provides a simple pass/fail verdict based on predefined thresholds.

    The evaluation focuses on three critical metrics:
    1. Overall accuracy - General model performance across all classes
    2. Crash detection rate (recall) - Ability to detect critical events
    3. False alarm rate - Frequency of incorrectly flagging normal conditions as anomalies

    Args:
        trainer: Trained TwoStageLSTMTrainer instance containing both stage models
                and the data processor with fitted scalers and encoders.
        df_test: Test dataset as DataFrame with datetime index and all required features.
        labels_mapping: Optional mapping for label conversion. Can be:
                       - Dict mapping indices to string labels
                       - Array of string labels
                       - None to use default numeric labels

    Returns:
        Tuple[float, float, float]: A tuple containing:
            - overall_accuracy: Overall classification accuracy (0.0 to 1.0)
            - crash_recall: Recall for detecting crash events (0.0 to 1.0)
            - false_alarm_rate: Rate of false positives for normal class (0.0 to 1.0)

    Note:
        The function includes robust error handling and returns sentinel values
        (0.0, 0.0, 1.0) in case of evaluation failure, indicating worst-case
        performance metrics.

    Example:
        >>> accuracy, crash_recall, false_alarms = evaluate_model(
        ...     trainer, test_df, labels_mapping=['normal', 'dip', 'rally', 'crash']
        ... )
        >>> print(f"Accuracy: {accuracy:.1%}, Crash Recall: {crash_recall:.1%}")

    Success Criteria:
        The system is considered ready for deployment if:
        - Overall accuracy > 80%
        - Crash detection rate > 70%
        - False alarm rate < 10%
    """
    logger.info("=== SIMPLE EVALUATION ===")

    try:
        # Prepare test data
        data_processor = trainer.data_processor
        X_test, y_test_original = data_processor.prepare_sequences(
            df_test, fit_scalers=False
        )
        y_true = np.argmax(y_test_original, axis=1)

        # Make predictions
        final_predictions, _ = make_two_stage_predictions(
            trainer, X_test, labels_mapping
        )

        # THE MOST IMPORTANT METRICS
        overall_accuracy = accuracy_score(y_true, final_predictions)

        # Critical event detection - handle both dict and array labels_mapping
        class_names = data_processor.label_encoder.classes_

        # Check if labels_mapping is a dictionary and contains strings
        if labels_mapping is not None and isinstance(labels_mapping, dict):
            # Dictionary case - check if values are strings
            if isinstance(list(labels_mapping.values())[0], str):
                # Handle string labels
                if "crash" in labels_mapping.values():
                    crash_label = [
                        k for k, v in labels_mapping.items() if v == "crash"
                    ][0]
                    crash_recall = recall_score(
                        y_true,
                        final_predictions,
                        labels=[crash_label],
                        average=None,
                        zero_division=0,
                    )[0]
                else:
                    crash_recall = 0.0
            else:
                # Handle numeric labels in dictionary
                if "crash" in class_names:
                    crash_idx = np.where(class_names == "crash")[0][0]
                    crash_recall = recall_score(
                        y_true,
                        final_predictions,
                        labels=[crash_idx],
                        average=None,
                        zero_division=0,
                    )[0]
                else:
                    crash_recall = 0.0
        else:
            # Handle case where labels_mapping is None or numpy array
            if "crash" in class_names:
                crash_idx = np.where(class_names == "crash")[0][0]
                crash_recall = recall_score(
                    y_true,
                    final_predictions,
                    labels=[crash_idx],
                    average=None,
                    zero_division=0,
                )[0]
            else:
                crash_recall = 0.0

        # False alarms (normal predicted as anomaly)
        normal_idx = 0
        false_alarms = np.sum(
            (y_true == normal_idx) & (final_predictions != normal_idx)
        )
        total_normal = np.sum(y_true == normal_idx)
        false_alarm_rate = false_alarms / total_normal if total_normal > 0 else 0

        # RESULTS
        logger.info(f"Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"Crash Detection Rate: {crash_recall:.1%}")
        logger.info(f"False Alarm Rate: {false_alarm_rate:.1%}")

        # Simple verdict
        if overall_accuracy > 0.8 and crash_recall > 0.7 and false_alarm_rate < 0.1:
            logger.info("SYSTEM READY")
        else:
            logger.warning("NEEDS IMPROVEMENT")

        return overall_accuracy, crash_recall, false_alarm_rate

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}", exc_info=True)
        # Return sentinel values to indicate failure
        return 0.0, 0.0, 1.0  # safe defaults: accuracy=0, recall=0, false alarms=high


def make_two_stage_predictions(
    trainer, X_test, labels_mapping: Dict = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions using the two-stage LSTM classification approach.

    This function implements the core two-stage prediction pipeline:

    Stage 1: Binary Anomaly Detection
    - Uses the first LSTM model to classify samples as normal vs anomaly
    - Outputs probability scores for anomaly detection confidence

    Stage 2: Anomaly Type Classification
    - For samples classified as anomalies in Stage 1, uses the second LSTM model
    - Classifies the specific type of anomaly (dip, rally, crash, etc.)
    - Only processes samples that were flagged as anomalies

    This approach improves performance on imbalanced datasets by first separating
    normal from abnormal conditions, then focusing detailed classification on
    anomalies only.

    Args:
        trainer: Trained TwoStageLSTMTrainer instance containing:
                - stage1_model: Binary classification model (normal vs anomaly)
                - stage2_model: Multi-class model for anomaly types
        X_test: Test input sequences of shape (samples, sequence_length, features).
        labels_mapping: Optional mapping for converting internal predictions to
                       meaningful labels. Can be:
                       - Dict: {index: label_name}
                       - Array: [label_name_0, label_name_1, ...]
                       - None: Use numeric indices

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - final_predictions: Array of final class predictions for each sample
            - anomaly_confidence: Array of anomaly probability scores from Stage 1

    Raises:
        Exception: If there's an error during prediction process.

    Note:
        The function handles label mapping intelligently:
        - String labels are converted to appropriate integer representations
        - Normal class is always mapped to index 0
        - Anomaly types are mapped to indices 1, 2, 3, etc.

    Example:
        >>> predictions, confidence = make_two_stage_predictions(
        ...     trainer, X_test, labels_mapping=['normal', 'dip', 'rally', 'crash']
        ... )
        >>> print(f"Detected {np.sum(predictions != 0)} anomalies")
        >>> print(f"Average confidence: {np.mean(confidence):.2f}")

    Processing Flow:
        1. Stage 1 processes all samples to binary predictions (normal/anomaly)
        2. Extract samples predicted as anomalies
        3. Stage 2 processes only anomaly samples to specific anomaly types
        4. Combine results: normal=0, anomaly types=1,2,3...
        5. Apply label mapping if provided
    """
    try:
        # Stage 1: Binary anomaly detection
        stage1_pred_proba = trainer.stage1_model.predict(X_test, verbose=0)
        is_anomaly = np.argmax(stage1_pred_proba, axis=1)
        anomaly_confidence = stage1_pred_proba[:, 1]

        logger.info(
            f"Stage 1 predictions: {np.sum(is_anomaly)} anomalies out of {len(X_test)} samples"
        )

        # Initialize all predictions as normal (0)
        final_predictions = np.zeros(len(X_test), dtype=int)

        # Stage 2: Classify detected anomalies
        anomaly_indices = np.where(is_anomaly == 1)[0]

        if len(anomaly_indices) > 0:
            X_anomalies = X_test[anomaly_indices]
            stage2_pred_proba = trainer.stage2_model.predict(X_anomalies, verbose=0)
            anomaly_types = np.argmax(stage2_pred_proba, axis=1)

            logger.info(
                f"Stage 2 classifying {len(anomaly_indices)} detected anomalies"
            )

            # Handle label mapping correctly for market events
            if labels_mapping is not None:
                labels_mapping = (
                    labels_mapping.tolist()
                    if isinstance(labels_mapping, np.ndarray)
                    else labels_mapping
                )
                # Check if labels_mapping contains strings (market event names)
                if isinstance(labels_mapping[0], str):
                    # Create reverse mapping: string label -> integer for final predictions
                    unique_labels = list(set(labels_mapping))
                    # Map market events to integers: crash=1, dip=2, rally=3, etc.
                    label_to_int = {
                        label: idx + 1
                        for idx, label in enumerate(sorted(unique_labels))
                    }

                    # Map stage2 predictions to string labels, then to integers
                    string_labels = [labels_mapping[i] for i in anomaly_types]
                    final_predictions[anomaly_indices] = [
                        label_to_int[label] for label in string_labels
                    ]
                else:
                    # If labels_mapping values are already numeric
                    mapped_types = np.array([labels_mapping[i] for i in anomaly_types])
                    final_predictions[anomaly_indices] = mapped_types + 1
            else:
                # No mapping, just add 1 to avoid 0 for anomalies
                final_predictions[anomaly_indices] = anomaly_types + 1

        return final_predictions, anomaly_confidence

    except Exception as e:
        logger.error(f"Error in two-stage prediction: {str(e)}")
        raise
