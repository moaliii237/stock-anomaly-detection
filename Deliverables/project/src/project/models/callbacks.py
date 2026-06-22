import logging

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class ClassificationReportCallback(tf.keras.callbacks.Callback):
    """Custom callback to print detailed classification reports during training.

    This callback generates and prints comprehensive classification reports at regular
    intervals during training, providing insights into model performance across different
    classes.

    The callback converts model predictions and true labels from one-hot encoded format
    to class indices and generates detailed metrics including precision, recall, and
    F1-score for each class.

    Attributes:
        X_val: Validation input data for generating predictions.
        y_val: Validation target data (one-hot encoded) for comparison.
        label_encoder: Label encoder with class names for meaningful report display.
    """

    def __init__(self, X_val, y_val, label_encoder):
        """Initialize the classification report callback.

        Args:
            X_val: Validation input data array of shape (samples, sequence_length, features).
            y_val: Validation target data array of shape (samples, num_classes) in one-hot format.
            label_encoder: Label encoder object with a 'classes_' attribute containing
                          class names for the classification report.

        Example:
            >>> callback = ClassificationReportCallback(X_val, y_val, label_encoder)
            >>> model.fit(X_train, y_train, callbacks=[callback])
        """
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.label_encoder = label_encoder

    def on_epoch_end(self, epoch, logs=None):
        """Generate and print classification report at the end of specified epochs.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing training metrics for the current epoch.

        Note:
            The classification report includes precision, recall, F1-score, and support
            for each class, providing a comprehensive view of model performance across
            all classes in the dataset.

        Example Output:
            === CLASSIFICATION REPORT (Epoch 10) ===
                          precision    recall  f1-score   support
            normal           0.9500    0.9200    0.9347       100
            dip              0.7800    0.8100    0.7949        37
            rally            0.8200    0.7900    0.8047        38
            crash            0.9100    0.9500    0.9295        20
        """
        if (epoch + 1) % 5 == 0:  # Print every 5 epochs
            from sklearn.metrics import classification_report

            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(self.y_val, axis=1)

            print(f"\n=== CLASSIFICATION REPORT (Epoch {epoch + 1}) ===")
            print(
                classification_report(
                    y_true_labels,
                    y_pred_labels,
                    target_names=self.label_encoder.classes_,
                    digits=4,
                )
            )


class OverfittingMonitorCallback(tf.keras.callbacks.Callback):
    """Monitor for overfitting and automatically stop training when detected.

    This callback monitors the difference between training and validation loss to detect
    overfitting patterns. When the validation loss becomes significantly higher than
    the training loss for a sustained period, it indicates that the model is overfitting
    to the training data and losing its ability to generalize.

    The callback implements a patience mechanism where overfitting must be detected
    for consecutive epochs before stopping training, preventing false positives from
    temporary fluctuations in the loss metrics.

    Attributes:
        patience: Number of consecutive epochs with overfitting before stopping training.
        threshold: Minimum difference between validation and training loss to consider overfitting.
        wait: Current count of consecutive epochs with detected overfitting.
    """

    def __init__(self, patience=3, threshold=0.2):
        """Initialize the overfitting monitor callback.

        Args:
            patience: Number of consecutive epochs with overfitting detection before
                     stopping training. Higher values make the callback less sensitive
                     to temporary fluctuations. Default is 3.
            threshold: Minimum difference between validation loss and training loss
                      to consider as overfitting. Higher values make the detection
                      less sensitive. Default is 0.2.

        Example:
            >>> # Conservative overfitting detection
            >>> callback = OverfittingMonitorCallback(patience=5, threshold=0.3)
            >>> model.fit(X_train, y_train, callbacks=[callback])

            >>> # Aggressive overfitting detection
            >>> callback = OverfittingMonitorCallback(patience=2, threshold=0.1)
        """
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        """Monitor training and validation loss to detect overfitting patterns.

        Args:
            epoch: Current epoch number (0-indexed).
            logs: Dictionary containing training metrics including 'loss' and 'val_loss'.

        Note:
            The callback waits until epoch 5 before starting monitoring to allow
            the model to stabilize its learning patterns. Early epochs often show
            high variance in loss values that could trigger false overfitting detection.

        Warning:
            When overfitting is detected and training is stopped, the callback logs
            both the detection event and the current loss values for debugging purposes.

        Example:
            If validation loss is 0.8 and training loss is 0.5 with threshold=0.2:
            - Difference: 0.8 - 0.5 = 0.3 > 0.2 (threshold)
            - Overfitting detected, increment wait counter
            - If wait >= patience, stop training
        """
        if epoch < 5:  # Wait a few epochs before monitoring
            return

        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        # Check if validation loss is significantly higher than training loss
        if val_loss - train_loss > self.threshold:
            self.wait += 1
            if self.wait >= self.patience:
                logger.warning(
                    f"Early stopping due to overfitting detected at epoch {epoch + 1}"
                )
                logger.info(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

            self.model.stop_training = True
        else:
            self.wait = 0
