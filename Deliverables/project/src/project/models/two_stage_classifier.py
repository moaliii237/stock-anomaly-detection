import logging
import os
import pickle
from typing import Tuple, Dict, List

import numpy as np
import tensorflow as tf
from keras import Sequential, regularizers, Input
from keras.src.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import (
    Conv1D,
    MaxPooling1D,
    Dropout,
    Bidirectional,
    LSTM,
    Dense,
    BatchNormalization,
)
from keras.src.metrics import Precision, Recall, F1Score
from sklearn.utils.class_weight import compute_class_weight

from src.project.config.model_config import Stage1Config, Stage2Config
from src.project.data.preprocessing.data_processor import TwoStageLSTMDataProcessor
from src.project.models.callbacks import (
    OverfittingMonitorCallback,
    ClassificationReportCallback,
)

logger = logging.getLogger(__name__)


def build_stage1_lstm_model(input_shape: Tuple[int, int], config: Stage1Config) -> Sequential:
    """Build a simplified LSTM model for binary anomaly detection.

    This function creates a convolutional-LSTM hybrid architecture optimized for
    binary classification of time series data into normal vs anomalous patterns.
    The model uses a combination of:
    - 1D convolution for local pattern detection
    - Bidirectional LSTM for temporal dependency modeling
    - Regularization techniques to prevent overfitting

    Args:
        input_shape: Tuple of (sequence_length, n_features) defining the input
                    dimensions for the LSTM model.

    Returns:
        Sequential: Compiled Keras model ready for training with binary
                   classification output (normal vs anomaly).

    Architecture Components:
        - Conv1D layer: Captures local temporal patterns
        - MaxPooling1D: Reduces sequence length for efficiency
        - Bidirectional LSTM: Models temporal dependencies in both directions
        - Dense layers: Final classification with regularization
        - Dropout/BatchNormalization: Prevents overfitting

    Example:
        >>> model = build_stage1_lstm_model((30, 10))
        >>> print(f"Model has {model.count_params()} parameters")
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=config.conv_units_stage1, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Dropout(config.dropout_rate),
            Bidirectional(
                LSTM(
                    config.lstm_units_stage1,
                    return_sequences=False,
                    recurrent_dropout=config.dropout_rate,
                    kernel_regularizer=regularizers.l2(0.001),
                )
            ),
            Dropout(config.dropout_rate),
            Dense(config.dense_units_stage1, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(config.dropout_rate),
            Dense(2, activation="softmax", dtype="float32"),
        ]
    )
    return model


def build_stage2_lstm_model(
    input_shape: Tuple[int, int], num_anomaly_classes: int, config: Stage2Config
) -> Sequential:
    """Build a specialized LSTM model for anomaly type classification.

    This function creates a more complex LSTM architecture designed specifically
    for multi-class classification of different anomaly types. The model is
    deeper and more sophisticated than Stage 1, as it needs to distinguish
    between subtle differences in anomaly patterns (e.g., crash vs rally vs dip).

    The architecture incorporates:
    - Multiple convolutional layers for complex pattern extraction
    - Stacked bidirectional LSTMs for deep temporal modeling
    - Extensive regularization for stable training on imbalanced data
    - Multiple dense layers for refined classification

    Args:
        input_shape: Tuple of (sequence_length, n_features) defining the input
                    dimensions for the LSTM model.
        num_anomaly_classes: Number of different anomaly types to classify
                           (e.g., 3 for crash/dip/rally).

    Returns:
        Sequential: Compiled Keras model ready for training with multi-class
                   classification output for anomaly types.

    Architecture Components:
        - Multiple Conv1D layers: Extract complex local patterns
        - Stacked bidirectional LSTMs: Deep temporal sequence modeling
        - Multiple dense layers: Hierarchical feature refinement
        - Extensive dropout/batch normalization: Robust training

    Example:
        >>> model = build_stage2_lstm_model((30, 10), 3)
        >>> print(f"Model classifies {num_anomaly_classes} anomaly types")
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=config.conv_units_stage2, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Dropout(config.dropout_rate),
            Bidirectional(LSTM(config.conv_units_stage2, return_sequences=True, recurrent_dropout=config.dropout_rate)),
            Dropout(config.dropout_rate),
            Bidirectional(LSTM(config.conv_units_stage2, return_sequences=False, recurrent_dropout=0.3)),
            Dropout(config.dropout_rate),
            Dense(config.dense_units_stage2, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(config.dropout_rate),
            Dense(config.dense_units_stage2, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(config.dropout_rate),
            Dense(num_anomaly_classes, activation="softmax", dtype="float32"),
        ]
    )
    return model


class TwoStageLSTMTrainer:
    """Comprehensive trainer for two-stage LSTM anomaly detection system.

    This class manages the complete training pipeline for a hierarchical anomaly
    detection system designed for financial time series. The two-stage approach
    addresses the challenges of imbalanced datasets common in financial markets:

    Stage 1: Binary Classification (Normal vs Anomaly)
    - Optimized for high sensitivity to detect any anomalous behavior
    - Uses balanced training with class weights
    - Focuses on not missing any potential anomalies

    Stage 2: Anomaly Type Classification (Crash/Dip/Rally/etc.)
    - Specialized for distinguishing between different anomaly types
    - Trained only on pre-identified anomalous samples
    - Can use techniques like SMOTE for better class balance

    The trainer handles all aspects of the training process including:
    - Data preprocessing and sequence preparation
    - Model architecture selection and compilation
    - Advanced callback management for robust training
    - Class weight computation for imbalanced datasets
    - Production artifact saving for deployment

    Attributes:
        stage1_config: Configuration object for binary classification stage.
        stage2_config: Configuration object for anomaly classification stage.
        stage1_model: Trained binary classification model.
        stage2_model: Trained anomaly type classification model.
        data_processor: Data preprocessing and sequence preparation component.
        anomaly_label_encoder: Encoder for anomaly type labels.
        stage1_history: Training history from Stage 1.
        stage2_history: Training history from Stage 2.
    """

    def __init__(self, stage1_config, stage2_config):
        """Initialize the two-stage LSTM trainer with configurations.

        Sets up the trainer with separate configurations for each stage,
        allowing for different hyperparameters, training strategies, and
        optimization approaches for each classification task.

        Args:
            stage1_config: Configuration object containing parameters for
                          binary anomaly detection including learning rate,
                          batch size, epochs, and model architecture settings.
            stage2_config: Configuration object containing parameters for
                          anomaly type classification including any special
                          settings like SMOTE usage or different regularization.
        """
        self.stage1_config = stage1_config
        self.stage2_config = stage2_config
        self.stage1_model = None
        self.stage2_model = None
        self.data_processor = TwoStageLSTMDataProcessor(stage1_config)
        self.anomaly_label_encoder = None
        self.stage1_history = None
        self.stage2_history = None

    def _compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Compute class weights for imbalanced dataset with proper format handling.

        This method calculates balanced class weights to address the inherent
        imbalance in financial anomaly detection datasets where normal conditions
        significantly outnumber anomalous events. The weights are computed using
        sklearn's balanced strategy and properly formatted for Keras training.

        Args:
            y_train: Training labels in various formats:
                    - 1D array of integer labels
                    - 2D one-hot encoded array
                    - 2D array with single column

        Returns:
            Dict[int, float]: Dictionary mapping class indices to weight values
                             suitable for use in Keras model.fit(class_weight=...).

        Raises:
            ValueError: If y_train has unexpected shape or format.


        Example:
            >>> weights = trainer._compute_class_weights(y_train)
            >>> print(weights)  # {0: 0.52, 1: 4.81}  # Normal class gets lower weight
        """
        if y_train.ndim == 1:
            y_train_int = y_train
        elif y_train.ndim == 2:
            if y_train.shape[1] > 1:
                y_train_int = np.argmax(y_train, axis=1)
            else:
                y_train_int = y_train.squeeze()
        else:
            raise ValueError(f"Unexpected y_train shape: {y_train.shape}")

        logger.debug(f"Computing class weights for labels with shape: {y_train.shape}")
        logger.debug(f"Converted to integer labels with shape: {y_train_int.shape}")
        logger.debug(f"Unique classes: {np.unique(y_train_int)}")

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train_int), y=y_train_int
        )
        class_weight_dict = dict(enumerate(class_weights))
        logger.info(f"Computed class weights: {class_weight_dict}")
        return class_weight_dict

    def _setup_stage1_callbacks(self, X_val: np.ndarray, y_val: np.ndarray) -> List:
        """Setup callbacks for Stage 1 binary classification training.

        Args:
            X_val: Validation input sequences for callback monitoring.
            y_val: Validation labels for callback evaluation.

        Returns:
            List: Configured callback objects ready for use in model.fit().

        Callback Components:
            - ModelCheckpoint: Saves best model based on validation loss
            - EarlyStopping: Prevents overfitting with patience mechanism
            - ReduceLROnPlateau: Adaptive learning rate reduction
            - OverfittingMonitorCallback: Custom overfitting detection
            - ClassificationReportCallback: Periodic performance reports

        Note:
            Creates a dummy binary label encoder for the classification report
            callback since Stage 1 only deals with normal vs anomaly classes.
        """

        # Create dummy binary encoder for callbacks
        class BinaryLabelEncoder:
            classes_ = ["normal", "anomaly"]

        binary_encoder = BinaryLabelEncoder()

        callbacks = [
            ModelCheckpoint(
                self.stage1_config.model_save_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.stage1_config.patience_early_stopping,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                patience=self.stage1_config.patience_lr_reduction,
                factor=self.stage1_config.lr_reduction_factor,
                monitor="val_loss",
                mode="min",
                min_lr=1e-7,
                verbose=1,
            ),
            OverfittingMonitorCallback(patience=3),
            ClassificationReportCallback(X_val, y_val, binary_encoder),
        ]
        return callbacks

    def _setup_stage2_callbacks(self, X_val: np.ndarray, y_val: np.ndarray) -> List:
        """Setup callbacks for Stage 2 anomaly type classification.

        Args:
            X_val: Validation input sequences for callback monitoring.
            y_val: Validation labels for callback evaluation.

        Returns:
            List: Configured callback objects ready for use in model.fit().

        Note:
            Uses the trained anomaly_label_encoder to provide meaningful class
            names in classification reports (e.g., 'crash', 'dip', 'rally').
        """
        callbacks = [
            ModelCheckpoint(
                self.stage2_config.model_save_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.stage2_config.patience_early_stopping,
                mode="min",
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                patience=self.stage2_config.patience_lr_reduction,
                factor=self.stage2_config.lr_reduction_factor,
                monitor="val_loss",
                mode="min",
                min_lr=1e-7,
                verbose=1,
            ),
            OverfittingMonitorCallback(patience=3),
            ClassificationReportCallback(X_val, y_val, self.anomaly_label_encoder),
        ]
        return callbacks

    def train_stage1(self, df_train, df_val):
        """Train binary anomaly detection model with callbacks.

        The training process includes:
        1. Data preparation with binary sequence generation
        2. Model architecture building and compilation
        3. Class weight computation for balanced training
        4. Comprehensive callback setup for robust training
        5. Model training with monitoring and optimization

        Args:
            df_train: Training dataset DataFrame with datetime index and features.
            df_val: Validation dataset DataFrame for monitoring training progress.

        Returns:
            Training history object containing loss and metric values throughout
            training, useful for plotting learning curves and diagnosing training.

        Raises:
            Exception: If training fails due to data issues, model problems, or
                      insufficient resources.

        Note:
            The method automatically handles class imbalance through computed weights.
            The trained model is automatically saved to the configured path.

        Example:
            >>> history = trainer.train_stage1(train_df, val_df)
            >>> print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
        """
        logger.info("=== Training Stage 1: Binary Anomaly Detection ===")

        try:
            # Prepare binary data
            X_train, y_train = self.data_processor.prepare_binary_sequences(df_train)
            X_val, y_val = self.data_processor.prepare_binary_sequences(
                df_val, fit_scalers=False
            )

            logger.info(
                f"Stage 1 - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
            )
            logger.info(
                f"Stage 1 - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}"
            )

            # Build and compile model directly
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.stage1_model = build_stage1_lstm_model(input_shape, config=self.stage1_config)

            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.stage1_config.learning_rate, weight_decay=0.01
            )

            self.stage1_model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=[
                    "accuracy",
                    Precision(name="precision"),
                    Recall(name="recall"),
                    F1Score(name="f1_score"),
                ],
            )

            logger.info(
                f"Stage 1 model built: {self.stage1_model.count_params()} parameters"
            )

            # Compute class weights and setup callbacks
            class_weights = self._compute_class_weights(y_train)
            callbacks = self._setup_stage1_callbacks(X_val, y_val)

            # Train model directly
            logger.info("Starting Stage 1 training...")
            self.stage1_history = self.stage1_model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.stage1_config.epochs,
                batch_size=self.stage1_config.batch_size,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )

            logger.info("Stage 1 training completed successfully")
            return self.stage1_history

        except Exception as e:
            logger.error(f"Error in Stage 1 training: {str(e)}")
            raise

    def train_stage2(self, df_train, df_val):
        """Train anomaly type classification model with callbacks.

        The training process includes:
        1. Anomaly-only data preparation and sequence generation
        2. Optional SMOTE oversampling for better class balance
        3. Complex model architecture building and compilation
        4. Class weight computation and callback setup
        5. Specialized training for anomaly type discrimination

        Args:
            df_train: Training dataset DataFrame with datetime index and features.
            df_val: Validation dataset DataFrame for monitoring training progress.

        Returns:
            Training history object containing loss and metric values throughout
            training, useful for analyzing training dynamics and model performance.

        Raises:
            Exception: If training fails due to insufficient anomaly samples,
                      SMOTE issues, or model training problems.

        Note:
            This stage only trains on anomalous samples, which may result in
            smaller training sets. SMOTE oversampling can be enabled via configuration
            to improve class balance among different anomaly types.

        Example:
            >>> history = trainer.train_stage2(train_df, val_df)
            >>> anomaly_types = trainer.anomaly_label_encoder.classes_
            >>> print(f"Trained to classify: {list(anomaly_types)}")
        """
        logger.info("=== Training Stage 2: Anomaly Classification ===")

        try:
            # Prepare anomaly-only data
            X_train, y_train = self.data_processor.prepare_anomaly_only_sequences(
                df_train
            )
            X_val, y_val = self.data_processor.prepare_anomaly_only_sequences(df_val)

            self.anomaly_label_encoder = self.data_processor.anomaly_label_encoder

            logger.info(
                f"Stage 2 - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
            )
            logger.info(
                f"Stage 2 - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}"
            )

            # Apply SMOTE if configured
            if self.stage2_config.use_smote and len(X_train) > 10:
                from imblearn.over_sampling import SMOTE
                from tensorflow.keras.utils import to_categorical

                logger.info("Applying SMOTE oversampling for Stage 2...")
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                y_train_1d = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train

                smote = SMOTE(random_state=42)
                X_train_smote, y_train_smote = smote.fit_resample(
                    X_train_2d, y_train_1d
                )

                X_train = X_train_smote.reshape(-1, X_train.shape[1], X_train.shape[2])
                num_classes = len(self.anomaly_label_encoder.classes_)
                y_train = to_categorical(y_train_smote, num_classes=num_classes)

                logger.info(
                    f"After SMOTE - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}"
                )

            # Build and compile model directly
            input_shape = (X_train.shape[1], X_train.shape[2])
            num_anomaly_classes = y_train.shape[1]
            self.stage2_model = build_stage2_lstm_model(
                input_shape, num_anomaly_classes, config=self.stage2_config
            )

            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.stage2_config.learning_rate, weight_decay=0.01
            )

            self.stage2_model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=[
                    "accuracy",
                    Precision(name="precision"),
                    Recall(name="recall"),
                    F1Score(name="f1_score"),
                ],
            )

            logger.info(
                f"Stage 2 model built: {self.stage2_model.count_params()} parameters"
            )

            # Compute class weights and setup callbacks
            class_weights = self._compute_class_weights(y_train)
            callbacks = self._setup_stage2_callbacks(X_val, y_val)

            # Train model directly
            logger.info("Starting Stage 2 training...")
            self.stage2_history = self.stage2_model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=self.stage2_config.epochs,
                batch_size=self.stage2_config.batch_size,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=1,
            )

            logger.info("Stage 2 training completed successfully")
            return self.stage2_history

        except Exception as e:
            logger.error(f"Error in Stage 2 training: {str(e)}")
            raise

    def save_production_artifacts(self, base_path: str = "models/saved_models/") -> str:
        """Save all components needed for production inference deployment.

        Creates a complete production deployment package containing all trained models
        and preprocessing components required for inference. This method ensures that
        the production predictor can operate independently without access to the
        training environment or training data.

        Args:
            base_path: Directory path where all production artifacts will be saved.
                      Directory will be created if it doesn't exist.

        Returns:
            str: Path to the saved preprocessing artifacts file.

        Raises:
            Exception: If saving fails due to file system issues or missing components.

        Saved Components:
            - stage1_lstm.keras: Binary anomaly detection model
            - stage2_lstm.keras: Anomaly type classification model
            - preprocessing_artifacts.pkl: All preprocessing components including:
              * Fitted data scalers
              * Label encoders
              * Configuration objects
              * Feature column definitions
              * Sequence length parameters

        Note:
            This method creates a complete, self-contained deployment package.
            The saved artifacts can be loaded by ProductionAnomalyPredictor
            for real-time inference without any dependencies on training code.

        Example:
            >>> artifacts_path = trainer.save_production_artifacts("deployment/")
            >>> print("Ready for production deployment!")
        """
        try:
            # Ensure directory exists
            os.makedirs(base_path, exist_ok=True)

            # Save the models
            if self.stage1_model is not None:
                stage1_path = os.path.join(base_path, "stage1_lstm.keras")
                self.stage1_model.save(stage1_path)
                logger.info(f"Stage 1 model saved to {stage1_path}")
            else:
                logger.warning("Stage 1 model is None, cannot save")

            if self.stage2_model is not None:
                stage2_path = os.path.join(base_path, "stage2_lstm.keras")
                self.stage2_model.save(stage2_path)
                logger.info(f"Stage 2 model saved to {stage2_path}")
            else:
                logger.warning("Stage 2 model is None, cannot save")

            # Prepare artifacts dictionary
            artifacts = {
                "scaler": self.data_processor.scaler,
                "label_encoder": self.data_processor.label_encoder,
                "anomaly_label_encoder": getattr(
                    self.data_processor, "anomaly_label_encoder", None
                ),
                "config": self.data_processor.config,
                "sequence_length": self.data_processor.config.sequence_length,
                "non_feature_cols": getattr(
                    self.data_processor.config, "non_feature_cols", []
                ),
                "target_column": getattr(
                    self.data_processor.config, "target_column", "event_in_30min"
                ),
                "stage1_config": self.stage1_config,
                "stage2_config": self.stage2_config,
            }

            # Save preprocessing artifacts
            artifacts_path = os.path.join(base_path, "preprocessing_artifacts.pkl")
            with open(artifacts_path, "wb") as f:
                pickle.dump(artifacts, f)

            logger.info(f"Preprocessing artifacts saved to {artifacts_path}")
            logger.info("All production artifacts saved successfully!")

            logger.info("Production deployment ready.")
            logger.info(f"Models and artifacts saved to: {base_path}")
            logger.info("Files created:")
            logger.info("   - stage1_lstm.keras - Binary anomaly detection model")
            logger.info("   - stage2_lstm.keras - Anomaly classification model")
            logger.info(
                "   - preprocessing_artifacts.pkl - All preprocessing components"
            )
            logger.info(f"Classes: {list(self.data_processor.label_encoder.classes_)}")
            if self.anomaly_label_encoder:
                logger.info(
                    f"Anomaly types: {list(self.anomaly_label_encoder.classes_)}"
                )
            logger.info(
                f"Sequence length: {self.data_processor.config.sequence_length}"
            )

            return artifacts_path

        except Exception as e:
            logger.error(f"Error saving production artifacts: {str(e)}")
            raise
