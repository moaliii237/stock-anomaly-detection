import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = logging.getLogger(__name__)


class LSTMDataProcessor:
    """Handles data loading and preprocessing for LSTM model.

    This class provides functionality for loading time series data, validating datasets,
    creating sequential data for LSTM training, and performing various preprocessing tasks
    including scaling and encoding.

    Attributes:
        config: Configuration object containing model parameters and file paths.
        scaler: MinMaxScaler instance for feature normalization.
        label_encoder: LabelEncoder instance for target variable encoding.
    """

    def __init__(self, config):
        """Initialize the LSTM data processor with configuration.

        Args:
            config: Configuration object containing model parameters including:
                - train_path: Path to training data CSV file
                - test_path: Path to test data CSV file
                - sequence_length: Length of input sequences for LSTM
                - target_column: Name of the target column
                - non_feature_cols: List of columns to exclude from features
                - val_ratio: Validation split ratio
        """
        self.config = config
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()

    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data with validation.

        Loads CSV files specified in the configuration, converts date columns to datetime,
        sets proper indexing, sorts by date, and validates compatibility between datasets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - df_train: Training dataset with datetime index
                - df_test: Test dataset with datetime index

        Raises:
            Exception: If there's an error loading files or processing data.

        Example:
            >>> processor = LSTMDataProcessor(config)
            >>> train_df, test_df = processor.load_and_validate_data()
        """
        try:
            df_train = pd.read_csv(self.config.train_path)
            df_test = pd.read_csv(self.config.test_path)

            # Convert dates and set index
            for df in [df_train, df_test]:
                df["Date"] = pd.to_datetime(
                    df["Date"], format="%Y-%m-%d %H:%M:%S", exact=False
                )
                df.set_index("Date", inplace=True)
                df.sort_index(inplace=True)

            self._validate_data_compatibility(df_train, df_test)

            logger.info(f"Loaded training data: {df_train.shape}")
            logger.info(f"Loaded test data: {df_test.shape}")

            return df_train, df_test

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data_compatibility(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame
    ):
        """Validate that train and test datasets are compatible.

        Checks for column consistency between training and test datasets,
        logging warnings for any missing columns in either dataset.

        Args:
            df_train: Training dataset to validate.
            df_test: Test dataset to validate.

        Note:
            This method logs warnings but does not raise exceptions for missing columns,
            allowing for flexible handling of dataset differences.
        """
        train_cols = set(df_train.columns)
        test_cols = set(df_test.columns)

        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols

        if missing_in_test:
            logger.warning(f"Columns missing in test: {missing_in_test}")
        if missing_in_train:
            logger.warning(f"Columns missing in train: {missing_in_train}")

    def prepare_sequences(
        self,
        df: pd.DataFrame,
        fit_scalers: bool = True,
        label_encoder: LabelEncoder = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM training.

        Transforms time series data into sequential format suitable for LSTM training,
        including feature scaling, label encoding, and sequence creation.

        Args:
            df: Input DataFrame with time series data.
            fit_scalers: Whether to fit scalers on this data (True for training,
                        False for validation/test data).
            label_encoder: Optional custom label encoder to use instead of the
                          instance's label_encoder.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X_seq: 3D array of shape (samples, sequence_length, features)
                        containing input sequences
                - y_categorical: 2D array of shape (samples, num_classes) containing
                               one-hot encoded target labels

        Raises:
            Exception: If there's an error during sequence preparation or scaling.

        Example:
            >>> X_train, y_train = processor.prepare_sequences(train_df, fit_scalers=True)
            >>> X_test, y_test = processor.prepare_sequences(test_df, fit_scalers=False)
        """
        try:
            # Separate features and target
            feature_cols = [
                col for col in df.columns if col not in self.config.non_feature_cols
            ]
            if self.config.target_column in feature_cols:
                feature_cols.remove(self.config.target_column)

            X = df[feature_cols].values
            y = df[self.config.target_column].values

            # Scale features
            if fit_scalers:
                X_scaled = self.scaler.fit_transform(X)
                used_label_encoder = label_encoder or self.label_encoder
                y_encoded = used_label_encoder.fit_transform(y)
            else:
                X_scaled = self.scaler.transform(X)
                used_label_encoder = label_encoder or self.label_encoder
                y_encoded = used_label_encoder.transform(y)

            # Create sequences
            X_seq, y_seq = self._create_sequences(X_scaled, y_encoded)

            # Convert to categorical for multi-class classification
            from tensorflow.keras.utils import to_categorical

            y_categorical = to_categorical(
                y_seq, num_classes=len(used_label_encoder.classes_)
            )

            logger.info(f"Created sequences: X{X_seq.shape}, y{y_categorical.shape}")

            return X_seq, y_categorical

        except Exception as e:
            logger.error(f"Error preparing sequences: {str(e)}")
            raise

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequential data from time series.

        Transforms flat time series data into sequences of specified length for LSTM input.
        Each sequence contains consecutive time steps, with the corresponding target being
        the value at the end of each sequence.

        Args:
            X: 2D array of shape (timesteps, features) containing scaled features.
            y: 1D array of shape (timesteps,) containing encoded target values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X_seq: 3D array of shape (samples, sequence_length, features)
                - y_seq: 1D array of shape (samples,) containing target values

        Note:
            The number of output samples will be len(X) - sequence_length because
            we need sequence_length historical points to create each sequence.
        """
        X_seq, y_seq = [], []

        for i in range(self.config.sequence_length, len(X)):
            X_seq.append(X[i - self.config.sequence_length : i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def split_temporal_validation(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split test data temporally for validation.

        Performs a temporal split of the dataset, preserving time order by taking
        the first portion for training/validation and the later portion for testing.

        Args:
            df: Input DataFrame to split temporally.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - First portion of the data (earlier timestamps)
                - Second portion of the data (later timestamps)

        Note:
            The split ratio is determined by self.config.val_ratio.
            This maintains temporal order which is crucial for time series validation.
        """
        split_idx = int(len(df) * self.config.val_ratio)
        return df.iloc[:split_idx], df.iloc[split_idx:]


class TwoStageLSTMDataProcessor(LSTMDataProcessor):
    """Extended data processor for two-stage approach.

    This class extends LSTMDataProcessor to support a two-stage classification approach:
    Stage 1: Binary classification (normal vs anomaly)
    Stage 2: Multi-class classification of anomaly types

    This approach can improve performance when dealing with imbalanced datasets where
    normal samples significantly outnumber anomaly samples.

    Attributes:
        anomaly_label_encoder: LabelEncoder specifically for anomaly types in Stage 2.
    """

    def prepare_binary_sequences(self, df: pd.DataFrame, fit_scalers: bool = True):
        """Prepare binary sequences for Stage 1 (Normal vs Anomaly).

        Converts multi-class labels into binary labels for the first stage of classification,
        where all non-normal classes are grouped as anomalies.

        Args:
            df: Input DataFrame with time series data.
            fit_scalers: Whether to fit scalers on this data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X_seq: 3D array of input sequences
                - y_binary_cat: 2D one-hot encoded binary labels (normal=0, anomaly=1)

        Note:
            This method assumes that "normal" is one of the classes in the label encoder.
            All other classes are treated as anomalies.

        Example:
            >>> X_binary, y_binary = processor.prepare_binary_sequences(train_df)
            >>> # y_binary will have shape (samples, 2) for binary classification
        """
        # Get original sequences first
        X_seq, y_original = self.prepare_sequences(df, fit_scalers=fit_scalers)

        # Convert original categorical labels back to class indices
        if y_original.ndim > 1:
            y_original_indices = np.argmax(y_original, axis=1)
        else:
            y_original_indices = y_original

        # Force to numpy array to avoid astype warning
        y_original_indices = np.array(y_original_indices)

        # normal class index
        normal_class_index = self.label_encoder.transform(["normal"])[0]
        # Create binary labels: 0 = normal, 1 = any anomaly
        y_binary = (y_original_indices != normal_class_index).astype(int)

        from tensorflow.keras.utils import to_categorical

        y_binary_cat = to_categorical(y_binary, num_classes=2)
        logger.info(
            f"Binary label distribution: Normal={np.sum(y_binary == 0)}, Anomaly={np.sum(y_binary == 1)}"
        )
        logger.info(f"Binary categorical shape: {y_binary_cat.shape}, ")

        return X_seq, y_binary_cat

    def prepare_anomaly_only_sequences(self, df: pd.DataFrame):
        """Prepare sequences with only anomaly samples for Stage 2.

        Filters the dataset to include only anomaly samples (excluding "normal" class)
        and creates a new label encoder specifically for anomaly types.

        Args:
            df: Input DataFrame with time series data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X_seq: 3D array of input sequences from anomaly samples only
                - y_seq: 2D one-hot encoded labels for anomaly types only

        Raises:
            ValueError: If no anomaly samples are found in the dataset.

        Note:
            This method creates and stores a new anomaly_label_encoder attribute
            that maps only the anomaly class types (excluding "normal").

        Example:
            >>> X_anomaly, y_anomaly = processor.prepare_anomaly_only_sequences(train_df)
            >>> # Only samples with non-"normal" labels are included
        """
        # Filter only anomaly samples (assuming 'normal' is the first class)
        target_col = self.config.target_column
        anomaly_mask = df[target_col] != "normal"
        df_anomaly = df[anomaly_mask].copy()

        if len(df_anomaly) == 0:
            raise ValueError("No anomaly samples found for Stage 2")

        logger.info(
            f"Stage 2 data: {len(df_anomaly)} anomaly samples from {len(df)} total"
        )

        # Re-encode labels for anomaly-only classification
        anomaly_classes = df_anomaly[target_col].unique()
        logger.info(f"Anomaly classes for Stage 2: {anomaly_classes}")

        # Create new label encoder for anomaly-only classes
        from sklearn.preprocessing import LabelEncoder

        anomaly_label_encoder = LabelEncoder()
        # Fix: Just fit the encoder, don't assign the result back
        anomaly_label_encoder.fit(df_anomaly[target_col])

        # Store the anomaly label encoder for later use
        self.anomaly_label_encoder = anomaly_label_encoder

        # Prepare sequences from anomaly-only data
        X_seq, y_seq = self.prepare_sequences(
            df_anomaly, fit_scalers=False, label_encoder=anomaly_label_encoder
        )

        return X_seq, y_seq
