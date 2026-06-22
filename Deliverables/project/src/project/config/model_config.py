from dataclasses import dataclass
from typing import List


class LSTMConfig:
    """Configuration class for LSTM market event detection model."""

    def __init__(
        self,
        train_path: str,
        test_path: str,
        model_save_path: str,
        non_feature_cols: List[str] = None,
    ):
        if self.non_feature_cols is None:
            self.non_feature_cols = []

        self.train_path = train_path
        self.test_path = test_path
        self.model_save_path = model_save_path

    # Data parameters
    sequence_length: int = 30
    val_ratio: float = 0.5
    target_column: str = "event_in_30min"
    non_feature_cols: List[str] = None

    # Model parameters
    dropout_rate: float = 0.2
    batch_norm: bool = True

    lstm_units_stage1: int = 16
    conv_units_stage1: int = 16
    dense_units_stage1: int = 16

    lstm_units_stage2: int = 32
    conv_units_stage2: int = 32
    dense_units_stage2: int = 32

    # Training parameters
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-4
    use_class_weights: bool = True

    # Path parameters
    train_path: str = None
    test_path: str = None
    model_save_path: str = None

    # Callback parameters
    patience_early_stopping: int = 8
    patience_lr_reduction: int = 8
    lr_reduction_factor: float = 0.5

    def __post_init__(self):
        if self.non_feature_cols is None:
            self.non_feature_cols = []


class Stage1Config(LSTMConfig):
    """Configuration for Stage 1 binary classifier."""

    def __init__(self, train_path: str, test_path: str, model_save_path: str, non_feature_cols: List[str] = None):
        super().__init__(train_path, test_path, model_save_path, non_feature_cols)
        if self.non_feature_cols is None:
            self.non_feature_cols = []

        self.train_path = train_path
        self.test_path = test_path
        self.model_save_path = model_save_path

    dropout_rate: float = 0.3
    epochs: int = 50

    def print_config(self):
        """Print the configuration parameters."""
        print(f"Train Path: {self.train_path}")
        print(f"Test Path: {self.test_path}")
        print(f"Model Save Path: {self.model_save_path}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Validation Ratio: {self.val_ratio}")
        print(f"Target Column: {self.target_column}")
        print(f"Non-Feature Columns: {self.non_feature_cols}")
        print(f"LSTM Units Stage 1: {self.lstm_units_stage1}")
        print(f"Conv Units Stage 1: {self.conv_units_stage1}")
        print(f"Dense Units Stage 1: {self.dense_units_stage1}")


class Stage2Config(LSTMConfig):
    """Configuration for Stage 2 anomaly classifier."""

    def __init__(self, train_path: str, test_path: str, model_save_path: str, non_feature_cols: List[str] = None):
        super().__init__(train_path, test_path, model_save_path, non_feature_cols)
        if self.non_feature_cols is None:
            self.non_feature_cols = []

        self.train_path = train_path
        self.test_path = test_path
        self.model_save_path = model_save_path

    dropout_rate: float = 0.4
    epochs: int = 80
    use_smote: bool = True  # Aggressive oversampling for rare classes

    def print_config(self):
        """Print the configuration parameters."""
        print(f"Train Path: {self.train_path}")
        print(f"Test Path: {self.test_path}")
        print(f"Model Save Path: {self.model_save_path}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Validation Ratio: {self.val_ratio}")
        print(f"Target Column: {self.target_column}")
        print(f"Non-Feature Columns: {self.non_feature_cols}")
        print(f"LSTM Units Stage 2: {self.lstm_units_stage2}")
        print(f"Conv Units Stage 2: {self.conv_units_stage2}")
        print(f"Dense Units Stage 2: {self.dense_units_stage2}")
