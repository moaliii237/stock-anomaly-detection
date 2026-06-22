import logging
from typing import Dict, Tuple

import pandas as pd

from src.project.utils.processing_helpers import get_event_in_horizon

# Configure logger
logger = logging.getLogger(__name__)


def load_and_prepare_data(input_file: str, cutoff_date: str) -> pd.DataFrame:
    """Load CSV data and prepare it with proper datetime indexing.

    Loads a CSV file containing time series data, converts the Date column to datetime format,
    sets it as the index, sorts the data chronologically, and applies a cutoff date filter.

    Args:
        input_file: Path to the input CSV file containing time series data.
        cutoff_date: ISO format date string (e.g., "2024-12-31") to filter data up to this date.

    Returns:
        pd.DataFrame: A DataFrame with datetime index, sorted chronologically and filtered
                     by the cutoff date.

    Raises:
        ValueError: If the DataFrame index is not properly sorted after processing.
        FileNotFoundError: If the input file doesn't exist.

    Example:
        >>> df = load_and_prepare_data("data.csv", "2024-12-31")
        >>> print(df.index.is_monotonic_increasing)  # True
    """
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Validate sorted index
    if not df.index.is_monotonic_increasing:
        logger.error("The index of the DataFrame is not sorted")
        raise ValueError(
            "The index of the DataFrame is not sorted. Please sort it before proceeding."
        )

    # Apply cutoff date filter
    cutoff_date = pd.to_datetime(cutoff_date)
    filtered_df = df[df.index <= cutoff_date]
    logger.info(f"Data loaded and filtered. Shape: {filtered_df.shape}")
    return filtered_df


def add_target_variable(
    df: pd.DataFrame, priority: Dict[str, int], horizon_minutes: int
) -> pd.DataFrame:
    """Add the target variable for prediction based on future events within a time horizon.

    Creates a new target column that contains the most important event occurring within
    a specified time horizon from each timestamp. The importance is determined by the
    priority dictionary, with higher values indicating higher priority events.

    Args:
        df: Input DataFrame with time series data containing an 'event' column.
        priority: Dictionary mapping event names to their priority levels (higher = more important).
                 Example: {"normal": 0, "dip": 1, "rally": 2, "crash": 3}
        horizon_minutes: Time horizon in minutes to look ahead for future events.

    Returns:
        pd.DataFrame: DataFrame with the original 'event' column removed and a new
                     'event_in_30min' column containing the target variable.

    Note:
        The target variable represents the most important event that will occur within
        the specified time horizon from each timestamp. If no events are found within
        the horizon, the lowest priority event is assigned.

    Example:
        >>> priority = {"normal": 0, "crash": 3}
        >>> df_with_target = add_target_variable(df, priority, 30)
        >>> # New column 'event_in_30min' contains future events within 30 minutes
    """
    logger.info(f"Adding target variable with {horizon_minutes} minute horizon")
    df = df.copy()
    df["event_in_30min"] = df.apply(
        lambda row: get_event_in_horizon(
            df, "event", row.name, horizon_minutes, priority
        ),
        axis=1,
    )
    result_df = df.drop(columns=["event"])
    logger.info("Target variable added successfully")
    return result_df


def split_dataset(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into training and testing sets using temporal splitting.

    Performs a temporal split of the dataset, preserving chronological order by taking
    the first portion for training and the later portion for testing. This is crucial
    for time series data to avoid data leakage.

    Args:
        df: Input DataFrame with time series data (must be chronologically sorted).
        test_size: Proportion of data to use for testing (between 0.0 and 1.0).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - df_train: Training dataset (earlier timestamps)
            - df_test: Testing dataset (later timestamps)

    Note:
        The split maintains temporal order, which is essential for time series modeling
        to ensure that the model is not trained on future data when predicting the past.

    Example:
        >>> train_df, test_df = split_dataset(df, test_size=0.2)
        >>> # 80% of data (earliest) for training, 20% (latest) for testing
    """
    train_size = int(len(df) * (1 - test_size))
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    logger.info(
        f"Dataset split - Train: {len(df_train)} samples, Test: {len(df_test)} samples"
    )
    return df_train, df_test


def log_distribution_summary(
    df_train: pd.DataFrame, df_test: pd.DataFrame, target_column: str
) -> None:
    """Log distribution summary of target variable in train and test sets.

    Analyzes and logs the distribution of the target variable across both training
    and testing datasets. This helps identify potential data imbalance issues and
    ensures that both datasets have representative distributions.

    Args:
        df_train: Training dataset containing the target variable.
        df_test: Testing dataset containing the target variable.
        target_column: Name of the target column to analyze.

    Note:
        The distributions are logged as proportions (normalized) to make it easy
        to compare class balance between training and testing sets.

    Example:
        >>> log_distribution_summary(train_df, test_df, "event_in_30min")
        # Logs proportions of each event type in both train and test sets
    """
    train_distribution = df_train[target_column].value_counts(normalize=True)
    test_distribution = df_test[target_column].value_counts(normalize=True)

    logger.info(f"Train distribution of {target_column}:")
    for value, proportion in train_distribution.items():
        logger.info(f"  {value}: {proportion:.4f}")

    logger.info(f"Test distribution of {target_column}:")
    for value, proportion in test_distribution.items():
        logger.info(f"  {value}: {proportion:.4f}")


def save_datasets(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    output_file_train: str,
    output_file_test: str,
) -> None:
    """Save training and testing datasets to CSV files.

    Exports the processed training and testing datasets to separate CSV files,
    preserving the datetime index for future loading and processing.

    Args:
        df_train: Training dataset to save.
        df_test: Testing dataset to save.
        output_file_train: File path for saving the training dataset.
        output_file_test: File path for saving the testing dataset.

    Note:
        The index (datetime) is preserved in the saved files to maintain
        temporal information when the datasets are loaded later.

    Example:
        >>> save_datasets(train_df, test_df, "train.csv", "test.csv")
        # Creates train.csv and test.csv with preserved datetime indices
    """
    logger.info(f"Saving training dataset to {output_file_train}")
    df_train.to_csv(output_file_train, index=True)

    logger.info(f"Saving testing dataset to {output_file_test}")
    df_test.to_csv(output_file_test, index=True)

    logger.info("Datasets saved successfully")


def generate_dataset(
    input_file: str,
    output_file_train: str,
    output_file_test: str,
    priority: Dict[str, int],
    horizon_minutes: int = 30,
    cutoff_date: str = "2024-12-31",
    test_size: float = 0.2,
) -> None:
    """Generate a dataset for LSTM training with temporal target variable creation.

    This is the main function that orchestrates the entire dataset generation process
    for time series prediction. It loads raw data, creates predictive target variables
    based on future events within a time horizon, splits the data temporally, and
    saves the processed datasets.

    The function is designed for financial time series prediction where the goal is
    to predict market events (normal, dip, rally, crash) based on historical features.

    Args:
        input_file: Path to input CSV file containing raw time series data with 'event' column.
        output_file_train: Path where the training dataset will be saved.
        output_file_test: Path where the testing dataset will be saved.
        priority: Dictionary mapping event names to priority levels for determining
                 the most important event within the time horizon.
                 Example: {"normal": 0, "dip": 1, "rally": 2, "crash": 3}
        horizon_minutes: Time horizon in minutes for looking ahead to create target variables.
                        Default is 30 minutes.
        cutoff_date: ISO format date string for filtering data up to this date.
                    Default is "2024-12-31".
        test_size: Proportion of data to use for testing (0.0 to 1.0). Default is 0.2 (20%).

    Raises:
        Exception: If any step in the dataset generation process fails.

    Note:
        This function implements a complete pipeline for creating supervised learning
        datasets from time series data. The target variable represents the most important
        event that will occur within the specified time horizon.

    Example:
        >>> priority = {"normal": 0, "dip": 1, "rally": 2, "crash": 3}
        >>> generate_dataset(
        ...     "raw_data.csv",
        ...     "train_data.csv",
        ...     "test_data.csv",
        ...     priority,
        ...     horizon_minutes=30,
        ...     test_size=0.2
        ... )
        # Creates training and testing datasets for 30-minute ahead prediction
    """
    logger.info("Starting dataset generation")
    logger.info(
        f"Parameters - horizon_minutes: {horizon_minutes}, cutoff_date: {cutoff_date}, test_size: {test_size}"
    )

    try:
        # Load and prepare data
        df = load_and_prepare_data(input_file, cutoff_date)

        # Add target variable
        df = add_target_variable(df, priority, horizon_minutes)

        # Split into train and test sets
        df_train, df_test = split_dataset(df, test_size)

        # Log distribution summary
        log_distribution_summary(df_train, df_test, "event_in_30min")

        # Save datasets
        save_datasets(df_train, df_test, output_file_train, output_file_test)

        logger.info("Dataset generation completed successfully")

    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_file = "../BKNG_engineering_train.csv"
    output_file_train = "../BKNG_engineering_train.csv"
    output_file_test = "../BKNG_engineering_test.csv"
    priority = {"normal": 0, "dip": 1, "rally": 2, "crash": 3}

    generate_dataset(input_file, output_file_train, output_file_test, priority)
    logger.info(
        f"Dataset generated and saved to {output_file_train} and {output_file_test}"
    )
