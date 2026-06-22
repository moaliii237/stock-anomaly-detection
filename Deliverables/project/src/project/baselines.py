import pandas as pd


def calculate_zero_r(df: pd.DataFrame, target: str) -> float:
    """Calculate the ZeroR baseline accuracy for anomaly detection evaluation.

    ZeroR is the simplest possible classifier that always predicts the most frequent
    class in the training data. This baseline represents the accuracy achieved by
    a naive approach that ignores all features and simply guesses the majority class
    for every prediction.

    In the context of financial anomaly detection, ZeroR typically predicts "normal"
    conditions for all samples since normal market behavior is much more common than
    anomalous events. This baseline is crucial for establishing a lower bound on
    model performance - any sophisticated model should significantly outperform ZeroR.

    ZeroR serves as an important sanity check in machine learning projects:
    - If a complex model performs worse than ZeroR, there are serious implementation issues
    - If a model only slightly beats ZeroR, the features may not be informative
    - For imbalanced datasets (common in anomaly detection), ZeroR accuracy can be misleadingly high

    Args:
        df: DataFrame containing the dataset with features and target labels.
           Should represent the same data distribution as the test set for
           meaningful baseline comparison.
        target: Name of the target column containing class labels. Must be a
               categorical column with discrete class values (e.g., 'normal',
               'crash', 'dip', 'rally').

    Returns:
        float: ZeroR baseline accuracy as a decimal between 0 and 1.
              This represents the proportion of samples that belong to the
              most frequent class in the dataset.

    Raises:
        ValueError: If target column is not found in DataFrame, is empty,
                   or contains no valid classes.

    Note:
        For highly imbalanced datasets common in anomaly detection, ZeroR accuracy
        can be very high (e.g., 95%+ if anomalies are rare) while being completely
        useless for actually detecting anomalies. This highlights the importance
        of using additional metrics like precision, recall, and F1-score.

    Example:
        >>> # Dataset with 950 normal samples and 50 anomalies
        >>> baseline_acc = calculate_zero_r(train_df, 'event_type')
        >>> print(f"ZeroR baseline: {baseline_acc:.1%}")
        ZeroR baseline: 95.0%
        >>> # This high accuracy is misleading - the model detects no anomalies!

    Mathematical Formula:
        ZeroR Accuracy = count(most_frequent_class) / total_samples

    Use Cases:
        - Establishing minimum performance thresholds for model validation
        - Comparing complex model performance against naive approaches
        - Understanding dataset class distribution and imbalance severity
        - Reporting baseline comparisons in research papers and model documentation
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    # Count the occurrences of each class in the target column
    class_counts = df[target].value_counts()
    if class_counts.empty:
        raise ValueError("Target column is empty or contains no valid classes.")
    most_frequent_count = class_counts.max()
    total_count = len(df)
    if total_count == 0:
        raise ValueError("DataFrame is empty.")
    # Calculate accuracy as the proportion of the most frequent class
    accuracy = most_frequent_count / total_count
    return accuracy


def calculate_weighted_random_guess(df: pd.DataFrame, target: str) -> float:
    """Calculate the Weighted Random Guess baseline accuracy for model evaluation.

    Weighted Random Guess represents the expected accuracy of a classifier that
    randomly predicts classes according to their frequency distribution in the
    training data. Unlike ZeroR which always predicts the same class, this baseline
    simulates random predictions that are biased toward more frequent classes.

    This baseline is particularly valuable for understanding the theoretical lower
    bound of classification performance on imbalanced datasets. It answers the
    question: "What accuracy would we expect from random guessing that respects
    the class distribution?"

    The weighted random guess accuracy is always between the uniform random guess
    (1/num_classes) and the ZeroR accuracy. For highly imbalanced datasets common
    in anomaly detection, this baseline provides a more realistic comparison point
    than uniform random guessing.

    Args:
        df: DataFrame containing the dataset with features and target labels.
           Should represent the same data distribution as the test set for
           meaningful baseline comparison.
        target: Name of the target column containing class labels. Must be a
               categorical column with discrete class values.

    Returns:
        float: Weighted random guess baseline accuracy as a decimal between 0 and 1.
              Always satisfies: (1/num_classes) <= result <= ZeroR_accuracy

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If target column is not found, is empty, or contains only NaN values.

    Example:
        >>> # Dataset: 80% normal, 15% dip, 5% crash
        >>> weighted_acc = calculate_weighted_random_guess(train_df, 'event_type')
        >>> print(f"Weighted random baseline: {weighted_acc:.1%}")
        Weighted random baseline: 67.0%  # 0.8² + 0.15² + 0.05² = 0.665

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    target_series = df[target].dropna()
    if target_series.empty:
        raise ValueError("Target column is empty or contains only NaNs.")

    class_probs = target_series.value_counts(normalize=True)
    return (class_probs**2).sum()


def calculate_persistent_baseline(df: pd.DataFrame, target: str) -> float:
    """Calculate the Persistent baseline accuracy for time series classification.

    The Persistent baseline (also known as "naive forecast" or "last value" baseline)
    predicts that the most recent observed class will continue indefinitely. This
    approach assumes temporal continuity - that market conditions tend to persist
    in the short term rather than change abruptly.

    For financial time series anomaly detection, this baseline captures the intuition
    that market regimes often exhibit persistence: normal conditions tend to continue
    being normal, and volatile periods may persist for some time. This makes the
    persistent baseline particularly relevant for financial applications where
    temporal dependencies are important.

    Args:
        df: DataFrame containing time series data with features and target labels.
           Should be ordered chronologically for meaningful persistence calculation.
           Typically represents training or validation data.
        target: Name of the target column containing class labels. Must be a
               categorical column representing time-dependent class labels
               (e.g., market conditions over time).

    Returns:
        float: Persistent baseline accuracy as a decimal between 0 and 1.
              Represents the proportion of samples in the dataset that match
              the most recent (last) observed class label.

    Raises:
        ValueError: If target column is not found in DataFrame, or if the last
                   value in the target column is NaN.

    Example:
        >>> # Time series: [normal, normal, crash, crash, normal, normal, dip]
        >>> # Last value is 'dip', appears 1/7 times in series
        >>> persistent_acc = calculate_persistent_baseline(time_series_df, 'market_state')
        >>> print(f"Persistent baseline: {persistent_acc:.1%}")
        Persistent baseline: 14.3%

        >>> # For a more persistent series: [normal, normal, normal, normal, normal]
        >>> # Last value 'normal' appears 5/5 times = 100% accuracy
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    if len(df) == 0 or df[target].empty:
        raise ValueError("DataFrame or target column is empty.")
    last_value = df[target].iloc[-1]
    if pd.isna(last_value):
        raise ValueError("Last value in target column is NaN.")
    count_last_value = (df[target] == last_value).sum()
    total_count = len(df)
    accuracy = count_last_value / total_count
    return accuracy
