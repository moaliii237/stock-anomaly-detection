import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def get_event_in_horizon(
    df: pd.DataFrame,
    y_col_name: str,
    timestamp: pd.Timestamp,
    horizon_minutes: int,
    priority: Dict[str, int],
) -> str:
    """Get the most important event label within a specified time horizon.

    Args:
        df: DataFrame containing time series data with datetime index and event labels.
           Must be sorted by timestamp for proper time-based slicing.
        y_col_name: Name of the column containing event labels (e.g., 'event_type',
                   'market_condition'). This column should contain categorical labels
                   representing different types of market events.
        timestamp: Starting timestamp for the look-ahead window. Must be a pandas
                  Timestamp object that exists in the DataFrame's index or can be
                  used for time-based slicing.
        horizon_minutes: Length of the forward-looking time window in minutes.
                        Typical values might be 30, 60, or 120 minutes depending
                        on the prediction horizon required for the application.
        priority: Dictionary mapping event label names to integer priority values.
                 Higher values indicate more important events. For example:
                 {'normal': 1, 'dip': 2, 'rally': 3, 'crash': 4}
                 This ensures crashes are prioritized over dips when both occur.

    Returns:
        str: Event label with the highest priority within the specified time horizon.
             If no events are found in the horizon, returns the event with the lowest
             priority value (typically 'normal' conditions).

    Raises:
        KeyError: If y_col_name doesn't exist in the DataFrame columns.
        ValueError: If timestamp is not valid for DataFrame slicing.
        TypeError: If priority dictionary values are not numeric.

    Example:
        >>> priority = {'normal': 1, 'dip': 2, 'rally': 3, 'crash': 4}
        >>> event = get_event_in_horizon(
        ...     df=market_data,
        ...     y_col_name='event_type',
        ...     timestamp=pd.Timestamp('2024-01-15 10:00:00'),
        ...     horizon_minutes=30,
        ...     priority=priority
        ... )
        >>> print(f"Most critical event in next 30 minutes: {event}")

    Technical Details:
        The time window is defined as (timestamp + 1 second, timestamp + horizon_minutes].
        This ensures the current timestamp is excluded from the search while including
        all events up to and including the end of the horizon period.
    """
    future_labels = df.loc[
        timestamp
        + pd.Timedelta(seconds=1) : timestamp
        + pd.Timedelta(minutes=horizon_minutes),
        y_col_name,
    ]
    if future_labels.empty:
        return min(priority, key=priority.get)
    return max(future_labels, key=lambda lbl: priority.get(lbl, 0))
