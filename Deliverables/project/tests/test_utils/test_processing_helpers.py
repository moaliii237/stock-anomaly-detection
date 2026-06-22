import pandas as pd
import pytest

from src.project.utils.processing_helpers import get_event_in_horizon


@pytest.fixture
def sample_priority():
    """Standard priority mapping for market events."""
    return {"normal": 1, "dip": 2, "rally": 3, "crash": 4}


@pytest.fixture
def sample_market_data():
    """Sample market data with datetime index and event labels."""
    timestamps = pd.date_range("2024-01-15 10:00:00", periods=10, freq="5min")
    data = {
        "event_type": [
            "normal",
            "normal",
            "dip",
            "normal",
            "rally",
            "normal",
            "crash",
            "normal",
            "normal",
            "dip",
        ],
        "price": [100, 101, 98, 99, 105, 104, 95, 96, 97, 94],
    }
    df = pd.DataFrame(data, index=timestamps)
    return df


def test_get_event_highest_priority_found(sample_market_data, sample_priority):
    """Test that the function returns the highest priority event within horizon."""
    timestamp = pd.Timestamp("2024-01-15 10:00:00")
    result = get_event_in_horizon(
        sample_market_data, "event_type", timestamp, 30, sample_priority
    )
    # Within 30 minutes, there should be a 'crash' event which has highest priority
    assert result == "crash"


def test_get_event_no_future_events(sample_market_data, sample_priority):
    """Test behavior when no events exist within the time horizon."""
    # Use timestamp at the very end of the data
    timestamp = pd.Timestamp("2024-01-15 10:45:00")  # Last timestamp in sample data
    result = get_event_in_horizon(
        sample_market_data, "event_type", timestamp, 5, sample_priority
    )
    # Should return the event with lowest priority (normal = 1)
    assert result == "normal"


def test_get_event_single_event_in_horizon(sample_priority):
    """Test with only one event in the horizon."""
    timestamps = pd.date_range("2024-01-15 10:00:00", periods=3, freq="10min")
    df = pd.DataFrame({"event_type": ["normal", "dip", "normal"]}, index=timestamps)

    timestamp = pd.Timestamp("2024-01-15 10:00:00")
    result = get_event_in_horizon(df, "event_type", timestamp, 15, sample_priority)
    assert result == "dip"


def test_get_event_multiple_same_priority(sample_priority):
    """Test with multiple events of the same priority."""
    timestamps = pd.date_range("2024-01-15 10:00:00", periods=4, freq="5min")
    df = pd.DataFrame(
        {"event_type": ["normal", "rally", "rally", "normal"]}, index=timestamps
    )

    timestamp = pd.Timestamp("2024-01-15 10:00:00")
    result = get_event_in_horizon(df, "event_type", timestamp, 20, sample_priority)
    assert result == "rally"


def test_get_event_time_boundary_inclusion():
    """Test that the time window boundaries are handled correctly."""
    timestamps = pd.date_range("2024-01-15 10:00:00", periods=5, freq="10min")
    df = pd.DataFrame(
        {"event_type": ["normal", "normal", "crash", "normal", "normal"]},
        index=timestamps,
    )
    priority = {"normal": 1, "crash": 4}

    timestamp = pd.Timestamp("2024-01-15 10:00:00")
    # 20 minutes horizon should include the crash at 10:20
    result = get_event_in_horizon(df, "event_type", timestamp, 20, priority)
    assert result == "crash"

    # 15 minutes horizon should NOT include the crash at 10:20
    result = get_event_in_horizon(df, "event_type", timestamp, 15, priority)
    assert result == "normal"


def test_get_event_invalid_column_name(sample_market_data, sample_priority):
    """Test error handling for non-existent column."""
    timestamp = pd.Timestamp("2024-01-15 10:00:00")
    with pytest.raises(KeyError):
        get_event_in_horizon(
            sample_market_data, "nonexistent_column", timestamp, 30, sample_priority
        )


def test_get_event_empty_dataframe(sample_priority):
    """Test behavior with empty DataFrame."""
    df = pd.DataFrame({"event_type": []})
    df.index = pd.DatetimeIndex([])
    timestamp = pd.Timestamp("2024-01-15 10:00:00")

    result = get_event_in_horizon(df, "event_type", timestamp, 30, sample_priority)
    assert result == "normal"  # Should return lowest priority


def test_get_event_priority_not_in_dict(sample_market_data):
    """Test behavior when event labels are not in priority dictionary."""
    priority = {"normal": 1, "dip": 2}  # Missing 'rally' and 'crash'
    timestamp = pd.Timestamp("2024-01-15 10:00:00")

    result = get_event_in_horizon(
        sample_market_data, "event_type", timestamp, 30, priority
    )
    # Should return the event with highest priority value
    # 'dip' has priority 2, which is higher than unlisted events (priority 0)
    assert result == "dip"


def test_get_event_zero_horizon():
    """Test with zero-minute horizon."""
    timestamps = pd.date_range("2024-01-15 10:00:00", periods=3, freq="1min")
    df = pd.DataFrame({"event_type": ["normal", "crash", "normal"]}, index=timestamps)
    priority = {"normal": 1, "crash": 4}

    timestamp = pd.Timestamp("2024-01-15 10:00:00")
    result = get_event_in_horizon(df, "event_type", timestamp, 0, priority)
    assert result == "normal"  # No future events in 0-minute window


def test_get_event_custom_priority_values(sample_market_data):
    """Test with custom priority values including negative numbers."""
    custom_priority = {"normal": -1, "dip": 0, "rally": 5, "crash": 10}
    timestamp = pd.Timestamp("2024-01-15 10:00:00")

    result = get_event_in_horizon(
        sample_market_data, "event_type", timestamp, 30, custom_priority
    )
    assert result == "crash"  # Highest priority value
