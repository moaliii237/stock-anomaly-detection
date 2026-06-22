import pandas as pd
import pytest

from src.project.baselines import (
    calculate_zero_r,
    calculate_weighted_random_guess,
    calculate_persistent_baseline,
)


def test_calculate_zero_r_majority():
    df = pd.DataFrame({"event_type": ["normal"] * 8 + ["crash"] * 2})
    result = calculate_zero_r(df, "event_type")
    assert result == 0.8


def test_calculate_zero_r_all_same():
    df = pd.DataFrame({"event_type": ["crash"] * 5})
    result = calculate_zero_r(df, "event_type")
    assert result == 1.0


def test_calculate_zero_r_empty():
    df = pd.DataFrame({"event_type": []})
    with pytest.raises(ValueError):
        calculate_zero_r(df, "event_type")


def test_weighted_random_guess():
    df = pd.DataFrame({"event_type": ["normal"] * 8 + ["crash"] * 2})
    result = calculate_weighted_random_guess(df, "event_type")
    assert abs(result - 0.68) < 1e-6  # 0.8^2 + 0.2^2


def test_weighted_random_guess_uniform():
    df = pd.DataFrame({"event_type": ["a", "b", "c", "d"]})
    result = calculate_weighted_random_guess(df, "event_type")
    assert abs(result - 0.25) < 1e-6  # 0.25^2 * 4


def test_persistent_baseline_last_value():
    df = pd.DataFrame(
        {"event_type": ["normal", "crash", "crash", "normal", "normal", "dip"]}
    )
    result = calculate_persistent_baseline(df, "event_type")
    assert abs(result - (1 / 6)) < 1e-6


def test_persistent_baseline_all_same():
    df = pd.DataFrame({"event_type": ["normal"] * 5})
    result = calculate_persistent_baseline(df, "event_type")
    assert result == 1.0


def test_persistent_baseline_empty():
    df = pd.DataFrame({"event_type": []})
    with pytest.raises(ValueError):
        calculate_persistent_baseline(df, "event_type")
