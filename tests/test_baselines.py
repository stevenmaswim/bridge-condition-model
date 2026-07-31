import numpy as np
import pandas as pd

from src.baselines import compute_metrics, persistence_baseline, age_curve_baseline


def test_compute_metrics_perfect():
    m = compute_metrics([5, 6, 7], [5, 6, 7])
    assert m["mae"] == 0 and m["within_1"] == 100.0 and m["r2"] == 1.0


def test_compute_metrics_clips_predictions():
    # a prediction of 12 is clipped to 9, so vs an actual 9 the error is 0
    m = compute_metrics([9.0], [12.0])
    assert m["mae"] == 0


def test_persistence_baseline_counts_transitions():
    panel = pd.DataFrame({
        "bridge_id": ["A", "A", "A", "B"],
        "d": ["2000-05-15", "2002-05-15", "2004-05-15", "2001-01-01"],
        "deck": [7, 7, 6, 8],
    })
    m, detail = persistence_baseline(panel, "bridge_id", "d", "deck")
    assert m["n"] == 2                         # A has 2 transitions; B (single obs) has none
    assert set(detail["prev_rating"]) == {7}   # predicted = previous rating


def test_age_curve_baseline_runs():
    panel = pd.DataFrame({"age": list(range(50)), "deck": [max(0, 9 - a / 10) for a in range(50)]})
    m, _ = age_curve_baseline(panel, "age", "deck")
    assert m is not None and m["mae"] >= 0 and 0 <= m["within_1"] <= 100


def test_age_curve_baseline_too_few_rows():
    panel = pd.DataFrame({"age": [1, 2, 3], "deck": [7, 6, 5]})
    m, _ = age_curve_baseline(panel, "age", "deck")
    assert m is None
