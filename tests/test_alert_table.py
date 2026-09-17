"""Tests for scripts/alert_table.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.alert_table import by_variant, pooled, variant_table


def res(variant, alerts, just, eps, warned, seconds=60.0):
    return {"variant": variant, "alerts": alerts, "justified_alerts": just, "episodes": eps,
            "warned_episodes": warned, "alert_precision": just / alerts if alerts else None,
            "episode_recall": warned / eps if eps else None,
            "unjustified_alerts_per_min": (alerts - just) / (seconds / 60),
            "alert_benchmark": {"total_seconds": seconds}}


def test_pooled_sums_counts_not_ratios():
    p = pooled([res("a", 10, 1, 10, 5), res("a", 90, 45, 2, 0)])
    assert p["precision"] == 46 / 100          # an average of ratios would give 0.3
    assert p["recall"] == 5 / 12
    assert p["unjustified_per_min"] == 54 / 2


def test_variant_table_reports_per_recording_direction():
    records = [
        {"recording": "r1", "results": [res("pre_ttc", 10, 1, 4, 1), res("ttc_fix", 10, 2, 4, 1)]},
        {"recording": "r2", "results": [res("pre_ttc", 10, 5, 4, 2), res("ttc_fix", 10, 4, 4, 3)]},
    ]
    table = variant_table(by_variant(records), ["pre_ttc", "ttc_fix"])
    assert "| pre_ttc | 20 | 30.0 % (6) | 8 | 37.5 % (3) |" in table
    assert "precision up in 1, down in 1, equal in 0 of 2" in table
    assert "recall up in 1, down in 0, equal in 1 of 2" in table


def test_empty_episodes_do_not_divide_by_zero():
    assert pooled([res("a", 0, 0, 0, 0)])["precision"] is None
