import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from monitoring.drift_report import psi


def test_psi_returns_numeric_score_for_shifted_distributions():
    score = psi(
        pd.Series([1, 2, 3, 4, 5, 6]),
        pd.Series([4, 5, 6, 7, 8, 9]),
        buckets=3,
    )

    assert isinstance(score, float)
    assert pd.notna(score)
    assert score >= 0
