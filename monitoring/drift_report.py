import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

REFERENCE_DATA = BASE_DIR / "data" / "processed" / "train.csv"
CURRENT_DATA = BASE_DIR / "data" / "processed" / "test.csv"
REPORT_PATH = BASE_DIR / "reports" / "monitoring_report.html"
SUMMARY_PATH = BASE_DIR / "reports" / "monitoring_summary.json"


def psi(expected, actual, buckets=10):
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if expected.empty or actual.empty:
        return None

    breakpoints = expected.quantile([i / buckets for i in range(buckets + 1)]).drop_duplicates().values.copy()

    if len(breakpoints) < 2:
        return 0.0

    breakpoints[0] = float("-inf")
    breakpoints[-1] = float("inf")

    expected_counts = pd.cut(expected, bins=breakpoints, include_lowest=True).value_counts(normalize=True)
    actual_counts = pd.cut(actual, bins=breakpoints, include_lowest=True).value_counts(normalize=True)

    expected_counts, actual_counts = expected_counts.align(actual_counts, fill_value=0)

    expected_counts = expected_counts.replace(0, 0.0001).astype(float)
    actual_counts = actual_counts.replace(0, 0.0001).astype(float)

    ratio = actual_counts / expected_counts

    return float(((actual_counts - expected_counts) * np.log(ratio)).sum())


def main():
    reference_df = pd.read_csv(REFERENCE_DATA)
    current_df = pd.read_csv(CURRENT_DATA)

    numeric_cols = reference_df.select_dtypes(include="number").columns.tolist()

    drift_results = []
    for col in numeric_cols:
        if col in current_df.columns:
            score = psi(reference_df[col], current_df[col])
            drift_results.append({
                "feature": col,
                "psi": score,
                "status": "drift" if score is not None and score > 0.2 else "stable"
            })

    drift_df = pd.DataFrame(drift_results).sort_values("psi", ascending=False)

    summary = {
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "features_checked": int(len(drift_df)),
        "drifted_features": int((drift_df["status"] == "drift").sum()) if not drift_df.empty else 0,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    html = f"""
    <html>
    <head><title>Customer Churn Monitoring Report</title></head>
    <body>
        <h1>Customer Churn Monitoring Report</h1>
        <h2>Summary</h2>
        <pre>{json.dumps(summary, indent=4)}</pre>
        <h2>Feature Drift PSI</h2>
        {drift_df.to_html(index=False)}
    </body>
    </html>
    """

    REPORT_PATH.write_text(html, encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("Monitoring report saved:", REPORT_PATH)
    print("Monitoring summary saved:", SUMMARY_PATH)
    print(summary)


if __name__ == "__main__":
    main()
