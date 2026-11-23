"""Re-run the whole simulation and monitor on several random seeds, so the
headline numbers are not one lucky draw.

Run:  python -m src.robustness
"""
import json

import pandas as pd

from src.config import REPORTS_DIR
from src.monitor import run
from src.simulate import simulate

SEEDS = [1, 2, 3, 4, 5]


def main() -> None:
    rows = []
    for seed in SEEDS:
        out = run(simulate(seed=seed))
        p = out["policies"]
        rows.append({
            "seed": seed,
            "drift_alert_month": out["drift_alert_month"],
            "performance_alert_known_month": out["performance_alert_known_month"],
            "never_retrain_capture_12_18": p["Never retrain"]["mean_capture_months_12_18"],
            "perf_alert_retrain_capture_12_18": p["Retrain on performance alert"]["mean_capture_months_12_18"],
            "drift_retrain_capture_12_18": p["Retrain on drift alert"]["mean_capture_months_12_18"],
        })
        print(rows[-1])
    df = pd.DataFrame(rows)
    df["drift_retrain_gain"] = df["drift_retrain_capture_12_18"] - df["never_retrain_capture_12_18"]
    summary = {
        "seeds": SEEDS,
        "runs": df.round(4).to_dict(orient="records"),
        "drift_retrain_gain_mean": round(float(df["drift_retrain_gain"].mean()), 4),
        "drift_retrain_gain_min": round(float(df["drift_retrain_gain"].min()), 4),
        "drift_retrain_gain_max": round(float(df["drift_retrain_gain"].max()), 4),
    }
    (REPORTS_DIR / "robustness.json").write_text(json.dumps(summary, indent=2))
    print({k: v for k, v in summary.items() if k != "runs"})


if __name__ == "__main__":
    main()
