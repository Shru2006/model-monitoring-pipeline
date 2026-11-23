"""Replay 12 months of production for the propensity model and monitor it.

Each month the monitor can see:
  * the new cohort's inputs and scores  -> feature and score PSI (immediately)
  * outcomes for the cohort scored 3 months ago -> AUC and top-decile capture

Three retraining policies are replayed on the same months:
  * never retrain
  * retrain when a performance alert is known (labels confirm the decay)
  * retrain when a drift alert fires (inputs shift; labels may still be stale)

Run:  python -m src.monitor
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd
from lightgbm import LGBMClassifier

from src.config import (CAPTURE_ALERT_RATIO, FEATURES, LABEL_LAG_MONTHS, METRICS_PATH, MONTH_COL,
                        MONTHLY_DATA_PATH, MONTHLY_TABLE_PATH, N_MONTHS, PSI_ALERT, RANDOM_STATE,
                        RETRAIN_WINDOW_MONTHS, TARGET, TRAIN_MONTHS, ensure_dirs)
from src.metrics import performance, psi, psi_table

DEPLOY_MONTHS = range(max(TRAIN_MONTHS) + 1, N_MONTHS + 1)
COOLDOWN_MONTHS = 3


def make_model() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=RANDOM_STATE, verbose=-1)


def fit(df: pd.DataFrame, months) -> LGBMClassifier:
    part = df[df[MONTH_COL].isin(list(months))]
    return make_model().fit(part[FEATURES], part[TARGET])


def score(model, frame: pd.DataFrame):
    return model.predict_proba(frame[FEATURES])[:, 1]


@dataclass
class Policy:
    name: str
    trigger: str                         # "never", "performance", "drift"
    model: LGBMClassifier = None
    last_retrain: int = -99
    retrains: list = field(default_factory=list)


def run(df: pd.DataFrame) -> dict:
    cohorts = {m: g for m, g in df.groupby(MONTH_COL)}

    # Time-based validation: fit on months 1-5, baseline = performance on month 6.
    train_months = list(TRAIN_MONTHS)
    base_model = fit(df, train_months[:-1])
    baseline = performance(cohorts[train_months[-1]][TARGET], score(base_model, cohorts[train_months[-1]]))
    capture_threshold = CAPTURE_ALERT_RATIO * baseline["top_decile_capture"]

    reference = df[df[MONTH_COL].isin(train_months)]
    ref_scores = score(base_model, reference)

    policies = [Policy("Never retrain", "never", base_model),
                Policy("Retrain on performance alert", "performance", base_model),
                Policy("Retrain on drift alert", "drift", base_model)]

    # Pass 1: what the *never retrain* monitor sees each month.
    rows = []
    for m in DEPLOY_MONTHS:
        cur = cohorts[m]
        feat_psi = psi_table(reference, cur, FEATURES)
        s = score(base_model, cur)
        perf = performance(cur[TARGET], s)
        rows.append({
            "month": m,
            "score_psi": psi(ref_scores, s),
            "max_feature_psi": feat_psi.max(),
            "top_drifting_feature": feat_psi.idxmax(),
            **{f"psi_{c}": v for c, v in feat_psi.items()},
            "auc": perf["auc"],
            "top_decile_capture": perf["top_decile_capture"],
            "labels_known_in_month": m + LABEL_LAG_MONTHS,
        })
    table = pd.DataFrame(rows)
    table["drift_alert"] = (table["max_feature_psi"] > PSI_ALERT) | (table["score_psi"] > PSI_ALERT)
    table["performance_alert"] = table["top_decile_capture"] < capture_threshold

    drift_alert_month = int(table.loc[table.drift_alert, "month"].min()) if table.drift_alert.any() else None
    perf_cohort = int(table.loc[table.performance_alert, "month"].min()) if table.performance_alert.any() else None
    perf_known_month = perf_cohort + LABEL_LAG_MONTHS if perf_cohort else None

    # Pass 2: replay each retraining policy month by month, using only what was knowable.
    capture_by_policy = {p.name: {} for p in policies}
    for m in DEPLOY_MONTHS:
        for p in policies:
            # Decide at the start of month m whether to retrain, from information known by then.
            labelled = [c for c in range(1, m) if c + LABEL_LAG_MONTHS <= m]
            known = table[table["labels_known_in_month"] <= m]
            triggered = False
            if p.trigger == "performance" and len(known):
                recent = known.iloc[-1]
                cur_capture = performance(cohorts[int(recent.month)][TARGET],
                                          score(p.model, cohorts[int(recent.month)]))["top_decile_capture"]
                triggered = cur_capture < capture_threshold
            elif p.trigger == "drift":
                prev = table[table.month == m - 1]
                triggered = bool(len(prev)) and bool(prev.drift_alert.iloc[0])
            if triggered and m - p.last_retrain >= COOLDOWN_MONTHS and labelled:
                p.model = fit(df, labelled[-RETRAIN_WINDOW_MONTHS:])
                p.last_retrain = m
                p.retrains.append({"month": m, "trained_on": [labelled[-RETRAIN_WINDOW_MONTHS:][0], labelled[-1]]})
            capture_by_policy[p.name][m] = performance(cohorts[m][TARGET], score(p.model, cohorts[m]))["top_decile_capture"]

    policy_df = pd.DataFrame(capture_by_policy)
    post = [m for m in DEPLOY_MONTHS if m >= 12]
    summary = {
        p.name: {
            "mean_capture_all_months": round(float(policy_df[p.name].mean()), 4),
            "mean_capture_months_12_18": round(float(policy_df.loc[post, p.name].mean()), 4),
            "retrains": p.retrains,
        } for p in policies
    }

    table = table.merge(policy_df.rename(columns=lambda c: f"capture_{c}").rename_axis("month").reset_index(),
                        on="month")
    return {
        "baseline": {k: round(v, 4) for k, v in baseline.items()},
        "capture_alert_threshold": round(capture_threshold, 4),
        "drift_alert_month": drift_alert_month,
        "performance_alert_cohort": perf_cohort,
        "performance_alert_known_month": perf_known_month,
        "lead_time_months": (perf_known_month - drift_alert_month) if (perf_known_month and drift_alert_month) else None,
        "policies": summary,
        "table": table,
    }


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(MONTHLY_DATA_PATH)
    out = run(df)
    table = out.pop("table")
    table.round(4).to_csv(MONTHLY_TABLE_PATH, index=False)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    cols = ["month", "score_psi", "max_feature_psi", "top_drifting_feature", "auc", "top_decile_capture",
            "drift_alert", "performance_alert"]
    print(table[cols].round(3).to_string(index=False))
    print(json.dumps(out, indent=2))

    from src.report import write_report
    write_report(table, out)


if __name__ == "__main__":
    main()
