# Monitoring report

- Baseline (time-based validation, month 6): AUC 0.749, top-decile capture 32.5%
- First drift alert (PSI > 0.2): month 10
- First performance alert confirmed by labels: month 19 (cohort 16)

| Month | Score PSI | Max feature PSI | Top drifting feature | AUC | Top-decile capture | Drift alert |
|---|---|---|---|---|---|---|
| 7 | 0.001 | 0.00 | annual_income | 0.745 | 31.6% |  |
| 8 | 0.008 | 0.04 | calculator_uses_90d | 0.743 | 36.0% |  |
| 9 | 0.007 | 0.18 | calculator_uses_90d | 0.727 | 31.3% |  |
| 10 | 0.017 | 0.37 | calculator_uses_90d | 0.719 | 30.3% | yes |
| 11 | 0.025 | 0.56 | calculator_uses_90d | 0.726 | 28.6% | yes |
| 12 | 0.023 | 0.78 | calculator_uses_90d | 0.729 | 33.0% | yes |
| 13 | 0.027 | 1.04 | calculator_uses_90d | 0.681 | 30.2% | yes |
| 14 | 0.027 | 1.31 | calculator_uses_90d | 0.680 | 29.6% | yes |
| 15 | 0.030 | 1.28 | calculator_uses_90d | 0.705 | 30.2% | yes |
| 16 | 0.033 | 1.26 | credit_card_utilisation | 0.658 | 26.3% | yes |
| 17 | 0.029 | 1.29 | calculator_uses_90d | 0.703 | 31.4% | yes |
| 18 | 0.032 | 1.29 | calculator_uses_90d | 0.700 | 29.7% | yes |

| Policy | Mean capture, months 7-18 | Mean capture, months 12-18 | Retrains |
|---|---|---|---|
| Never retrain | 30.7% | 30.1% | none |
| Retrain on performance alert | 30.7% | 30.1% | none |
| Retrain on drift alert | 34.6% | 36.4% | 11, 14, 17 |
