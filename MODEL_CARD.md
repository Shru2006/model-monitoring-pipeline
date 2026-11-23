# Model card: first home buyer propensity model (monitored)

| | |
|---|---|
| **Purpose** | Rank existing everyday-banking customers by how likely they are to apply for their first home loan within 90 days, so lending specialists can prioritise conversations. |
| **Not for** | Credit decisions, pricing, or anything that changes what a customer is offered. It is a marketing propensity model. |
| **Model** | LightGBM classifier, 10 behavioural features (savings, deposit readiness, research activity, credit use, tenure). No protected attributes. |
| **Training data** | Synthetic monthly cohorts, months 1-5. Validated on month 6 (time-based split). |
| **Baseline** | AUC 0.749, top-decile capture 32.5% (month 6). |
| **Label** | Applied for a home loan within 90 days. Known **3 months** after scoring. |

## Monitoring

| Signal | Available | Rule |
|---|---|---|
| Feature drift (PSI per feature vs training) | Immediately | Alert if any feature PSI > 0.2 |
| Score drift (PSI of model scores) | Immediately | Alert if PSI > 0.2 |
| Performance (AUC, top-decile capture) | 3 months later | Alert if capture < 85% of baseline |

## Retraining policy

Retrain on the latest 6 **labelled** months when a drift alert fires, at most every 3 months. Replayed over 12 months of a simulated rate shock on 5 random seeds, this kept 5 to 11 points more applicants in the top 10% than never retraining (8.3 on average).

## Known limitations

- Synthetic data: the shock and its timing are designed, not observed.
- Retraining on drift uses labels that may still pre-date the shift. It worked here because the shift built up gradually.
- PSI flags change, not harm. A drift alert should trigger a review and a retrain candidate, which then has to beat the current model on the latest labelled month before it is promoted. That promotion check is not implemented yet.
