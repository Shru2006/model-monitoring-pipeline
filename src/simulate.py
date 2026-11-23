"""Simulate 18 monthly cohorts of everyday-banking customers with a macro shock.

Months 1-7 are stable. From month 8, rising interest rates hit the base:
  * Data drift: savings growth falls, credit card use and buy-now-pay-later rise,
    and people use the borrowing power calculator more (checking what they can
    still afford).
  * Concept drift: research activity predicts applying less well than before,
    and deposit readiness matters more, because affordability, not interest,
    now decides who applies.

The label (applied within 90 days) only becomes known 3 months after scoring,
which is why input drift is the early-warning signal.

Run:  python -m src.simulate
"""
import numpy as np
import pandas as pd

from src.config import CUSTOMERS_PER_MONTH, MONTHLY_DATA_PATH, N_MONTHS, RANDOM_STATE, ensure_dirs


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def shock(month: int) -> float:
    """0 before month 8, ramping to 1 by month 14."""
    return float(np.clip((month - 7) / 7, 0, 1))


def simulate_month(month: int, n: int, rng: np.random.Generator) -> pd.DataFrame:
    s = shock(month)
    income = rng.lognormal(np.log(92_000), 0.35, n).clip(40_000, 400_000).round(-2)
    deposit_ratio = (rng.beta(2, 5, n) * 0.7 * (1 - 0.15 * s)).round(3)
    savings_growth = rng.normal(0.06 - 0.10 * s, 0.10 + 0.03 * s, n).round(3)
    calculator = rng.poisson(1.5 + 2.0 * s, n)
    page_views = rng.poisson(4.0, n)
    rent_to_income = rng.normal(0.28 + 0.03 * s, 0.07, n).clip(0.08, 0.7).round(3)
    cc_util = rng.beta(2 + 1.5 * s, 4 - 1.0 * s, n).round(3)
    bnpl = rng.poisson(1.2 + 1.5 * s, n)
    tenure = rng.gamma(2, 30, n).clip(1, 300).round()
    spoke = rng.binomial(1, 0.06, n)

    intent = 0.35 * calculator + 0.15 * page_views
    logit = (
        -4.4
        + (1.30 - 1.10 * s) * intent          # research matters less after the shock
        + (3.0 + 7.0 * s) * deposit_ratio     # affordability matters more
        + 3.0 * savings_growth
        - 1.5 * cc_util
        - 0.12 * bnpl
        - 2.0 * (rent_to_income - 0.28)
        + 1.3 * spoke
        + 0.003 * (income / 1000 - 92)
        - 0.6 * s                             # fewer applications overall
    )
    applied = rng.binomial(1, _sigmoid(logit))
    return pd.DataFrame({
        "customer_id": [f"M{month:02d}-{i:05d}" for i in range(n)],
        "month": month,
        "annual_income": income,
        "deposit_ratio": deposit_ratio,
        "savings_growth_6m": savings_growth,
        "calculator_uses_90d": calculator,
        "home_loan_page_views_30d": page_views,
        "rent_to_income": rent_to_income,
        "credit_card_utilisation": cc_util,
        "bnpl_txn_3m": bnpl,
        "tenure_months": tenure,
        "spoke_to_lender": spoke,
        "applied_90d": applied,
    })


def simulate(n_months: int = N_MONTHS, per_month: int = CUSTOMERS_PER_MONTH,
             seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.concat([simulate_month(m, per_month, rng) for m in range(1, n_months + 1)], ignore_index=True)


def main() -> None:
    ensure_dirs()
    df = simulate()
    df.to_csv(MONTHLY_DATA_PATH, index=False)
    print(f"Wrote {len(df):,} rows -> {MONTHLY_DATA_PATH}")
    print(df.groupby("month")["applied_90d"].mean().round(4).to_string())


if __name__ == "__main__":
    main()
