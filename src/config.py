"""Central settings for the monitoring pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MONTHLY_DATA_PATH = DATA_DIR / "monthly_customers_synthetic.csv"
METRICS_PATH = REPORTS_DIR / "metrics.json"
MONTHLY_TABLE_PATH = REPORTS_DIR / "monitoring_by_month.csv"
REPORT_PATH = REPORTS_DIR / "monitoring_report.md"

ID_COL = "customer_id"
MONTH_COL = "month"
TARGET = "applied_90d"

FEATURES = [
    "annual_income",
    "deposit_ratio",
    "savings_growth_6m",
    "calculator_uses_90d",
    "home_loan_page_views_30d",
    "rent_to_income",
    "credit_card_utilisation",
    "bnpl_txn_3m",
    "tenure_months",
    "spoke_to_lender",
]

N_MONTHS = 18                 # months 1..18
CUSTOMERS_PER_MONTH = 6_000
TRAIN_MONTHS = range(1, 7)    # model trained on months 1-6, deployed from month 7
LABEL_LAG_MONTHS = 3          # "applied within 90 days" is only known 3 months later

# Alert rules
PSI_ALERT = 0.20              # population stability index: > 0.2 is a material shift
PSI_WATCH = 0.10
CAPTURE_ALERT_RATIO = 0.85    # alert if top-decile capture falls below 85% of baseline
RETRAIN_WINDOW_MONTHS = 6     # retrain on the latest 6 labelled months

RANDOM_STATE = 42


def ensure_dirs() -> None:
    for d in (DATA_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
