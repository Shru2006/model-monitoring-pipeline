"""Charts and a markdown monitoring report from the monitor's output."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, PSI_ALERT, PSI_WATCH, REPORT_PATH

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
POLICY_COLORS = {"Never retrain": "#eb6834", "Retrain on performance alert": "#eda100",
                 "Retrain on drift alert": "#2a78d6"}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _drift_chart(table: pd.DataFrame, out: dict):
    psi_cols = [c for c in table.columns if c.startswith("psi_")]
    top = table[psi_cols].max().sort_values(ascending=False).head(4).index
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=150)
    _style(ax)
    for col, colr in zip(top, SERIES):
        ax.plot(table.month, table[col], color=colr, linewidth=2, marker="o", markersize=4,
                label=col.removeprefix("psi_").replace("_", " "))
    ax.plot(table.month, table.score_psi, color=MUTED, linewidth=2, linestyle="--", label="model score")
    ax.axhline(PSI_ALERT, color=INK, linewidth=0.8, linestyle=":")
    ax.axhline(PSI_WATCH, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate("alert (0.2)", (table.month.min(), PSI_ALERT), xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=INK)
    if out["drift_alert_month"]:
        ax.axvline(out["drift_alert_month"], color=INK, linewidth=0.8)
        ax.annotate(f"first drift alert: month {out['drift_alert_month']}", (out["drift_alert_month"], ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Month in production", color=INK, fontsize=10)
    ax.set_ylabel("Population stability index vs training", color=INK, fontsize=10)
    ax.set_title("Inputs drift months before the score distribution does", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="center left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_psi.png")
    plt.close(fig)


def _policy_chart(table: pd.DataFrame, out: dict):
    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=150)
    _style(ax)
    never = table["capture_Never retrain"]
    for name, colr in POLICY_COLORS.items():
        label = name
        same = name != "Never retrain" and (table[f"capture_{name}"] - never).abs().max() < 1e-9
        if same:
            label += " (no alert in time: same as never)"
        ax.plot(table.month, table[f"capture_{name}"] * 100, color=colr, marker="o", markersize=4, label=label,
                linewidth=2, linestyle=(0, (4, 3)) if same else "-", zorder=2 if same else 1)
        for r in out["policies"][name]["retrains"]:
            ax.scatter([r["month"]], [table.loc[table.month == r["month"], f"capture_{name}"].iloc[0] * 100],
                       s=70, facecolor="white", edgecolor=colr, linewidth=2, zorder=3)
    ax.axhline(out["capture_alert_threshold"] * 100, color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate("performance alert threshold", (table.month.max(), out["capture_alert_threshold"] * 100),
                xytext=(0, -12), textcoords="offset points", fontsize=8, color=MUTED, ha="right")
    ax.set_xlabel("Month in production (open circle = retrain)", color=INK, fontsize=10)
    ax.set_ylabel("Applicants captured in top 10% (%)", color=INK, fontsize=10)
    ax.set_title("Retraining when inputs drift keeps the model useful", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "retrain_policies.png")
    plt.close(fig)


def write_report(table: pd.DataFrame, out: dict) -> None:
    _drift_chart(table, out)
    _policy_chart(table, out)
    known = out["performance_alert_known_month"]
    lines = [
        "# Monitoring report", "",
        f"- Baseline (time-based validation, month 6): AUC {out['baseline']['auc']:.3f}, "
        f"top-decile capture {out['baseline']['top_decile_capture']:.1%}",
        f"- First drift alert (PSI > {PSI_ALERT}): month {out['drift_alert_month']}",
        f"- First performance alert confirmed by labels: "
        + (f"month {known} (cohort {out['performance_alert_cohort']})" if known else "not within the window"),
        "", "| Month | Score PSI | Max feature PSI | Top drifting feature | AUC | Top-decile capture | Drift alert |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(f"| {r.month} | {r.score_psi:.3f} | {r.max_feature_psi:.2f} | {r.top_drifting_feature} | "
                     f"{r.auc:.3f} | {r.top_decile_capture:.1%} | {'yes' if r.drift_alert else ''} |")
    lines += ["", "| Policy | Mean capture, months 7-18 | Mean capture, months 12-18 | Retrains |", "|---|---|---|---|"]
    for name, p in out["policies"].items():
        lines.append(f"| {name} | {p['mean_capture_all_months']:.1%} | {p['mean_capture_months_12_18']:.1%} | "
                     f"{', '.join(str(r['month']) for r in p['retrains']) or 'none'} |")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
