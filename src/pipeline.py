"""Run the whole project end to end with one command.

Run:  python -m src.pipeline            # includes the 5-seed robustness check (~30 s)
      python -m src.pipeline --quick    # skip it
"""
import sys

from src import monitor, robustness, simulate


def main() -> None:
    quick = "--quick" in sys.argv
    sys.argv = sys.argv[:1]
    steps = [("1/3 Simulate 18 monthly cohorts", simulate.main),
             ("2/3 Monitor drift and performance, replay retraining policies", monitor.main)]
    if not quick:
        steps.append(("3/3 Repeat on 5 random seeds", robustness.main))
    for title, step in steps:
        print(f"\n=== {title} ===")
        step()


if __name__ == "__main__":
    main()
