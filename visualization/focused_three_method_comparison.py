import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from baseline_comparison_sim import SimConfig, run_single_trial, average_results, save_csv


FOCUSED_MODES = [
    "fixed_intent",
    "cbba_like_consensus",
    "ours_layered",
]

MODE_LABELS = {
    "fixed_intent": "Fixed\nIntent",
    "cbba_like_consensus": "CBBA-like\nConsensus",
    "ours_layered": "Our\nLayered BT",
}

METRICS_TO_PLOT = [
    ("mission_completion_time", "Mission Completion Time"),
    ("planning_allocation_time", "Planning / Allocation Time"),
    ("messages", "Message Count"),
    ("total_distance_energy_proxy", "Distance / Energy Proxy"),
    ("pre_duplicates", "PRE Duplicates"),
    ("crit_violations", "CRIT Violations"),
]


def make_plot(avg_rows, output_dir):
    rows_by_mode = {r["mode"]: r for r in avg_rows}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    labels = [MODE_LABELS[m] for m in FOCUSED_MODES]

    for ax, (metric, title) in zip(axes, METRICS_TO_PLOT):
        values = [rows_by_mode[m][metric] for m in FOCUSED_MODES]

        ax.bar(labels, values)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle(
        "Focused Comparison: Fixed Intent vs. CBBA-like Consensus vs. Our Layered BT",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = output_dir / "focused_three_method_comparison.png"
    plt.savefig(out, dpi=240)
    print(f"Saved focused comparison figure: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--delay_mean", type=float, default=3.0)
    parser.add_argument("--delay_jitter", type=float, default=1.5)
    parser.add_argument("--drop_rate", type=float, default=0.15)
    parser.add_argument("--grasp_zone", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "phase1a_ws/results/focused_three_method_comparison"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    avg_rows = []

    for mode in FOCUSED_MODES:
        mode_rows = []

        for i in range(args.trials):
            cfg = SimConfig(
                mode=mode,
                seed=1000 + i,
                delay_mean=args.delay_mean,
                delay_jitter=args.delay_jitter,
                drop_rate=args.drop_rate,
                use_grasp_zone=args.grasp_zone,
            )

            row = run_single_trial(cfg)
            mode_rows.append(row)
            all_rows.append(row)

        avg_rows.append(average_results(mode_rows))

    raw_csv = output_dir / "focused_raw_trials.csv"
    avg_csv = output_dir / "focused_averages.csv"

    save_csv(all_rows, raw_csv)
    save_csv(avg_rows, avg_csv)

    print(f"Saved raw trials: {raw_csv}")
    print(f"Saved averages: {avg_csv}")

    print("\n=== Focused Average Results ===")
    for row in avg_rows:
        print(row)

    make_plot(avg_rows, output_dir)


if __name__ == "__main__":
    main()
