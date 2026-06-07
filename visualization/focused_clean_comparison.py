import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from baseline_comparison_sim import SimConfig, run_single_trial, average_results, save_csv


MODES = [
    "fixed_intent",
    "cbba_like_consensus",
    "ours_layered_no_guard",
    "ours_layered_full",
]

MODE_LABELS = {
    "fixed_intent": "Fixed\nIntent",
    "cbba_like_consensus": "CBBA-like\nConsensus",
    "ours_layered_no_guard": "Our Layered\nNo Guard",
    "ours_layered_full": "Our Full\nLayered",
}

METRICS = [
    ("mission_completion_time", "Mission Completion Time"),
    ("planning_allocation_time", "Planning / Allocation Time"),
    ("messages", "Message Count"),
    ("total_distance_energy_proxy", "Distance / Energy Proxy"),
    ("pre_duplicates", "PRE Duplicates"),
    ("crit_violations", "CRIT Violations"),
    ("grasp_denials", "Grasp-zone Denials"),
]


def run_mode(mode, seed, args):
    if mode == "ours_layered_no_guard":
        cfg_mode = "ours_layered"
        use_grasp_zone = False
    elif mode == "ours_layered_full":
        cfg_mode = "ours_layered"
        use_grasp_zone = True
    else:
        cfg_mode = mode
        use_grasp_zone = False

    cfg = SimConfig(
        mode=cfg_mode,
        seed=seed,
        delay_mean=args.delay_mean,
        delay_jitter=args.delay_jitter,
        drop_rate=args.drop_rate,
        use_grasp_zone=use_grasp_zone,
    )

    row = run_single_trial(cfg)
    row["mode"] = mode
    return row


def make_plot(avg_rows, output_dir):
    rows_by_mode = {r["mode"]: r for r in avg_rows}
    labels = [MODE_LABELS[m] for m in MODES]

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, METRICS):
        values = [rows_by_mode[m][metric] for m in MODES]

        ax.bar(labels, values)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    # Hide unused axes
    for j in range(len(METRICS), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Clean Focused Comparison: Fixed Intent vs CBBA vs Our Layered Method",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = output_dir / "focused_clean_comparison.png"
    plt.savefig(out, dpi=240)
    print(f"Saved figure: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--delay_mean", type=float, default=5.0)
    parser.add_argument("--delay_jitter", type=float, default=2.0)
    parser.add_argument("--drop_rate", type=float, default=0.15)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "phase1a_ws/results/focused_clean_comparison"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    avg_rows = []

    for mode in MODES:
        mode_rows = []

        for i in range(args.trials):
            row = run_mode(mode, seed=5000 + i, args=args)
            mode_rows.append(row)
            all_rows.append(row)

        avg_rows.append(average_results(mode_rows))

    raw_csv = output_dir / "focused_clean_raw_trials.csv"
    avg_csv = output_dir / "focused_clean_averages.csv"

    save_csv(all_rows, raw_csv)
    save_csv(avg_rows, avg_csv)

    print(f"Saved raw trials: {raw_csv}")
    print(f"Saved averages: {avg_csv}")

    print("\n=== Clean Focused Average Results ===")
    for row in avg_rows:
        print(row)

    make_plot(avg_rows, output_dir)


if __name__ == "__main__":
    main()
