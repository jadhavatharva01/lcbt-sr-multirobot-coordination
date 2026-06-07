import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt


SELECTED_MODES = [
    "auction_only",
    "fixed_intent",
    "conservative_wait",
    "cbba_like_consensus",
    "event_triggered_consensus",
    "ours_layered",
]

MODE_LABELS = {
    "auction_only": "Auction\nOnly",
    "fixed_intent": "Fixed\nIntent",
    "conservative_wait": "Conservative\nWait",
    "cbba_like_consensus": "CBBA-like\nConsensus",
    "event_triggered_consensus": "Event-triggered\nConsensus",
    "ours_layered": "Our\nLayered BT",
}

METRICS = [
    ("mission_completion_time", "Mission Completion Time"),
    ("total_distance_energy_proxy", "Distance / Energy Proxy"),
    ("messages", "Message Count"),
    ("pre_duplicates", "PRE Duplicates"),
    ("crit_violations", "Crit Voilations"),
    ("winner_disagreement", "Winner Disagreement"),
]


def load_rows(csv_path):
    rows = {}

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            mode = row["mode"]
            if mode not in SELECTED_MODES:
                continue

            rows[mode] = row

    return rows


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 make_focused_comparison_figure.py <comparison_averages.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    rows = load_rows(csv_path)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.flatten()

    labels = [MODE_LABELS[m] for m in SELECTED_MODES]

    for ax, (metric, title) in zip(axes, METRICS):
        values = [float(rows[m][metric]) for m in SELECTED_MODES]

        ax.bar(labels, values)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "State-of-the-Art-Inspired Baselines vs. Our Risk-Aware Layered BT",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    output_path = csv_path.parent / "focused_state_of_art_comparison.png"
    plt.savefig(output_path, dpi=240)

    print(f"Saved focused comparison figure to: {output_path}")


if __name__ == "__main__":
    main()
