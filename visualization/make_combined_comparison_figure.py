import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt


METRICS = [
    ("mission_completion_time", "Mission Completion Time"),
    ("total_distance_energy_proxy", "Distance / Energy Proxy"),
    ("pre_duplicates", "PRE Duplicates"),
    ("messages", "Messages"),
]


def load_rows(csv_path):
    rows = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            clean = {"mode": row["mode"]}

            for key, _ in METRICS:
                clean[key] = float(row[key])

            rows.append(clean)

    return rows


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 make_combined_comparison_figure.py <comparison_averages.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    rows = load_rows(csv_path)

    modes = [r["mode"] for r in rows]
    display_modes = [
        m.replace("_", "\n").title()
        for m in modes
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, METRICS):
        values = [r[metric] for r in rows]

        ax.bar(display_modes, values)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Representative Baseline Comparison Under Asynchronous Communication",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = csv_path.parent / "combined_comparison_figure.png"
    plt.savefig(output_path, dpi=220)

    print(f"Saved combined figure to: {output_path}")


if __name__ == "__main__":
    main()
