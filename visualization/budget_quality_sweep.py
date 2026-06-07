import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt

from baseline_comparison_sim import SimConfig, run_single_trial


METHODS = [
    "cbba_like_consensus",
    "ours_layered_full",
]

METHOD_LABELS = {
    "cbba_like_consensus": "CBBA-like Consensus",
    "ours_layered_full": "Our Full Layered BT",
}

BANDWIDTH_QUALITIES = {
    "good": {
        "delay_mean": 2.0,
        "delay_jitter": 1.0,
        "drop_rate": 0.05,
    },
    "medium": {
        "delay_mean": 5.0,
        "delay_jitter": 2.0,
        "drop_rate": 0.15,
    },
    "poor": {
        "delay_mean": 8.0,
        "delay_jitter": 3.0,
        "drop_rate": 0.25,
    },
}


def run_method(method, seed, delay_mean, delay_jitter, drop_rate):
    if method == "ours_layered_full":
        cfg_mode = "ours_layered"
        use_grasp_zone = True
    else:
        cfg_mode = method
        use_grasp_zone = False

    cfg = SimConfig(
        mode=cfg_mode,
        seed=seed,
        delay_mean=delay_mean,
        delay_jitter=delay_jitter,
        drop_rate=drop_rate,
        use_grasp_zone=use_grasp_zone,
    )

    row = run_single_trial(cfg)
    row["method"] = method
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument(
        "--budgets",
        type=str,
        default="20,30,40,50,60,80,100",
        help="Comma-separated message budgets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "phase1a_ws/results/budget_quality_sweep"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    budgets = [float(x.strip()) for x in args.budgets.split(",")]

    rows = []

    for quality_name, params in BANDWIDTH_QUALITIES.items():
        delay_mean = params["delay_mean"]
        delay_jitter = params["delay_jitter"]
        drop_rate = params["drop_rate"]

        for method in METHODS:
            trial_rows = []

            for i in range(args.trials):
                row = run_method(
                    method=method,
                    seed=9000 + i,
                    delay_mean=delay_mean,
                    delay_jitter=delay_jitter,
                    drop_rate=drop_rate,
                )
                trial_rows.append(row)

            for budget in budgets:
                safe_success_count = 0
                bandwidth_feasible_safe_success_count = 0
                budget_exceeded_count = 0
                total_messages = 0.0
                total_mission_time = 0.0
                total_crit_violations = 0.0

                for row in trial_rows:
                    messages = row["messages"]
                    mission_success = row["mission_success"]
                    crit_violations = row["crit_violations"]

                    safe_success = (
                        mission_success == 1
                        and crit_violations == 0
                    )

                    bandwidth_feasible_safe_success = (
                        safe_success
                        and messages <= budget
                    )

                    safe_success_count += 1 if safe_success else 0
                    bandwidth_feasible_safe_success_count += 1 if bandwidth_feasible_safe_success else 0
                    budget_exceeded_count += 1 if messages > budget else 0

                    total_messages += messages
                    total_mission_time += row["mission_completion_time"]
                    total_crit_violations += crit_violations

                n = len(trial_rows)

                rows.append({
                    "bandwidth_quality": quality_name,
                    "delay_mean": delay_mean,
                    "delay_jitter": delay_jitter,
                    "drop_rate": drop_rate,
                    "method": method,
                    "message_budget": budget,
                    "safe_success_rate": safe_success_count / n,
                    "bandwidth_feasible_safe_success_rate": bandwidth_feasible_safe_success_count / n,
                    "budget_exceeded_rate": budget_exceeded_count / n,
                    "avg_messages": total_messages / n,
                    "avg_mission_time": total_mission_time / n,
                    "avg_crit_violations": total_crit_violations / n,
                })

    csv_path = output_dir / "budget_quality_sweep.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved sweep CSV: {csv_path}")

    # Plot 1: bandwidth-feasible safe success rate
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, quality_name in zip(axes, BANDWIDTH_QUALITIES.keys()):
        for method in METHODS:
            method_rows = [
                r for r in rows
                if r["bandwidth_quality"] == quality_name
                and r["method"] == method
            ]

            method_rows = sorted(method_rows, key=lambda r: r["message_budget"])

            x = [r["message_budget"] for r in method_rows]
            y = [r["bandwidth_feasible_safe_success_rate"] for r in method_rows]

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=METHOD_LABELS[method],
            )

        ax.set_title(f"{quality_name.title()} Bandwidth")
        ax.set_xlabel("Message Budget")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Bandwidth-Feasible Safe Success Rate")
    axes[-1].legend(loc="lower right")

    fig.suptitle(
        "Safe Success Under Message Budget and Bandwidth Quality",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out1 = output_dir / "budget_quality_safe_success.png"
    plt.savefig(out1, dpi=240)
    print(f"Saved graph: {out1}")

    # Plot 2: budget exceeded rate
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, quality_name in zip(axes, BANDWIDTH_QUALITIES.keys()):
        for method in METHODS:
            method_rows = [
                r for r in rows
                if r["bandwidth_quality"] == quality_name
                and r["method"] == method
            ]

            method_rows = sorted(method_rows, key=lambda r: r["message_budget"])

            x = [r["message_budget"] for r in method_rows]
            y = [r["budget_exceeded_rate"] for r in method_rows]

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=METHOD_LABELS[method],
            )

        ax.set_title(f"{quality_name.title()} Bandwidth")
        ax.set_xlabel("Message Budget")
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Budget Exceeded Rate")
    axes[-1].legend(loc="upper right")

    fig.suptitle(
        "Message Budget Exceedance Under Bandwidth Constraints",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out2 = output_dir / "budget_quality_exceeded_rate.png"
    plt.savefig(out2, dpi=240)
    print(f"Saved graph: {out2}")

    print("\n=== Summary Rows ===")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
