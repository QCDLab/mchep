import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os


def plot_benchmarks(file_path):
    """
    Reads benchmark data from a JSON file, calculates accuracy metrics,
    and generates plots for performance and accuracy comparison.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    with open(file_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    # --- Data Processing ---
    # Combine backend and algorithm for a clearer legend
    df["integrator"] = df["algorithm"] + "-" + df["backend"]

    # Calculate sigma distance: |value - analytical| / error
    # This measures how many standard deviations the result is from the truth.
    df["sigma_dist"] = abs(df["value"] - df["analytical_result"]) / df["error"]
    df["rel_error"] = abs(df["value"] - df["analytical_result"]) / abs(
        df["analytical_result"]
    )

    # Create output directory
    output_dir = "analysis/plots"
    os.makedirs(output_dir, exist_ok=True)

    # --- Plotting ---
    sns.set_theme(style="whitegrid")

    # Plot 1: Execution Time
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x="integrand", y="time_s", hue="integrator")
    plt.title("Benchmark: Execution Time")
    plt.ylabel("Time (seconds, log scale)")
    plt.yscale("log")
    plt.xticks(rotation=15)
    plt.tight_layout()
    time_plot_path = os.path.join(output_dir, "benchmark_time.png")
    plt.savefig(time_plot_path)
    print(f"Execution time plot saved to {time_plot_path}")
    plt.close()

    # Plot 2: Accuracy (Sigma Distance)
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x="integrand", y="sigma_dist", hue="integrator")
    plt.axhline(1.0, color="r", linestyle="--", label="1-sigma deviation")
    plt.axhline(2.0, color="g", linestyle="--", label="2-sigma deviation")
    plt.title("Benchmark: Accuracy (Sigma Distance)")
    plt.ylabel("Sigma Distance (|Value - Analytical| / Error)")
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    sigma_plot_path = os.path.join(output_dir, "benchmark_sigma_dist.png")
    plt.savefig(sigma_plot_path)
    print(f"Sigma distance plot saved to {sigma_plot_path}")
    plt.close()

    # Plot 3: Relative Error
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x="integrand", y="rel_error", hue="integrator")
    plt.title("Benchmark: Relative Error")
    plt.ylabel("Relative Error (log scale)")
    plt.yscale("log")
    plt.xticks(rotation=15)
    plt.tight_layout()
    rel_error_plot_path = os.path.join(output_dir, "benchmark_rel_error.png")
    plt.savefig(rel_error_plot_path)
    print(f"Relative error plot saved to {rel_error_plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from mchep.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the benchmark_results.json file.",
        default="benchmark_results.json",
        nargs="?",
    )
    args = parser.parse_args()
    plot_benchmarks(args.file_path)
