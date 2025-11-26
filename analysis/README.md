# Benchmark Analysis

This directory contains scripts to analyze the benchmark results produced by the `mchep` library.

## How to Run

1.  **Run the Rust Benchmark Runner:**

    First, you need to generate the benchmark data. From the root of the `mchep` repository, run the `full_benchmark` bench target. You can configure the benchmark parameters using command-line arguments.

    The `--features` flag is used to enable the `benchmarking` code and other optional features like `gpu` or `mpi`.

    ```bash
    # Example: Run benchmarks for 4 dimensions with 1M evaluations per iteration
    # This will run the scalar, SIMD, and GPU backends.
    cargo bench --bench full_benchmark --features="benchmarking gpu" -- --dim 4 --n-eval 1000000
    ```

    This will produce a `benchmark_results.json` file in the root of the repository.

    To run the MPI benchmarks, you need to have an MPI implementation installed (e.g., OpenMPI). You can then run the benchmark with `mpirun`:

    ```bash
    # Example: Run MPI benchmarks with 4 processes
    mpirun -n 4 cargo bench --bench full_benchmark --features="benchmarking mpi" -- --backend mpi --dim 4 --n-eval 1000000
    ```

2.  **Run the Python Analysis Script:**

    Next, run the Python script to generate plots from the benchmark data. Make sure you have the required Python packages installed:

    ```bash
    pip install pandas matplotlib seaborn
    ```

    Then, run the script:

    ```bash
    python analysis/plot_benchmarks.py benchmark_results.json
    ```

    The script will create a directory `analysis/plots` containing the generated plots.
