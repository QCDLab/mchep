#!/bin/bash

# Exit on error
set -e

# Number of runs for averaging
N_RUNS=5

echo "Starting comprehensive benchmark..."
echo "==================================="

# --- MCHEP Scalar Benchmark ---
echo ""
echo "--- Benchmarking MCHEP (Scalar, Optimized) ---"

# Build benchmark
echo "Building mchep scalar benchmark..."
make benchmark_mchep

# Run benchmark
echo "Running mchep scalar benchmark ($N_RUNS runs)..."
MCHEP_SCALAR_TIMES=()
for i in $(seq 1 $N_RUNS); do
    TIME_OUTPUT=$( (time -p ./benchmark_mchep) 2>&1 )
    REAL_TIME=$(echo "$TIME_OUTPUT" | grep real | awk '{print $2}' | tr ',' '.')
    MCHEP_SCALAR_TIMES+=($REAL_TIME)
    echo "Run $i: $REAL_TIME s"
done
MCHEP_SCALAR_AVG_TIME=$(echo "${MCHEP_SCALAR_TIMES[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; print s/NF}')
echo "Average MCHEP (Scalar) time: $MCHEP_SCALAR_AVG_TIME s"

# --- MCHEP SIMD Benchmark ---
echo ""
echo "--- Benchmarking MCHEP (SIMD) ---"

# Build benchmark
echo "Building mchep SIMD benchmark..."
make benchmark_mchep_simd

# Run benchmark
echo "Running mchep SIMD benchmark ($N_RUNS runs)..."
MCHEP_SIMD_TIMES=()
for i in $(seq 1 $N_RUNS); do
    TIME_OUTPUT=$( (time -p ./benchmark_mchep_simd) 2>&1 )
    REAL_TIME=$(echo "$TIME_OUTPUT" | grep real | awk '{print $2}' | tr ',' '.')
    MCHEP_SIMD_TIMES+=($REAL_TIME)
    echo "Run $i: $REAL_TIME s"
done
MCHEP_SIMD_AVG_TIME=$(echo "${MCHEP_SIMD_TIMES[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; print s/NF}')
echo "Average MCHEP (SIMD) time: $MCHEP_SIMD_AVG_TIME s"

# --- LIBCUBA Benchmark ---
echo ""
echo "--- Benchmarking LIBCUBA ---"

# Build benchmark
echo "Building libcuba benchmark..."
make cuba_example

# Run benchmark
echo "Running libcuba benchmark ($N_RUNS runs)..."
LIBCUBA_TIMES=()
for i in $(seq 1 $N_RUNS); do
    TIME_OUTPUT=$( (time -p make run_cuba) 2>&1 )
    REAL_TIME=$(echo "$TIME_OUTPUT" | grep real | awk '{print $2}' | tr ',' '.')
    LIBCUBA_TIMES+=($REAL_TIME)
    echo "Run $i: $REAL_TIME s"
done
LIBCUBA_AVG_TIME=$(echo "${LIBCUBA_TIMES[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; print s/NF}')
echo "Average LIBCUBA time: $LIBCUBA_AVG_TIME s"

# --- Summary ---
echo ""
echo "--- Benchmark Summary ---"
echo "MCHEP (Scalar, Optimized): $MCHEP_SCALAR_AVG_TIME s"
echo "MCHEP (SIMD):              $MCHEP_SIMD_AVG_TIME s"
echo "LIBCUBA (Vegas):           $LIBCUBA_AVG_TIME s"
echo "======================="
