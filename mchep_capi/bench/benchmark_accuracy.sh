#!/bin/bash

# Exit on error
set -e

# Number of runs for averaging
N_RUNS=5

# Target accuracies to test (percentage)
ACCURACIES=(5.0 2.0 1.0 0.5 0.1)

echo "Starting accuracy-based benchmark..."
echo "====================================="
echo ""

# Build all benchmarks
echo "Building benchmarks..."
make benchmark_mchep_accuracy benchmark_mchep_simd_accuracy cuba_accuracy
echo ""

# CSV header
echo "Method,TargetAccuracy(%),Value,Error,ActualAccuracy(%),Time(ms),Extra"
echo "------,------------------,-----,-----,------------------,--------,-----"

for TARGET_ACC in "${ACCURACIES[@]}"; do
    echo "" >&2
    echo "--- Testing target accuracy: ${TARGET_ACC}% ---" >&2

    # --- MCHEP Scalar ---
    MCHEP_TIMES=()
    for i in $(seq 1 $N_RUNS); do
        OUTPUT=$(./benchmark_mchep_accuracy $TARGET_ACC)
        TIME_MS=$(echo "$OUTPUT" | cut -d',' -f6)
        MCHEP_TIMES+=($TIME_MS)
    done
    MCHEP_AVG_TIME=$(echo "${MCHEP_TIMES[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; print s/NF}')
    # Get last run's full output for values
    LAST_OUTPUT=$(./benchmark_mchep_accuracy $TARGET_ACC)
    VALUE=$(echo "$LAST_OUTPUT" | cut -d',' -f3)
    ERROR=$(echo "$LAST_OUTPUT" | cut -d',' -f4)
    ACTUAL_ACC=$(echo "$LAST_OUTPUT" | cut -d',' -f5)
    CHI2=$(echo "$LAST_OUTPUT" | cut -d',' -f7)
    echo "MCHEP_Scalar,$TARGET_ACC,$VALUE,$ERROR,$ACTUAL_ACC,$MCHEP_AVG_TIME,chi2=$CHI2"

    # --- MCHEP SIMD ---
    MCHEP_SIMD_TIMES=()
    for i in $(seq 1 $N_RUNS); do
        OUTPUT=$(./benchmark_mchep_simd_accuracy $TARGET_ACC)
        TIME_MS=$(echo "$OUTPUT" | cut -d',' -f6)
        MCHEP_SIMD_TIMES+=($TIME_MS)
    done
    MCHEP_SIMD_AVG_TIME=$(echo "${MCHEP_SIMD_TIMES[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; print s/NF}')
    # Get last run's full output for values
    LAST_OUTPUT=$(./benchmark_mchep_simd_accuracy $TARGET_ACC)
    VALUE=$(echo "$LAST_OUTPUT" | cut -d',' -f3)
    ERROR=$(echo "$LAST_OUTPUT" | cut -d',' -f4)
    ACTUAL_ACC=$(echo "$LAST_OUTPUT" | cut -d',' -f5)
    CHI2=$(echo "$LAST_OUTPUT" | cut -d',' -f7)
    echo "MCHEP_SIMD,$TARGET_ACC,$VALUE,$ERROR,$ACTUAL_ACC,$MCHEP_SIMD_AVG_TIME,chi2=$CHI2"

    # --- CUBA Vegas ---
    CUBA_TIMES=()
    for i in $(seq 1 $N_RUNS); do
        OUTPUT=$(./cuba_accuracy $TARGET_ACC)
        TIME_MS=$(echo "$OUTPUT" | cut -d',' -f6)
        CUBA_TIMES+=($TIME_MS)
    done
    CUBA_AVG_TIME=$(echo "${CUBA_TIMES[@]}" | awk '{for(i=1;i<=NF;i++)s+=$i; print s/NF}')
    # Get last run's full output for values
    LAST_OUTPUT=$(./cuba_accuracy $TARGET_ACC)
    VALUE=$(echo "$LAST_OUTPUT" | cut -d',' -f3)
    ERROR=$(echo "$LAST_OUTPUT" | cut -d',' -f4)
    ACTUAL_ACC=$(echo "$LAST_OUTPUT" | cut -d',' -f5)
    NEVAL=$(echo "$LAST_OUTPUT" | cut -d',' -f7)
    echo "CUBA_Vegas,$TARGET_ACC,$VALUE,$ERROR,$ACTUAL_ACC,$CUBA_AVG_TIME,neval=$NEVAL"
done

echo ""
echo "--- Benchmark Summary ---"
echo "Note: Exact integral value is 1.0"
echo "========================="
