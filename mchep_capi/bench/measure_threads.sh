#!/bin/bash

# Measure thread usage for each benchmark

echo "System: $(sysctl -n hw.ncpu) CPU cores available"
echo ""

measure_threads() {
    local name="$1"
    local cmd="$2"

    echo "--- $name ---"

    # Run command in background
    $cmd &
    PID=$!

    MAX_THREADS=0
    while kill -0 $PID 2>/dev/null; do
        # Count threads for this process (macOS)
        THREADS=$(ps -M -p $PID 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
        if [ -n "$THREADS" ] && [ "$THREADS" -gt "$MAX_THREADS" ]; then
            MAX_THREADS=$THREADS
        fi
        sleep 0.01
    done

    wait $PID
    echo "Max threads observed: $MAX_THREADS"
    echo ""
}

# Build first
echo "Building benchmarks..."
make benchmark_mchep benchmark_mchep_simd cuba_example 2>/dev/null
echo ""

# Measure each benchmark
measure_threads "MCHEP Scalar" "./benchmark_mchep"
measure_threads "MCHEP SIMD" "./benchmark_mchep_simd"
measure_threads "CUBA Vegas" "./cuba_example"

echo "=== Summary ==="
echo "Check RAYON_NUM_THREADS env var to control MCHEP parallelism"
echo "Current RAYON_NUM_THREADS: ${RAYON_NUM_THREADS:-not set (uses all cores)}"
