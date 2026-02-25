# MCHEP

**MCHEP** is a highly parallelizable Monte Carlo integration routine. Specifically, it
supports multi-threads/cores parallelization, Single Instruction Multiple Data (SIMD)
instructions, and GPU acceleration. Currently, it implements two adaptive multidimensional
integrations, namely VEGAS and VEGAS+ (with adaptive stratified sampling) as presented in
the paper [arXiv:2009.05112](https://arxiv.org/pdf/2009.05112).

## Performance

The following benchmark compares MCHEP against the [CUBA](https://feynarts.de/cuba/)
library across different integrand complexities, simulating typical High Energy Physics (HEP)
workloads:

![MCHEP vs CUBA Scaling Benchmark](benchmark_scaling.png)

**Left plot**: Integration throughput (evaluations per millisecond) vs. computational cost
per evaluation. MCHEP with SIMD+AVX consistently outperforms both MCHEP scalar and CUBA
across all complexity levels.

**Right plot**: Speedup factor of MCHEP implementations compared to CUBA Vegas.
MCHEP SIMD+AVX achieves **4-6x speedup** over CUBA for typical HEP workloads.

| QCD Complexity | Typical Cost (FLOPS) | MCHEP Scalar (evals/ms) | MCHEP SIMD+AVX (evals/ms) | CUBA (evals/ms) | Speedup vs CUBA |
|----------------|----------------------|-------------------------|---------------------------|-----------------|-----------------|
| LO             | ~10k                 | 2,500                   | 7,500                     | 1,800           | 4.2x            |
| NLO            | ~1M                  | 48                      | 153                       | 34              | 4.5x            |
| NNLO           | ~10M                 | 5                       | 18                        | 5               | 3.9x            |

Combined with multi-core parallelization (16 cores), MCHEP can reduce month-long calculations
to days.

## Feature Availability

| Feature | Rust API (Vegas) | Rust API (VegasPlus) | C/C++ API (Vegas) | C/C++ API (VegasPlus) | Python API (Vegas) | Python API (VegasPlus) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Multi-threaded (Rayon) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SIMD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| GPU | ✓ | ✓ | | | | |
| MPI | | ✓ | | ✓ | | ✓ |
| MPI+SIMD | | ✓ | | ✓ | | |
